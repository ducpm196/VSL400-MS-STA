# OS-level watchdog for MS-STA retrain-v3 (independent of Cursor).
# Snapshot every LoopSeconds; resume orchestrator at most MaxRestarts times.
# Restart count is persisted on disk. Never resume while an old process tree is alive.
param(
  [string]$Policy = "Docs\results\ms_sta_retrain_v3_policy\RETRAIN_POLICY.json",
  [int]$LoopSeconds = 600,
  [int]$MaxRestarts = 2,
  [string]$Phase = "hpo_screen",
  [string]$Python = ".\_venvs\train_cuda\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$StatusDir = Join-Path $Root "logs\ms_sta_retrain_v3"
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null
$WatchLog = Join-Path $StatusDir "watchdog.log"
$RestartStatePath = Join-Path $StatusDir "watchdog_restarts.json"

function Write-Watch([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date).ToString("o"), $msg
  Add-Content -Path $WatchLog -Value $line
  Write-Host $line
}

function Get-RestartCount {
  if (Test-Path $RestartStatePath) {
    $obj = Get-Content $RestartStatePath -Raw | ConvertFrom-Json
    return [int]$obj.restart_count
  }
  return 0
}

function Set-RestartCount([int]$count, [string]$reason) {
  $payload = @{
    restart_count = $count
    max_restarts = $MaxRestarts
    updated_at = (Get-Date).ToString("o")
    last_reason = $reason
  } | ConvertTo-Json
  Set-Content -Path $RestartStatePath -Value $payload -Encoding UTF8
}

function Get-DescendantPids([int]$ParentId) {
  $all = @()
  $queue = New-Object System.Collections.Generic.Queue[int]
  $queue.Enqueue($ParentId)
  while ($queue.Count -gt 0) {
    $curId = $queue.Dequeue()
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$curId" -ErrorAction SilentlyContinue
    foreach ($c in $children) {
      $cid = [int]$c.ProcessId
      if ($all -notcontains $cid) {
        $all += $cid
        $queue.Enqueue($cid)
      }
    }
  }
  return $all
}

function Test-PidAlive([int]$ProcessId) {
  if ($ProcessId -le 0) { return $false }
  try {
    $p = Get-Process -Id $ProcessId -ErrorAction Stop
    return $null -ne $p
  } catch {
    return $false
  }
}

function Get-LiveTrainerPids {
  $pids = @()
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
  foreach ($p in $procs) {
    if ($p.CommandLine -and ($p.CommandLine -match 'train_ms_sta_vsl400')) {
      $pids += [int]$p.ProcessId
    }
  }
  return $pids
}

function Wait-ProcessTreeDead([int]$RootPid, [int]$TimeoutSec = 600) {
  if ($RootPid -le 0) { return $true }
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    $alive = @()
    if (Test-PidAlive $RootPid) { $alive += $RootPid }
    foreach ($cid in (Get-DescendantPids $RootPid)) {
      if (Test-PidAlive $cid) { $alive += $cid }
    }
    if ($alive.Count -eq 0) { return $true }
    Write-Watch ("waiting for process tree to exit: $($alive -join ',')")
    Start-Sleep -Seconds 5
  }
  return $false
}

$RestartCount = Get-RestartCount
Write-Watch "watchdog start policy=$Policy loop=$LoopSeconds maxRestarts=$MaxRestarts phase=$Phase restart_count=$RestartCount"

while ($true) {
  try {
    & $Python "scripts\monitor_ms_sta_retrain_v3.py" --policy $Policy | Out-Null
    $latestPath = Join-Path $StatusDir "monitor_latest.json"
    if (-not (Test-Path $latestPath)) {
      Write-Watch "missing monitor_latest.json"
      Start-Sleep -Seconds $LoopSeconds
      continue
    }
    $snap = Get-Content $latestPath -Raw | ConvertFrom-Json
    Write-Watch ("verdict={0} gpu={1} disk={2}" -f $snap.verdict, $snap.gpu.util_pct, $snap.disk_free_gb)

    if ($snap.verdict -eq "COMPLETED") {
      Write-Watch "COMPLETED; exiting watchdog"
      break
    }

    if ($snap.verdict -eq "STALLED" -or $snap.verdict -eq "FAILED") {
      $RestartCount = Get-RestartCount
      if ($RestartCount -ge $MaxRestarts) {
        Write-Watch "BLOCKED after $MaxRestarts restarts; last verdict=$($snap.verdict)"
        break
      }

      $pidInfoPath = Join-Path $StatusDir "orchestrator.pid.json"
      $oldPid = 0
      if (Test-Path $pidInfoPath) {
        $pidInfo = Get-Content $pidInfoPath -Raw | ConvertFrom-Json
        $oldPid = [int]$pidInfo.pid
      }

      # Refuse resume while any process in the old tree is still alive.
      if ($oldPid -gt 0) {
        $stillAlive = @()
        if (Test-PidAlive $oldPid) { $stillAlive += $oldPid }
        foreach ($cid in (Get-DescendantPids $oldPid)) {
          if (Test-PidAlive $cid) { $stillAlive += $cid }
        }
        if ($stillAlive.Count -gt 0) {
          Write-Watch ("refuse resume; live process tree: $($stillAlive -join ',')")
          Start-Sleep -Seconds $LoopSeconds
          continue
        }
        $dead = Wait-ProcessTreeDead -RootPid $oldPid -TimeoutSec 30
        if (-not $dead) {
          Write-Watch "refuse resume; timed out waiting for old tree"
          Start-Sleep -Seconds $LoopSeconds
          continue
        }
      }

      # Also refuse if an orphan trainer still holds the GPU (parent orchestrator died).
      $trainerPids = Get-LiveTrainerPids
      if ($trainerPids.Count -gt 0) {
        Write-Watch ("refuse resume; live trainer pids: $($trainerPids -join ',')")
        Start-Sleep -Seconds $LoopSeconds
        continue
      }

      # Never change hyperparameters; only resume orchestrator.
      $RestartCount++
      Set-RestartCount -count $RestartCount -reason $snap.verdict
      Write-Watch "attempting --resume restart #$RestartCount"
      # UseShellExecute breakaway so watchdog-spawned orch survives parent shell/job teardown.
      $proc = Start-Process -FilePath $Python -ArgumentList @(
        "-u", "scripts\run_ms_sta_retrain_v3.py",
        "--phase", $Phase,
        "--policy", $Policy,
        "--resume"
      ) -PassThru -WindowStyle Hidden -WorkingDirectory $Root
      Write-Watch "restarted orchestrator pid=$($proc.Id)"
    }
  }
  catch {
    Write-Watch ("monitor error: {0}" -f $_.Exception.Message)
  }
  Start-Sleep -Seconds $LoopSeconds
}
