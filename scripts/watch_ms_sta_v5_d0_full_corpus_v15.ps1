# Continuous watchdog for D0 full-corpus v15.
# Cold-start order (fail-closed):
#   1) read-only monitor (no binary-root writes)
#   2) launch runner first while binary root absent/empty
#   3) wait bounded time for runner-owned immutable _run_contract.json
#   4) validate v15 identity
#   5) only then persist WATCHDOG_STATE / MONITOR_TICK
# Resume only when prior process tree is dead. Never start a second worker.
# Does NOT launch extraction unless -Launch is passed.
param(
  [switch]$Launch,
  [switch]$Resume,
  [string]$Device = "cuda",
  [int]$IntervalSec = 600,
  [int]$MaxRestarts = 20,
  [int]$ContractWaitSec = 180,
  [int]$ContractPollSec = 2,
  [switch]$Once,
  # Test overrides (optional; production leaves unset)
  [string]$RootOverride = "",
  [string]$F0Override = "",
  [string]$RunnerOverride = "",
  [string]$MonitorOverride = "",
  [string]$ApprovalRelOverride = ""
)

$ErrorActionPreference = "Stop"
if ($RootOverride) {
  $Root = $RootOverride
} else {
  $Root = Split-Path -Parent $PSScriptRoot
}
Set-Location $Root

$F0 = if ($F0Override) { $F0Override } else { Join-Path $Root "_venvs\f0_v5\Scripts\python.exe" }
$Runner = if ($RunnerOverride) { $RunnerOverride } else { Join-Path $Root "scripts\run_ms_sta_v5_d0_full_corpus.py" }
$Monitor = if ($MonitorOverride) { $MonitorOverride } else { Join-Path $Root "scripts\monitor_ms_sta_v5_d0_full_corpus.py" }
$CampaignToken = "D0_FULL_CORPUS_V15"
$OutputVersion = "v15"
$EvidenceV15 = Join-Path $Root "Docs\results\ms_sta_v5_d0_full\v15"
$ApprovalRel = if ($ApprovalRelOverride) { $ApprovalRelOverride } else { "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_APPROVAL_V15.json" }

if (-not (Test-Path $F0)) { throw "missing_f0_python" }

# In-memory restart state until a valid immutable contract exists (no pre-contract disk writes).
$script:MemRestartState = $null

function New-RestartState {
  return [pscustomobject]@{
    restart_count = 0
    last_action   = $null
    last_at       = $null
  }
}

function Get-Monitor {
  param([switch]$WriteTick)
  # Exact v15 identity only — never fall back to hermetic v14 defaults.
  # Read-only by default: --write-tick only when caller opts in after valid contract.
  $monitorArgs = @($Monitor, "--json", "--output-version", $OutputVersion, "--campaign-token", $CampaignToken)
  if ($WriteTick) { $monitorArgs += "--write-tick" }
  $raw = & $F0 @monitorArgs 2>$null
  if ($LASTEXITCODE -ne 0) { throw "monitor_failed:$LASTEXITCODE" }
  $mon = ($raw | ConvertFrom-Json)
  if ($mon.campaign_token -ne $CampaignToken) {
    throw "monitor_identity_mismatch:campaign_token=$($mon.campaign_token)"
  }
  if ($mon.output_version -ne $OutputVersion) {
    throw "monitor_identity_mismatch:output_version=$($mon.output_version)"
  }
  if ($mon.binary_root -match 'ms_sta_v5_d0_full_v14_') {
    throw "monitor_v15_refuses_v14_binary_root"
  }
  return $mon
}

function Get-RestartState([string]$RuntimeDir) {
  $path = Join-Path $RuntimeDir "WATCHDOG_STATE.json"
  if (Test-Path $path) {
    $obj = (Get-Content -Raw $path | ConvertFrom-Json)
    # Normalize schema so last_at assignment never fails on older files.
    $count = 0
    if ($null -ne $obj.restart_count) { $count = [int]$obj.restart_count }
    return [pscustomobject]@{
      restart_count = $count
      last_action   = $(if ($null -ne $obj.last_action) { $obj.last_action } else { $null })
      last_at       = $(if ($null -ne $obj.last_at) { $obj.last_at } else { $null })
    }
  }
  if ($null -ne $script:MemRestartState) {
    return $script:MemRestartState
  }
  return New-RestartState
}

function Save-RestartState([string]$RuntimeDir, $state) {
  if (-not (Test-Path (Join-Path $RuntimeDir "_run_contract.json"))) {
    throw "watchdog_state_requires_run_contract"
  }
  $path = Join-Path $RuntimeDir "WATCHDOG_STATE.json"
  New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
  ($state | ConvertTo-Json -Depth 5) | Set-Content -Path $path -Encoding utf8NoBOM
}

function Test-BinaryRootAbsentOrEmpty([string]$BinaryRoot) {
  if (-not (Test-Path $BinaryRoot)) { return $true }
  $files = @(Get-ChildItem -LiteralPath $BinaryRoot -Recurse -File -ErrorAction SilentlyContinue)
  return ($files.Count -eq 0)
}

function Get-ApprovalObject {
  $path = Join-Path $Root $ApprovalRel
  if (-not (Test-Path $path)) { throw "approval_v15_missing:$path" }
  return (Get-Content -Raw $path | ConvertFrom-Json)
}

function Test-ValidRunContract([string]$RuntimeDir) {
  $contractPath = Join-Path $RuntimeDir "_run_contract.json"
  if (-not (Test-Path $contractPath)) { return $false }
  $contract = Get-Content -Raw $contractPath | ConvertFrom-Json
  $approval = Get-ApprovalObject
  if ($contract.schema -ne "ms_sta_v5_d0_full_corpus_immutable_run_contract_v15") { return $false }
  if ($contract.campaign_token -ne $CampaignToken) { return $false }
  if ($contract.output_version -ne $OutputVersion) { return $false }
  if ($contract.code_lock_git_revision -ne $approval.code_lock_git_revision) { return $false }
  if ($contract.executable_tree_sha256 -ne $approval.executable_tree_sha256) { return $false }
  $cBin = [string]$contract.binary_root_rel
  $aBin = [string]$approval.binary_root_rel
  if ($cBin.Replace("\", "/") -ne $aBin.Replace("\", "/")) { return $false }
  return $true
}

function Wait-ForValidRunContract {
  param(
    $Proc,
    [string]$RuntimeDir,
    [int]$WaitSec,
    [int]$PollSec
  )
  $deadline = (Get-Date).AddSeconds($WaitSec)
  while ($true) {
    if (Test-ValidRunContract $RuntimeDir) { return $true }
    if ($Proc.HasExited) { return $false }
    if ((Get-Date) -ge $deadline) { return $false }
    Start-Sleep -Seconds $PollSec
  }
}

function Assert-NoSecondWorker($Mon) {
  if ($Mon.live_orchestrator) {
    throw "live_process_tree_blocks_second_worker:pid=$($Mon.orchestrator_pid)"
  }
}

if (-not $Launch) {
  $mon = Get-Monitor
  Write-Host "watchdog_v15_ready_no_launch"
  Write-Host ($mon | ConvertTo-Json -Depth 6)
  exit 0
}

while ($true) {
  $mon = Get-Monitor
  $runtimeDir = [string]$mon.runtime_dir
  $binaryRoot = [string]$mon.binary_root
  $hasContract = Test-ValidRunContract $runtimeDir

  if ($hasContract) {
    # Contract exists: ticks/state persistence allowed.
    $mon = Get-Monitor -WriteTick
    $state = Get-RestartState $runtimeDir
  } else {
    $state = Get-RestartState $runtimeDir
  }

  $haltPath = Join-Path $runtimeDir "HALT.json"
  $detFailPath = Join-Path $runtimeDir "DETERMINISTIC_PHASE_FAILURE.json"
  $phaseFailPath = Join-Path $runtimeDir "PHASE_FAILURE.json"
  if (Test-Path $haltPath) {
    Write-Host "halt_sentinel_present_no_restart"
    exit 0
  }
  if (Test-Path $detFailPath) {
    Write-Host "deterministic_phase_failure_no_restart"
    exit 3
  }
  if (Test-Path $phaseFailPath) {
    Write-Host "phase_failure_no_restart"
    exit 3
  }
  if ($mon.startup_stall_alert -eq $true) {
    Write-Host "startup_stall_material_no_restart"
    exit 3
  }
  if ($mon.run_state -and $mon.run_state.status -match 'HALTED_') {
    Write-Host "halted_status_no_restart status=$($mon.run_state.status)"
    exit 0
  }

  if ($mon.run_state -and $mon.run_state.status -match 'CLOSED_') {
    Write-Host "completion_detected_verify_only"
    $completion = Join-Path $runtimeDir "COMPLETION.json"
    $seal = Join-Path $EvidenceV15 "D0_FULL_CORPUS_COMPLETION_SEAL.json"
    if ((Test-Path $completion) -and (Test-Path $seal)) {
      Write-Host "watchdog_idle_completed"
      exit 0
    }
    Write-Host "completion_missing_heal_scan"
    & $F0 $Runner --phase quality_scan --launch --bound-worker --resume --campaign-token $CampaignToken
    exit $LASTEXITCODE
  }

  if ($mon.live_orchestrator) {
    Write-Host "live_process_tree_blocks_watchdog pid=$($mon.orchestrator_pid)"
    if ($Once) { exit 2 }
    Start-Sleep -Seconds $IntervalSec
    continue
  }

  if ([int]$state.restart_count -ge $MaxRestarts) {
    Write-Host "max_restarts_exceeded:$($state.restart_count)"
    exit 2
  }

  if (-not $hasContract) {
    # --- Cold start: runner first; no binary-root / watchdog writes beforehand ---
    if (-not (Test-BinaryRootAbsentOrEmpty $binaryRoot)) {
      Write-Host "cold_start_refused_binary_root_not_empty:$binaryRoot"
      exit 2
    }
    $state.restart_count = [int]$state.restart_count + 1
    $state.last_action = "cold_start_launch"
    $state.last_at = (Get-Date).ToUniversalTime().ToString("o")
    $script:MemRestartState = $state
    # Explicitly do NOT Save-RestartState / write-tick before contract.

    $runnerArgs = @($Runner, "--phase", "all", "--device", $Device, "--launch", "--campaign-token", $CampaignToken)
    Write-Host "watchdog_v15_cold_start_restart=$($state.restart_count)"
    $proc = Start-Process -FilePath $F0 -ArgumentList $runnerArgs -PassThru -WorkingDirectory $Root
    $gotContract = Wait-ForValidRunContract -Proc $proc -RuntimeDir $runtimeDir -WaitSec $ContractWaitSec -PollSec $ContractPollSec
    if (-not $gotContract) {
      Write-Host "cold_start_runner_failed_before_contract"
      if (-not $proc.HasExited) {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      }
      # Fail closed: do not fabricate runtime state; do not retry indefinitely.
      exit 3
    }
    if (-not (Test-ValidRunContract $runtimeDir)) {
      Write-Host "cold_start_contract_identity_invalid"
      exit 3
    }
    Save-RestartState $runtimeDir $state
    $script:MemRestartState = $null
    Write-Host "watchdog_v15_contract_bound"
  } else {
    # --- Resume: attach to existing contract; never second worker ---
    Assert-NoSecondWorker $mon
    $runnerArgs = @($Runner, "--phase", "all", "--device", $Device, "--launch", "--campaign-token", $CampaignToken, "--resume")
    $state.restart_count = [int]$state.restart_count + 1
    $state.last_action = "resume_launch"
    $state.last_at = (Get-Date).ToUniversalTime().ToString("o")
    Save-RestartState $runtimeDir $state
    Write-Host "watchdog_v15_resume_restart=$($state.restart_count)"
    $proc = Start-Process -FilePath $F0 -ArgumentList $runnerArgs -PassThru -WorkingDirectory $Root
  }

  while (-not $proc.HasExited) {
    $mon = Get-Monitor -WriteTick
    Write-Host "watchdog_tick phase=$($mon.phase) disk_free_gb=$($mon.disk_free_gb) workers_alive=$($mon.worker_registry.alive)"
    if (Test-Path (Join-Path $mon.runtime_dir "DETERMINISTIC_PHASE_FAILURE.json")) {
      Write-Host "deterministic_phase_failure_mid_run_no_restart"
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      exit 3
    }
    if (Test-Path (Join-Path $mon.runtime_dir "PHASE_FAILURE.json")) {
      Write-Host "phase_failure_mid_run_no_restart"
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      exit 3
    }
    if ($mon.startup_stall_alert -eq $true) {
      Write-Host "startup_stall_mid_run_no_restart"
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      exit 3
    }
    if (Test-Path (Join-Path $mon.runtime_dir "HALT.json")) {
      Write-Host "halt_sentinel_mid_run_no_restart"
      try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
      exit 0
    }
    Start-Sleep -Seconds $IntervalSec
  }
  $code = $proc.ExitCode
  if (Test-Path (Join-Path $runtimeDir "DETERMINISTIC_PHASE_FAILURE.json")) {
    Write-Host "deterministic_phase_failure_after_exit_no_restart code=$code"
    exit 3
  }
  if (Test-Path (Join-Path $runtimeDir "HALT.json")) {
    Write-Host "halt_sentinel_after_exit_no_restart code=$code"
    exit 0
  }
  if ($code -eq 0) {
    Write-Host "watchdog_runner_exit_0"
    if ($Once) { exit 0 }
    Start-Sleep -Seconds $IntervalSec
    continue
  }
  Write-Host "watchdog_runner_exit_$code"
  if ($Once) { exit $code }
  Start-Sleep -Seconds $IntervalSec
}
