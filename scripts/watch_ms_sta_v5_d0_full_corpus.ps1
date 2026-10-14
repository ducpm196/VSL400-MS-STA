# Continuous watchdog for D0 full-corpus v14.
# Loop every 10 minutes. Resume only when prior process tree is dead.
# After completion: verify/heal only — never re-infer.
# Does NOT launch extraction unless -Launch is passed.
param(
  [switch]$Launch,
  [switch]$Resume,
  [string]$Device = "cuda",
  [int]$IntervalSec = 600,
  [int]$MaxRestarts = 20,
  [switch]$Once
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$F0 = Join-Path $Root "_venvs\f0_v5\Scripts\python.exe"
$Runner = Join-Path $Root "scripts\run_ms_sta_v5_d0_full_corpus.py"
$Monitor = Join-Path $Root "scripts\monitor_ms_sta_v5_d0_full_corpus.py"
$CampaignToken = "D0_FULL_CORPUS_V14"
$EvidenceV14 = Join-Path $Root "Docs\results\ms_sta_v5_d0_full\v14"

if (-not (Test-Path $F0)) { throw "missing_f0_python" }

function Get-Monitor {
  $raw = & $F0 $Monitor --json --write-tick 2>$null
  if ($LASTEXITCODE -ne 0) { throw "monitor_failed:$LASTEXITCODE" }
  return ($raw | ConvertFrom-Json)
}

function Get-RestartState($mon) {
  $path = Join-Path $mon.runtime_dir "WATCHDOG_STATE.json"
  if (Test-Path $path) {
    return (Get-Content -Raw $path | ConvertFrom-Json)
  }
  return [pscustomobject]@{ restart_count = 0; last_action = $null }
}

function Save-RestartState($mon, $state) {
  $path = Join-Path $mon.runtime_dir "WATCHDOG_STATE.json"
  New-Item -ItemType Directory -Force -Path $mon.runtime_dir | Out-Null
  ($state | ConvertTo-Json -Depth 5) | Set-Content -Path $path -Encoding utf8NoBOM
}

if (-not $Launch) {
  $mon = Get-Monitor
  Write-Host "watchdog_ready_no_launch"
  Write-Host ($mon | ConvertTo-Json -Depth 6)
  exit 0
}

while ($true) {
  $mon = Get-Monitor
  $state = Get-RestartState $mon

  $haltPath = Join-Path $mon.runtime_dir "HALT.json"
  $detFailPath = Join-Path $mon.runtime_dir "DETERMINISTIC_PHASE_FAILURE.json"
  if (Test-Path $haltPath) {
    Write-Host "halt_sentinel_present_no_restart"
    exit 0
  }
  if (Test-Path $detFailPath) {
    Write-Host "deterministic_phase_failure_no_restart"
    exit 3
  }
  if ($mon.run_state -and $mon.run_state.status -match 'HALTED_') {
    Write-Host "halted_status_no_restart status=$($mon.run_state.status)"
    exit 0
  }

  if ($mon.run_state -and $mon.run_state.status -match 'CLOSED_') {
    Write-Host "completion_detected_verify_only"
    $completion = Join-Path $mon.runtime_dir "COMPLETION.json"
    $seal = Join-Path $EvidenceV14 "D0_FULL_CORPUS_COMPLETION_SEAL.json"
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

  $runnerArgs = @($Runner, "--phase", "all", "--device", $Device, "--launch", "--campaign-token", $CampaignToken)
  if ($Resume -or $mon.run_contract_exists) { $runnerArgs += "--resume" }

  $state.restart_count = [int]$state.restart_count + 1
  $state.last_action = "launch_or_resume"
  $state.last_at = (Get-Date).ToUniversalTime().ToString("o")
  Save-RestartState $mon $state

  Write-Host "watchdog_start_restart=$($state.restart_count)"
  $proc = Start-Process -FilePath $F0 -ArgumentList $runnerArgs -PassThru -WorkingDirectory $Root
  while (-not $proc.HasExited) {
    $mon = Get-Monitor
    Write-Host "watchdog_tick phase=$($mon.phase) disk_free_gb=$($mon.disk_free_gb) workers_alive=$($mon.worker_registry.alive)"
    Start-Sleep -Seconds $IntervalSec
  }
  $code = $proc.ExitCode
  if (Test-Path (Join-Path $mon.runtime_dir "DETERMINISTIC_PHASE_FAILURE.json")) {
    Write-Host "deterministic_phase_failure_after_exit_no_restart code=$code"
    exit 3
  }
  if (Test-Path (Join-Path $mon.runtime_dir "HALT.json")) {
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
