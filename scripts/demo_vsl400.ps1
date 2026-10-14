# Start VSL400 demo (API + Streamlit).
# Default: verify the locked B4-V6 Defense GO registry and checkpoint.
# Historical Defense-GO demo only via: demo_vsl400_historical.ps1 -AllowHistoricalDefense
param(
  [switch]$AllowHistoricalDefense
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-Sha256([string]$Path) {
  $stream = [System.IO.File]::OpenRead($Path)
  $hasher = [System.Security.Cryptography.SHA256]::Create()
  try { return ([System.BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
  finally { $hasher.Dispose(); $stream.Dispose() }
}

$py = $null
$venvUsed = $null
foreach ($venvName in @("env_demo", "_venvs\demo", "_venvs/demo")) {
  $candidate = Join-Path $Root ($venvName + "\Scripts\python.exe")
  $st = Join-Path $Root ($venvName + "\Scripts\streamlit.exe")
  if ((Test-Path $candidate) -and (Test-Path $st)) {
    Write-Host ("pip check (" + $venvName + ")...")
    & $candidate -m pip check
    if ($LASTEXITCODE -ne 0) {
      Write-Host ("WARN: pip check failed for " + $venvName + " - skipping")
      continue
    }
    $py = $candidate
    $venvUsed = $venvName
    Write-Host ("Using venv: " + $venvName)
    break
  }
}
if (-not $py) {
  throw "Missing clean demo venv (env_demo / _venvs\demo). Do not use legacy env."
}

# A transferred demo bundle carries the pretrained backbone used by B0 startup.
$bundledTorch = Join-Path $Root "data\pretrained\torch"
if (Test-Path (Join-Path $bundledTorch "hub\checkpoints\SLOWFAST_8x8_R50.pyth")) {
  $env:TORCH_HOME = $bundledTorch
}
if (Test-Path (Join-Path $Root "SUBMISSION_MANIFEST.json")) {
  & $py (Join-Path $Root "scripts\verify_submission.py")
  if ($LASTEXITCODE -ne 0) { throw "Submission integrity check failed. Re-extract the verified bundle." }
}
foreach ($tool in @("ffmpeg", "ffprobe")) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    throw ("Missing " + $tool + " on PATH; required for browser video preview.")
  }
}

# Active registry is authoritative SoT (not historical ms_sta_registry.json).
$activePtr = Join-Path $Root "data\processed\ms_sta_registry_active.json"
if (-not (Test-Path $activePtr)) {
  throw "Missing active registry pointer: data/processed/ms_sta_registry_active.json"
}
$ptrJson = Get-Content -Raw -Encoding UTF8 $activePtr | ConvertFrom-Json
$regRel = [string]$ptrJson.registry_path
if (-not $regRel) { throw "Active pointer missing registry_path" }
$regPath = Join-Path $Root ($regRel -replace '/', '\')
if (-not (Test-Path $regPath)) { throw ("Active registry missing: " + $regRel) }
$reg = Get-Content -Raw -Encoding UTF8 $regPath | ConvertFrom-Json
$registrySha = Get-Sha256 $regPath
if ($registrySha -ne ([string]$ptrJson.registry_sha256).ToLowerInvariant()) {
  throw "Active registry SHA mismatch"
}
if ([string]$reg.schema -ne "ms_sta_registry_v6_defense_go_v1") {
  throw ("Defense demo requires v6 registry; got " + [string]$reg.schema)
}
if ([string]$reg.active_variant -ne "b4_v6_ctr") {
  throw ("Defense demo requires b4_v6_ctr; got " + [string]$reg.active_variant)
}
if ($reg.e4_ready -eq $true) {
  throw "Defense demo refuse: e4_ready must be false (E4 is appendix-only NO-GO)"
}
$ckptPath = Join-Path $Root (([string]$reg.checkpoint) -replace '/', '\')
if (-not (Test-Path -LiteralPath $ckptPath)) { throw ("Missing B4-V6 checkpoint: " + $ckptPath) }
$ckptSha = Get-Sha256 $ckptPath
if ($ckptSha -ne ([string]$reg.checkpoint_sha256).ToLowerInvariant()) {
  throw "B4-V6 checkpoint SHA mismatch"
}
if ($ckptSha -ne "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1") {
  throw "B4-V6 checkpoint SHA is not the locked Defense GO identity"
}
$env:MS_STA_REGISTRY_PATH = $regPath
$status = [string]($(if ($reg.campaign_status) { $reg.campaign_status } else { $reg.status }))
$demoAllowed = $false
if ($null -ne $reg.demo_allowed) { $demoAllowed = [bool]$reg.demo_allowed }
$defenseReady = $false
if ($null -ne $reg.defense_ready) { $defenseReady = [bool]$reg.defense_ready }

if ($status -eq "VALIDATION_NO_GO" -or (-not $demoAllowed) -or (-not $defenseReady)) {
  if (-not $AllowHistoricalDefense) {
    Write-Host "============================================================"
    Write-Host "DEMO REFUSED — campaign status: $status"
    Write-Host "Active registry: $regRel"
    Write-Host "Current-best (active): B4-V6 — see Docs/results/ms_sta_v6_defense_go/"
    Write-Host "No MS-STA defense checkpoint is approved for this registry state."
    Write-Host "Historical Defense-GO demo (provenance only / V5 rollback):"
    Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo_vsl400_historical.ps1 -AllowHistoricalDefense"
    Write-Host "============================================================"
    exit 2
  }
  Write-Host "WARN: -AllowHistoricalDefense set; proceeding is NOT canonical SoT."
}

$demoSetRel = if ($reg.demo_set_rel) { [string]$reg.demo_set_rel } else { "data/processed/demo_set_ms_sta.json" }
$demoSetPath = Join-Path $Root ($demoSetRel -replace '/', '\')
if (-not (Test-Path -LiteralPath $demoSetPath)) {
  throw ("Preflight missing demo set: " + $demoSetRel)
}
foreach ($rel in @(
  "data\processed\vsl400_classifier.pth",
  "data\processed\vsl400_labels.json",
  "data\processed\ms_sta_registry_active.json",
  "metadata.csv"
)) {
  if (-not (Test-Path (Join-Path $Root $rel))) {
    throw ("Preflight missing: " + $rel)
  }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

# Never stop unrelated Python services or another checkout's demo.
foreach ($port in @(8000, 8501)) {
  $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
  try { $listener.Start() }
  catch { throw ("Port " + $port + " is busy. Stop the existing service before starting this demo.") }
  finally { $listener.Stop() }
}

$env:PYTHONPATH = $Root

$apiProcess = Start-Process -FilePath $py -ArgumentList @(
  "-m", "uvicorn", "src.deployment.backend_api:app",
  "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $Root "logs\demo_api.log") `
  -RedirectStandardError (Join-Path $Root "logs\demo_api.err.log")

try {
Write-Host ("Waiting for API health [" + $venvUsed + "] — require B4-V6 Defense GO...")
$probe = Join-Path $Root "scripts\demo_health_probe.py"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
  if ($apiProcess.HasExited) { throw "API exited during startup. See logs/demo_api.err.log" }
  Start-Sleep -Seconds 2
  # isolated readiness only; defense mode must fail under NO_GO
  & $py $probe --mode defense
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
}
if (-not $ok) {
  throw "API health failed. See logs/demo_api.err.log"
}
Write-Host "API OK (B4-V6, exact checkpoint and cache identity; e4_ready=false). Starting Streamlit..."
$app = Join-Path $Root "src\deployment\app.py"
& $py -m streamlit run $app --server.headless true --server.port 8501
if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with an error." }
}
finally {
  # Only release the process created by this invocation.
  if (-not $apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id -ErrorAction SilentlyContinue }
}
