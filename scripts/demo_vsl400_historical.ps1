# EXPLICIT historical Defense-GO demo only. Not primary SoT.
# Requires -AllowHistoricalDefense and temporarily points MS_STA_REGISTRY_PATH
# at data/processed/ms_sta_registry.json (historical_not_primary).
param(
  [Parameter(Mandatory = $true)]
  [switch]$AllowHistoricalDefense
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $AllowHistoricalDefense) {
  throw "Refusing historical demo without -AllowHistoricalDefense"
}

Write-Host "WARNING: historical Defense-GO demo — NOT canonical after retrain-v3 VALIDATION_NO_GO."

foreach ($rel in @(
  "data\processed\ms_sta_defense\b4_seed42_defense.pth",
  "data\processed\vsl400_classifier.pth",
  "data\processed\vsl400_labels.json",
  "data\processed\ms_sta_registry.json",
  "data\processed\demo_set_ms_sta.json",
  "metadata.csv"
)) {
  if (-not (Test-Path (Join-Path $Root $rel))) {
    throw ("Historical preflight missing: " + $rel)
  }
}

$env:MS_STA_REGISTRY_PATH = (Join-Path $Root "data\processed\ms_sta_registry.json")
$env:MS_STA_ALLOW_HISTORICAL_KPI = "1"

$py = $null
foreach ($venvName in @("env_demo", "_venvs\demo", "_venvs/demo")) {
  $candidate = Join-Path $Root ($venvName + "\Scripts\python.exe")
  $st = Join-Path $Root ($venvName + "\Scripts\streamlit.exe")
  if ((Test-Path $candidate) -and (Test-Path $st)) {
    & $candidate -m pip check
    if ($LASTEXITCODE -eq 0) { $py = $candidate; break }
  }
}
if (-not $py) { throw "Missing clean demo venv" }

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match 'uvicorn.*backend_api' -or $_.CommandLine -match 'streamlit.*app\.py') } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

$env:PYTHONPATH = $Root
Start-Process -FilePath $py -ArgumentList @(
  "-m", "uvicorn", "src.deployment.backend_api:app",
  "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $Root -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $Root "logs\demo_api_historical.log") `
  -RedirectStandardError (Join-Path $Root "logs\demo_api_historical.err.log")

$probe = Join-Path $Root "scripts\demo_health_probe.py"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 2
  & $py $probe --mode defense
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
}
if (-not $ok) { throw "Historical defense health failed" }
Write-Host "Historical defense API OK (not SoT). Starting Streamlit..."
& $py -m streamlit run (Join-Path $Root "src\deployment\app.py") --server.headless true
