# Field Kit live v1: set up, build the knowledge pack (first run only), and start the server.
#   .\run.ps1                 start (builds the pack automatically if it's missing)
#   .\run.ps1 -Ingest         rebuild the knowledge pack
#   .\run.ps1 -MaxDocs 5000   smaller/faster pack
param(
    [switch]$Ingest,
    [int]$MaxDocs = 20000,
    [string]$Model = "smollm2-1.7b",
    [string]$Venv = "$env:USERPROFILE\.venvs\llm-chat"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating Python environment in $Venv"
    python -m venv $Venv
    & $py -m pip install --upgrade pip
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}
& $py -m pip install -q -r (Join-Path $here "requirements.txt")

$env:DATA_DIR = Join-Path $here "data"
$env:CHAT_MODEL = $Model
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

if ($Ingest -or -not (Test-Path (Join-Path $env:DATA_DIR "vectors.npy"))) {
    Write-Host "Building the knowledge pack ($MaxDocs articles of Simple English Wikipedia)..."
    $env:KNOWLEDGE_MAX_DOCS = "$MaxDocs"
    & $py (Join-Path $here "ingest.py")
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.PrefixOrigin -in 'Dhcp', 'Manual' -and $_.IPAddress -notlike '169.*' -and
                      $_.InterfaceAlias -notmatch 'vEthernet|WSL|VPN|Loopback' } |
       Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "Field Kit is starting:  http://localhost:8080"
if ($ip) { Write-Host "From a phone on the same Wi-Fi:  http://${ip}:8080  (allow port 8080 in Windows Firewall)" }
& $py (Join-Path $here "server.py")
