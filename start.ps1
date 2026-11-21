# Sound Spy: start the whole app (backend + the built page) and open it in the browser.
#
#   Right-click > "Run with PowerShell", or in a terminal:  .\start.ps1
#   (if scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\start.ps1)
#
# Runs `soundspy app` from the project's own virtual environment (backend\.venv): no activation,
# no admin rights; the project path may contain spaces. Extra arguments are passed on, e.g.
#   .\start.ps1 --no-browser
# Stop the app with Ctrl+C in this window.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root "backend\.venv\Scripts\soundspy.exe"
$page = Join-Path $root "frontend\dist\index.html"

if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "The backend is not installed yet. Once, in this folder:" -ForegroundColor Yellow
    Write-Host "  cd backend; py -3.11 -m venv .venv; .venv\Scripts\pip install -e `".[api,data,ml,spatial,dev]`""
    exit 1
}
if (-not (Test-Path -LiteralPath $page)) {
    Write-Host "The page is not built yet. Once (and after changing it), in this folder:" -ForegroundColor Yellow
    Write-Host "  cd frontend; npm install; npm run build"
    exit 1
}

Set-Location -LiteralPath (Join-Path $root "backend")
& $exe app @args
exit $LASTEXITCODE
