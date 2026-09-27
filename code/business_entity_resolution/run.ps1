# One-command end-to-end run on Windows.
#   powershell -ExecutionPolicy Bypass -File run.ps1
#   powershell -ExecutionPolicy Bypass -File run.ps1 --from stage2_fit
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
python src/run_all.py --check
if ($LASTEXITCODE -ne 0) { exit 1 }
python src/run_all.py @args
