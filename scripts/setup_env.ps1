# One-time setup on the RTX 5080 machine (Windows PowerShell).
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe scripts\check_gpu.py
Write-Host "`nSetup done. Next:  .\.venv\Scripts\python.exe scripts\run_day.py 1"
