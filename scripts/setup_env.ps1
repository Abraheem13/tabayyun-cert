# One-time setup on the RTX 5080 machine (Windows PowerShell).
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python scripts\check_gpu.py | Tee-Object -FilePath results\env.txt
pytest -q
