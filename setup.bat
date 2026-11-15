@echo off
rem One-time setup on Windows: virtualenv, dependencies, Tesseract model.
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv || goto :error
)
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements-dev.txt || goto :error
.venv\Scripts\python scripts\download_models.py || goto :error
echo.
echo Setup complete. Start the app with run.bat and open http://localhost:8000
exit /b 0
:error
echo Setup failed.
exit /b 1
