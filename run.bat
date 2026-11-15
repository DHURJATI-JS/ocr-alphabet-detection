@echo off
rem Start the server. Uses port 8000, or the next free one if 8000 is taken.
rem Settings come from the environment or .env (see .env.example). Extra args: --port, --reload
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo No virtualenv yet - run setup.bat first.
  exit /b 1
)
.venv\Scripts\python -m app %*
