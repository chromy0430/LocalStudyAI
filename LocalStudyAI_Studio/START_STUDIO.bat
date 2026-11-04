@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0\.."
set "HF_HOME=%CD%\models\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "TORCH_HOME=%CD%\models\torch"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo Run from an existing E:\LocalStudyAI project.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run "LocalStudyAI_Studio\studio_app.py" --server.address 127.0.0.1 --server.port 8501
pause
