@echo off
setlocal

cd /d "%~dp0"
set "EXIT_CODE=0"

if exist "%CD%\.venv\Scripts\python.exe" (
  "%CD%\.venv\Scripts\python.exe" "%CD%\solid_inpaint_ui.py"
  set "EXIT_CODE=%ERRORLEVEL%"
) else if exist "%CD%\..\.venv\Scripts\python.exe" (
  "%CD%\..\.venv\Scripts\python.exe" "%CD%\solid_inpaint_ui.py"
  set "EXIT_CODE=%ERRORLEVEL%"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 "%CD%\solid_inpaint_ui.py"
    set "EXIT_CODE=%ERRORLEVEL%"
  ) else (
    python "%CD%\solid_inpaint_ui.py"
    set "EXIT_CODE=%ERRORLEVEL%"
  )
)

if not "%EXIT_CODE%"=="0" (
  echo.
  echo 塗白啟動失敗，請先安裝依賴：
  echo   cd /d "%CD%"
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  echo.
  pause
)

exit /b %EXIT_CODE%
