@echo off
rem Builds DocExtractor.exe (folder: dist\DocExtractor). Python is needed ONLY on the machine that builds it.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 - 3.12 is required to BUILD the exe. Install it from python.org, or use the GitHub build described in README.md
  pause
  exit /b 1
)

py -3 -m venv .venv || goto :fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :fail
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :fail
pip install -r requirements.txt pyinstaller || goto :fail
python prefetch_models.py || goto :fail
pyinstaller --noconfirm --clean --onedir --name DocExtractor ^
  --add-data "static;static" --add-data "models;models" ^
  --collect-all easyocr server.py || goto :fail

echo.
echo Done. Run: dist\DocExtractor\DocExtractor.exe
pause
exit /b 0

:fail
echo Build failed - see the messages above.
pause
exit /b 1
