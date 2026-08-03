@echo off
title Background Remover
cd /d "%~dp0"
echo Starting Background Remover...
echo.
echo When it says "Running on http://127.0.0.1:5000", open that link in your browser.
echo Close this window to stop the app.
echo.
python app.py
pause
