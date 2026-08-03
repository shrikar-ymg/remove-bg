@echo off
REM Creates a Desktop shortcut named "Background Removal Studio" that launches the app.
setlocal
set "APPDIR=%~dp0"
set "TARGET=%APPDIR%start.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\Background Removal Studio.lnk"

powershell -NoProfile -Command ^
  "$w = New-Object -ComObject WScript.Shell;" ^
  "$s = $w.CreateShortcut('%SHORTCUT%');" ^
  "$s.TargetPath = '%TARGET%';" ^
  "$s.WorkingDirectory = '%APPDIR%';" ^
  "$s.IconLocation = 'shell32.dll,238';" ^
  "$s.Description = 'AI background removal - local web app';" ^
  "$s.Save()"

echo.
echo Created shortcut on your Desktop: "Background Removal Studio"
echo Double-click it any time to start the app.
echo.
pause
