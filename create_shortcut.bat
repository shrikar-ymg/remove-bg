@echo off
REM Creates a Desktop shortcut named "Cutout Studio" that launches the app.
setlocal
set "APPDIR=%~dp0"
set "TARGET=%APPDIR%start.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\Cutout Studio.lnk"

powershell -NoProfile -Command ^
  "$w = New-Object -ComObject WScript.Shell;" ^
  "$s = $w.CreateShortcut('%SHORTCUT%');" ^
  "$s.TargetPath = '%TARGET%';" ^
  "$s.WorkingDirectory = '%APPDIR%';" ^
  "$s.IconLocation = 'shell32.dll,238';" ^
  "$s.Description = 'Cutout Studio 2.2 - local AI background removal';" ^
  "$s.Save()"

echo.
echo Created shortcut on your Desktop: "Cutout Studio"
echo Double-click it any time to start the app.
echo.
pause
