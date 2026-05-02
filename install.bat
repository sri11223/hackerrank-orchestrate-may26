@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
for /f "usebackq delims=" %%D in (`py -3 -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user') or sysconfig.get_path('scripts'))" 2^>nul`) do set "TRIAGE_SCRIPTS=%%D"
if not defined TRIAGE_SCRIPTS (
  for /f "usebackq delims=" %%D in (`python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user') or sysconfig.get_path('scripts'))" 2^>nul`) do set "TRIAGE_SCRIPTS=%%D"
)
if defined TRIAGE_SCRIPTS (
  echo %PATH% | find /I "%TRIAGE_SCRIPTS%" >nul || set "PATH=%PATH%;%TRIAGE_SCRIPTS%"
)
where triage >nul 2>nul && triage --help
