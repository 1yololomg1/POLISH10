@echo off
REM ============================================================
REM Build the standalone POLISH distribution for Windows.
REM
REM All build settings live in WirelinePreprocessing.spec. This script only
REM invokes it. Do not duplicate PyInstaller flags here: a previous version of
REM this script carried its own copy of the options, drifted out of sync with
REM the spec, and shipped a build that crashed on launch because of a stale
REM --optimize=2 flag.
REM
REM Output: dist\WirelinePreprocessing\  (a folder, not a single .exe)
REM         plus WirelinePreprocessing-windows.zip ready to send to a user.
REM ============================================================

setlocal
cd /d "%~dp0.."

echo ============================================
echo Building POLISH standalone distribution
echo ============================================
echo.

python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo Building. This takes roughly 15-20 minutes.
echo.

python -m PyInstaller --noconfirm --clean ^
    --distpath "distribution\dist" ^
    --workpath "build" ^
    "distribution\WirelinePreprocessing.spec"

if errorlevel 1 (
    echo.
    echo ERROR: Build failed.
    pause
    exit /b 1
)

if not exist "distribution\dist\WirelinePreprocessing\WirelinePreprocessing.exe" (
    echo.
    echo ERROR: Expected executable was not produced.
    pause
    exit /b 1
)

REM Ship the end-user instructions inside the folder the recipient extracts,
REM so the log location and privacy statement travel with the application.
copy /Y "distribution\README_FIRST.txt" "distribution\dist\WirelinePreprocessing\README_FIRST.txt" >nul

echo.
echo Packaging distribution archive...
powershell -NoProfile -Command ^
    "Compress-Archive -Path 'distribution\dist\WirelinePreprocessing\*' -DestinationPath 'distribution\dist\WirelinePreprocessing-windows.zip' -Force"

REM The work directory is intermediate output that --clean regenerates on every
REM run, so keeping it only wastes disk. Removing it here stops build artifacts
REM accumulating across repeated builds.
if exist "build" rmdir /S /Q "build"

echo.
echo ============================================
echo Build complete
echo ============================================
echo.
echo Folder:  distribution\dist\WirelinePreprocessing\
echo Archive: distribution\dist\WirelinePreprocessing-windows.zip
echo.
echo Send the .zip. The recipient extracts it once, then runs
echo WirelinePreprocessing.exe from inside the extracted folder.
echo.
echo IMPORTANT: Always launch the produced .exe once on this machine before
echo sending it. A packaged build can fail on launch for reasons that never
echo occur when running from source.
echo.
pause
endlocal
