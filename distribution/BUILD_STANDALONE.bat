@echo off
REM Build script to create standalone executable with Python bundled
REM This creates a .exe file that users can run without Python installed

echo ============================================
echo Building Standalone Executable
echo ============================================
echo.
echo This will create an .exe file with Python bundled.
echo Users won't need to install Python separately!
echo.

REM Check if PyInstaller is installed
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

echo.
echo Building executable...
echo This may take several minutes...
echo.

REM Change to parent directory to build from project root
cd ..

REM Build with PyInstaller - Code is compiled to bytecode (not source .py files)
REM Users will NOT have access to readable source code
pyinstaller --name="WirelinePreprocessing" ^
    --onefile ^
    --console ^
    --add-data "core;core" ^
    --add-data "ui;ui" ^
    --add-data "petrophysics;petrophysics" ^
    --hidden-import=numpy ^
    --hidden-import=pandas ^
    --hidden-import=matplotlib ^
    --hidden-import=tkinter ^
    --hidden-import=lasio ^
    --hidden-import=scipy ^
    --hidden-import=sklearn ^
    --hidden-import=pywt ^
    --hidden-import=psutil ^
    --hidden-import=platform ^
    --collect-all=matplotlib ^
    --collect-all=numpy ^
    --collect-all=pandas ^
    --optimize=2 ^
    advanced_preprocessing_system10.py

REM Move output to distribution folder
if exist "dist\WirelinePreprocessing.exe" (
    if not exist "distribution\dist" mkdir "distribution\dist"
    move /Y "dist\WirelinePreprocessing.exe" "distribution\dist\WirelinePreprocessing.exe"
    echo Moved executable to distribution\dist\
)

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Build Complete!
echo ============================================
echo.
echo The executable is in: dist\WirelinePreprocessing.exe
echo.
echo You can now distribute this .exe file - users don't need Python!
echo.
pause

