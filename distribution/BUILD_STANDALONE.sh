#!/bin/bash
# Build script to create standalone executable with Python bundled
# This creates a binary file that users can run without Python installed

echo "============================================"
echo "Building Standalone Executable"
echo "============================================"
echo ""
echo "This will create a binary file with Python bundled."
echo "Users won't need to install Python separately!"
echo ""

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install PyInstaller"
        exit 1
    fi
fi

echo ""
echo "Building executable..."
echo "This may take several minutes..."
echo ""

# Build with PyInstaller - Code is compiled to bytecode (not source .py files)
# Users will NOT have access to readable source code
pyinstaller --name="WirelinePreprocessing" \
    --onefile \
    --windowed \
    --noconsole \
    --add-data "core:core" \
    --add-data "ui:ui" \
    --add-data "petrophysics:petrophysics" \
    --hidden-import=numpy \
    --hidden-import=pandas \
    --hidden-import=matplotlib \
    --hidden-import=tkinter \
    --hidden-import=lasio \
    --hidden-import=scipy \
    --hidden-import=sklearn \
    --hidden-import=pywt \
    --hidden-import=psutil \
    --collect-all=matplotlib \
    --collect-all=numpy \
    --collect-all=pandas \
    --optimize=2 \
    advanced_preprocessing_system10.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo ""
echo "The executable is in: dist/WirelinePreprocessing"
echo ""
echo "You can now distribute this binary - users don't need Python!"
echo ""

