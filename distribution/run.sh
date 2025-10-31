#!/bin/bash
# Launcher script for Advanced Wireline Data Preprocessing System
# Linux/Mac Shell Script

echo "Advanced Wireline Data Preprocessing System"
echo "============================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Some dependencies may be missing"
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        echo "Please run: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo ""
echo "Starting application..."
echo ""

# Run the application
python3 advanced_preprocessing_system10.py

exit $?

