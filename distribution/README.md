# Advanced Wireline Data Preprocessing System

## Distribution Package

This is a complete, ready-to-run distribution of the Advanced Wireline Data Preprocessing System.

## Quick Start

### ⚠️ IMPORTANT: Python Not Included
**Python must be installed separately before using this application.**

### 1. Install Python (Required First Step)
- **Download Python**: https://www.python.org/downloads/
- **Version Required**: Python 3.8 or higher (3.11+ recommended)
- **Installation Tip**: Check "Add Python to PATH" during installation!
- **Verify Installation**: Open terminal and type `python --version`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python advanced_preprocessing_system10.py
```

**Windows Users:** Double-click `run.bat` (if available) or use the command above.

## Package Contents

```
distribution/
├── advanced_preprocessing_system10.py  ← Main application (run this)
├── core/                               ← Core processing modules
│   ├── __init__.py
│   ├── reporting.py                   ← Standardization reporting
│   └── petrophysical_models.py       ← Petrophysical calculations
├── ui/                                 ← User interface components
│   ├── __init__.py
│   ├── visualization.py              ← Plotting and visualization
│   └── status.py                     ← Status management
├── petrophysics/                      ← Petrophysical constants
│   ├── __init__.py
│   └── constants.py                  ← Industry standards
├── requirements.txt                   ← Python dependencies
└── README.md                         ← This file
```

## Features

- **Multi-Format Support**: Load LAS, CSV, and Excel files
- **Advanced Gap Filling**: Linear, spline, Gaussian Process, kriging, and multi-curve correlation
- **Signal Denoising**: Wavelet, bilateral, Savitzky-Golay, and median filtering
- **Industry Standards**: Automatic curve identification and unit standardization
- **Quality Control**: Comprehensive validation, outlier detection, and QC metrics
- **Visualization**: Industry-standard log displays with QC indicators
- **Multi-Well Processing**: Process multiple wells with proper state isolation
- **Security**: Path validation, file size limits, and secure file handling

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 4GB minimum (8GB+ recommended for large datasets)
- **Disk Space**: 500MB for installation + space for data files

## Installation

### ⚠️ Prerequisites
**Python 3.8+ must be installed first!**
- Download from: https://www.python.org/downloads/
- During installation, check "Add Python to PATH"

### Option 1: Standard Installation (Recommended)

1. **Install Python first** (see Prerequisites above)
2. Extract this distribution folder to your desired location
3. Open a terminal/command prompt in the distribution folder
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python advanced_preprocessing_system10.py
   ```

### Option 2: Virtual Environment (Isolated Installation)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate it:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python advanced_preprocessing_system10.py
   ```

## Usage Guide

### Loading Data
1. Click the **Data** tab
2. Click **"Load File"** or **"Load Multiple Files"**
3. Select your wireline data file(s) (LAS, CSV, or Excel)
4. The system will automatically identify curves

### Processing
1. Go to the **Processing** tab
2. Configure gap filling and denoising options
3. Click **"Start Processing"**
4. Monitor progress in the status area

### Visualization
1. Go to the **Visualization** tab
2. Select curves and visualization type
3. View industry-standard log displays with QC indicators

### Exporting
1. After processing, use **"Export Data"** to save results
2. Choose format: LAS, CSV, or Excel
3. Generate comprehensive reports in the **Report** tab

## Troubleshooting

### Application Won't Start
- Verify Python is installed: `python --version`
- Check tkinter availability: `python -m tkinter`
- Install missing dependencies: `pip install -r requirements.txt`

### Import Errors
- Most optional packages have graceful fallbacks
- Install missing packages individually if needed
- Advanced features require: `scipy`, `scikit-learn`, `pywavelets`

### File Loading Issues
- Maximum file size: 500MB (configurable in code)
- Supported formats: `.las`, `.csv`, `.xlsx`, `.xls`
- Check file permissions and path validity

### Performance
- Large datasets may take time to process
- Close other applications to free memory
- Process wells individually for better performance

## Advanced Configuration

### Optional Features
The application works with basic dependencies but gains advanced features with:
- **scipy**: Advanced gap filling (Gaussian Process, Kriging)
- **scikit-learn**: Machine learning-based interpolation
- **pywavelets**: Wavelet denoising

### File Size Limits
Default maximum file size is 500MB. This can be adjusted in the code if needed.

## Support

For issues, questions, or feature requests:
- Check the security audit report for security information
- Review the project status report for known limitations
- Consult the "How to Run" documentation for detailed instructions

## License

This software is provided as-is for wireline data preprocessing and analysis.

## Version Information

- **Application**: Advanced Preprocessing System v10
- **Distribution Date**: 2024
- **Python Compatibility**: 3.8+

---

**Note**: This is a scientific/research application. Ensure proper data backup before processing critical datasets.

