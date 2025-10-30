# How to Run POLISH (Advanced Preprocessing System)

## Quick Start

### Option 1: Direct Python Execution (Recommended)
```bash
python advanced_preprocessing_system10.py
```

### Option 2: Run as Module
```bash
python -m advanced_preprocessing_system10
```

## Prerequisites

### Required Python Packages
The program uses these core libraries (most are standard or commonly installed):
- `tkinter` - GUI framework (usually included with Python)
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `lasio` - LAS file reading (optional but recommended)

### Optional Advanced Features
For full functionality, install these additional packages:
```bash
pip install scipy scikit-learn pywavelets
```

These enable:
- Advanced gap filling (Gaussian Process, Kriging)
- Machine learning features
- Wavelet denoising

## Running the Application

### Windows (PowerShell or Command Prompt)
```powershell
cd C:\Users\achav\OneDrive\TraceSeis5\polish10
python advanced_preprocessing_system10.py
```

### What to Expect
1. The application will start with a Tkinter GUI window
2. You'll see tabs for:
   - **Data**: Load and view wireline data (LAS/CSV/Excel)
   - **Processing**: Configure gap filling and denoising
   - **Visualization**: View curves and analysis
   - **Report**: Generate processing reports

3. **First Steps:**
   - Click "Load File" to import your wireline data
   - The system will automatically identify curves
   - Configure processing options as needed
   - Click "Start Processing" to run the pipeline

## Troubleshooting

### If the window doesn't appear:
- Check that Python is properly installed: `python --version`
- Verify tkinter is available: `python -m tkinter`
- Check for error messages in the console

### If imports fail:
- Install missing packages: `pip install <package-name>`
- The program will run with limited functionality if optional packages are missing

### Memory Issues:
- The program handles large datasets, but if you encounter memory issues:
  - Process wells individually rather than in batch
  - Close other applications to free memory

## Application Structure

```
polish10/
├── advanced_preprocessing_system10.py  ← Main application (run this)
├── core/                                 ← Core modules
│   ├── reporting.py                     ← Standardization reporting
│   └── petrophysical_models.py         ← Petrophysical calculations
├── ui/                                  ← UI components
│   ├── visualization.py                ← Plotting and visualization
│   └── status.py                       ← Status updates and logging
└── petrophysics/                        ← Petrophysical constants
    └── constants.py                    ← Industry standards and parameters
```

## Notes

- The application saves processing history and supports undo/redo
- All standardization operations are logged and auditable
- Visualizations include industry-standard log displays with QC indicators
- The system supports multi-well processing with proper state isolation

