# Troubleshooting Guide - Advanced Wireline Data Preprocessing System

## Table of Contents
1. [General Troubleshooting](#general-troubleshooting)
2. [Installation and Setup Issues](#installation-and-setup-issues)
3. [File Loading Problems](#file-loading-problems)
4. [Processing Issues](#processing-issues)
5. [Visualization Problems](#visualization-problems)
6. [Export and Reporting Issues](#export-and-reporting-issues)
7. [Performance Problems](#performance-problems)
8. [Memory and Resource Issues](#memory-and-resource-issues)
9. [Multi-Well Operations](#multi-well-operations)
10. [Error Messages Reference](#error-messages-reference)
11. [Diagnostic Tools](#diagnostic-tools)
12. [Getting Additional Help](#getting-additional-help)

---

## General Troubleshooting

### Before You Start
1. **Check System Requirements**: Ensure your system meets minimum requirements
2. **Update Dependencies**: Make sure all Python packages are up to date
3. **Restart the Application**: Many issues resolve with a simple restart
4. **Check Available Memory**: Ensure sufficient RAM is available
5. **Review Error Messages**: Read error messages carefully for specific guidance

### Basic Diagnostic Steps

#### Step 1: System Information Check
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check system memory
python -c "import psutil; print(f'Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')"
```

#### Step 2: Application Logs
Check the processing log in the application for detailed error information:
- Look for specific error messages
- Note the sequence of operations leading to the error
- Check for memory warnings or resource constraints

#### Step 3: Test with Sample Data
Try processing with a simple, known-good dataset to isolate the issue:
- Use the sample data creation script from Quick Start Guide
- Test with a small LAS file from a logging company
- Verify basic functionality before troubleshooting complex issues

---

## Installation and Setup Issues

### Issue: Application Won't Start

#### Problem: "ModuleNotFoundError" or Import Errors
**Symptoms**: Application fails to start with import errors
**Causes**: Missing Python packages or incorrect Python environment

**Solutions**:
1. **Install Required Packages**:
   ```bash
   pip install tkinter matplotlib numpy pandas scipy lasio openpyxl ipykernel
   ```

2. **Check Python Environment**:
   ```bash
   # Verify you're using the correct Python
   which python
   python -c "import sys; print(sys.executable)"
   ```

3. **Virtual Environment Issues**:
   ```bash
   # If using virtual environment, activate it first
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

#### Problem: "Permission Denied" Errors
**Symptoms**: Cannot install packages or access files
**Solutions**:
1. **Run as Administrator** (Windows) or use `sudo` (Linux/Mac)
2. **Check File Permissions**: Ensure you have write access to installation directory
3. **Use User Installation**:
   ```bash
   pip install --user package_name
   ```

### Issue: GUI Won't Display

#### Problem: Tkinter Display Issues
**Symptoms**: Application starts but GUI doesn't appear or is corrupted
**Causes**: Display driver issues, remote desktop problems, or Tkinter configuration

**Solutions**:
1. **Check Display Settings**:
   ```bash
   # Test basic Tkinter functionality
   python -c "import tkinter; tkinter.Tk().mainloop()"
   ```

2. **Remote Desktop Issues** (if applicable):
   ```bash
   # Set display environment variable
   export DISPLAY=:0.0  # Linux
   ```

3. **Update Graphics Drivers**: Ensure graphics drivers are current

### Issue: Application Crashes on Startup

#### Problem: Memory or Resource Issues
**Symptoms**: Application crashes immediately after starting
**Solutions**:
1. **Check Available Memory**: Close other applications to free memory
2. **Check Disk Space**: Ensure sufficient disk space for temporary files
3. **Run with Debug Mode**:
   ```bash
   python -u advanced_preprocessing_system10.py 2>&1 | tee startup.log
   ```

---

## File Loading Problems

### Issue: LAS Files Won't Load

#### Problem: "LAS file not compliant" Error
**Symptoms**: LAS file fails to load with compliance error
**Causes**: Non-standard LAS format, corrupted file, or version incompatibility

**Solutions**:
1. **Check LAS File Format**:
   - Verify file follows LAS 2.0 or 3.0 standards
   - Check for proper section headers (~VERSION, ~WELL, ~CURVE, ~DATA)
   - Ensure proper line endings (CRLF for Windows, LF for Unix)

2. **Try Manual LAS Parsing**:
   - Use a text editor to check file structure
   - Look for missing sections or malformed headers
   - Verify data section format

3. **Convert to CSV**:
   - Use external tools to convert LAS to CSV
   - Load CSV file instead of LAS
   - Re-export as LAS after processing

#### Problem: "No curves detected" Warning
**Symptoms**: LAS loads but no petrophysical curves are identified
**Causes**: Non-standard curve mnemonics or missing curve information

**Solutions**:
1. **Check Curve Section**:
   - Verify ~CURVE section exists and is properly formatted
   - Check curve mnemonics against industry standards
   - Ensure curve descriptions are present

2. **Manual Curve Identification**:
   - Use the curve identification tools in the Data tab
   - Manually specify curve types if automatic detection fails
   - Check curve statistics to verify data quality

3. **Update Mnemonic Library**:
   - Add custom curve mnemonics to the system
   - Contact support for non-standard curve types

### Issue: CSV Files Won't Load

#### Problem: "Delimiter detection failed" Error
**Symptoms**: CSV file loads but data appears incorrect
**Causes**: Non-standard delimiter, encoding issues, or header problems

**Solutions**:
1. **Check File Encoding**:
   - Ensure file is saved as UTF-8 or ASCII
   - Check for special characters in headers
   - Verify line ending format

2. **Manual Delimiter Specification**:
   - Use the delimiter selection tool in file loading
   - Try different delimiter options (comma, semicolon, tab)
   - Check for mixed delimiters in the file

3. **Header Row Issues**:
   - Verify headers are in the first row
   - Check for empty header cells
   - Ensure headers don't contain special characters

#### Problem: "Depth column not found" Warning
**Symptoms**: CSV loads but depth curve is not identified
**Solutions**:
1. **Manual Depth Column Selection**:
   - Use the depth column selector in the Data tab
   - Specify which column contains depth data
   - Verify depth data is numeric and properly formatted

2. **Check Depth Data Format**:
   - Ensure depth values are numeric (no units in data)
   - Check for consistent depth intervals
   - Verify depth is in ascending order

### Issue: Excel Files Won't Load

#### Problem: "Worksheet not found" Error
**Symptoms**: Excel file fails to load or shows empty data
**Solutions**:
1. **Check Worksheet Selection**:
   - Verify correct worksheet is selected
   - Check if data starts from the correct row
   - Ensure worksheet contains data

2. **Excel Format Issues**:
   - Save as .xlsx format (not .xls)
   - Check for merged cells or complex formatting
   - Ensure data is in tabular format

3. **Large File Issues**:
   - Excel files with >1M rows may cause memory issues
   - Consider splitting large files or converting to CSV
   - Use 64-bit Python for large Excel files

---

## Processing Issues

### Issue: Processing Fails to Start

#### Problem: "Insufficient data for processing" Error
**Symptoms**: Processing button is disabled or processing fails immediately
**Causes**: No data loaded, insufficient curves, or data quality issues

**Solutions**:
1. **Verify Data Loading**:
   - Check that data is properly loaded and displayed
   - Verify at least one depth curve is present
   - Ensure data contains valid numeric values

2. **Check Data Quality**:
   - Review data preview for obvious issues
   - Check for all-NaN curves or empty data
   - Verify depth range is reasonable

3. **Minimum Requirements**:
   - Ensure at least 10 data points per curve
   - Check that depth intervals are consistent
   - Verify curves have reasonable value ranges

### Issue: Gap Filling Produces Poor Results

#### Problem: Unrealistic or Oscillating Filled Values
**Symptoms**: Gap-filled values appear unrealistic or create oscillations
**Causes**: Inappropriate gap filling method or poor parameter selection

**Solutions**:
1. **Try Different Methods**:
   - Use Linear interpolation for simple gaps
   - Try Cubic Spline for smooth geological variations
   - Use Multi-curve correlation when other curves are available

2. **Adjust Parameters**:
   - Reduce gap size limit for interpolation methods
   - Increase smoothing parameters for noisy data
   - Use geological zone awareness if zones are detected

3. **Check Data Quality**:
   - Verify surrounding data points are reliable
   - Check for outliers that might affect interpolation
   - Consider manual gap filling for critical sections

#### Problem: "Gap filling method failed" Error
**Symptoms**: Specific gap filling method fails with error message
**Solutions**:
1. **Method-Specific Troubleshooting**:
   - **Gaussian Process**: Reduce data size or increase memory
   - **Kriging**: Check for sufficient data points or try different variogram
   - **Polynomial**: Reduce polynomial degree or check for outliers

2. **Fallback Options**:
   - Use Linear interpolation as fallback
   - Try different gap filling methods
   - Process gaps individually with different methods

### Issue: Denoising Removes Important Features

#### Problem: Geological Features Lost During Denoising
**Symptoms**: Important geological boundaries or features are smoothed out
**Solutions**:
1. **Adjust Denoising Parameters**:
   - Reduce smoothing strength
   - Use edge-preserving methods (Bilateral filtering)
   - Increase noise threshold parameters

2. **Method Selection**:
   - Use Median filtering to preserve sharp boundaries
   - Try Savitzky-Golay for feature preservation
   - Avoid aggressive wavelet denoising

3. **Selective Denoising**:
   - Apply different denoising to different curves
   - Skip denoising for curves with important features
   - Use geological zone awareness

### Issue: Processing Takes Too Long

#### Problem: Processing Hangs or Takes Excessive Time
**Symptoms**: Processing appears to freeze or takes hours to complete
**Solutions**:
1. **Check Data Size**:
   - Large datasets (>100K points) may take longer
   - Consider subsampling for initial testing
   - Use faster processing methods for large datasets

2. **Method Selection**:
   - Use Linear interpolation instead of Gaussian Process
   - Choose Bilateral filtering over Wavelet denoising
   - Avoid Kriging for very large datasets

3. **System Resources**:
   - Close other applications to free CPU and memory
   - Check system temperature and CPU usage
   - Restart application if memory usage is high

---

## Visualization Problems

### Issue: Plots Appear Blank or Incorrect

#### Problem: Empty Plot Windows
**Symptoms**: Plot windows open but show no data
**Solutions**:
1. **Check Data Selection**:
   - Verify curves are selected for plotting
   - Check depth range settings
   - Ensure data is loaded and processed

2. **Plot Parameters**:
   - Adjust depth range to match your data
   - Check curve scaling settings
   - Verify plot type is appropriate for data

3. **Display Issues**:
   - Try different plot types
   - Check if data is within expected ranges
   - Restart visualization if plots are corrupted

#### Problem: Incorrect Data Display
**Symptoms**: Plots show wrong data or incorrect scaling
**Solutions**:
1. **Data Verification**:
   - Check that correct curves are selected
   - Verify depth range matches your data
   - Ensure processing completed successfully

2. **Scaling Issues**:
   - Check curve units and ranges
   - Verify depth units (meters vs feet)
   - Adjust plot scaling manually

3. **Active Well Issues**:
   - Ensure correct well is selected (multi-well mode)
   - Check that data is loaded for selected well
   - Verify well switching completed successfully

### Issue: 3D Visualization Problems

#### Problem: 3D Plots Don't Display
**Symptoms**: 3D visualization fails or shows errors
**Solutions**:
1. **Check 3D Toolkit Installation**:
   ```bash
   python -c "from mpl_toolkits.mplot3d import Axes3D; print('3D toolkit available')"
   ```

2. **Data Requirements**:
   - Ensure at least 3 curves are selected
   - Check that data has sufficient points for 3D display
   - Verify data ranges are appropriate for 3D visualization

3. **Display Driver Issues**:
   - Update graphics drivers
   - Check OpenGL support
   - Try different 3D rendering backends

### Issue: Interactive Features Don't Work

#### Problem: Zoom, Pan, or Export Features Fail
**Symptoms**: Interactive plot controls don't respond
**Solutions**:
1. **Toolbar Issues**:
   - Check if navigation toolbar is visible
   - Try different toolbar configurations
   - Restart visualization if toolbar is corrupted

2. **Event Handling**:
   - Check for conflicting mouse events
   - Verify plot canvas is active
   - Try clicking on plot area to activate

3. **Export Problems**:
   - Check file permissions for export directory
   - Verify export format is supported
   - Ensure sufficient disk space

---

## Export and Reporting Issues

### Issue: LAS Export Fails

#### Problem: "LAS export failed" Error
**Symptoms**: Cannot export processed data as LAS file
**Solutions**:
1. **LAS Compliance Issues**:
   - Check curve names comply with LAS standards
   - Ensure all required sections are present
   - Verify data format is compatible with LAS

2. **File System Issues**:
   - Check write permissions for export directory
   - Ensure sufficient disk space
   - Verify file path doesn't contain invalid characters

3. **Data Issues**:
   - Check for invalid values (NaN, inf) in data
   - Verify curve information is complete
   - Ensure depth data is properly formatted

#### Problem: Exported LAS File is Invalid
**Symptoms**: Exported LAS file cannot be opened by other software
**Solutions**:
1. **LAS Format Verification**:
   - Use LAS validation tools to check format
   - Compare with original LAS file structure
   - Check section headers and formatting

2. **Data Validation**:
   - Verify all curves have proper descriptions
   - Check units are correctly specified
   - Ensure NULL values are properly handled

### Issue: Report Generation Fails

#### Problem: "Report generation failed" Error
**Symptoms**: Cannot generate processing report
**Solutions**:
1. **Template Issues**:
   - Check report template files are present
   - Verify template format is valid
   - Try generating different report types

2. **Data Issues**:
   - Ensure processing completed successfully
   - Check that required data is available
   - Verify processing results are valid

3. **Resource Issues**:
   - Check available memory for report generation
   - Ensure sufficient disk space
   - Close other applications if needed

### Issue: Excel Export Creates Large Files

#### Problem: Excel files are too large or slow to open
**Symptoms**: Exported Excel files are hundreds of MB or won't open
**Solutions**:
1. **Data Reduction**:
   - Export only essential curves
   - Reduce data precision (fewer decimal places)
   - Use CSV format for large datasets

2. **Excel Optimization**:
   - Split data across multiple worksheets
   - Use Excel binary format (.xlsb)
   - Compress data before export

3. **Alternative Formats**:
   - Use CSV for large datasets
   - Export as LAS format
   - Use compressed formats when possible

---

## Performance Problems

### Issue: Slow Processing Performance

#### Problem: Processing Takes Much Longer Than Expected
**Symptoms**: Processing is significantly slower than normal
**Solutions**:
1. **System Resource Check**:
   ```bash
   # Check CPU usage
   python -c "import psutil; print(f'CPU Usage: {psutil.cpu_percent()}%')"
   
   # Check memory usage
   python -c "import psutil; print(f'Memory Usage: {psutil.virtual_memory().percent}%')"
   ```

2. **Processing Optimization**:
   - Use faster processing methods (Linear vs Gaussian Process)
   - Reduce data size for initial testing
   - Process wells individually instead of batch processing

3. **System Maintenance**:
   - Close unnecessary applications
   - Restart application to clear memory
   - Check for background processes consuming resources

### Issue: Memory Usage is Too High

#### Problem: System Runs Out of Memory
**Symptoms**: Application crashes or becomes unresponsive
**Solutions**:
1. **Memory Management**:
   - Close unused wells in multi-well mode
   - Process smaller datasets at a time
   - Restart application periodically

2. **Data Optimization**:
   - Use data compression where possible
   - Reduce data precision for non-critical operations
   - Archive processed results to disk

3. **System Upgrades**:
   - Increase system RAM if possible
   - Use 64-bit Python for large datasets
   - Consider using SSD storage for better performance

### Issue: UI Becomes Unresponsive

#### Problem: Interface Freezes During Processing
**Symptoms**: UI stops responding while processing continues
**Solutions**:
1. **Threading Issues**:
   - Processing should run in background thread
   - Check if processing thread is still active
   - Wait for processing to complete

2. **Memory Issues**:
   - High memory usage can cause UI freezing
   - Check system memory usage
   - Restart application if needed

3. **Display Issues**:
   - Check graphics driver issues
   - Try different display settings
   - Restart application with different display options

---

## Memory and Resource Issues

### Issue: Out of Memory Errors

#### Problem: "MemoryError" or "Out of Memory" Messages
**Symptoms**: Application crashes with memory-related errors
**Solutions**:
1. **Immediate Actions**:
   - Close other applications immediately
   - Restart the application
   - Reduce dataset size

2. **Long-term Solutions**:
   - Increase system RAM
   - Use 64-bit Python
   - Process data in smaller chunks

3. **Memory Monitoring**:
   ```bash
   # Monitor memory usage
   python -c "
   import psutil
   import time
   while True:
       mem = psutil.virtual_memory()
       print(f'Memory: {mem.percent}% used, {mem.available/1024**3:.1f}GB available')
       time.sleep(5)
   "
   ```

### Issue: Disk Space Problems

#### Problem: Insufficient Disk Space Errors
**Symptoms**: Cannot save files or export data
**Solutions**:
1. **Free Up Space**:
   - Delete temporary files
   - Move large files to external storage
   - Clear application cache

2. **Optimize Storage**:
   - Use compressed file formats
   - Export only essential data
   - Use external storage for large datasets

### Issue: CPU Overload

#### Problem: System Becomes Unresponsive Due to High CPU Usage
**Symptoms**: Computer becomes slow or unresponsive
**Solutions**:
1. **Process Management**:
   - Check for stuck processing threads
   - Restart application if needed
   - Use task manager to monitor CPU usage

2. **Processing Optimization**:
   - Use less CPU-intensive methods
   - Process smaller datasets
   - Reduce processing complexity

---

## Multi-Well Operations

### Issue: Multi-Well Loading Problems

#### Problem: "Failed to load multiple files" Error
**Symptoms**: Batch loading of multiple wells fails
**Solutions**:
1. **File Format Issues**:
   - Ensure all files are in supported formats
   - Check for corrupted files in the batch
   - Try loading files individually to identify problem files

2. **Memory Issues**:
   - Reduce number of files loaded simultaneously
   - Load wells in smaller batches
   - Check available system memory

3. **File Path Issues**:
   - Ensure all files are accessible
   - Check for files with invalid characters in names
   - Verify file permissions

### Issue: Active Well Selection Problems

#### Problem: Cannot Switch Between Wells
**Symptoms**: Well selection doesn't work or shows wrong data
**Solutions**:
1. **Well List Issues**:
   - Verify wells are properly loaded
   - Check well identification is correct
   - Refresh well list display

2. **Data Synchronization**:
   - Ensure data is loaded for selected well
   - Check that well switching completed successfully
   - Verify active well indicator is correct

### Issue: Cross-Well Analysis Problems

#### Problem: Cross-well statistics are incorrect
**Symptoms**: Statistics don't match individual well data
**Solutions**:
1. **Data Consistency**:
   - Ensure all wells have same curve types
   - Check for units consistency across wells
   - Verify depth ranges are comparable

2. **Statistical Issues**:
   - Check for outliers affecting statistics
   - Verify statistical calculations are correct
   - Ensure sufficient data points per well

---

## Error Messages Reference

### Common Error Messages and Solutions

#### File Loading Errors
```
"File format not supported"
→ Check file extension and format
→ Convert to supported format (LAS, CSV, Excel)

"LAS file not compliant"
→ Verify LAS format compliance
→ Check section headers and structure
→ Try converting to CSV first

"No curves detected"
→ Check curve section in LAS file
→ Verify curve mnemonics are standard
→ Manually identify curves if needed
```

#### Processing Errors
```
"Insufficient data for processing"
→ Ensure data is loaded and valid
→ Check minimum data requirements
→ Verify depth curve is present

"Gap filling method failed"
→ Try different gap filling method
→ Check gap size and surrounding data
→ Use simpler interpolation method

"Denoising failed"
→ Try different denoising method
→ Check data quality and noise level
→ Adjust denoising parameters
```

#### Visualization Errors
```
"Plot creation failed"
→ Check data selection and ranges
→ Verify curves are available
→ Try different plot type

"3D visualization not available"
→ Check 3D toolkit installation
→ Verify sufficient data for 3D
→ Update graphics drivers
```

#### Export Errors
```
"Export failed"
→ Check file permissions
→ Ensure sufficient disk space
→ Verify data is valid for export

"LAS export failed"
→ Check LAS compliance
→ Verify curve information
→ Ensure proper data format
```

---

## Diagnostic Tools

### Built-in Diagnostic Features

#### System Information
The application provides system information through the Help menu:
- Python version and packages
- System memory and CPU information
- Graphics driver details
- File system capabilities

#### Processing Logs
Detailed processing logs are available in the Report tab:
- Step-by-step processing information
- Error messages with context
- Performance metrics
- Memory usage statistics

#### Data Validation Tools
Built-in tools for data validation:
- Curve statistics and quality metrics
- Range validation results
- Completeness analysis
- Correlation analysis

### External Diagnostic Tools

#### Python Environment Check
```bash
# Check Python installation
python --version
python -c "import sys; print(sys.executable)"

# Check package versions
pip list | grep -E "(matplotlib|numpy|pandas|scipy|lasio)"

# Check system resources
python -c "import psutil; print(psutil.virtual_memory())"
```

#### File Format Validation
```bash
# Check LAS file format
python -c "
import lasio
try:
    las = lasio.read('your_file.las')
    print('LAS file is valid')
except Exception as e:
    print(f'LAS file error: {e}')
"

# Check CSV file format
python -c "
import pandas as pd
try:
    df = pd.read_csv('your_file.csv')
    print(f'CSV loaded: {len(df)} rows, {len(df.columns)} columns')
except Exception as e:
    print(f'CSV file error: {e}')
"
```

---

## Getting Additional Help

### Self-Help Resources

1. **Documentation**:
   - User Manual for comprehensive guidance
   - Technical Guide for advanced features
   - Quick Start Guide for basic operations
   - API Reference for developers

2. **Built-in Help**:
   - Help menu in application
   - Tooltips on UI elements
   - Error message explanations
   - Processing log analysis

3. **Sample Data**:
   - Use provided sample datasets
   - Create test data with sample scripts
   - Practice with known-good files

### Professional Support

1. **Technical Support**:
   - Provide detailed error messages
   - Include system information
   - Attach sample data files
   - Describe exact steps to reproduce issue

2. **Community Support**:
   - Connect with other users
   - Share experiences and solutions
   - Participate in user forums
   - Contribute to knowledge base

3. **Training and Consulting**:
   - Attend training workshops
   - Schedule consultation sessions
   - Get custom configuration help
   - Learn advanced workflows

### Information to Provide When Seeking Help

1. **System Information**:
   - Operating system and version
   - Python version and packages
   - Available memory and CPU
   - Graphics driver information

2. **Error Details**:
   - Complete error messages
   - Processing log excerpts
   - Steps leading to error
   - Expected vs actual behavior

3. **Data Information**:
   - File format and size
   - Number of curves and data points
   - Data quality and characteristics
   - Processing parameters used

---

*This troubleshooting guide covers the most common issues encountered when using the Advanced Wireline Data Preprocessing System. For issues not covered here, consult the technical documentation or contact professional support.*
