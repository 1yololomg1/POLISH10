# Quick Start Guide - Advanced Wireline Data Preprocessing System

## Get Up and Running in 5 Minutes

This guide will help you load your first well data file and run basic processing in just a few steps.

---

## Step 1: Launch the Application

```bash
python advanced_preprocessing_system10.py
```

The application window will open with four main tabs: **Data**, **Processing**, **Visualization**, and **Report**.

---

## Step 2: Load Your Data File

### In the Data Tab:
1. **Click "Browse"** to select your data file
2. **Supported formats**: LAS (.las), CSV (.csv), Excel (.xlsx, .xls)
3. **Click "Load File"** to import your data

### What Happens:
- System automatically analyzes your curves
- Well information appears in the green card (if LAS file)
- Data preview shows in the table below
- Processing options become available

---

## Step 3: Configure Processing (Optional)

### In the Processing Tab:
1. **Gap Filling**: Choose method (Linear is good for beginners)
2. **Denoising**: Select method (Bilateral is recommended)
3. **Quality Control**: Leave default settings for first run

### Quick Settings for Beginners:
- **Gap Filling**: Linear Interpolation
- **Denoising**: Bilateral Filtering
- **Quality Control**: Enable all validations

---

## Step 4: Run Processing

1. **Click "Start Processing"** in the Processing tab
2. **Watch the progress bar** - processing typically takes 30 seconds to 2 minutes
3. **Wait for "Processing Complete"** message

### What the System Does:
- Validates depth data
- Fills gaps in missing data
- Reduces noise in curves
- Performs quality checks
- Prepares processed data for analysis

---

## Step 5: View Results

### In the Visualization Tab:
1. **Click "Plot Comparison"** to see before/after processing
2. **Try "Plot Multi-Curve"** to see all curves together
3. **Use "Plot Correlation Matrix"** to see curve relationships

### In the Report Tab:
1. **Click "Generate Report"** to create processing summary
2. **View "LAS Preview"** to see processed data format
3. **Use "Export Data"** to save your processed file

---

## Common First-Time Scenarios

### Scenario 1: LAS File from Logging Company
**What to expect:**
- Well information automatically populated
- Curves identified by standard mnemonics
- Processing runs smoothly with default settings

**Best settings:**
- Gap Filling: Cubic Spline
- Denoising: Wavelet (if available) or Bilateral

### Scenario 2: CSV File from Database
**What to expect:**
- May need to specify which column is depth
- Curve names might need manual identification
- Check data preview before processing

**Best settings:**
- Gap Filling: Linear or Polynomial
- Denoising: Savitzky-Golay or Median

### Scenario 3: Excel File with Multiple Sheets
**What to expect:**
- Select correct worksheet during loading
- Headers might be in different rows
- Verify data starts from correct row

**Best settings:**
- Gap Filling: Multi-Curve Correlation (if multiple curves)
- Denoising: Bilateral or Median

---

## Quick Tips for Success

### ✅ Do This:
- **Start with default settings** - they work for most data
- **Check the data preview** before processing
- **Look at the well information card** to verify correct well
- **Save your processed data** using Export function
- **Read any warning messages** - they provide helpful guidance

### ❌ Avoid This:
- **Don't skip the data preview** - verify your data loaded correctly
- **Don't ignore error messages** - they indicate real problems
- **Don't process without understanding** - know what each method does
- **Don't forget to save** - processed data is valuable

---

## Troubleshooting Common Issues

### Issue: "File failed to load"
**Solution:**
- Check file format is supported (.las, .csv, .xlsx, .xls)
- Ensure file is not corrupted
- Try a different file to test the system

### Issue: "No curves detected"
**Solution:**
- Check if data preview shows your data
- Verify headers are in the first row
- Try specifying depth column manually

### Issue: "Processing failed"
**Solution:**
- Check data quality in preview
- Try different processing methods
- Reduce data size if very large file

### Issue: "Plots appear blank"
**Solution:**
- Verify data loaded correctly
- Check depth range settings
- Try different plot types

---

## Next Steps After Quick Start

### Learn More:
1. **Read the full User Manual** for detailed feature explanations
2. **Try different processing methods** to understand their effects
3. **Experiment with visualization options** to explore your data
4. **Load multiple wells** to try multi-well operations

### Advanced Features:
1. **Unit Standardization** - Convert between measurement units
2. **Multi-Well Analysis** - Process and compare multiple wells
3. **Custom Processing Parameters** - Fine-tune processing methods
4. **Batch Operations** - Process multiple files automatically

---

## Sample Data for Practice

If you don't have your own data yet, you can practice with:

### Create Sample LAS Data:
```python
# Simple sample data for testing
import pandas as pd
import numpy as np

# Create sample depth and curves
depth = np.arange(1000, 2000, 0.5)  # 1000-2000m depth
gr = 50 + 30 * np.sin(depth/100) + np.random.normal(0, 5, len(depth))  # Gamma Ray
rhob = 2.65 + 0.3 * np.cos(depth/200) + np.random.normal(0, 0.05, len(depth))  # Density
rt = 10 * np.exp(-depth/1000) + np.random.normal(0, 0.5, len(depth))  # Resistivity

# Create DataFrame
data = pd.DataFrame({
    'DEPT': depth,
    'GR': gr,
    'RHOB': rhob,
    'RT': rt
})

# Save as CSV for practice
data.to_csv('sample_well_data.csv', index=False)
```

### Use This Sample Data:
1. Save the above code as a Python script and run it
2. Load the resulting `sample_well_data.csv` file
3. Practice the processing workflow
4. Experiment with different visualization options

---

## Success Indicators

You'll know you're successful when:
- ✅ Data loads without errors
- ✅ Well information displays correctly
- ✅ Processing completes successfully
- ✅ Plots show meaningful results
- ✅ Report generates with quality metrics
- ✅ You can export processed data

---

## Need Help?

### Quick Reference:
- **User Manual**: Comprehensive guide with all features
- **Technical Guide**: Advanced configuration and API details
- **Troubleshooting**: Common problems and solutions
- **Best Practices**: Professional workflows and recommendations

### Getting Support:
1. **Check error messages** - they often contain the solution
2. **Review processing logs** - detailed information about what happened
3. **Try different settings** - many issues are parameter-related
4. **Contact support** - for complex technical issues

---

*Congratulations! You've completed the Quick Start Guide. You now have the basic skills to load data, run processing, and view results. Continue with the full User Manual to learn advanced features and professional workflows.*
