# Creating a Standalone Executable

## Why Create a Standalone Executable?

A standalone executable bundles Python and all dependencies into a single `.exe` file (Windows) or application bundle (Mac/Linux). 

**Key Benefits:**
- ✅ Users won't need to install Python separately
- ✅ **Source code is protected** - compiled to bytecode, not readable
- ✅ Professional distribution suitable for proprietary software
- ✅ Single file to distribute

## Quick Build (Windows)

1. Make sure you have Python installed with all dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Run the build script:
   ```bash
   BUILD_STANDALONE.bat
   ```

3. Find your executable:
   - Location: `dist/WirelinePreprocessing.exe`
   - Size: ~100-200 MB (includes Python + all libraries)

## Quick Build (Linux/Mac)

1. Make sure you have Python installed with all dependencies:
   ```bash
   pip3 install -r requirements.txt
   pip3 install pyinstaller
   ```

2. Make the script executable:
   ```bash
   chmod +x BUILD_STANDALONE.sh
   ```

3. Run the build script:
   ```bash
   ./BUILD_STANDALONE.sh
   ```

4. Find your executable:
   - Location: `dist/WirelinePreprocessing`
   - Size: ~100-200 MB

## Manual Build with PyInstaller

If you prefer to build manually:

### Windows:
```bash
pyinstaller --name="WirelinePreprocessing" --onefile --windowed --add-data "core;core" --add-data "ui;ui" --add-data "petrophysics;petrophysics" advanced_preprocessing_system10.py
```

### Linux/Mac:
```bash
pyinstaller --name="WirelinePreprocessing" --onefile --windowed --add-data "core:core" --add-data "ui:ui" --add-data "petrophysics:petrophysics" advanced_preprocessing_system10.py
```

## Using the Spec File

For more control, edit `pyinstaller_spec.spec` and run:
```bash
pyinstaller pyinstaller_spec.spec
```

## What Gets Created

After building, you'll find:
- **dist/WirelinePreprocessing.exe** (Windows) or **dist/WirelinePreprocessing** (Linux/Mac)
- This is a complete, standalone application
- Users can just double-click to run - no Python needed!
- **Source code is NOT included** - code is compiled to bytecode for protection

## File Size

The executable will be large (100-200 MB) because it includes:
- Python interpreter
- All Python libraries (numpy, pandas, matplotlib, etc.)
- Your application code
- All modules (core, ui, petrophysics)

This is normal for standalone executables.

## Distribution

Once built, you can:
1. Test the executable: `dist/WirelinePreprocessing.exe`
2. Zip just the executable
3. Send to users - they can run it directly!

## Troubleshooting

### Build Fails
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Make sure PyInstaller is installed: `pip install pyinstaller`
- Try building with the spec file for more control

### Executable Won't Run
- Make sure you included the data folders (core, ui, petrophysics)
- Check that hidden imports are included
- Test on a clean system without Python installed

### Missing Modules Error
- Add missing modules to `--hidden-import` flags
- Or edit the spec file to include them

## Advanced Options

### Add Icon
Edit `pyinstaller_spec.spec` and change:
```python
icon=None,  # Change to 'path/to/icon.ico'
```

### Reduce Size (OneDir mode)
Change `--onefile` to `--onedir` in build scripts. This creates a folder with the executable and dependencies (easier to debug, but multiple files).

### Debug Mode
Remove `--windowed` flag to see console output for debugging.

## Notes

- First build takes longer (5-10 minutes)
- Subsequent builds are faster
- The executable is platform-specific (Windows .exe won't run on Mac/Linux)
- You'll need to build separately for each platform

