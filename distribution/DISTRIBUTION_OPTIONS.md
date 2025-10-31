# Distribution Options

You have **two options** for distributing this application:

## Option 1: Source Distribution (Current Package)
**Users need Python installed**

- ✅ Smaller package size (~1 MB)
- ✅ Easy to update code
- ❌ Users must install Python 3.8+
- ❌ Users must install dependencies

**Best for:** Developers, users comfortable with Python

## Option 2: Standalone Executable (Recommended for End Users)
**Users DON'T need Python installed**

- ✅ No Python installation required
- ✅ Just double-click to run
- ✅ All dependencies bundled
- ❌ Larger file size (100-200 MB)
- ❌ Platform-specific (Windows .exe won't run on Mac)

**Best for:** End users, non-technical users, production deployment

## How to Create Standalone Executable

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build Executable

**Windows:**
```bash
BUILD_STANDALONE.bat
```

**Linux/Mac:**
```bash
chmod +x BUILD_STANDALONE.sh
./BUILD_STANDALONE.sh
```

### Step 3: Distribute

The executable will be in `dist/` folder:
- Windows: `dist/WirelinePreprocessing.exe`
- Linux/Mac: `dist/WirelinePreprocessing`

Just zip and send the executable file!

## Recommendation

**For end users:** Create a standalone executable
- More professional
- Easier for users
- No installation headaches

**For developers:** Use source distribution
- Easier to modify
- Smaller size
- Can see and modify code

See `CREATE_STANDALONE_README.md` for detailed build instructions.

