# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# Get the parent directory (project root) for building from distribution folder
# When building from distribution folder, go up one level to project root
spec_dir = os.path.dirname(os.path.abspath(SPEC))
project_root = os.path.dirname(spec_dir)

# Path to main script from distribution folder
main_script = os.path.join(project_root, 'advanced_preprocessing_system10.py')

# Data directories from project root
datas = [
    (os.path.join(project_root, 'core'), 'core'),
    (os.path.join(project_root, 'ui'), 'ui'),
    (os.path.join(project_root, 'petrophysics'), 'petrophysics'),
]
binaries = []
hiddenimports = [
    'numpy', 'pandas', 'matplotlib', 'tkinter', 'lasio', 'scipy', 'sklearn', 'pywt', 'psutil',
    'platform',  # Added for system information
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'mpl_toolkits.mplot3d',
]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [main_script],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],  # Fixed: removed invalid option tuples
    name='WirelinePreprocessing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for production (no console window) - set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
