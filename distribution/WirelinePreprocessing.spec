# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for the POLISH wireline preprocessing application.

BUILD MODE
----------
This produces a ONE-FOLDER build (EXE + COLLECT), not a one-file build.

A one-file executable must decompress its entire payload into a temporary
directory on every single launch. Measured on this project that cost 67 seconds
of the 82 seconds required to reach the main window, and it is repeated every
time the user opens the program. The one-folder layout loads its dependencies
directly from disk and starts in a few seconds.

Distribute by zipping the output folder. The user extracts it once and runs
WirelinePreprocessing.exe from inside.

CRITICAL SETTINGS
-----------------
optimize=0  Level 2 strips docstrings from bundled modules. NumPy's
            array_function dispatch feeds dispatcher.__doc__ into the C
            function add_docstring(), which rejects None, so a level 2 build
            dies while importing numpy before any application code runs.

upx=False   UPX compression is a frequent trigger for antivirus heuristics.
            A false positive on a customer machine in a corporate environment
            is far more costly than the disk space it saves, and it saves
            little in a one-folder layout.
"""
import os
from PyInstaller.utils.hooks import collect_all

# Resolve the project root so the spec can be invoked from either directory.
spec_dir = os.path.dirname(os.path.abspath(SPEC))
project_root = os.path.dirname(spec_dir)

main_script = os.path.join(project_root, 'advanced_preprocessing_system10.py')

# Application packages are shipped alongside the bundle so that any dynamic
# lookups resolve identically to a source checkout.
datas = [
    (os.path.join(project_root, 'core'), 'core'),
    (os.path.join(project_root, 'ui'), 'ui'),
    (os.path.join(project_root, 'petrophysics'), 'petrophysics'),
]
binaries = []
hiddenimports = [
    'numpy', 'pandas', 'matplotlib', 'tkinter', 'lasio', 'scipy', 'sklearn',
    'pywt', 'psutil', 'seaborn', 'platform',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'mpl_toolkits.mplot3d',
]

# Packages that exist in the build environment but that this application never
# imports. Every name here was verified absent from the source tree.
#
# This list is not optional housekeeping. PyInstaller bundles whatever reaches
# its import graph, and a developer machine typically has far more installed
# than the application needs. Without these exclusions a build on this machine
# pulls in PyTorch alone, which is 494 MB before anything else is counted.
excludes = [
    # Machine learning stacks. scikit-learn is used and stays; these are not.
    'torch', 'torchvision', 'torchaudio', 'transformers', 'tokenizers',
    'safetensors', 'tensorflow', 'keras', 'numba', 'sympy',
    # Alternative GUI toolkits. This application is Tkinter only.
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'kivy',
    # Alternative plotting and dataframe backends.
    'plotly', 'bokeh', 'pyarrow', 'dask', 'statsmodels', 'vaex',
    # Interactive and notebook tooling.
    'IPython', 'jedi', 'parso', 'prompt_toolkit', 'notebook', 'nbconvert',
    'nbformat', 'jupyter', 'jupyter_core', 'ipykernel', 'ipywidgets',
    'tornado',
    # Networking. The application performs no network access by design, so
    # excluding these also keeps HTTP machinery out of the shipped product.
    'yt_dlp', 'websockets', 'requests', 'urllib3', 'mutagen', 'brotli',
    'curl_cffi', 'certifi', 'secretstorage', 'Cryptodome', 'googleapiclient',
    'google', 'grpc',
    # Developer tooling that has no place in a customer build.
    'semgrep', 'sphinx', 'pytest', '_pytest', 'clang', 'gurobipy',
    'setuptools._distutils', 'pip',
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
    excludes=excludes,
    noarchive=False,
    # Must remain 0. See CRITICAL SETTINGS above; level 2 breaks numpy import.
    optimize=0,
)
pyz = PYZ(a.pure)

# exclude_binaries=True keeps dependencies out of the executable itself so that
# COLLECT can place them next to it, which is what makes startup fast.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WirelinePreprocessing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WirelinePreprocessing',
)
