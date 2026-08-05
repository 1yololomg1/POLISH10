#!/bin/bash
# ============================================================
# Build the standalone POLISH distribution for Linux/macOS.
#
# All build settings live in WirelinePreprocessing.spec. This script only
# invokes it. Do not duplicate PyInstaller flags here: a previous version of
# this script carried its own copy of the options, drifted out of sync with the
# spec, and shipped a build that crashed on launch because of a stale
# --optimize=2 flag.
#
# Output: dist/WirelinePreprocessing/  (a folder, not a single binary)
#         plus WirelinePreprocessing-<platform>.tar.gz ready to distribute.
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Building POLISH standalone distribution"
echo "============================================"
echo ""

if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

echo "Building. This takes roughly 15-20 minutes."
echo ""

python3 -m PyInstaller --noconfirm --clean \
    --distpath "distribution/dist" \
    --workpath "build" \
    "distribution/WirelinePreprocessing.spec"

APP_DIR="distribution/dist/WirelinePreprocessing"
if [ ! -x "$APP_DIR/WirelinePreprocessing" ]; then
    echo ""
    echo "ERROR: Expected executable was not produced."
    exit 1
fi

echo ""
echo "Packaging distribution archive..."
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
tar -czf "distribution/dist/WirelinePreprocessing-${PLATFORM}.tar.gz" \
    -C "distribution/dist" "WirelinePreprocessing"

# Intermediate output that --clean regenerates on every run. Removing it stops
# build artifacts accumulating across repeated builds.
rm -rf build

echo ""
echo "============================================"
echo "Build complete"
echo "============================================"
echo ""
echo "Folder:  $APP_DIR/"
echo "Archive: distribution/dist/WirelinePreprocessing-${PLATFORM}.tar.gz"
echo ""
echo "IMPORTANT: Always launch the produced binary once on this machine before"
echo "distributing it. A packaged build can fail on launch for reasons that"
echo "never occur when running from source."
echo ""
