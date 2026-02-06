#!/bin/bash
# Setup script to create library symlinks
# This fixes broken symlinks when cloning on Windows or systems with core.symlinks=false

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

echo "Setting up library symlinks in $LIB_DIR..."

cd "$LIB_DIR"

# Remove existing symlinks (they may be broken text files on Windows)
rm -f libjpeg.so libturbojpeg.so

# Create symlinks to versioned libraries
ln -sf libjpeg.so.62.4.0 libjpeg.so
ln -sf libturbojpeg.so.0.3.0 libturbojpeg.so

echo "Library symlinks created successfully:"
ls -lh libjpeg.so libturbojpeg.so
