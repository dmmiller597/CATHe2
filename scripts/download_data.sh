#!/bin/bash
# scripts/download_data.sh
# This script downloads the CATHe Dataset from Zenodo, unzips it,
# reorganizes the files by moving all contents from the top-level folder
# into the "data" directory, and then removes the original folder.
# Ensure you are running this script from your desired working directory.

# Zenodo dataset URL
ZENODO_URL="https://zenodo.org/records/6327572/files/CATHe%20Dataset.zip?download=1"
OUTPUT_ZIP="CATHe_Dataset.zip"

# Check if file already exists
if [ -f "$OUTPUT_ZIP" ]; then
    echo "File '$OUTPUT_ZIP' already exists. Skipping download."
else
    echo "Downloading CATHe Dataset from Zenodo..."
    # Try wget first; if not, try curl.
    if command -v wget > /dev/null 2>&1; then
        wget -O "$OUTPUT_ZIP" "$ZENODO_URL"
    elif command -v curl > /dev/null 2>&1; then
        curl -L -o "$OUTPUT_ZIP" "$ZENODO_URL"
    else
        echo "Error: Neither wget nor curl is installed."
        exit 1
    fi
fi

# Unzip the dataset
if command -v unzip > /dev/null 2>&1; then
    echo "Unzipping '$OUTPUT_ZIP'..."
    unzip -o "$OUTPUT_ZIP"
else
    echo "Error: 'unzip' utility is not available on this system."
    exit 1
fi

# Reorganize the extracted contents:
# If the zip file created a single top-level directory (e.g., "CATHe Dataset"),
# move its contents into the "data" folder and remove that directory.
EXTRACTED_DIR=$(unzip -Z -1 "$OUTPUT_ZIP" | head -n 1 | cut -d/ -f1)

if [ -d "$EXTRACTED_DIR" ]; then
    echo "Reorganizing files: moving contents of '$EXTRACTED_DIR' into 'data' directory..."
    mkdir -p data
    # Enable dotglob to include hidden files
    shopt -s dotglob
    mv "$EXTRACTED_DIR"/* data/
    shopt -u dotglob
    rm -rf "$EXTRACTED_DIR"
else
    echo "No top-level directory detected; assuming files are already in place."
fi

echo "Dataset downloaded, extracted, and reorganized successfully."
rm "$OUTPUT_ZIP" 