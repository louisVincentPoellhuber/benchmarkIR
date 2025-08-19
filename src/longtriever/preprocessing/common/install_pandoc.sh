#!/usr/bin/env bash

# -----------------------------
# CONFIGURATION
# -----------------------------
PANDOC_VERSION="3.2.1"
INSTALL_DIR="$HOME/pandoc"
mkdir -p "$INSTALL_DIR"

# Detect OS
OS="$(uname | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

# Set download URL
FILE="pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz"

URL="https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/${FILE}"

# -----------------------------
# DOWNLOAD
# -----------------------------
echo "Downloading Pandoc $PANDOC_VERSION..."
wget -q --show-progress "$URL" -O "/tmp/$FILE"

# -----------------------------
# EXTRACT
# -----------------------------
echo "Extracting..."
if [[ "$FILE" == *.tar.gz ]]; then
    tar -xzf "/tmp/$FILE" -C "$INSTALL_DIR" --strip-components=1
elif [[ "$FILE" == *.zip ]]; then
    unzip -q "/tmp/$FILE" -d "$INSTALL_DIR"
fi

# -----------------------------
# UPDATE PATH
# -----------------------------
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "Adding $INSTALL_DIR to PATH in ~/.bashrc"
    echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.bashrc
    export PATH="$INSTALL_DIR:$PATH"
fi

# -----------------------------
# VERIFY INSTALL
# -----------------------------
echo "Pandoc installed at $INSTALL_DIR"
pandoc --version
