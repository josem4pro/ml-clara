#!/bin/bash

# CLaRa Stage 3 Setup Script
# This script installs and downloads the CLaRa-7B-E2E model

set -e  # Exit on error

echo "=========================================="
echo "CLaRa Stage 3 Installation Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Miniconda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}⚠ Conda not found. Installing Miniconda...${NC}"
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    echo "123456" | sudo -S bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
    source /opt/miniconda3/etc/profile.d/conda.sh
fi

# Initialize conda
echo -e "${BLUE}→ Initializing conda...${NC}"
source /opt/miniconda3/etc/profile.d/conda.sh

# Create environment
echo -e "${BLUE}→ Creating conda environment (clara)...${NC}"
conda create -n clara python=3.10 -y || true
conda activate clara

# Install dependencies
echo -e "${BLUE}→ Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt --no-cache-dir

# Install HuggingFace CLI
echo -e "${BLUE}→ Installing HuggingFace Hub CLI...${NC}"
pip install huggingface-hub -q

# Create models directory
mkdir -p models

# Download model
echo -e "${BLUE}→ Downloading CLaRa-7B-E2E model from HuggingFace...${NC}"
source ~/.env
huggingface-cli download apple/CLaRa-7B-E2E --local-dir ./models/clara-e2e

echo -e "${GREEN}✓ Installation completed successfully!${NC}"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Activate the environment:"
echo "   source /opt/miniconda3/etc/profile.d/conda.sh && conda activate clara"
echo ""
echo "2. Run the demo:"
echo "   python demo_clara.py"
echo ""
echo "3. Export PYTHONPATH (optional):"
echo "   export PYTHONPATH=/home/jose/Repositorios/ml-clara:\$PYTHONPATH"
echo "=========================================="
