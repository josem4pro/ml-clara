#!/bin/bash

# Complete CLaRa Setup: Download Model and Run Test
# Run this script after pip installation completes

set -e

cd /home/jose/Repositorios/ml-clara

echo "=========================================="
echo "CLaRa Stage 3 - Download & Test"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Activate environment
echo -e "${BLUE}→ Activating clara environment...${NC}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara

# Step 2: Load HF token
echo -e "${BLUE}→ Loading HuggingFace token...${NC}"
if [ -f ~/.env ]; then
    source ~/.env
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}⚠ HF_TOKEN not found in ~/.env${NC}"
    else
        echo -e "${GREEN}✓ HF_TOKEN loaded${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ~/.env not found${NC}"
fi

# Step 3: Create models directory
mkdir -p ./models

# Step 4: Download model
echo -e "${BLUE}→ Downloading CLaRa-7B-E2E model...${NC}"
echo "  This may take 15-30 minutes depending on connection speed"
huggingface-cli download apple/CLaRa-7B-E2E --local-dir ./models/clara-e2e

echo -e "${GREEN}✓ Model downloaded successfully!${NC}"

# Step 5: Verify installation
echo -e "${BLUE}→ Verifying installation...${NC}"
python -c "
import torch
import transformers
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# Step 6: Run demo
echo -e "${BLUE}→ Running demo...${NC}"
python demo_clara.py

echo ""
echo -e "${GREEN}=========================================="
echo "✓ CLaRa installation complete!"
echo "==========================================${NC}"
