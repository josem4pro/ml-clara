#!/bin/bash

# Activar ambiente Clara
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara

# Cargar variables de entorno
source ~/.env

# Descargar modelo CLaRa Stage 3
echo "Descargando CLaRa-7B-E2E desde HuggingFace..."
huggingface-cli download apple/CLaRa-7B-E2E --local-dir ./models/clara-e2e

echo "✅ Descarga completada"
