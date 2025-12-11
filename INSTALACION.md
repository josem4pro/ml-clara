# CLaRa Stage 3 - Guía de Instalación Rápida

## Estado Actual

**Instalación en progreso:**
- ✅ Conda instalado en `/opt/miniconda3`
- ✅ Ambiente `clara` creado con Python 3.10
- ✅ Dependencias corregidas (versiones compatibles)
- ⏳ Instalando dependencias (ID: 615bcd)
- ⏳ Descargando modelo CLaRa-7B-E2E (próximo paso)

**Versiones finales:**
- PyTorch 2.2.0
- Transformers 4.40.0
- TorchVision 0.17.0

## Paso 1: Esperar Instalación

La instalación de dependencies puede tomar **30-60 minutos**. Puedes monitorear:

```bash
# Ver status de proceso
ps aux | grep pip
```

## Paso 2: Activar Ambiente (Una vez instalado)

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara
```

## Paso 3: Descargar Modelo

Una vez que pip termine, ejecuta:

```bash
cd /home/jose/Repositorios/ml-clara
bash install_and_download_clara.sh
```

Este script:
- Verifica que todo esté instalado correctamente
- Descarga el modelo CLaRa-7B-E2E (~15GB)
- Toma 15-30 minutos (depende de velocidad de internet)

## Paso 4: Probar Instalación

Una vez descargado el modelo:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara
python demo_clara.py
```

## Estructura de Directorios

```
/home/jose/Repositorios/ml-clara/
├── models/
│   └── clara-e2e/          # Modelo descargado (~15GB)
├── scripts/                 # Scripts de entrenamiento
├── openrlhf/                # Framework core
├── evaluation/              # Evaluación
├── demo_clara.py            # Script de demostración
├── download_model.sh        # Descarga modelo
├── setup_clara.sh           # Setup completo
└── requirements.txt         # Dependencias (corregido)
```

## Uso Básico

### Python Script

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./models/clara-e2e",
    trust_remote_code=True
).to('cuda')

documents = [[
    "Document 1 content...",
    "Document 2 content...",
]]

questions = ["Your question here"]

output, topk_indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print("Answer:", output[0])
print("Selected docs:", topk_indices)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"
```bash
conda activate clara
pip install transformers torch
```

### "CUDA out of memory"
Reduce `max_new_tokens` o usa CPU:
```python
model = model.to('cpu')  # CPU mode
```

### Problema descargando modelo
Verifica HF_TOKEN:
```bash
source ~/.env
echo $HF_TOKEN
huggingface-cli login  # Si es necesario
```

## Recursos

- **Paper**: https://arxiv.org/abs/2511.18659
- **Models**: https://huggingface.co/apple/CLaRa-7B-E2E
- **GitHub**: https://github.com/apple/CLaRa

## Próximos Pasos

1. ✅ Esperar instalación (en progreso)
2. ✅ Descargar modelo
3. 📝 Crear scripts personalizados
4. 🚀 Entrenar con tus datos (opcional)
