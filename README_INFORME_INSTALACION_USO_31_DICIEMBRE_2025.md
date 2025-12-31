# CLaRa - Informe Completo de Instalación y Uso

**Fecha:** 31 de Diciembre de 2025
**Sistema:** Ubuntu 24.04 LTS | RTX 3090 24GB | Python 3.12
**Repositorio:** `/home/jose/Repositorios/ml-clara`

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Estructura del Repositorio](#estructura-del-repositorio)
4. [Instalación Paso a Paso](#instalación-paso-a-paso)
5. [Modelos Disponibles](#modelos-disponibles)
6. [Guía de Uso por Stage](#guía-de-uso-por-stage)
7. [Ejemplos de Código Completos](#ejemplos-de-código-completos)
8. [Scripts de Entrenamiento](#scripts-de-entrenamiento)
9. [Evaluación](#evaluación)
10. [Troubleshooting](#troubleshooting)
11. [Configuración del Token HuggingFace](#configuración-del-token-huggingface)
12. [Comandos Rápidos](#comandos-rápidos)

---

## Resumen Ejecutivo

**CLaRa (Continuous Latent Reasoning)** es un modelo de Retrieval-Augmented Generation (RAG) desarrollado por Apple que unifica retrieval y generación en un espacio latente continuo. El modelo logra tasas de compresión de 32x-64x mientras preserva información semántica esencial.

### Características Principales

- **Compresión Eficiente:** Reduce documentos hasta 128x sin pérdida significativa de información
- **Three-Stage Training:** Pretraining → Instruction Tuning → End-to-End
- **Retrieval Unificado:** El mismo modelo hace retrieval y generación
- **Base Mistral-7B:** Construido sobre Mistral-7B-Instruct-v0.2

### Estado Actual de la Instalación

| Componente | Estado | Versión |
|------------|--------|---------|
| Entorno Virtual | ✅ Activo | `venv_clara` (Python 3.12) |
| PyTorch | ✅ Instalado | 2.9.1+cu128 |
| CUDA | ✅ Funcionando | 12.8 (PyTorch) / 12.0 (Sistema) |
| Flash Attention | ✅ Compilado | 2.8.3 |
| Transformers | ✅ Instalado | 4.57.3 |
| DeepSpeed | ✅ Instalado | 0.18.3 |
| Modelos CLaRa | ✅ Descargados | Base, Instruct, E2E |
| Mistral-7B Base | ✅ Descargado | v0.2 |

---

## Requisitos del Sistema

### Hardware Mínimo

| Componente | Mínimo | Recomendado | Este Sistema |
|------------|--------|-------------|--------------|
| GPU VRAM | 16 GB | 24 GB | 24 GB (RTX 3090) |
| RAM | 32 GB | 64 GB | - |
| Almacenamiento | 50 GB | 100 GB | ~35 GB usados |
| CUDA Compute | 7.0+ | 8.0+ | 8.6 |

### Software

```
Ubuntu 24.04 LTS
Python 3.12.3
NVIDIA Driver 535.274.02
CUDA Toolkit 12.0
```

---

## Estructura del Repositorio

```
/home/jose/Repositorios/ml-clara/
├── venv_clara/                          # Entorno virtual Python
├── models/                              # Modelos descargados (35 GB)
│   ├── mistral-7b-instruct-v0.2/       # Modelo base Mistral (28 GB)
│   ├── clara-base/                      # Stage 1: Compression Pretraining (3.3 GB)
│   │   ├── compression-16/              # Tasa de compresión 16x
│   │   └── compression-128/             # Tasa de compresión 128x
│   ├── clara-instruct/                  # Stage 2: Instruction Tuning (2.5 GB)
│   │   ├── compression-16/
│   │   └── compression-128/
│   └── clara-e2e/                       # Stage 3: End-to-End (1.5 GB)
│       ├── compression-16/
│       └── compression-128/
├── openrlhf/                            # Framework de entrenamiento
│   ├── models/
│   │   └── modeling_clara.py           # Definición del modelo CLaRa
│   ├── datasets/
│   │   └── sft_dataset.py              # Dataset para entrenamiento
│   ├── trainer/
│   │   └── sft_trainer.py              # Trainer SFT
│   └── cli/
│       └── train_sft.py                # Script principal de entrenamiento
├── scripts/                             # Scripts de entrenamiento
│   ├── train_pretraining.sh            # Stage 1
│   ├── train_instruction_tuning.sh     # Stage 2
│   ├── train_stage_end_to_end.sh       # Stage 3
│   ├── evaluation_end_to_end.sh        # Evaluación E2E
│   └── evaluation_instruction_tuning.sh # Evaluación Stage 2
├── evaluation/                          # Framework de evaluación
│   └── evaluation_data/                # Datos de evaluación
├── example/                             # Datos de ejemplo
│   ├── pretrain_data.jsonl
│   ├── instruction_tuning_data.jsonl
│   └── end_to_end_data.jsonl
├── demo_clara.py                        # Script de demostración
├── inference.ipynb                      # Notebook de inferencia
├── requirements.txt                     # Dependencias completas
└── requirements-minimal.txt             # Dependencias mínimas
```

---

## Instalación Paso a Paso

### Opción A: Instalación desde Cero

```bash
# 1. Clonar repositorio
git clone https://github.com/apple/ml-clara.git
cd ml-clara

# 2. Crear entorno virtual
python3.12 -m venv venv_clara
source venv_clara/bin/activate

# 3. Instalar CUDA toolkit del sistema (necesario para Flash Attention)
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# 4. Instalar dependencias base
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Instalar dependencias del proyecto
pip install transformers accelerate peft datasets deepspeed
pip install einops scipy pandas tqdm rich click python-dotenv
pip install safetensors sentencepiece bitsandbytes

# 6. Instalar Flash Attention (requiere CUDA toolkit)
export CUDA_HOME=/usr
pip install flash-attn --no-build-isolation

# 7. Descargar modelos (requiere HF_TOKEN)
export HF_TOKEN="tu_token_aqui"
cd models
huggingface-cli download apple/CLaRa-7B-Base --local-dir clara-base
huggingface-cli download apple/CLaRa-7B-Instruct --local-dir clara-instruct
huggingface-cli download apple/CLaRa-7B-E2E --local-dir clara-e2e
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir mistral-7b-instruct-v0.2
```

### Opción B: Usar Instalación Existente (Este Sistema)

```bash
# Activar entorno existente
source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate

# Configurar PYTHONPATH
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH

# Configurar token HuggingFace (ya está en ~/.env)
export HF_TOKEN=$(grep HF_TOKEN /home/jose/.env | cut -d'"' -f2)

# Verificar instalación
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Script de Activación Rápida

Crear archivo `activate_clara.sh`:

```bash
#!/bin/bash
# /home/jose/Repositorios/ml-clara/activate_clara.sh

source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH
export HF_TOKEN=$(grep HF_TOKEN /home/jose/.env | cut -d'"' -f2)
export CUDA_HOME=/usr

echo "CLaRa environment activated!"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
```

---

## Modelos Disponibles

### Arquitectura de los Stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLaRa Three-Stage Training                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: Compression Pretraining (SCP)                                 │
│  ├── Modelo: clara-base/compression-{16,128}                            │
│  ├── Objetivo: Aprender a comprimir documentos preservando semántica    │
│  ├── Loss: MSE + QA Loss                                                │
│  └── Output: Paráfrasis del documento comprimido                        │
│                                                                          │
│  Stage 2: Compression Instruction Tuning                                │
│  ├── Modelo: clara-instruct/compression-{16,128}                        │
│  ├── Objetivo: Fine-tune para tareas de QA con documentos comprimidos   │
│  ├── Input: Pregunta + Documentos comprimidos                           │
│  └── Output: Respuesta en texto                                         │
│                                                                          │
│  Stage 3: End-to-End (CLaRa)                                            │
│  ├── Modelo: clara-e2e/compression-{16,128}                             │
│  ├── Objetivo: Retrieval + Generación unificados                        │
│  ├── Input: Pregunta + Pool de documentos candidatos                    │
│  ├── Output: Respuesta + Índices de documentos seleccionados            │
│  └── Diferenciable: Usa top-k diferenciable para entrenamiento E2E      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tabla de Modelos

| Modelo | Stage | Compresión | Tamaño | Uso Principal |
|--------|-------|------------|--------|---------------|
| `clara-base/compression-16` | 1 | 16x | 1.5 GB | Paráfrasis, embeddings |
| `clara-base/compression-128` | 1 | 128x | 1.8 GB | Compresión agresiva |
| `clara-instruct/compression-16` | 2 | 16x | 1.2 GB | QA con compresión |
| `clara-instruct/compression-128` | 2 | 128x | 1.3 GB | QA ultra-comprimido |
| `clara-e2e/compression-16` | 3 | 16x | 0.75 GB | RAG completo |
| `clara-e2e/compression-128` | 3 | 128x | 0.75 GB | RAG ultra-eficiente |
| `mistral-7b-instruct-v0.2` | Base | - | 28 GB | Modelo base |

### Parámetros de Configuración (config.json)

```json
{
  "compr_rate": 16,              // Tasa de compresión (16 o 128)
  "doc_max_length": 256,         // Longitud máxima de documento
  "generation_top_k": 5,         // Top-K documentos para generación (Stage 3)
  "max_new_tokens": 128,         // Tokens máximos a generar
  "lora": true,                  // Usar LoRA adapters
  "lora_r": 16,                  // Rank de LoRA
  "training_stage": "stage2",    // stage1, stage1_2, stage2
  "quantization": "no"           // no, 4bit, 8bit
}
```

---

## Guía de Uso por Stage

### Stage 1: Compression Pretraining

**Propósito:** Generar paráfrasis de documentos desde representaciones comprimidas.

```python
from transformers import AutoModel
import torch

# Cargar modelo Stage 1
model = AutoModel.from_pretrained(
    "/home/jose/Repositorios/ml-clara/models/clara-base/compression-16",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")

# Documentos a comprimir
documents = [[
    "La inteligencia artificial es un campo de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
    "El aprendizaje profundo es una rama del machine learning basada en redes neuronales artificiales.",
    "Los transformers revolucionaron el procesamiento del lenguaje natural en 2017."
]]

# Generar paráfrasis (pregunta vacía para Stage 1)
questions = [""]

output = model.generate_from_paraphrase(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print("Paráfrasis generada:", output[0])
```

### Stage 2: Instruction Tuning

**Propósito:** Responder preguntas usando documentos comprimidos.

```python
from transformers import AutoModel
import torch

# Cargar modelo Stage 2
model = AutoModel.from_pretrained(
    "/home/jose/Repositorios/ml-clara/models/clara-instruct/compression-16",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")

# Documentos y pregunta
documents = [[
    "Python fue creado por Guido van Rossum y lanzado en 1991.",
    "Python es conocido por su sintaxis clara y legible.",
    "El nombre Python viene del grupo de comedia Monty Python."
]]

questions = ["¿Quién creó Python y en qué año?"]

# Generar respuesta
output = model.generate_from_text(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print("Respuesta:", output[0])
```

### Stage 3: End-to-End (RAG Completo)

**Propósito:** Retrieval automático + Generación de respuestas.

```python
from transformers import AutoModel
import torch

# Cargar modelo Stage 3 (E2E)
model = AutoModel.from_pretrained(
    "/home/jose/Repositorios/ml-clara/models/clara-e2e/compression-16",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")

# Pool de documentos candidatos (típicamente 20+)
documents = [[
    "Apple fue fundada el 1 de abril de 1976 por Steve Jobs, Steve Wozniak y Ronald Wayne.",
    "La primera computadora Apple, conocida como Apple I, fue diseñada por Wozniak.",
    "La sede de Apple está ubicada en Cupertino, California.",
    "La empresa es famosa por sus productos iPhone, iPad y MacBook.",
    "Tim Cook es el CEO de Apple desde el 24 de agosto de 2011.",
    "Apple lanzó el primer iPhone en 2007.",
    "El sistema operativo de Apple se llama macOS.",
    "Apple Music fue lanzado en 2015.",
    "La App Store se lanzó en 2008.",
    "Apple Watch fue presentado en 2015."
]]

questions = ["¿Dónde está la sede de Apple y quién la fundó?"]

# Generar con retrieval automático
output, topk_indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print("Respuesta:", output[0])
print("Documentos seleccionados (índices):", topk_indices[0].tolist())
print("\nDocumentos utilizados:")
for idx in topk_indices[0]:
    print(f"  [{idx}]: {documents[0][idx][:80]}...")
```

---

## Ejemplos de Código Completos

### Ejemplo 1: Inferencia Básica con Todos los Stages

```python
#!/usr/bin/env python3
"""
Ejemplo completo de inferencia con CLaRa
Archivo: examples/inference_all_stages.py
"""

import torch
from transformers import AutoModel
import time

def load_model(model_path):
    """Carga un modelo CLaRa con configuración óptima."""
    return AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

def main():
    base_path = "/home/jose/Repositorios/ml-clara/models"

    # Documentos de prueba
    docs = [[
        "El Sol es una estrella de tipo G en el centro del Sistema Solar.",
        "La Tierra orbita alrededor del Sol a una distancia promedio de 150 millones de km.",
        "La Luna es el único satélite natural de la Tierra.",
        "Marte es el cuarto planeta del Sistema Solar.",
        "Júpiter es el planeta más grande del Sistema Solar."
    ]]

    question = ["¿Cuál es la distancia de la Tierra al Sol?"]

    print("=" * 60)
    print("CLaRa - Demostración de los Tres Stages")
    print("=" * 60)

    # Stage 1: Paráfrasis
    print("\n[Stage 1] Compression Pretraining - Paráfrasis")
    print("-" * 40)
    model1 = load_model(f"{base_path}/clara-base/compression-16")
    start = time.time()
    output1 = model1.generate_from_paraphrase(
        questions=[""],
        documents=docs,
        max_new_tokens=64
    )
    print(f"Tiempo: {time.time()-start:.2f}s")
    print(f"Paráfrasis: {output1[0]}")
    del model1
    torch.cuda.empty_cache()

    # Stage 2: QA
    print("\n[Stage 2] Instruction Tuning - QA")
    print("-" * 40)
    model2 = load_model(f"{base_path}/clara-instruct/compression-16")
    start = time.time()
    output2 = model2.generate_from_text(
        questions=question,
        documents=docs,
        max_new_tokens=64
    )
    print(f"Tiempo: {time.time()-start:.2f}s")
    print(f"Pregunta: {question[0]}")
    print(f"Respuesta: {output2[0]}")
    del model2
    torch.cuda.empty_cache()

    # Stage 3: E2E
    print("\n[Stage 3] End-to-End - RAG Completo")
    print("-" * 40)
    model3 = load_model(f"{base_path}/clara-e2e/compression-16")
    start = time.time()
    output3, topk = model3.generate_from_questions(
        questions=question,
        documents=docs,
        max_new_tokens=64
    )
    print(f"Tiempo: {time.time()-start:.2f}s")
    print(f"Pregunta: {question[0]}")
    print(f"Respuesta: {output3[0]}")
    print(f"Docs seleccionados: {topk[0].tolist()}")
    del model3
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Demostración completada!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

### Ejemplo 2: Procesamiento por Lotes

```python
#!/usr/bin/env python3
"""
Procesamiento por lotes con CLaRa Stage 3
"""

import torch
from transformers import AutoModel
from typing import List, Tuple

class CLaRaRAG:
    def __init__(self, model_path: str, compression_rate: int = 16):
        self.model = AutoModel.from_pretrained(
            f"{model_path}/clara-e2e/compression-{compression_rate}",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda")
        self.model.eval()

    def answer(
        self,
        questions: List[str],
        documents: List[List[str]],
        max_tokens: int = 64
    ) -> List[Tuple[str, List[int]]]:
        """
        Procesa múltiples preguntas en batch.

        Args:
            questions: Lista de preguntas
            documents: Lista de listas de documentos (una por pregunta)
            max_tokens: Tokens máximos por respuesta

        Returns:
            Lista de tuplas (respuesta, índices_documentos_usados)
        """
        with torch.no_grad():
            outputs, topk_indices = self.model.generate_from_questions(
                questions=questions,
                documents=documents,
                max_new_tokens=max_tokens
            )

        results = []
        for i, (output, indices) in enumerate(zip(outputs, topk_indices)):
            results.append((output, indices.tolist()))

        return results

# Uso
if __name__ == "__main__":
    rag = CLaRaRAG("/home/jose/Repositorios/ml-clara/models")

    # Batch de preguntas
    questions = [
        "¿Qué es Python?",
        "¿Cuántos planetas hay en el Sistema Solar?"
    ]

    documents = [
        ["Python es un lenguaje de programación creado en 1991.",
         "Python es interpretado y de alto nivel.",
         "Guido van Rossum creó Python."],
        ["El Sistema Solar tiene 8 planetas.",
         "Plutón fue reclasificado como planeta enano en 2006.",
         "Los planetas interiores son Mercurio, Venus, Tierra y Marte."]
    ]

    results = rag.answer(questions, documents)

    for q, (answer, docs_used) in zip(questions, results):
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"Docs: {docs_used}\n")
```

### Ejemplo 3: Integración con Base de Conocimientos

```python
#!/usr/bin/env python3
"""
CLaRa con base de conocimientos local
"""

import torch
from transformers import AutoModel
import json
from pathlib import Path

class KnowledgeBase:
    def __init__(self, kb_path: str):
        self.documents = []
        self.load_knowledge_base(kb_path)

    def load_knowledge_base(self, path: str):
        """Carga documentos desde archivo JSONL."""
        with open(path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc['text'])

    def get_documents(self, max_docs: int = 20) -> List[str]:
        """Retorna hasta max_docs documentos."""
        return self.documents[:max_docs]

class CLaRaKB:
    def __init__(self, model_path: str, kb_path: str):
        self.model = AutoModel.from_pretrained(
            f"{model_path}/clara-e2e/compression-16",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda")
        self.kb = KnowledgeBase(kb_path)

    def query(self, question: str) -> dict:
        """
        Consulta la base de conocimientos.
        """
        documents = [self.kb.get_documents()]

        with torch.no_grad():
            output, topk = self.model.generate_from_questions(
                questions=[question],
                documents=documents,
                max_new_tokens=128
            )

        return {
            "question": question,
            "answer": output[0],
            "sources": [
                {"index": idx, "text": documents[0][idx]}
                for idx in topk[0].tolist()
            ]
        }

# Uso
# kb_rag = CLaRaKB(
#     "/home/jose/Repositorios/ml-clara/models",
#     "/path/to/knowledge_base.jsonl"
# )
# result = kb_rag.query("¿Cuál es la capital de Francia?")
```

---

## Scripts de Entrenamiento

### Stage 1: Compression Pretraining

```bash
#!/bin/bash
# scripts/train_pretraining.sh

source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH

deepspeed --num_gpus 1 openrlhf/cli/train_sft.py \
    --pretrain mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset /path/to/pretrain_data.jsonl \
    --save_path ./checkpoints/stage1 \
    --stage stage1 \
    --compress_rate 16 \
    --doc_max_length 256 \
    --max_len 2048 \
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --learning_rate 1e-4 \
    --max_epochs 3 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --mse_loss \
    --qa_loss
```

### Stage 2: Instruction Tuning

```bash
#!/bin/bash
# scripts/train_instruction_tuning.sh

source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH

deepspeed --num_gpus 1 openrlhf/cli/train_sft.py \
    --pretrain mistralai/Mistral-7B-Instruct-v0.2 \
    --pretrain_checkpoint ./checkpoints/stage1 \
    --dataset /path/to/instruction_tuning_data.jsonl \
    --save_path ./checkpoints/stage2 \
    --stage stage1_2 \
    --compress_rate 16 \
    --generation_top_k 5 \
    --max_len 2048 \
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --learning_rate 1e-4 \
    --max_epochs 3 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --mse_loss \
    --do_eval_gen
```

### Stage 3: End-to-End

```bash
#!/bin/bash
# scripts/train_stage_end_to_end.sh

source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH

deepspeed --num_gpus 1 openrlhf/cli/train_sft.py \
    --pretrain mistralai/Mistral-7B-Instruct-v0.2 \
    --pretrain_checkpoint ./checkpoints/stage2 \
    --dataset /path/to/end_to_end_data.jsonl \
    --save_path ./checkpoints/stage3 \
    --stage stage2 \
    --compress_rate 16 \
    --generation_top_k 5 \
    --max_len 1024 \
    --train_batch_size 2 \
    --micro_train_batch_size 1 \
    --learning_rate 5e-6 \
    --max_epochs 3 \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --do_eval_gen
```

### Formato de Datos de Entrenamiento

**Stage 1 (Pretraining):**
```json
{
    "data_type": "qa",
    "question": ["¿Pregunta de ejemplo?"],
    "answers": ["Respuesta de ejemplo"],
    "docs": ["Documento con información relevante"]
}
```

**Stage 2 y 3:**
```json
{
    "question": "¿Pregunta de ejemplo?",
    "docs": ["Doc 1", "Doc 2", "Doc 3"],
    "gold_answer": "Respuesta correcta"
}
```

---

## Evaluación

### Ejecutar Evaluación End-to-End

```bash
#!/bin/bash
source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH

bash scripts/evaluation_end_to_end.sh
```

### Datasets de Evaluación Soportados

| Dataset | Descripción | Tipo |
|---------|-------------|------|
| HotpotQA | QA multi-hop | Razonamiento |
| MuSiQue | QA multi-hop diverso | Razonamiento |
| 2WikiMultiHopQA | QA sobre Wikipedia | Razonamiento |
| Natural Questions | QA open-domain | Factual |

### Métricas

- **Exact Match (EM):** Coincidencia exacta con respuesta gold
- **F1 Score:** Overlap de tokens entre predicción y gold
- **Recall@K:** Precisión del retrieval

---

## Troubleshooting

### Error: CUDA out of memory

```python
# Solución 1: Usar dtype más pequeño
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16  # o torch.bfloat16
)

# Solución 2: Limitar documentos
documents = [docs[:10]]  # Máximo 10 documentos

# Solución 3: Reducir max_new_tokens
output = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=32  # Reducir de 64
)

# Solución 4: Limpiar caché
torch.cuda.empty_cache()
```

### Error: Flash Attention no disponible

```bash
# Verificar CUDA toolkit
nvcc --version

# Si no está instalado:
sudo apt-get install nvidia-cuda-toolkit

# Reinstalar flash-attn
export CUDA_HOME=/usr
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### Error: Module not found 'openrlhf'

```bash
# Asegurar PYTHONPATH
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH
```

### Error: Token de HuggingFace

```bash
# Configurar token
export HF_TOKEN=$(grep HF_TOKEN /home/jose/.env | cut -d'"' -f2)

# O directamente
huggingface-cli login
```

### Warning: PYTORCH_CUDA_ALLOC_CONF deprecated

```bash
# Usar la nueva variable
export PYTORCH_ALLOC_CONF="expandable_segments:True"
```

---

## Configuración del Token HuggingFace

El token de HuggingFace está almacenado en `/home/jose/.env`:

```bash
# Extraer token
HF_TOKEN=$(grep HF_TOKEN /home/jose/.env | cut -d'"' -f2)

# Usar en scripts
export HF_TOKEN="$HF_TOKEN"

# O configurar globalmente
huggingface-cli login --token $HF_TOKEN
```

---

## Comandos Rápidos

### Activar Entorno

```bash
source /home/jose/Repositorios/ml-clara/venv_clara/bin/activate
export PYTHONPATH=/home/jose/Repositorios/ml-clara:$PYTHONPATH
```

### Ejecutar Demo

```bash
cd /home/jose/Repositorios/ml-clara
python demo_clara.py
```

### Verificar GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Verificar Paquetes

```bash
pip list | grep -E "torch|transformers|flash|deep|peft"
```

### Limpiar Caché CUDA

```python
import torch
torch.cuda.empty_cache()
```

### Descargar Modelo Adicional

```bash
export HF_TOKEN=$(grep HF_TOKEN /home/jose/.env | cut -d'"' -f2)
huggingface-cli download apple/CLaRa-7B-E2E --local-dir models/clara-e2e
```

---

## Información del Sistema

```
Sistema Operativo: Ubuntu 24.04 LTS
Kernel: 6.14.0-37-generic
Python: 3.12.3
GPU: NVIDIA GeForce RTX 3090 (24 GB VRAM)
Driver NVIDIA: 535.274.02
CUDA (Sistema): 12.0
CUDA (PyTorch): 12.8
Entorno Virtual: venv_clara
Ubicación: /home/jose/Repositorios/ml-clara
```

---

## Referencias

- **Paper:** [CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning](https://arxiv.org/abs/2511.18659)
- **GitHub:** [apple/ml-clara](https://github.com/apple/ml-clara)
- **HuggingFace:** [apple/CLaRa-7B-E2E](https://huggingface.co/apple/CLaRa-7B-E2E)
- **Video Tutorial:** [Fahd Mirza - Installation Guide](https://youtu.be/al2VoAKn8GU)

---

*Documento generado el 31 de Diciembre de 2025*
*Última verificación exitosa: `python demo_clara.py` - OK*
