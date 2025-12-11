# CLaRa Stage 3 - Guía de Uso Rápido

## Activar el Ambiente

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara
cd /home/jose/Repositorios/ml-clara
```

---

## Uso Básico en Python

### 1. Importar y Cargar el Modelo

```python
from transformers import AutoModel
import torch

# Cargar el modelo
model = AutoModel.from_pretrained(
    "./models/clara-e2e/compression-128",
    trust_remote_code=True
).to('cuda' if torch.cuda.is_available() else 'cpu')

print("✓ Modelo cargado exitosamente")
```

### 2. Hacer una Pregunta con Documentos

```python
# Documentos (lista de listas - un batch de documentos)
documents = [[
    "Apple fue fundada el 1 de abril de 1976 por Steve Jobs, Steve Wozniak y Ronald Wayne.",
    "El primer Apple Computer, conocido como Apple I, fue diseñado por Wozniak.",
    "La sede de Apple está ubicada en Cupertino, California.",
    "La empresa es famosa por iPhone, iPad y MacBook.",
    "Tim Cook ha sido CEO de Apple desde el 24 de agosto de 2011."
]]

# Preguntas (lista de preguntas)
questions = ["¿Dónde está ubicada Apple y quién la fundó?"]

# Generar respuesta
output, topk_indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

# Mostrar resultados
print(f"Pregunta: {questions[0]}")
print(f"Respuesta: {output[0]}")
print(f"Documentos seleccionados: {topk_indices[0]}")

for idx in topk_indices[0]:
    print(f"  - Doc {idx}: {documents[0][idx][:80]}...")
```

### 3. Ejecutar el Demo Completo

```bash
python demo_clara.py
```

---

## Parámetros Principales

### `generate_from_questions()`

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `questions` | `List[str]` | Preguntas a responder | Requerido |
| `documents` | `List[List[str]]` | Documentos para cada pregunta | Requerido |
| `max_new_tokens` | `int` | Máximo de tokens a generar | 128 |
| `generation_top_k` | `int` | Top-k para sampling | 5 |

### Opciones de Compresión

El modelo descargado usa **compresión 128x** (comprime 256 tokens de documento en ~2 tokens)

Para usar compresión diferente:

```python
# Compresión 16x (mejor calidad, menos compresión)
model = AutoModel.from_pretrained(
    "./models/clara-e2e/compression-16",
    trust_remote_code=True
).to('cuda')
```

---

## Ejemplos Prácticos

### Ejemplo 1: Pregunta Única

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./models/clara-e2e/compression-128",
    trust_remote_code=True
).to('cuda')

docs = [[
    "Python es un lenguaje de programación interpretado.",
    "Fue creado por Guido van Rossum en 1991.",
    "Es ampliamente usado en ciencia de datos y ML."
]]

q = ["¿Cuándo fue creado Python?"]
answer, indices = model.generate_from_questions(questions=q, documents=docs)

print(answer[0])
```

### Ejemplo 2: Batch de Preguntas

```python
# Múltiples documentos y preguntas
documents = [
    ["El Everest es la montaña más alta...", "...con 8,849 metros..."],
    ["Paris es la capital de Francia...", "...ubicada en el río Sena..."]
]

questions = [
    "¿Cuál es la altura del Everest?",
    "¿En qué país está Paris?"
]

answers, selected_docs = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=32
)

for q, a, idx in zip(questions, answers, selected_docs):
    print(f"Q: {q}\nA: {a}\nSeleccionado doc: {idx}\n")
```

### Ejemplo 3: Usar CPU

```python
# Útil si se agota la VRAM de GPU
model = AutoModel.from_pretrained(
    "./models/clara-e2e/compression-128",
    trust_remote_code=True
).to('cpu')

output, indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=32
)
```

### Ejemplo 4: Muchos Documentos

```python
# CLaRa puede procesar hasta 128 documentos simultáneamente
documents = [[
    f"Documento número {i}: información sobre tema {i}"
    for i in range(100)  # 100 documentos
]]

questions = ["¿Qué dice el documento 50?"]

output, indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print(f"Respuesta: {output[0]}")
print(f"Docs seleccionados: {indices[0]}")  # Índices de docs relevantes
```

---

## Características Clave

### Compresión de Documentos
- **32x-128x** compresión sin pérdida significativa
- Procesa documentos largos eficientemente
- Mantiene información relevante

### Recuperación Integrada
- Selecciona automáticamente documentos relevantes
- No requiere re-ranking separado
- Retorna índices de documentos seleccionados

### Generación de Calidad
- Respuestas basadas en documentos proporcionados
- Soporte para múltiples documentos simultáneamente
- Configuración flexible de longitud de respuesta

---

## Benchmarks del Modelo

Exactitud en datasets estándar (CR=4, compresión 128x):

| Dataset | Exactitud |
|---------|-----------|
| NQ | 57.05% |
| HotpotQA | 45.09% |
| MuSiQue | 10.34% |
| 2Wiki | 46.94% |

---

## Solución de Problemas

### "CUDA out of memory"

```python
# Opción 1: Usar CPU
model = model.to('cpu')

# Opción 2: Reducir tokens generados
output = model.generate_from_questions(
    questions=q,
    documents=docs,
    max_new_tokens=32  # Reducir de 128 a 32
)

# Opción 3: Usar menos documentos
docs_subset = [docs[0][:5]]  # Solo primeros 5 docs
```

### "Model loading is slow"

- Primera ejecución descarga modelos (normal)
- Próximas ejecuciones serán más rápidas
- Verifica conexión a internet para HuggingFace

### "TypeError: unsupported operand type(s)"

Asegúrate de que:
1. Las preguntas son `List[str]`
2. Los documentos son `List[List[str]]` (lista de listas)
3. Todos los strings son texto válido

---

## Estructura de Directorio

```
/home/jose/Repositorios/ml-clara/
├── models/
│   └── clara-e2e/
│       ├── compression-128/     # Modelo principal
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── modeling_clara.py
│       │   └── tokenizer.model
│       └── compression-16/       # Alternativa
├── demo_clara.py                 # Script de demostración
├── requirements-minimal.txt      # Dependencias
├── INSTALACION_COMPLETA.md      # Guía de instalación
└── USO_RAPIDO.md                # Este archivo
```

---

## Información de Versiones

```
Python: 3.10.19
PyTorch: 2.2.0
Transformers: 4.57.3
PEFT: 0.18.0
Datasets: 4.4.1
HuggingFace Hub: 0.36.0
CUDA: 12.1
```

---

## Recursos Adicionales

- **Paper**: https://arxiv.org/abs/2511.18659
- **Modelos HF**: https://huggingface.co/apple/CLaRa-7B-E2E
- **GitHub**: https://github.com/apple/CLaRa
- **Documentación**: `INSTALACION_COMPLETA.md`

---

**Última actualización**: 11 de Diciembre 2025
**Estado**: ✅ Operativo
