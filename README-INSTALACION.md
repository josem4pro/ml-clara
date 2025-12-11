# CLaRa Stage 3 - Guía de Instalación y Configuración

**Estado**: ✅ **Instalación completada y verificada** (11 Dic 2025)

> **Nota**: Este proyecto ya está completamente instalado en `/home/jose/Repositorios/ml-clara/`. Sigue los comandos de abajo para usar el modelo.

---

## 🚀 Inicio Rápido (30 segundos)

```bash
# 1. Activar el ambiente
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate clara
cd /home/jose/Repositorios/ml-clara

# 2. Ejecutar demo
python demo_clara.py

# 3. Ver documentación
cat USO_RAPIDO.md
```

---

## 📋 Documentación Disponible

| Documento | Contenido |
|-----------|----------|
| **INSTALACION_COMPLETA.md** | Reporte exhaustivo de instalación, problemas solucionados, test de verificación |
| **USO_RAPIDO.md** | Guía práctica con ejemplos de código para usar el modelo |
| **TROUBLESHOOTING.md** | Solución a 15+ problemas comunes |
| **README.md** | Documentación original del proyecto Apple CLaRa |

---

## 📦 Lo que Está Instalado

### Modelos
- ✅ **CLaRa-7B-E2E**: 745 MB descargado y verificado
- ✅ **Mistral-7B-Instruct-v0.2**: 14 GB descargado y verificado

### Dependencias
- ✅ **PyTorch**: 2.2.0 con CUDA 12.1
- ✅ **Transformers**: 4.57.3
- ✅ **PEFT**: 0.18.0
- ✅ **+50 paquetes**: Todas las dependencias resueltas

### Ambiente Conda
- ✅ **Conda**: v25.9.1 en `/opt/miniconda3`
- ✅ **Ambiente "clara"**: Python 3.10.19, completamente funcional

---

## ✅ Test de Verificación

Se realizó test exitoso el **11 de Diciembre 2025, 03:21 UTC**:

```
Pregunta: "Where is Apple headquartered and who founded it?"

Documentos (5):
  1. Fundación de Apple
  2. Diseño Apple I
  3. Sede de Apple        ← Seleccionado
  4. Productos Apple
  5. CEO Tim Cook         ← Seleccionado

Respuesta Generada:
  "Apple is headquartered in Cupertino, California,
   and was founded by Steve Jobs, Steve Wozniak,
   and Ronald Wayne."

✅ RESULTADO: Correcto y coherente
```

**Detalles completos**: Ver sección 4 en `INSTALACION_COMPLETA.md`

---

## 💻 Ejemplo Básico en Python

```python
from transformers import AutoModel
import torch

# Cargar modelo
model = AutoModel.from_pretrained(
    "./models/clara-e2e/compression-128",
    trust_remote_code=True
).to('cuda')

# Documentos
documents = [[
    "Apple fue fundada el 1 de abril de 1976 por Steve Jobs.",
    "La sede está ubicada en Cupertino, California.",
    "CEO es Tim Cook desde 2011."
]]

# Pregunta
questions = ["¿Dónde está Apple y quién la fundó?"]

# Generar respuesta
output, indices = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print(output[0])      # Respuesta
print(indices[0])     # Docs seleccionados
```

---

## 🔧 Problemas Solucionados Durante Instalación

| Problema | Solución |
|----------|----------|
| pytorch-triton-rocm no existe | Removido (no necesario) |
| torch 2.8.0+cu118 inválido | Cambiar a 2.2.0 |
| torchvision incompatible | Downgrade a 0.17.0 |
| fastapi/starlette conflicto | Removidos |
| **numpy 2.2.6 rompe PyTorch** | **Downgrade a numpy<2** |
| **PEFT incompatible** | **Transformers 4.57.3** |
| **config.json paths hardcoded** | **HuggingFace IDs** |

**Ver detalles**: Sección 2 en `INSTALACION_COMPLETA.md`

---

## 🗂️ Estructura del Repositorio

```
/home/jose/Repositorios/ml-clara/
├── models/clara-e2e/
│   ├── compression-128/          ← MODELO PRINCIPAL (745 MB)
│   │   ├── config.json           (✅ Corregido)
│   │   ├── adapters.pth
│   │   ├── decoder_first_last_layers.pth
│   │   ├── tokenizer.model
│   │   └── modeling_clara.py
│   └── compression-16/           (Alternativa)
├── scripts/                      (Entrenamiento)
├── evaluation/                   (Evaluación)
├── example/                      (Datos ejemplo)
├── requirements-minimal.txt      (✅ Versiones resueltas)
├── demo_clara.py                 (✅ Demo funcional)
├── INSTALACION_COMPLETA.md      (Este proceso)
├── USO_RAPIDO.md                (Guía usuario)
├── TROUBLESHOOTING.md           (Solución problemas)
└── README.md                    (Original Apple)
```

---

## 🖥️ Sistema

| Aspecto | Especificación |
|---------|----------------|
| **GPU** | NVIDIA RTX 3090 (25.3 GB VRAM) |
| **CPU** | AMD Ryzen (múltiples núcleos) |
| **RAM** | 32+ GB |
| **OS** | Ubuntu 24.04 LTS |
| **Almacenamiento usado** | ~31.5 GB (modelos + dependencias) |

---

## 📖 Próximos Pasos

### Para Usar el Modelo
1. Lee `USO_RAPIDO.md` para ejemplos prácticos
2. Ejecuta `python demo_clara.py` para verificar
3. Adapta los ejemplos a tus datos

### Si Hay Problemas
1. Consulta `TROUBLESHOOTING.md`
2. Revisa `INSTALACION_COMPLETA.md` para contexto técnico
3. Verifica `requirements-minimal.txt` para dependencias

### Para Entrenar
1. Ver `scripts/` para scripts de training
2. Consultar `evaluation/` para benchmarks
3. Usar datos en `example/`

---

## 🔗 Referencias

- **Paper**: https://arxiv.org/abs/2511.18659
- **Modelos HuggingFace**: https://huggingface.co/apple/CLaRa-7B-E2E
- **GitHub Original**: https://github.com/apple/CLaRa

---

## ❓ Preguntas Frecuentes

**P: ¿Dónde están los archivos de documentación?**
R: Todo está en este directorio (`/home/jose/Repositorios/ml-clara/`):
   - `INSTALACION_COMPLETA.md` - Proceso completo
   - `USO_RAPIDO.md` - Cómo usar el modelo
   - `TROUBLESHOOTING.md` - Solucionar problemas

**P: ¿Necesito reinstalar?**
R: No. Todo está preconfigurado. Solo activa el ambiente conda:
   ```bash
   source /opt/miniconda3/etc/profile.d/conda.sh
   conda activate clara
   ```

**P: ¿Cuánto espacio necesita?**
R: ~31.5 GB total (modelos 15GB + dependencias 2.5GB + caché 14GB)

**P: ¿Qué GPU necesito?**
R: Mínimo 8GB VRAM. La RTX 3090 (25GB) tiene mucho headroom.

**P: ¿Puedo usar CPU en lugar de GPU?**
R: Sí, pero será ~100x más lento. Ver `USO_RAPIDO.md` ejemplo 3.

---

## 📊 Benchmarks

Desempeño esperado del modelo:

| Dataset | Exactitud (CR=4, 128x) |
|---------|------------------------|
| NQ | 57.05% |
| HotpotQA | 45.09% |
| MuSiQue | 10.34% |
| 2Wiki | 46.94% |

**Velocidad** (RTX 3090): 150-200 tokens/segundo

---

**Instalación completada**: 11 de Diciembre 2025, 03:21 UTC
**Estado**: ✅ **OPERATIVO Y VERIFICADO**
**Próximo paso**: Leer `USO_RAPIDO.md` o ejecutar `python demo_clara.py`
