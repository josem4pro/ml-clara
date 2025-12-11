# CLaRa Stage 3 - Guía de Solución de Problemas

## Problemas Comunes y Soluciones

### 1. CUDA out of memory

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Soluciones**:

```python
# Opción 1: Usar CPU en lugar de GPU
model = model.to('cpu')

# Opción 2: Reducir tokens generados
output = model.generate_from_questions(
    questions=questions,
    documents=documents,
    max_new_tokens=32  # Reducir de 128 a 32
)

# Opción 3: Usar menos documentos
documents_subset = [docs[0][:20]]  # Solo primeros 20 docs

# Opción 4: Limpiar caché CUDA
import torch
torch.cuda.empty_cache()
model = model.to('cuda')
```

---

### 2. "Model loading is slow"

**Causa**: Primera carga descarga dependencias de HuggingFace

**Soluciones**:

```bash
# La primera ejecución es normal que sea lenta
# Las próximas serán más rápidas (caché local)

# Verificar conexión a HuggingFace
ping huggingface.co

# Pre-descargar modelo manualmente
huggingface-cli download apple/CLaRa-7B-E2E --local-dir ./models/clara-e2e
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
```

---

### 3. "ModuleNotFoundError: No module named 'transformers'"

**Causa**: Ambiente conda no activado correctamente

**Solución**:

```bash
# Verificar que el ambiente está activo
conda activate clara

# Si no funciona, reinstalar
pip install transformers torch
```

---

### 4. TypeError: unsupported operand type(s)

**Error**:
```
TypeError: unsupported operand type(s) for operation
```

**Causa**: Formato incorrecto de entrada (preguntas/documentos no son listas)

**Solución**:

```python
# ❌ INCORRECTO
questions = "¿Pregunta?"           # String
documents = ["doc1", "doc2"]       # Lista plana

# ✅ CORRECTO
questions = ["¿Pregunta?"]         # Lista de strings
documents = [["doc1", "doc2"]]     # Lista de listas de strings
```

---

### 5. "numpy<2 compatibility issue"

**Error**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Solución**:

```bash
# Downgrade NumPy
pip install 'numpy<2'

# Verificar instalación
python -c "import numpy; print(numpy.__version__)"
```

---

### 6. "PEFT ImportError"

**Error**:
```
ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'
```

**Causa**: Transformers muy antigua para PEFT 0.18.0

**Solución**:

```bash
# Actualizar transformers
pip install --upgrade transformers

# Verificar versión (debe ser 4.45.0+)
python -c "import transformers; print(transformers.__version__)"
```

---

### 7. "Config.json not found"

**Error**:
```
OSError: ./models/clara-e2e does not appear to have a file named config.json
```

**Causa**: Ruta incorrecta del modelo

**Solución**:

```python
# ❌ INCORRECTO
model_path = "./models/clara-e2e"

# ✅ CORRECTO (especificar compresión)
model_path = "./models/clara-e2e/compression-128"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True
).to('cuda')
```

---

### 8. "HuggingFace authentication error"

**Error**:
```
OSError: HF_TOKEN not found. You need to authenticate
```

**Solución**:

```bash
# Opción 1: Usar token almacenado
source ~/.env
export HF_TOKEN="your_token_here"

# Opción 2: Login interactivo
huggingface-cli login

# Opción 3: Verificar token
echo $HF_TOKEN
```

---

### 9. "Transformers version incompatibility"

**Error**:
```
ValueError: transformers_version mismatch
```

**Causa**: Versión de transformers no coincide

**Solución**:

```bash
# Usar versión compatible especificada
pip install transformers==4.57.3

# Verificar
python -c "import transformers; print(transformers.__version__)"
```

---

### 10. "CUDA not available"

**Error**:
```
torch.cuda.is_available() returns False
```

**Causa**: GPU no detectada o CUDA/cuDNN no configurado

**Soluciones**:

```bash
# Verificar NVIDIA drivers
nvidia-smi

# Verificar CUDA en PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Si falla, verificar instalación de CUDA
nvcc --version

# Como alternativa, usar CPU
model = model.to('cpu')
```

---

### 11. "Permission denied" en descarga de modelos

**Error**:
```
PermissionError: [Errno 13] Permission denied: './models/clara-e2e'
```

**Causa**: Permisos insuficientes en directorio

**Solución**:

```bash
# Crear directorio con permisos
mkdir -p ./models/clara-e2e
chmod 755 ./models

# O descargar desde otro directorio
huggingface-cli download apple/CLaRa-7B-E2E --cache-dir /tmp/hf_cache
```

---

### 12. "Descarga lenta desde HuggingFace"

**Causa**: Conexión lenta o limitación de ancho de banda

**Soluciones**:

```bash
# Verificar conexión
ping huggingface.co

# Usar WiFi de mejor calidad o conexión Ethernet

# Reintenta la descarga
rm -rf ./models/clara-e2e/.incomplete
huggingface-cli download apple/CLaRa-7B-E2E --local-dir ./models/clara-e2e

# Usar espejo (si disponible)
# Algunos repositorios ofrecen mirrors en regiones específicas
```

---

### 13. "Out of disk space"

**Error**:
```
IOError: [Errno 28] No space left on device
```

**Solución**:

```bash
# Verificar espacio disponible
df -h

# Limpiar caché de HuggingFace
rm -rf ~/.cache/huggingface/

# O descargar a ubicación alternativa
huggingface-cli download apple/CLaRa-7B-E2E --local-dir /mnt/large_disk/clara-e2e
```

---

### 14. "Python version mismatch"

**Error**:
```
ERROR: Python 3.9 requires torch... but you have Python 3.8
```

**Solución**:

```bash
# Crear ambiente con Python 3.10
conda create -n clara python=3.10 -y
conda activate clara

# O verificar versión actual
python --version
```

---

### 15. "Memory leak during inference"

**Síntoma**: Memoria VRAM aumenta con cada inferencia

**Solución**:

```python
# Limpiar caché después de cada inferencia
import torch
torch.cuda.empty_cache()

# O usar context manager
with torch.no_grad():
    output = model.generate_from_questions(
        questions=questions,
        documents=documents,
        max_new_tokens=64
    )
    torch.cuda.empty_cache()
```

---

## Verificación de Sistema

### Verificar instalación completa

```bash
# 1. Verificar Conda
conda --version

# 2. Verificar ambiente
conda activate clara
python --version

# 3. Verificar PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Verificar Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 5. Verificar PEFT
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# 6. Verificar modelo
python -c "from transformers import AutoModel; print('✓ Modelo cargable')"

# 7. Ejecutar demo
python demo_clara.py
```

---

## Comandos Útiles

### Limpiar y Resetear

```bash
# Limpiar caché de pip
pip cache purge

# Limpiar caché de HuggingFace
rm -rf ~/.cache/huggingface/

# Limpiar CUDA
python -c "import torch; torch.cuda.empty_cache()"

# Resetear ambiente (cuidado: elimina instalaciones)
conda remove -n clara --all
conda create -n clara python=3.10
```

### Monitorear Sistema

```bash
# Ver uso de GPU en tiempo real
watch -n 1 nvidia-smi

# Ver procesos de Python
ps aux | grep python

# Ver uso de memoria
free -h
```

### Debugging

```bash
# Activar verbose logging
export TRANSFORMERS_VERBOSITY=debug

# Ver archivos descargados
ls -lh ./models/clara-e2e/compression-128/

# Verificar integridad de descarga
sha256sum ./models/clara-e2e/compression-128/model.safetensors
```

---

## Contacto y Recursos

- **Repositorio GitHub**: https://github.com/apple/CLaRa
- **Documentación Local**: `INSTALACION_COMPLETA.md`, `USO_RAPIDO.md`
- **HuggingFace Community**: https://huggingface.co/apple/CLaRa-7B-E2E/discussions

---

**Última actualización**: 11 de Diciembre 2025
**Versión**: 1.0
