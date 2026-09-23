<div align="center">

# 🛡️ Recognition-Facial

### Sistema de Reconocimiento Facial · gRPC

![BTS](https://img.shields.io/badge/BUNKER-TECHNOLOGIES%20SOLUTIONS-0a0a23?style=for-the-badge&labelColor=14142e)
![Sandra](https://img.shields.io/badge/plataforma-Sandra%20Server-1a1a4e?style=for-the-badge)
![Uso](https://img.shields.io/badge/uso-interno-0a0a23?style=for-the-badge)
![Version](https://img.shields.io/badge/v0.1.0-2a2a5e?style=for-the-badge)
![License](https://img.shields.io/badge/Apache--2.0-1a1a4e?style=for-the-badge)

**Propiedad de `BUNKER TECHNOLOGIES SOLUTIONS`** — para uso interno exclusivo
de la plataforma **Sandra Server**.

```
╔══════════════════════════════════════════════════════════════════╗
║   ██████╗ ██╗   ██╗███╗   ██╗██╗  ██╗███████╗██████╗             ║
║   ██╔══██╗██║   ██║████╗  ██║██║ ██╔╝██╔════╝██╔══██╗            ║
║   ██████╔╝██║   ██║██╔██╗ ██║█████╔╝ █████╗  ██████╔╝            ║
║   ██╔══██╗██║   ██║██║╚██╗██║██╔═██╗ ██╔══╝  ██╔══██╗            ║
║   ██████╔╝╚██████╔╝██║ ╚████║██║  ██╗███████╗██║  ██║            ║
║   ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝            ║
╚══════════════════════════════════════════════════════════════════╝
```

Servicio **gRPC** de reconocimiento facial del ecosistema **Sandra**.
Recibe imágenes crudas (`jpg`, `png`, `bmp`, `webp`), detecta rostros,
genera embeddings 512-d y los compara contra un índice de personas conocido.

</div>

---

## 🧠 Qué hace

```
imagen (bytes)
   │
   ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  YOLO        │────►│  InsightFace     │────►│  FAISS / NumPy   │
│  detección   │     │  embeddings 512d │     │  búsqueda coseno │
└──────────────┘     └──────────────────┘     └──────────────────┘
   │                     │                       │
   └── rostros ──────────┴── vector p/persona ───┴── FaceResult[]
```

| Función | RPC | Descripción |
|---------|-----|-------------|
| **Identificar** | `ProcessImage` | Detecta todos los rostros de una imagen y los compara contra el índice. |
| **Registrar** | `RegisterFace` | Enrola una persona: guarda la **foto original** en `data/imagenes/<cedula>.<ext>` y su embedding en el índice. |
| **Verificar 1:1** | `VerifyFace` | Valida que la **imagen de cámara** pertenece a la persona de una **cédula** (usa la foto guardada como referencia). |
| **Eliminar** | `RemovePerson` | Quita a una persona del índice y sus vectores. |
| **Listar** | `ListPersons` | Devuelve quiénes están indexados. |
| **Health check** | `Check` | Estado del servidor (liveness, modelos, aceleración). |

---

## 🎯 Caso de uso principal (verificación 1:1)

Un servicio externo envía:

- `cedula` → por ejemplo `"17818665.jpg"` (nombre de la imagen guardada)
- `imagen` → `"nueva-imagen.jpg"` (captura de cámara)

y debe validar que **la persona frente a la cámara es el titular de la cédula**.

```
┌──────────────┐      ┌───────────────────────────────┐      ┌──────────────┐
│  cedula      │      │  Server recognition-facial    │      │   imagen     │
│ 17818665.jpg │ ───► │  1. buscar data/imagenes/     │ ◄─── │ camera.jpg   │
└──────────────┘      │     17818665.jpg              │      └──────────────┘
                      │  2. comparar embeddings       │
                      │  3. devolver score (0–1)      │
                      └──────────────┬────────────────┘
                                     ▼
                     ✔ MATCH  score=0.8177 · umbral=0.50
                     ✖ NO_MATCH score=0.3120 · umbral=0.50
```

---

## 📋 Requisitos

- **Python 3.10+** (probado en 3.14 / macOS ARM64)
- macOS ARM64 (MPS) o Linux x86_64 (CPU/GPU)
- Modelos que se descargan en primer uso:
  - **YOLO**: `models/yolov8n-face.pt` (~6 MB)
  - **InsightFace**: `rec.onnx` de `buffalo_l` (~12 MB → `~/.insightface/models/`)

---

## 🔧 Instalación

```bash
cd python/recognition-facial

# 1) Entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2) Dependencias
pip install -r requirements.txt

# 3) Configuración (opcional)
cp .env.example .env
```

### Descargar pesos YOLO (si no existen)

```bash
mkdir -p models
curl -L -o models/yolov8n-face.pt \
  "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
```

> **Tip (red inestable):** usa resume: `curl -L -C - -o models/yolov8n-face.pt <URL>`.

InsightFace descarga automáticamente `buffalo_l/rec.onnx` en la primera ejecución.
Si tu red es inestable, descárgalo aparte:

```bash
curl -L -o /tmp/buffalo_l.zip \
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
unzip -o /tmp/buffalo_l.zip -d ~/.insightface/models/buffalo_l/
```

> Puedes usar el reconocedor alternativo **`w600k_r50.onnx`** (512-d, compatible)
> si ya lo tienes en `~/.insightface/models/buffalo_l/` — funciona igual para
> embeddings y verificación.

---

## ⚙️ Configuración (`.env`)

Todo es opcional (zero-config). Ver `.env.example` con los valores completos.

| Variable | Default | Descripción |
|----------|---------|-------------|
| `GRPC_HOST` | `0.0.0.0` | Host del servidor gRPC |
| `GRPC_PORT` | `50052` | Puerto del servidor gRPC |
| `GRPC_MAX_WORKERS` | `4` | Hilos de inferencia (con GPU → 1 recomendado) |
| `GRPC_MAX_MESSAGE_MB` | `32` | Tamaño máximo de imagen recibida |
| `DETECTOR_MODEL` | `models/yolov8n-face.pt` | Pesos YOLO |
| `DETECTOR_CONF` | `0.40` | Umbral de confianza de detección |
| `DETECTOR_IOU` | `0.45` | NMS IoU |
| `DETECTOR_IMG_SIZE` | `640` | Tamaño de entrada del modelo |
| `DETECTOR_USE_GPU` | `true` | MPS/CUDA si disponible |
| `EMBEDDER_MODEL` | `buffalo_l/rec.onnx` | Modelo InsightFace (512-d) |
| `EMBEDDER_USE_GPU` | `true` | GPU para ONNX Runtime |
| `INDEX_ENGINE` | `auto` | `auto` \| `faiss` \| `numpy` |
| `INDEX_DIM` | `512` | Dimensiones del embedding |
| `INDEX_METRIC` | `cosine` | `cosine` \| `l2` |
| `INDEX_FILE` | `data/faces.index` | Índice persistido (FAISS/NumPy) |
| `INDEX_META` | `data/faces.ndjson` | Metadatos (persona → embeddings) |
| **`IMAGES_DIR`** | `data/imagenes` | **Fotos originales de los titulares** (`<cedula>.<ext>`) |
| `MATCH_THRESHOLD` | `0.50` | Similitud mínima para considerar match |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `SERVICE_VERSION` | `0.1.0` | Versión reportada por `Check` |

---

## 🚀 Uso

### 1) Arrancar el servidor

```bash
cd python/recognition-facial
.venv/bin/python -m face_service.server --port 50052
```

Salida esperada:

```
INFO:faiss.loader:Successfully loaded faiss.
INFO:recognition-facial:gRPC · ejecutandose en 0.0.0.0:50052 (pipeline 0.1.0)
```

---

### 2) Cliente CLI (incluido)

**Registrar (subir y guardar la foto del titular):**

```bash
.venv/bin/python -m face_service.client \
  --host localhost --port 50052 \
  register 17818665 /ruta/foto-del-titular.png --label "Andres Rodriguez"
```

```
INFO:recognition-facial:Imagen guardada: .../data/imagenes/17818665.jpg (235472 bytes, formato jpg)
INFO:recognition-facial:Pipeline listo: 7 rostros en el índice · engine=faiss dim=512
```

**Verificar 1:1 (cédula ↔ cámara):**

```bash
.venv/bin/python -m face_service.client \
  --host localhost --port 50052 \
  verify 17818665 /ruta/camara/nueva-imagen.jpg
```

```
✅ MATCH · score=0.8177 · threshold=0.500
   persona: 17818665 · -
   referencia: 17818665.jpg · 62.0 ms
   detalle: MATCH
```

**Otros subcomandos:**

```bash
.venv/bin/python -m face_service.client --port 50052 list
.venv/bin/python -m face_service.client --port 50052 identify /ruta/foto.jpg
.venv/bin/python -m face_service.client --port 50052 remove 17818665
```

Opciones de `verify`:

| Opción | Descripción |
|--------|-------------|
| `--threshold 0.65` | Eleva el umbral de match (coseno 0–1); default `0.50` |
| `person_id` | Acepta `17818665` **o** `17818665.jpg` |

---

### 3) `grpcurl` (integración externa)

El servidor expone **reflection**, como descubres el schema en vivo:

```bash
grpcurl -plaintext <host>:50052 list
#   grpc.reflection.v1alpha.ServerReflection
#   sandra.face.FaceRecognition

grpcurl -plaintext <host>:50052 describe sandra.face.FaceRecognition
```

**Health check:**

```bash
grpcurl -plaintext -d '{}' <host>:50052 sandra.face.FaceRecognition/Check
```

**Registrar (subir imagen):** el campo `image_data` es `bytes` → **base64**.

```bash
B64=$(base64 -i /ruta/foto-titular.png | tr -d '\n')

grpcurl -plaintext -d "{
  \"person_id\": \"17818665\",
  \"label\": \"Andres Rodriguez\",
  \"image_data\": \"$B64\",
  \"format\": \"png\"
}" <host>:50052 sandra.face.FaceRecognition/RegisterFace
```

**Verificar 1:1 (cédula ↔ cámara):**

```bash
B64=$(base64 -i /ruta/camara/nueva-imagen.jpg | tr -d '\n')

grpcurl -plaintext -d "{
  \"person_id\": \"17818665.jpg\",
  \"image_data\": \"$B64\",
  \"format\": \"jpg\",
  \"match_threshold\": 0.5
}" <host>:50052 sandra.face.FaceRecognition/VerifyFace
```

Respuesta típica:

```json
{
  "success": true,
  "match": true,
  "score": 0.8177,
  "threshold": 0.5,
  "person_id": "17818665",
  "reference_image": "17818665.jpg",
  "message": "MATCH",
  "processing_ms": 87
}
```

**Identificar todos los rostros de una imagen:**

```bash
B64=$(base64 -i /ruta/foto.jpg | tr -d '\n')
grpcurl -plaintext -d "{\"image_data\": \"$B64\"}" <host>:50052 sandra.face.FaceRecognition/ProcessImage
```

**Listar / eliminar:**

```bash
grpcurl -plaintext -d '{}' <host>:50052 sandra.face.FaceRecognition/ListPersons
grpcurl -plaintext -d '{"person_id": "17818665"}' <host>:50052 sandra.face.FaceRecognition/RemovePerson
```

---

## 🔌 Contrato gRPC

- **Package:** `sandra.face`
- **Archivo fuente:** `proto/face_recognition.proto`
- **Generados:** `face_service/pb2/*_pb2.py` y `*_pb2_grpc.py`

| RPC | Request | Response |
|-----|---------|----------|
| `ProcessImage` | `ImageRequest` | `ImageReply` |
| `RegisterFace` | `RegisterRequest` | `RegisterReply` |
| `RemovePerson` | `RemoveRequest` | `RemoveReply` |
| `ListPersons` | `ListRequest` | `ListReply` |
| `Check` | `HealthCheckRequest` | `HealthCheckResponse` |
| `VerifyFace` | `VerifyRequest` | `VerifyReply` |

### Mensajes clave

**`VerifyRequest`**
```proto
string person_id      = 1;  // "17818665" o "17818665.jpg"
bytes  image_data     = 2;  // captura de cámara (jpg/png/...)
string format         = 3;  // pista de formato (opcional)
optional float min_confidence   = 4;
optional float match_threshold  = 5;  // default 0.50
```

**`VerifyReply`**
```proto
bool   success = 1;
bool   match   = 2;
float  score   = 3;
float  threshold = 4;
string person_id = 5;
string label = 6;
string reference_image = 7;  // "17818665.jpg"
string message = 8;
double processing_ms = 9;
```

**`RegisterRequest`**
```proto
string person_id   = 1;
string label       = 2;
bytes  image_data  = 3;      // foto con UN rostro principal
string format      = 4;
repeated float embedding = 5;  // alternativa directa a imagen
optional float min_confidence = 6;
```

---

## 📁 Estructura del proyecto

```
recognition-facial/
├── face_service/
│   ├── server.py        # Servidor gRPC aio + RPCs (VerifyFace, RegisterFace, …)
│   ├── client.py        # CLI: register / identify / verify / remove / list
│   ├── pipeline.py      # Orquesta: detect → embed → search
│   ├── detector.py      # YOLO (ultralytics) — device mps/cuda/cpu automático
│   ├── embedder.py      # InsightFace ONNX — get_feat() → embedding 512-d
│   ├── repository.py    # Índice FAISS/NumPy + persistencia NDJSON
│   ├── imagenes.py      # Guardado de fotos originales + find_reference()
│   ├── config.py        # Settings inmutable (zero-config, .env, env vars)
│   ├── common.py        # Tipos (BBox, DetectedFace, FaceMatch, errores)
│   └── pb2/             # Stubs gRPC generados
├── proto/
│   └── face_recognition.proto
├── data/
│   ├── imagenes/        # Fotos de titulares (<cedula>.<ext>)
│   ├── faces.index      # Índice persistido
│   └── faces.ndjson     # Metadatos de personas
├── models/
│   └── yolov8n-face.pt  # Pesos YOLO
├── scripts/
├── tests/
├── .env.example
└── requirements.txt
```

---

## 📝 Notas técnicas y solución de problemas

### Compatibilidad nativa (importante)

El proyecto carga varias librerías nativas que compiten por recursos:

- **`torch` + `faiss`** → cada una trae su propia `libomp` (OpenMP). En macOS,
  ejecutarlas en la **misma región paralela** puede producir **segfault**
  (`exit 139`). El fix está embebido al importar `pipeline.py`:
  ```python
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  os.environ["OMP_NUM_THREADS"]      = "1"
  os.environ["MKL_NUM_THREADS"]      = "1"
  torch.set_num_threads(1)
  ```

- **`import cv2` después de cargar YOLO/MPS** también crashea ("Fatal Python error").
  Por eso `cv2` se importa **al inicio** de `pipeline.py`, nunca inline tras la carga.

- **Device de YOLO:** `device="auto"` falla si no hay CUDA (macOS).
  Ahora se resuelve **MPS → CUDA → CPU**.

> Si ves `exit 139` o "Fatal Python error" al procesar, es este conflicto;
> no hay error de código — las variables ya están seteadas por el módulo.

### Errores comunes

| Síntoma | Causa | Solución |
|---------|-------|----------|
| `grpc: server does not support the reflection API` | Server viejo sin reflection | Usa el código actualizado, o usa el CLI `face_service.client` |
| `Method not found: .../VerifyFace` | Build anterior sin VerifyFace | Actualiza `face_service/pb2/` y reinicia |
| `No existe imagen de referencia para la cedula X` | Aún no registrada | Ejecuta `register` antes de `verify` |
| `File is not a zip file` al instalar torch | Descarga corrupta/truncada | Descarga por `curl -C -` (resume) y valida con `zipfile` |

### Performance (macOS M3, medido)

- Carga de modelos: **~3 s** (primera llamada)
- Detección YOLO MPS: **~0.8 s/imagen**
- Embedding: **~4 ms/rostro**
- Verificación completa: **~60–90 ms**

---

## 🚢 Despliegue

### Local (desarrollo)

```bash
cd python/recognition-facial
.venv/bin/python -m face_service.server --port 50052
```

### Como servicio (systemd — Linux)

```ini
# /etc/systemd/system/recognition-facial.service
[Unit]
Description=Recognition Facial gRPC (Sandra)
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/recognition-facial
ExecStart=/opt/recognition-facial/.venv/bin/python -m face_service.server --port 50052
Restart=always
RestartSec=3
Environment=OMP_NUM_THREADS=1
Environment=KMP_DUPLICATE_LIB_OK=TRUE

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now recognition-facial
```

### Binario autónomo (PyInstaller)

```bash
.venv/bin/pip install pyinstaller
.venv/bin/pyinstaller --onefile --name sandra-face \
  --add-data "models:models" face_service/__main__.py
```

> Nota: empaquetar torch + onnxruntime + faiss en one-file es pesado;
> se recomienda one-folder y desplegar los `.dylib/.so` junto al binario.

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Modelos (descarga en build para evitar descargas en runtime)
RUN mkdir -p models && \
    curl -sL -o models/yolov8n-face.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

EXPOSE 50052
CMD ["python", "-m", "face_service.server", "--port", "50052"]
```

```bash
docker build -t sandra/recognition-facial .
docker run -d --name recognition-facial \
  -p 50052:50052 \
  -v $(pwd)/data:/app/data \
  -v $HOME/.insightface:/root/.insightface \
  sandra/recognition-facial
```

---

## 🧪 Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

**Smoke test manual** (sin modelos, solo health):

```bash
.venv/bin/python -m face_service.server --port 50053 &
grpcurl -plaintext localhost:50053 list
grpcurl -plaintext -d '{}' localhost:50053 sandra.face.FaceRecognition/Check
# → {"status": "SERVING"|"LOADING", "version": "0.1.0", ...}
```

---

<div align="center">

## 🏢 BUNKER TECHNOLOGIES SOLUTIONS

**© BUNKER TECHNOLOGIES SOLUTIONS — Todos los derechos reservados.**

> Propiedad de **BUNKER TECHNOLOGIES SOLUTIONS** para uso interno exclusivo
> de la plataforma **Sandra Server**.
>
> Apache-2.0 · Sandra · Code-Epic
>
> `recognition-facial` — Sistema de Reconocimiento Facial · gRPC v0.1.0

```
 ██████╗ ██╗   ██╗███╗   ██╗██╗  ██╗███████╗██████╗
 ██╔══██╗██║   ██║████╗  ██║██║ ██╔╝██╔════╝██╔══██╗
 ██████╔╝██║   ██║██╔██╗ ██║█████╔╝ █████╗  ██████╔╝
 ██╔══██╗██║   ██║██║╚██╗██║██╔═██╗ ██╔══╝  ██╔══██╗
 ██████╔╝╚██████╔╝██║ ╚████║██║  ██╗███████╗██║  ██║
 ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
```

</div>
