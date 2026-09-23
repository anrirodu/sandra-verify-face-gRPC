"""Configuración del servicio de reconocimiento facial.

Carga la configuración con precedencia a tres niveles (de mayor a menor
prioridad — el estándar de los servicios del ecosistema Sandra):

    1. Variables de entorno del proceso (SERVICE_*, INDEX_*, …)
    2. Archivo ``.env`` en el directorio de trabajo
    3. Valores por defecto razonables para un despliegue local

La inmutabilidad de las constantes se garantiza con ``@dataclass(frozen=True)``
y los valores son tipados con *dataclass transform* para que tanto `mypy`
como los editores ofrezcan autocompletado seguro.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv as _dotenv_load
except Exception:  # pragma: no cover — dotenv es opcional
    _dotenv_load = None  # type: ignore[assignment]


# ─── Rutas canónicas del proyecto ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROTO_DIR = BASE_DIR / "proto"
PB2_DIR = BASE_DIR / "face_service" / "pb2"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_DIR = DATA_DIR
DEFAULT_IMAGES_DIR = DATA_DIR / "imagenes"

# Embedding por defecto: modelo buffalo_l/rec.onnx de InsightFace → 512-d.
DEFAULT_EMBEDDING_DIM = 512


# ════════════════════════════════════════════════════════════
# Helpers de conversión tipada
# ════════════════════════════════════════════════════════════
def _as_bool(value: Optional[str], default: bool) -> bool:
    """Convierte un valor string a booleano con tolerancia a formatos."""
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _as_float(value: Optional[str], default: float) -> float:
    """Float estricto; si falla el parseo devuelve el default."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Optional[str], default: int) -> int:
    """Entero estricto; si falla el parseo devuelve el default."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(value: Optional[str], default: Path) -> Path:
    """Resuelve una ruta relativa contra BASE_DIR si no es absoluta."""
    if not value:
        return default
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

def _as_list(value: Optional[str], default: List[str]) -> List[str]:
    """Convierte un string de lista (JSON o separado por comas)."""
    if value is None or value == "":
        return default
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in str(value).split(",") if x.strip()]


# ════════════════════════════════════════════════════════════
# Modelo de configuración (inmutable)
# ════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Settings:
    """Configuración inmutable del servicio.

    Todos los campos tienen valor por defecto, de modo que el proyecto
    arranca incluso sin ``.env`` ni variables de entorno (modo "zero-config").
    """

    # ─── Servidor gRPC ────────────────────────────────────
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50052
    grpc_max_workers: int = 4
    grpc_max_message_mb: int = 32

    # ─── Detección (YOLO / ultralytics) ───────────────────
    # Ruta a los pesos. Se espera un modelo YOLO afinado para rostros
    # (ej. yolov8n-face.pt). Relativa a BASE_DIR si no es absoluta.
    detector_model: Path = MODELS_DIR / "yolov8n-face.pt"
    detector_conf: float = 0.40
    detector_iou: float = 0.45
    detector_img_size: int = 640
    detector_gpu: bool = True                      # MPS (Mac) / CUDA si existe

    # ─── Embedding (InsightFace / ONNX) ──────────────────
    # Nombre del modelo de reconocimiento (rec.onnx). Si es un nombre
    # liso se busca dentro del HOME de modelos de InsightFace.
    embedder_model: str = "buffalo_l/rec.onnx"
    embedder_gpu: bool = True

    # ─── Índice de similitud ─────────────────────────────
    # auto → faiss si está instalado, si no numpy (caída limpia).
    index_engine: str = "auto"
    index_dim: int = DEFAULT_EMBEDDING_DIM
    index_metric: str = "cosine"                    # cosine | inner | l2
    index_file: Path = DEFAULT_INDEX_DIR / "faces.index"
    index_meta: Path = DEFAULT_INDEX_DIR / "faces.ndjson"
    images_dir: Path = DEFAULT_IMAGES_DIR          # fotos originales (PNG/JPG)
    match_threshold: float = 0.50                   # similitud mínima (0–1)
    faiss_dims_warn: bool = True                    # aviso si faiss ausente

    # ─── Misc ────────────────────────────────────────────
    log_level: str = "INFO"
    service_name: str = "recognition-facial"
    service_version: str = "0.1.0"
    register_faces_default: bool = False            # auto-registro por defecto
    # Devuelve embeddings en las respuestas (más bytes, útil en depuración).
    include_embeddings: bool = False

    # ─── Parser / producer (avanzado, opcional) ──────────
    # Pre/pos-procesadores invocables para p. ej. normalización de color.
    # Se inyectan vacíos; los módulos concretos pueden sobrescribirlos.
    preprocessors: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    postprocessors: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)

    # ─── Propiedades derivadas ───────────────────────────
    @property
    def grpc_max_message_bytes(self) -> int:
        return self.grpc_max_message_mb * 1024 * 1024

    @property
    def b2_index_file(self) -> Path:
        """Ruta del índice FAISS/NumPy (debe poder crearse el directorio)."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        return self.index_file

    def require_index_dir(self) -> Path:
        """Garantiza y devuelve el directorio de datos."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        return self.index_file.parent

    def threat_score_label(self, score: float) -> str:
        return "match"


def load_env(path: Optional[Path] = None) -> None:
    """Carga el archivo ``.env`` si existe (idempotente)."""
    if _dotenv_load is None:
        return
    candidate = path or (BASE_DIR / ".env")
    if candidate.exists():
        _dotenv_load(dotenv_path=str(candidate))


def get_settings(environ: Optional[Dict[str, str]] = None) -> Settings:
    """Construye ``Settings`` desde el entorno.

    Los nombres de variable siguen el patrón de prefijos de Sandra:
      - ``FACE_``   → grpc/embedder (ej. FACE_PORT)
      - ``INDEX_``  → índice de similitud
      - ``DETECTOR_`` → detección YOLO
      - ``EMBEDDER_`` → embeddings InsightFace
    """
    load_env()
    env = environ if environ is not None else dict(os.environ)

    # Rutas por defecto pero resolubles desde entorno.
    detector_model = _resolve_path(
        env.get("DETECTOR_MODEL"), MODELS_DIR / "yolov8n-face.pt"
    )
    index_file = _resolve_path(env.get("INDEX_FILE"), DEFAULT_INDEX_DIR / "faces.index")
    index_meta = _resolve_path(env.get("INDEX_META"), DEFAULT_INDEX_DIR / "faces.ndjson")
    images_dir = _resolve_path(env.get("IMAGES_DIR"), DEFAULT_IMAGES_DIR)

    return Settings(
        grpc_host=env.get("GRPC_HOST", "0.0.0.0"),
        grpc_port=_as_int(env.get("GRPC_PORT"), 50052),
        grpc_max_workers=_as_int(env.get("GRPC_MAX_WORKERS"), 4),
        grpc_max_message_mb=_as_int(env.get("GRPC_MAX_MESSAGE_MB"), 32),
        detector_model=detector_model,
        detector_conf=_as_float(env.get("DETECTOR_CONF"), 0.40),
        detector_iou=_as_float(env.get("DETECTOR_IOU"), 0.45),
        detector_img_size=_as_int(env.get("DETECTOR_IMG_SIZE"), 640),
        detector_gpu=_as_bool(env.get("DETECTOR_USE_GPU"), True),
        embedder_model=env.get("EMBEDDER_MODEL", "buffalo_l/rec.onnx"),
        embedder_gpu=_as_bool(env.get("EMBEDDER_USE_GPU"), True),
        index_engine=env.get("INDEX_ENGINE", "auto"),
        index_dim=_as_int(env.get("INDEX_DIM"), DEFAULT_EMBEDDING_DIM),
        index_metric=env.get("INDEX_METRIC", "cosine"),
        index_file=index_file,
        index_meta=index_meta,
        images_dir=images_dir,
        match_threshold=_as_float(env.get("MATCH_THRESHOLD"), 0.50),
        log_level=env.get("LOG_LEVEL", "INFO"),
        service_name=env.get("SERVICE_NAME", "recognition-facial"),
        service_version=env.get(
            "SERVICE_VERSION", getattr(sys.modules[__name__], "__version__", "0.1.0")
        ),
        register_faces_default=_as_bool(
            env.get("REGISTER_FACES_DEFAULT"), False
        ),
        include_embeddings=_as_bool(env.get("INCLUDE_EMBEDDINGS"), False),
    )


# Instancia por defecto: el resto del proyecto importa `settings` aquí.
settings = get_settings()

# Conveniencia: dimensiones y umbral usados en todo el código.
INDEX_DIM = settings.index_dim
MATCH_THRESHOLD = settings.match_threshold
