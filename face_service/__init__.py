"""recognition-facial · Servicio Sandra de reconocimiento facial.

Exporta la API pública estable del paquete — la que consumen el servidor
gRPC, los CLIs y los tests. Mantener estos símbolos congelados es un
contrato: cualquiera de los sub-agentes del ecosistema (SERF, Valquiria,
Freya) puede importar el paquete y depender de que estos nombres existan.
"""

from __future__ import annotations

# ─── versión (singular: la lee config, pyproject y Docker) ──
__version__ = "0.1.0"
__service_name__ = "recognition-facial"
__service_version__ = __version__

# ─── configuración ────────────────────────────────────────
from .config import Settings, get_settings, settings

# ─── tipos de dominio ─────────────────────────────────────
from .common import (
    BBox,
    DecodeError,
    DetectedFace,
    FaceServiceError,
    FaceMatch,
    MultipleFacesError,
    NoFaceDetectedError,
    PersonNotFoundError,
)
from .detector import YoloFaceDetector
from .embedder import InsightFaceEmbedder
from .pipeline import ImageResult, StageTiming, RecognitionPipeline
from .repository import PersonMatch, PersonRepository

# ─── utilidades ───────────────────────────────────────────
from .common import now_ms, elapsed_ms

__all__ = [
    # versión
    "__version__",
    "__service_name__",
    "__service_version__",
    # config
    "Settings",
    "get_settings",
    "settings",
    # dominio
    "BBox",
    "DetectedFace",
    "FaceMatch",
    "FaceServiceError",
    "DecodeError",
    "MultipleFacesError",
    "NoFaceDetectedError",
    "PersonNotFoundError",
    # modelos
    "YoloFaceDetector",
    "InsightFaceEmbedder",
    # pipeline
    "ImageResult",
    "StageTiming",
    "RecognitionPipeline",
    # repositorio
    "PersonMatch",
    "PersonRepository",
    # utilidades
    "now_ms",
    "elapsed_ms",
]
