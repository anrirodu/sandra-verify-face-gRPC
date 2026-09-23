"""Pipeline de reconocimiento: imagen cruda → resultados tipados.

Orquesta los cuatro pasos y expone una única función pura
``process_bytes`` que convierte los bytes recibidos por gRPC en una
estructura ``ImageResult`` lista para serializar. Mantiene la *métrica*
(ms por etapa) para el servidor y para las herramientas CLI.

Flujo:

    bytes → OpenCV (decode) → YOLO (rostros, conf≥umbral)
         → InsightFace (embedding 512-d por rostro)
         → repository (búsqueda top-K) → FaceMatch[]

El módulo **no toca a gRPC ni a los pb2**: el server solo convierte
``ImageResult`` → ``ImageReply``.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional


# ── Fix de compatibilidad nativa ──
# torch + faiss cargan libomp/OpenMP duplicadas y crashean (segfault) en la
# misma región paralela. Forzamos 1 hilo OpenMP y permitimos libomp duplicada.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch as _torch

    _torch.set_num_threads(1)
except Exception:
    pass

import numpy as np
import cv2 as _cv2  # import temprano: cargar cv2 tras torch/MPS causa segfault

from . import config
from .common import (
    BBox,
    DecodeError,
    DetectedFace,
    ElapsedTimer,
    FaceMatch,
    ModelNotLoadedError,
    NoFaceDetectedError,
    now_ms,
)
from .detector import YoloFaceDetector
from .embedder import InsightFaceEmbedder
from .repository import PersonRepository

log = logging.getLogger(config.settings.service_name)


@dataclass
class StageTiming:
    """Cronometraje de cada etapa del pipeline (ms)."""

    decode_ms: float = 0.0
    detect_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.decode_ms + self.detect_ms + self.embed_ms + self.search_ms


@dataclass
class ImageResult:
    """Salida completa de un ``ProcessImage`` (sin tocar pb2)."""

    # Rostros ya resueltos (match_score/known rellenos).
    faces: List[FaceMatch] = field(default_factory=list)

    # Rostros auto-registrados en esta llamada (índices de personas).
    registered: List[str] = field(default_factory=list)

    # Dimensiones de la imagen de entrada.
    width: int = 0
    height: int = 0

    # Tiempos en ms por etapa.
    timing: StageTiming = field(default_factory=StageTiming)

    # El índice cambió (el cliente debe invalidar caches).
    index_changed: bool = False

    # Embeddings detectados en esta imagen (para auto-registro).
    embeddings: List[np.ndarray] = field(default_factory=list)


# ════════════════════════════════════════════════════════════
# Pipeline concreto
# ════════════════════════════════════════════════════════════
class RecognitionPipeline:
    """Compone detector + embedder + repository y ejecuta el flujo.

    La carga de los tres modelos es *lazy* (primera llamada) y se hace
    dentro de un ciclo de trabajo dedicado. Todas las etapas son síncronas:
    el servidor llama ``process_bytes`` dentro de un *executor*.
    """

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings
        self._detector = YoloFaceDetector(settings)
        self._embedder = InsightFaceEmbedder(settings)
        self._repository = PersonRepository(settings)
        self._loaded = False
        self._load_error: Optional[str] = None

    # ─── Estado de carga ─────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> None:
        """Carga los tres componentes. Lanza ante el primer fallo."""
        if self._loaded:
            return
        first_error: Optional[Exception] = None

        # Detector (YOLO) — crítico para cualquier operación.
        try:
            self._detector.load()
        except Exception as exc:
            first_error = exc
            self._load_error = f"Detector: {exc}"
            log.error("Falló la carga del detector: %s", exc)

        # Embedder (InsightFace) — crítico si el detector no está, ya que
        # necesita embeder para enrolar/reconocer.
        try:
            self._embedder.load()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            self._load_error = f"Embedder: {exc}"
            log.error("Falló la carga del embedder: %s", exc)

        # Repository (índice) — nunca falla al cargar (reconstruye vacío).
        self._repository.load()

        if first_error is not None:
            raise first_error

        self._loaded = True
        log.info(
            "Pipeline listo: %s rostros en el índice · engine=%s dim=%d",
            len(self._repository),
            self._repository.engine_name,
            self._settings.index_dim,
        )

    # ─── Operación principal ─────────────────────────────
    def process_bytes(self, image_bytes: bytes) -> ImageResult:
        """Procesa bytes crudos de imagen y devuelve el resultado.

        Es un método *sincrónico* de propósito general: lo usan tanto el
        servidor gRPC (dentro de un executor) como los CLIs ``register.py``
        e ``identify.py``. No lanza excepciones de dominio: las engrapa en
        ``ImageResult`` vacío con ``faces_registered=0`` y el error en una
        propiedad, para que el servidor las traduzca a status gRPC.
        """
        if not self._loaded:
            self.load()

        t0 = now_ms()
        try:
            # 1 ─ Decodificar
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if img is None:
                raise DecodeError("No se pudo decodificar la imagen con OpenCV")
            height, width = img.shape[:2]
            t_decode = now_ms()

            # 2 ─ Detectar (YOLO)
            detections = self._detector.detect(img)
            t_detect = now_ms()

            # 3 ─ Embedder + búsqueda
            faces: List[FaceMatch] = []
            embeddings: List[np.ndarray] = []
            for det in detections:
                face_crop = self._crop_face(img, det)
                embedding = self._embedder.embed_face(face_crop)
                embeddings.append(embedding)
                match = self._repository.match_embedding(
                    embedding, threshold=self._settings.match_threshold
                )
                faces.append(
                    FaceMatch(
                        person_id=match.person_id,
                        label=match.label,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        match_score=match.score,
                        embedding=embedding,   # el embedding real generado
                        known=match.known,
                    )
                )
            t_search = now_ms()

            return ImageResult(
                faces=faces,
                width=width,
                height=height,
                timing=StageTiming(
                    decode_ms=t_decode - t0,
                    detect_ms=t_detect - t_decode,
                    embed_ms=t_search - t_detect,
                    search_ms=now_ms() - t_search,
                ),
            )
        except (DecodeError, NoFaceDetectedError) as exc:
            log.info("Imagen inválida o sin rostros: %s", exc)
            return ImageResult(width=0, height=0, timing=StageTiming())

    def compare_images(
        self,
        reference_bytes: bytes,
        probe_bytes: bytes,
        threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        """Verificación 1:1 — �¿es la imagen de la cámara la misma persona?

        Compara dos imágenes (referencia guardada ↔ captura de cámara)
        usando la similitud coseno de sus embeddings (rostro principal de
        cada una). Devuelve un dict con ``match``, ``score``, ``threshold``
        y mensaje, sin lanzar excepciones de dominio (el servidor lo
        traduce a status gRPC).
        """
        def _all_ready() -> None:
            if not self._loaded:
                self.load()

        def _embed_best(image_bytes: bytes) -> Optional[np.ndarray]:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if img is None:
                return None
            dets = self._detector.detect(img)
            if not dets:
                return None
            # Rostro principal = mayor confianza de detección.
            best = max(dets, key=lambda d: d.confidence)
            crop = self._crop_face(img, best)
            return self._embedder.embed_face(crop)

        t0 = now_ms()
        try:
            _all_ready()
            ref = _embed_best(reference_bytes)
            probe = _embed_best(probe_bytes)
        except (ModelNotLoadedError, DecodeError, NoFaceDetectedError) as exc:
            log.warning("Verificación no ejecutable: %s", exc)
            return {
                "success": False,
                "match": False,
                "score": 0.0,
                "threshold": float(threshold or self._settings.match_threshold),
                "message": str(exc),
                "processing_ms": now_ms() - t0,
            }

        thr = float(
            threshold if threshold is not None else self._settings.match_threshold
        )
        if ref is None or probe is None:
            which = "referencia" if ref is None else "cámara"
            return {
                "success": True,
                "match": False,
                "score": 0.0,
                "threshold": thr,
                "message": f"Sin rostro detectable en la {which}",
                "processing_ms": now_ms() - t0,
            }

        # Similitud coseno sobre vectores normalizados (0–1).
        def _norm(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return v if n == 0 else v / n

        score = float(np.dot(_norm(ref), _norm(probe)))
        score = max(-1.0, min(1.0, score))
        match = score >= thr
        return {
            "success": True,
            "match": bool(match),
            "score": round(score, 4),
            "threshold": thr,
            "message": "MATCH" if match else "NO_MATCH",
            "processing_ms": round(now_ms() - t0, 2),
        }

    @staticmethod
    def _crop_face(img: np.ndarray, det: DetectedFace) -> np.ndarray:
        """Recorta la caja trasladando YOLO→ImgH×ImgW y aplicando margen."""
        H, W = img.shape[:2]
        x1 = max(0, int(det.bbox.x))
        y1 = max(0, int(det.bbox.y))
        x2 = min(W, int(det.bbox.x + det.bbox.width))
        y2 = min(H, int(det.bbox.y + det.bbox.height))
        if x2 <= x1 or y2 <= y1:
            raise NoFaceDetectedError("Caja YOLO degenerada tras recorte")
        return img[y1:y2, x1:x2]
