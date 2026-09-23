"""Cálculo de embeddings faciales con InsightFace (ONNX Runtime).

Una vez YOLO localiza un rostro, lo recortamos y lo pasamos por el modelo
de reconocimiento ``rec.onnx`` (buffalo_l), que devuelve un vector de
**512 dimensiones** que codifica la *identidad* de la persona (invariante
a luz, pose y expresión hasta cierto punto).

Este módulo encaja con el SERF (``sandra-facial-recognition``) y reutiliza
las mismas convenciones: InsightFace 0.7.x + ONNX Runtime, con aceleración
MPS/CUDA cuando está disponible y CoreML en macOS.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from . import config
from .common import ModelNotLoadedError

log = logging.getLogger(config.settings.service_name)


class InsightFaceEmbedder:
    """Wrapper tipado del modelo de reconocimiento de InsightFace.

    Detecta/carga el modelo ``rec.onnx`` en la primera llamada (lazy) y
    expone un único método público: ``embed_single(image_bgr) -> np.ndarray``.
    """

    _HOME_KEYS = ("INSIGHTFACE_HOME", "INSIGHTFACEFACE_HOME", "HF_HOME")

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings
        self._model = None          # insightface.model_zoo.FaceRecognition | None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ─── Ciclo de vida ──────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._loaded

    def _resolve_model_path(self) -> str:
        """Devuelve la ruta al archivo ``rec.onnx`` (existente o no)."""
        name = self._settings.embedder_model  # ej. "buffalo_l/rec.onnx"
        # Si la ruta es absoluta o relativa a un dir existente se usa tal cual.
        p = np_path_or_none(name)
        if p is not None and Path_exists(p):
            return str(p)
        # Busca en el HOME de InsightFace (ubicación estándar de buffalo_l).
        home = self._first_existing_home()
        if home:
            candidate = home / name
            if candidate.exists():
                return str(candidate)
        return name  # InsightFace sabrá resolver el resto

    @staticmethod
    def _first_existing_home():
        import os
        from pathlib import Path
        for key in InsightFaceEmbedder._HOME_KEYS:
            value = os.environ.get(key)
            if value:
                path = Path(value).expanduser()
                if path.exists():
                    return path
        # InsightFace usa ~/.insightface por defecto.
        default = Path.home() / ".insightface"
        return default if default.exists() else None

    def load(self) -> None:
        """Carga el modelo de embeddings (o relanza el error con contexto)."""
        if self._loaded:
            return
        try:
            import insightface.model_zoo as model_zoo  # import pesado
        except Exception as exc:   # pragma: no cover
            self._load_error = "insightface no está instalado (pip install insightface)"
            raise ModelNotLoadedError(self._load_error) from exc

        try:
            path = self._resolve_model_path()
            self._model = model_zoo.get_model(path, download=True)
            # insightface gestiona providers; elegimos GPU si está activa.
            self._model.prepare(
                ctx_id=0 if self._settings.embedder_gpu else -1,
                det_thresh=0.3,
                use_kps=False,
            )
            self._loaded = True
            log.info("Embedder InsightFace cargado: %s", path)
        except Exception as exc:  # pragma: no cover — descarga/red/versión
            self._load_error = str(exc)
            raise ModelNotLoadedError(
                f"No se pudo cargar el embedder ({self._settings.embedder_model}): {exc}"
            ) from exc

    # ─── Entrada única ──────────────────────────────────
    def embed_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Devuelve el embedding 512-d de un recorte de rostro (BGR).

        Parámetros
        ----------
        face_bgr:
            Recorte del rostro ya alineado y normalizado por el pipeline
            (BGR, uint8, tamaño razonable 112×112 o similar).

        Devuelve un ``np.ndarray`` de forma ``(512,)`` normalizado a norma L2.
        """
        if not self._loaded:
            self.load()
        assert self._model is not None

        try:
            # get(img, face) exige keypoints del objeto Face; nosotros ya
            # alineamos/cortamos el rostro, así que usamos get_feat directo
            # (hace interno el blob 112×112). w600k_r50 y buffalo_l lo soportan.
            embedding = self._model.get_feat(face_bgr)  # (512,) float32
        except Exception as exc:  # pragma: no cover — API interna distinta
            raise RuntimeError(f"El embedder falló en get_feat(): {exc}") from exc

        arr = np.asarray(embedding, dtype=np.float32).ravel()
        if arr.size == 0:
            raise RuntimeError("El embedder devolvió un embedding vacío")
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr


# ─── utilidades (sin importar Path/os en la cabecera) ───────
def np_path_or_none(value: str):
    from pathlib import Path
    try:
        return Path(value)
    except Exception:
        return None


def Path_exists(path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False
