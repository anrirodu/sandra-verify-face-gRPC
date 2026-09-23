"""Detección de rostros con YOLO (Ultralytics).

El detector YOLO localiza *dónde* está cada rostro. Los pesos esperados
son un modelo YOLO afinado sobre rostros (ej. ``yolov8n-face.pt``); si se
usa un peso COCO genérico (``yolov8n.pt``) este módulo se limita a la
clase 0 (``person``) y descarta los demás objetos, lo que en la práctica
sirve como *detector de personas* de respaldo cuando no hay un modelo de
rostros disponible.

Carga *lazy*: el modelo se instancia en la primera detección y queda en
caché. Esto mantiene el import del paquete ligero (los tests de config y
repository no arrastran Ultralytics) y permite que PyInstaller encuentre
los datos del wrapper con ``--collect-all ultralytics``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config
from .common import BBox, DetectedFace, ModelNotLoadedError

log = logging.getLogger(config.settings.service_name)


class YoloFaceDetector:
    """Wrapper tipado sobre ``ultralytics.YOLO`` para detección de rostros.

    Atributos de clase por convención del ecosistema Sandra: identificador
    en inglés (``_model``), comentarios en español.
    """

    # Órbitas de runtime que Ultralytics detecta automáticamente.
    _CPU_HINT = "CPU (fallback)"

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings
        self._model = None          # caché del modelo YOLO
        self._device = self._CPU_HINT
        self._loaded = False
        self._load_error: Optional[str] = None

    # ─── Ciclo de vida ──────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def _resolve_weights(self) -> Path:
        """Devuelve la ruta de pesos; lanza error claro si falta."""
        path = self._settings.detector_model
        if isinstance(path, str):
            path = Path(path)
        if not path.is_absolute():
            path = (config.BASE_DIR / path)
        if not path.exists():
            raise ModelNotLoadedError(
                f"No existe el modelo del detector: {path}. "
                "Descárgalo con: scripts/download_weights.sh"
            )
        return path

    def load(self) -> None:
        """Carga (o recarga) el modelo YOLO en memoria."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO  # import pesado, dentro del método
        except Exception as exc:  # pragma: no cover — depende del entorno
            self._load_error = (
                "ultralytics no está instalado (pip install ultralytics)"
            )
            raise ModelNotLoadedError(self._load_error) from exc

        try:
            weights = self._resolve_weights()
            self._model = YOLO(str(weights))
            self._loaded = True
            self._device = self._pick_device()
            log.info("Detector YOLO cargado: %s · device=%s", weights.name, self._device)
        except Exception as exc:  # pragma: no cover — I/O / versión
            self._load_error = str(exc)
            raise ModelNotLoadedError(f"No se pudo cargar el detector: {exc}") from exc

    @staticmethod
    def _pick_device() -> str:
        """Selecciona el dispositivo de inferencia usable de verdad.

        Ultralytics rechaza ``device=auto`` cuando no hay CUDA (Mac). Aquí
        resolvemos: MPS si torch lo reporta, luego CUDA, y CPU de respaldo.
        """
        if not config.settings.detector_gpu:
            return "cpu"
        try:
            import torch  # noqa: F401
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    # ─── Inferencia ─────────────────────────────────────
    def detect(self, image_bgr: np.ndarray, min_conf: Optional[float] = None) -> List[DetectedFace]:
        """Detección de rostros en una imagen BGR (OpenCV).

        Parámetros
        ----------
        image_bgr:
            Imagen decodificada por OpenCV (formato BGR, uint8).
        min_conf:
            Umbral de confianza; si es ``None`` usa la configuración.

        Devuelve una lista (posiblemente vacía) de ``DetectedFace``
        ordenada por confianza descendente.
        """
        if not self._loaded:
            self.load()
        assert self._model is not None, "modelo disponible tras load()"

        threshold = float(min_conf) if min_conf is not None else self._settings.detector_conf
        device = self._pick_device()

        try:
            results = self._model.predict(
                source=image_bgr,
                conf=threshold,
                iou=self._settings.detector_iou,
                imgsz=self._settings.detector_img_size,
                device=device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover — runtime del modelo
            raise RuntimeError(f"YOLO falló en predict: {exc}") from exc

        if not results or len(results) == 0:
            return []

        return self._extract(results[0])

    def _extract(self, result) -> List[DetectedFace]:
        """Transforma el resultado ``ultralytics`` a ``DetectedFace``."""
        faces: List[DetectedFace] = []
        try:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return faces
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, conf, cls):
                # Rostros (clase 0) en modelos afinados; en COCO genérico
                # la clase 0 es "person" – la tratamos igual (fallback).
                if k != 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                faces.append(
                    DetectedFace(
                        bbox=BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                        confidence=float(c),
                        class_id=int(k),
                    )
                )
        except Exception as exc:  # pragma: no cover — shape inesperada
            log.warning("No se pudieron parsear los boxes YOLO: %s", exc)
            return []

        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces
