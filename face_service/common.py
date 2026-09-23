"""Tipos compartidos, excepciones y utilidades de bajo nivel.

Este módulo es intencionalmente independiente de cualquier dependencia
pesada (``numpy``, ``ultralytics``, ``insightface``, …). Cualquier módulo
del paquete puede importarlo de forma segura sin arrastrar librerías.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ════════════════════════════════════════════════════════════
# Excepciones de dominio
# ════════════════════════════════════════════════════════════
class FaceServiceError(Exception):
    """Error base de todo el servicio."""


class ModelNotLoadedError(FaceServiceError):
    """Los pesos del detector/embedder no están cargados o no existen."""


class DecodeError(FaceServiceError):
    """No fue posible decodificar los bytes de imagen con OpenCV."""


class NoFaceDetectedError(FaceServiceError):
    """YOLO no encontró rostros con la confianza mínima configurada."""


class MultipleFacesError(FaceServiceError):
    """Se esperaba un único rostro (enrolamiento) y hay varios."""


class InvalidVectorError(FaceServiceError):
    """El embedding recibido no coincide con las dimensiones del índice."""


class PersonNotFoundError(FaceServiceError):
    """``person_id`` no existe en el índice."""


# ════════════════════════════════════════════════════════════
# Tipos de dominio (estructuras puras, sin lógica)
# ════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BBox:
    """Caja delimitadora de un rostro en la imagen original (px)."""

    x: float            # esquina superior-izquierda X
    y: float            # esquina superior-izquierda Y
    width: float        # anchura
    height: float       # altura

    @property
    def area(self) -> float:
        return self.width * self.height

    def xyxy(self) -> List[float]:
        """Devuelve ``[x1, y1, x2, y2]`` (formato YOLO)."""
        return [self.x, self.y, self.x + self.width, self.y + self.height]


@dataclass
class DetectedFace:
    """Rostro salido del detector YOLO (antes del embedding)."""

    # Caja en la imagen ORIGINAL (ya re-escalada desde la entrada YOLO).
    bbox: BBox
    # Confianza de la detección YOLO (0.0–1.0).
    confidence: float
    # Índice de clase del modelo (ej. 0 para "face" en modelos afinados).
    class_id: int
    track_id: Optional[int] = None


@dataclass
class FaceMatch:
    """Resultado ya resuelto contra el índice (salida de la búsqueda)."""

    person_id: str
    label: str
    bbox: BBox
    confidence: float          # confianza de detección YOLO
    match_score: float         # similitud coseno del mejor candidato (0–1)
    embedding: Optional[List[float]] = None   # 512-d (si se pide)
    known: bool = True


# ════════════════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════════════════
def now_ms() -> float:
    """Sello de tiempo de alta resolución en milisegundos."""
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Milisegundos transcurridos desde un sello previo de ``now_ms()``."""
    return now_ms() - start_ms


@dataclass
class ElapsedTimer:
    """Cronómetro de alta resolución para medir una etapa del pipeline.

    Es la pieza entre ``now_ms()`` (sello puntual) y ``elapsed_ms``
    (diferencia): ofrece un objeto con estado que cuenta el tiempo entre
    ``start()`` y ``stop()``/``lap()``. Lo consume ``pipeline``/``server``
    para desglosar los ms por etapa sin repetir el molde
    ``t0 = now_ms(); ...; ms = elapsed_ms(t0)``.
    """

    _start: Optional[float] = None

    def start(self) -> "ElapsedTimer":
        """Arranca (o reinicia) el cronómetro. Devuelve self para encadenar."""
        self._start = now_ms()
        return self

    def stop(self) -> float:
        """Detiene y devuelve los ms transcurridos; deja el crono en cero."""
        if self._start is None:
            return 0.0
        ms = elapsed_ms(self._start)
        self._start = None
        return ms

    def lap(self) -> float:
        """Devuelve los ms desde el último reset sin detener el crono."""
        if self._start is None:
            return 0.0
        return elapsed_ms(self._start)

    @property
    def running(self) -> bool:
        return self._start is not None
