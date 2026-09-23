"""Almacenamiento persistente de imágenes originales para reconocimiento
facial posterior (servicio externo / SERF).

El servidor gRPC recibe la foto en varios formatos aceptados por el
cliente: bytes de imagen crudos (``png``/``jpg``), ``base64`` (texto) o el
formato explícito. Este módulo normaliza y guarda la foto en
``DATA_DIR/IMAGES_DIR`` (carpeta ``imagenes/``) con el patrón
``<person_id>_<timestamp>.<ext>``, de modo que otro servicio pueda leerla
y aplicar su propio reconocimiento sin acoplarse al índice interno.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import time
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger(config.settings.service_name)

# Firmas de encabezado mágicas (magic bytes) para detectar el formato
# real aunque el campo ``format`` venga vacío o mal indicado.
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}

_BASE64_RE = re.compile(
    rb"^[A-Za-z0-9+/]+=*$"
)
_DATA_URL_RE = re.compile(rb"^data:image/(\w+);base64,(.*)$", re.S)

# Extensiones que se escriben tal cual (sin re-codificar).
_REMAP = {"jpeg": "jpg"}


class ImageStoreError(Exception):
    """No se pudo guardar la imagen recibida."""


def detect_format(data: bytes, declared: str = "") -> str:
    """Devuelve el formato real de la imagen (png/jpg/gif/bmp/tiff/base64).

    Prioridad: él ``declared`` explícito (normalizado) → firma mágica →
    detecta base64 (texto). Si no hay coincidencias devuelve ``bin``.
    """
    if declared:
        norm = declared.strip().lower().lstrip(".")
        if norm:
            return _REMAP.get(norm, norm)
    for magic, fmt in _MAGIC.items():
        if data.startswith(magic):
            return fmt
    if _is_base64(data):
        return "base64"
    return "bin"


def _is_base64(data: bytes) -> bool:
    """Reconoce base64 (texto ASCII válido, sin whitespace inesperado)."""
    probe = data[:128]
    if not probe or len(data) < 8:
        return False
    import string  # noqa: PLC0415
    allowed = set(string.ascii_letters + string.digits + "+/=\r\n\t ")
    if any(c not in allowed for c in data[:256]):
        return False
    return bool(_BASE64_RE.match(data.strip()))


def decode_image(data: bytes, fmt: str) -> bytes:
    """Devuelve los bytes crudos de imagen (decodiﬁca base64 si hace falta)."""
    if fmt == "base64" or _is_base64(data):
        body = data
        m = _DATA_URL_RE.match(body)
        if m:
            body = m.group(2)
        try:
            return base64.b64decode(body)
        except (binascii.Error, ValueError) as exc:
            raise ImageStoreError(f"base64 inválido: {exc}") from exc
    return data


def save_image(
    image_data: bytes,
    person_id: str,
    fmt: str = "",
    images_dir: Optional[Path] = None,
) -> Path:
    """Guarda una imagen recibida en ``imagenes/`` y devuelve su ruta.

    - Detecta formato real (declarado o por firma mágica).
    - Si viene base64 lo decodiﬁca a bytes.
    - Nombre: ``<person_id>_<unix_ms>.<ext>`` (colisiones improbables).
    """
    if not image_data:
        raise ImageStoreError("imagen vacía")

    images_dir = Path(images_dir or config.settings.images_dir)
    fmt_real = detect_format(image_data, fmt)
    raw = decode_image(image_data, fmt_real)

    if fmt_real == "bin":
        fmt_real = detect_format(raw, "")
    ext = "bin" if fmt_real in {"bin", "base64"} else fmt_real

    images_dir.mkdir(parents=True, exist_ok=True)
    sanitized = re.sub(r"[^A-Za-z0-9_\-]", "_", person_id)[:64] or "unknown"
    name = f"{sanitized}.{ext}"          # nombre estable: 17818665.jpg
    path = images_dir / name

    path.write_bytes(raw)
    log.info("Imagen guardada: %s (%d bytes, formato %s)", path, len(raw), fmt_real)
    return path

def find_reference(person_id: str, images_dir: Optional[Path] = None) -> Optional[Path]:
    """Localiza la foto de referencia de una persona en ``imagenes/``.

    Busca ``<person_id>.<ext>`` con cualquier extensión soportada. Devuelve
    ``None`` si no existe. Es la imagen usada por ``VerifyFace`` para la
    comparación 1:1 (la cédula ↔ la persona en la cámara).
    """
    images_dir = Path(images_dir or config.settings.images_dir)
    if not images_dir.is_dir():
        return None
    sanitized = re.sub(r"[^A-Za-z0-9_\-]", "_", person_id)[:64] or person_id
    # Preferencia 1: coincidencia exacta con la extensión que tenga.
    for ext in ("jpg", "jpeg", "png", "bmp", "webp", "gif", "bin", "tiff"):
        cand = images_dir / f"{sanitized}.{ext}"
        if cand.exists():
            return cand
    # Preferencia 2: cualquier archivo que empiece por <person_id>.
    matches = sorted(images_dir.glob(f"{sanitized}.*"))
    return matches[0] if matches else None
