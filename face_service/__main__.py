"""Punto de entrada alternativo: ``python -m face_service``.

Habilita el modo por defecto del paquete sin instalar dependencias gRPC
adicionales: si la ruta ``scripts/*.py`` no está disponible (porque el
paquete se importó desde un entorno mínimo), este módulo reexporta el CLI
real que vive en ``server.py``/``client.py``. Mantenemos aquí SOLO la lógica
de arranque — el trabajo pesado está en los módulos hermanos.
"""

from __future__ import annotations

import sys


def main(argv: Optional[List[str]] = None) -> int:
    """Delega en ``face_service.client.main`` si es invocable directamente."""
    from .client import main as client_main  # import pesado, perezoso

    return client_main(argv)


if __name__ == "__main__":
    sys.exit(main())
