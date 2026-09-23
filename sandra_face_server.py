"""Entrypoint del binario PyInstaller para el servidor gRPC de reconocimiento facial.

Importar desde la raíz (y no lanzar face_service/server.py como script) evita
el clásico fallo de PyInstaller con imports relativos (``from . import pb2``).
El módulo ``face_service.server`` expone ``main()``; aquí solo delegamos.
"""

from face_service.server import main

if __name__ == "__main__":
    raise SystemExit(main())
