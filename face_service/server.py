"""Servidor gRPC aio (gRPC async) con health check integrado.

Servicio *de cara a la red*: recibe ``bytes`` de imagen, ejecuta el
pipeline en un executor (hilos) para no bloquear el bucle de eventos, y
traduce ``ImageResult`` → ``ImageReply``. Separa con claridad:

* **Capa de transporte** (este módulo): conversión pb ↔ dominio, política
  de errores → status gRPC, health check, arranque/parada.
* **Capa de dominio** (``pipeline.py``): imágenes crudas → ``ImageResult``,
  sin tocar ni pb2 ni gRPC.

El pipeline mantiene la *métrica* (ms por etapa) y auto-registra rostros
nuevos si ``register_faces`` viene activo, tal y como documenta el proto.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as _futures
import logging
import signal as _signal
from typing import Dict, List, Optional

import grpc
import grpc_reflection.v1alpha.reflection as _reflection

from . import config, imagenes, pb2
from .common import FaceServiceError, now_ms
from .pipeline import ImageResult, RecognitionPipeline

log = logging.getLogger(config.settings.service_name)


# ════════════════════════════════════════════════════════════
# Helpers de traducción pb ↔ dominio
# ════════════════════════════════════════════════════════════
def _to_bbox_pb(bbox) -> "pb2.face_recognition_pb2.BBox":
    return pb2.face_recognition_pb2.BBox(
        x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height
    )


def _to_face_result_pb(face) -> "pb2.face_recognition_pb2.FaceResult":
    return pb2.face_recognition_pb2.FaceResult(
        person_id=face.person_id,
        label=face.label,
        confidence=face.confidence,
        match_score=face.match_score,
        bbox=_to_bbox_pb(face.bbox),
        known=face.known,
        embedding=[float(x) for x in face.embedding],
    )


def _to_image_reply(result: ImageResult) -> "pb2.face_recognition_pb2.ImageReply":
    return pb2.face_recognition_pb2.ImageReply(
        faces=[_to_face_result_pb(f) for f in result.faces],
        processing_ms=result.processing_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        faces_registered=result.faces_registered,
        index_changed=result.index_changed,
    )


# ════════════════════════════════════════════════════════════
# Implementación del servicio gRPC
# ════════════════════════════════════════════════════════════
class FaceRecognitionServicer(pb2.face_recognition_pb2_grpc.FaceRecognitionServicer):
    """Mapea llamadas gRPC → pipeline y respuestas pb.

    Un único executor de hilos compartido para todas las RPC: la carga de
    modelos es *lazy* (primera llamada) y el pipeline es síncrono, así que
    ejecutamos ``process_image_bytes`` dentro del executor y esperamos el
    futuro sin bloquear el loop aio.
    """

    def __init__(self, pipeline: RecognitionPipeline) -> None:
        self._pipeline = pipeline
        self._started_ms = now_ms()
        self._executor = _futures.ThreadPoolExecutor(
            max_workers=config.settings.grpc_max_workers,
            thread_name_prefix="face-pipeline",
        )

    # ─── RPC: ProcessImage ──────────────────────────────
    async def ProcessImage(
        self, request: "pb2.face_recognition_pb2.ImageRequest", context
    ) -> "pb2.face_recognition_pb2.ImageReply":
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._pipeline.process_image_bytes,
            request.image_data,
        )
        if result.last_error is not None:
            # Error de dominio → status de error claro para el cliente.
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, result.last_error)
        return _to_image_reply(result)

    # ─── RPC: RegisterFace ──────────────────────────────
    async def RegisterFace(
        self, request: "pb2.face_recognition_pb2.RegisterRequest", context
    ) -> "pb2.face_recognition_pb2.RegisterReply":
        """Registra un rostro: detecta→embebe→añade al índice.

        ``RecognitionPipeline`` no expone ``register_face``; esta RPC usa
        el contrato real de ``PersonRepository`` (``add_embedding``) y el
        pipeline (``process_bytes``) para obtener el embedding del rostro
        principal de la imagen. Además, la imagen original recibida se
        guarda en ``imagenes/`` (binario/base64/PNG/JPG autodetectado)
        para reconocimiento posterior por otro servicio.
        """
        # Persistimos la imagen original antes de procesar: siempre queda el
        # original disponible aunque el rostro no llegue a embeberse.
        try:
            stored = imagenes.save_image(
                request.image_data, request.person_id, fmt=request.format
            )
        except imagenes.ImageStoreError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb2.face_recognition_pb2.RegisterReply(success=False)
        loop = asyncio.get_running_loop()

        def _register_sync() -> Dict[str, object]:
            # 1) Detectar+embeder el rostro principal de la imagen.
            result = self._pipeline.process_bytes(request.image_data)
            if not result.faces:
                raise RuntimeError(
                    "No se detectó ningún rostro con la confianza mínima"
                )
            main = result.faces[0]          # FaceMatch real (primer rostro)
            emb = main.embedding
            if emb is None:
                raise RuntimeError("El pipeline no devolvió embedding")
            # 2) Añadir al repositorio real.
            pos = self._pipeline._repository.add_embedding(
                emb, person_id=request.person_id, label=request.label or request.person_id
            )
            self._pipeline._repository.save()
            return {
                "success": True,
                "person_id": request.person_id,
                "label": request.label or request.person_id,
                "entry_id": int(pos),
                "index_size": len(self._pipeline._repository),
            }

        try:
            reply = await loop.run_in_executor(self._executor, _register_sync)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb2.face_recognition_pb2.RegisterReply(success=False)
        return pb2.face_recognition_pb2.RegisterReply(
            entry_id=int(reply["entry_id"]),
            index_size=int(reply["index_size"]),
            success=True,
            message="Rostro registrado",
            status=0,                      # 0 = creado
        )

    # ─── RPC: RemovePerson ──────────────────────────────
    async def RemovePerson(
        self, request: "pb2.face_recognition_pb2.RemoveRequest", context
    ) -> "pb2.face_recognition_pb2.RemoveReply":
        loop = asyncio.get_running_loop()

        def _remove_sync() -> Dict[str, object]:
            removed = self._pipeline._repository.remove_person(request.person_id)
            self._pipeline._repository.save()
            return {
                "removed": int(removed),
                "index_size": len(self._pipeline._repository),
            }

        try:
            reply = await loop.run_in_executor(self._executor, _remove_sync)
        except Exception as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            return pb2.face_recognition_pb2.RemoveReply(success=False)
        return pb2.face_recognition_pb2.RemoveReply(
            success=True, removed=int(reply["removed"]), index_size=int(reply["index_size"])
        )

    # ─── RPC: ListPersons ───────────────────────────────
    async def ListPersons(
        self, request: "pb2.face_recognition_pb2.ListRequest", context
    ) -> "pb2.face_recognition_pb2.ListReply":
        loop = asyncio.get_running_loop()

        def _persons_sync() -> List[Dict[str, object]]:
            return self._pipeline._repository.persons()

        try:
            persons = await loop.run_in_executor(self._executor, _persons_sync)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return pb2.face_recognition_pb2.ListReply()
        return pb2.face_recognition_pb2.ListReply(
            persons=[
                pb2.face_recognition_pb2.PersonInfo(
                    person_id=str(p["person_id"]),
                    label=str(p.get("label", "")),
                    face_count=int(p.get("face_count", 0)),
                    registered=True,
                )
                for p in persons
            ]
        )

    # ─── RPC: Check (health) ────────────────────────────
    async def Check(
        self, request: "pb2.face_recognition_pb2.HealthCheckRequest", context
    ) -> "pb2.face_recognition_pb2.HealthCheckResponse":
        settings = config.settings
        ready = self._pipeline.loaded       # contrato real del pipeline
        status = "SERVING" if ready else "LOADING" if self._pipeline.load_error is None else "NOT_SERVING"
        uptime = int((now_ms() - _server_started_ms) / 1000)
        return pb2.face_recognition_pb2.HealthCheckResponse(
            status=status,
            version=settings.service_version,
            started_at=int(_server_started_ms // 1000),
            uptime_seconds=uptime,
            model_loaded=ready,
            accelerated=settings.detector_gpu or settings.embedder_gpu,
        )

    # --- RPC: VerifyFace -----------------------------------------------
    async def VerifyFace(
        self, request: "pb2.face_recognition_pb2.VerifyRequest", context
    ) -> "pb2.face_recognition_pb2.VerifyReply":
        """Verificacion 1:1 (cedula <-> camara).

        El servicio externo envia la *cedula* del titular (ej. "17818665"
        o "17818665.jpg") y la *imagen capturada* por la camara. Localiza
        la foto de referencia guardada en ``imagenes/<cedula>.<ext>`` y la
        compara con la captura usando ``RecognitionPipeline.compare_images``.
        """
        person_id = str(request.person_id or "").strip()
        if not person_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "person_id requerido")
            return pb2.face_recognition_pb2.VerifyReply(success=False)

        # Normaliza la cedula: acepta "17818665.jpg" -> "17818665".
        cid = person_id.split(".")[0].strip()

        # 1) Foto de referencia guardada (la imagen registrada del titular).
        ref_path = imagenes.find_reference(cid)
        if ref_path is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"No existe imagen de referencia para la cedula {cid}",
            )
            return pb2.face_recognition_pb2.VerifyReply(success=False, person_id=cid)

        # 2) Captura de camara (la imagen a validar).
        if not request.image_data:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image_data requerido")
            return pb2.face_recognition_pb2.VerifyReply(success=False, person_id=cid)
        probe_bytes = request.image_data

        ref_bytes: bytes = ref_path.read_bytes()
        loop = asyncio.get_running_loop()

        def _verify_sync() -> Dict[str, object]:
            return self._pipeline.compare_images(
                ref_bytes,
                probe_bytes,
                threshold=request.match_threshold if request.HasField("match_threshold") else None,
            )

        try:
            result: Dict[str, object] = await loop.run_in_executor(
                self._executor, _verify_sync
            )
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return pb2.face_recognition_pb2.VerifyReply(success=False, person_id=cid)

        if not result.get("success", False):
            await context.abort(grpc.StatusCode.INTERNAL, str(result.get("message", "")))
            return pb2.face_recognition_pb2.VerifyReply(success=False, person_id=cid)

        label = self._pipeline._repository.label_of(cid) \
            if hasattr(self._pipeline._repository, "label_of") else ""
        return pb2.face_recognition_pb2.VerifyReply(
            success=True,
            match=bool(result.get("match", False)),
            score=float(result.get("score", 0.0)),
            threshold=float(result.get("threshold", 0.0)),
            person_id=cid,
            label=str(label or ""),
            reference_image=str(ref_path.name),
            message=str(result.get("message", "")),
            processing_ms=int(result.get("processing_ms", 0)),
        )


# ════════════════════════════════════════════════════════════
# Arranque / claves públicas
# ════════════════════════════════════════════════════════════
# Sello del arranque del módulo (para uptime del health check).
_server_started_ms: float = now_ms()

def build_servicer(pipeline: Optional[RecognitionPipeline] = None):
    """Fábrica del servicer (perezosa: construye el pipeline de verdad solo
    cuando hace falta; los tests inyectan uno fake)."""
    if pipeline is None:
        pipeline = RecognitionPipeline(config.settings)
    return FaceRecognitionServicer(pipeline)


# ════════════════════════════════════════════════════════════
# Bootstrap: arranque del servidor gRPC aio
# ════════════════════════════════════════════════════════════
def _build_aio_server():
    """Construye el ``grpc.aio.Server`` con el servicer real y opciones.

    Separado de ``serve`` para que los tests lo compongan sin necesitar
    un event loop (y para poder inyectar un servicer fake).
    """
    s = config.settings
    server = grpc.aio.server(
        options=[
            # Límite de mensaje grande en bytes (32 MB por defecto).
            (
                "grpc.max_send_message_length",
                s.grpc_max_message_mb * 1024 * 1024,
            ),
            (
                "grpc.max_receive_message_length",
                s.grpc_max_message_mb * 1024 * 1024,
            ),
        ],
    )
    servicer = build_servicer()
    pb2.face_recognition_pb2_grpc.add_FaceRecognitionServicer_to_server(
        servicer, server
    )
    # Reflection API → permite ``grpcurl list``/``describe`` sin stubs.
    _service_names = (
        pb2.face_recognition_pb2.DESCRIPTOR.services_by_name[
            "FaceRecognition"
        ].full_name,
        _reflection.SERVICE_NAME,
    )
    _reflection.enable_server_reflection(_service_names, server)
    return server


def serve(host: str = "", port: int = 0) -> None:
    """Arranca el servidor gRPC **aio** y bloquea hasta Ctrl+C.

    Usa los valores de ``config.settings`` salvo que el llamador
    inyecte ``host``/``port``.

    **Importante (gRPC aio):** TODA la construcción (``grpc.aio.server``,
    bind, start, stop) debe ocurrir DENTRO del event loop con el que el
    servidor corre. Por eso ``serve`` es asíncrona y ``main`` la lanza
    con ``asyncio.run``: si se crea el server fuera del loop, gRPC le
    engancha los futures al loop *anterior* y revienta con
    *"attached to a different loop"*.

    Modelos *lazy*: el pipeline solo carga el detector+embedder en la
    primera RPC real, así que el servidor responde al health check sin
    haber descargado ningún peso.
    """

    async def _serve_aio() -> None:
        s = config.settings
        bind_host = host or s.grpc_host
        bind_port = port or s.grpc_port

        server = _build_aio_server()  # dentro del loop que corre
        bound = server.add_insecure_port(f"{bind_host}:{bind_port}")
        if bound == 0:
            raise RuntimeError(f"No se pudo vincular {bind_host}:{bind_port}")

        await server.start()
        log.info(
            "gRPC · ejecutandose en %s:%d (pipeline %s)",
            bind_host, bind_port, s.service_version,
        )
        try:
            await server.wait_for_termination()
        finally:
            await server.stop(grace=2.0)

    try:
        asyncio.run(_serve_aio())
    except KeyboardInterrupt:
        log.info("Se a apagado el servidor gRPC con Ctrl+C")



def main(argv: Optional[list] = None) -> int:
    """CLI de arranque: ``python -m face_service.server [--host] [--port]``.

    Sin argumentos usa ``config.settings`` (0.0.0.0:50052 por defecto).
    """
    s = config.settings
    p = argparse.ArgumentParser(
        prog="face-service-server",
        description="Servidor gRPC de reconocimiento facial (Bunker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=s.grpc_host, help="Host a vincular.")
    p.add_argument("--port", type=int, default=s.grpc_port, help="Puerto gRPC.")
    args = p.parse_args(argv)

    logging.basicConfig(level=s.log_level)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
