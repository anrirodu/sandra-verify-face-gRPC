"""Cliente CLI del servicio de reconocimiento facial (gRPC).

Expone los cuatro flujos documentados en el README:

* ``register <person_id> <imagen> [--label]``  → enrola a una persona.
* ``identify <imagen> [--min-conf] [--threshold]`` → identifica rostros.
* ``remove <person_id>`` → elimina a la persona del índice.
* ``list`` → lista personas indexadas.

Usa ``face_service.client`` internamente y mantiene el manejador de
timing de cada etapa. Es "open-box": imprime la métrica por rostro para
que un humano (o una integración CI) pueda auditar el costo del pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import grpc

from . import config, pb2


def _make_stub(channel) -> "pb2.face_recognition_pb2_grpc.FaceRecognitionStub":
    return pb2.face_recognition_pb2_grpc.FaceRecognitionStub(channel)


def register_face(
    stub, person_id: str, image_path: Path, label: str = "" 
) -> "pb2.face_recognition_pb2.RegisterReply":
    data = image_path.read_bytes()
    req = pb2.face_recognition_pb2.RegisterRequest(
        person_id=person_id, label=label, image_data=data
    )
    return stub.RegisterFace(req)


def identify_image(
    stub,
    image_path: Path,
    min_conf: float | None = None,
    threshold: float | None = None,
) -> "pb2.face_recognition_pb2.ImageReply":
    data = image_path.read_bytes()
    req = pb2.face_recognition_pb2.ImageRequest(image_data=data)
    if min_conf is not None:
        req.min_confidence = min_conf
    if threshold is not None:
        req.match_threshold = threshold
    return stub.ProcessImage(req)


def remove_person(stub, person_id: str) -> "pb2.face_recognition_pb2.RemoveReply":
    return stub.RemovePerson(pb2.face_recognition_pb2.RemoveRequest(person_id=person_id))


def list_persons(stub) -> "pb2.face_recognition_pb2.ListReply":
    return stub.ListPersons(pb2.face_recognition_pb2.ListRequest())


def verify_face(
    stub,
    person_id: str,
    image_path: Optional[Path] = None,
    image_data: bytes | None = None,
    threshold: float | None = None,
) -> "pb2.face_recognition_pb2.VerifyReply":
    """Verificación 1:1: cédula (con o sin extensión, ej. "17818665.jpg")
    contra la imagen de cámara. Usa la foto guardada en ``imagenes/`` como
    referencia en el servidor."""
    data = image_data if image_data is not None else image_path.read_bytes()
    req = pb2.face_recognition_pb2.VerifyRequest(
        person_id=person_id, image_data=data
    )
    if threshold is not None:
        req.match_threshold = threshold
    return stub.VerifyFace(req)


def _print_reply(reply) -> None:
    """Imprime de forma compacta el resultado (open-box)."""
    header = f"gRPC {config.settings.grpc_host}:{config.settings.grpc_port}"
    print(f"\n── {header} ──")
    if getattr(reply, "match", None) is not None and hasattr(reply, "score"):
        verdict = "✅ MATCH" if reply.match else "❌ NO_MATCH"
        print(f"  {verdict} · score={reply.score:.4f} · threshold={reply.threshold:.3f}")
        print(f"  persona: {reply.person_id} · {reply.label or '-'}")
        print(f"  referencia: {reply.reference_image} · {reply.processing_ms} ms")
        if reply.message:
            print(f"  detalle: {reply.message}")
    # Bloque de RegisterReply/RemoveReply (verificaciones no tienen
    # entry_id; ese bloque solo aplica cuando el mensaje lo define).
    if hasattr(reply, "entry_id") and getattr(reply, "person_id", ""):
        print(f"  persona: {reply.person_id} · {reply.label or '-caja-'}")
        print(
            f"  entry_id={reply.entry_id} · índice={reply.index_size} "
            f"· status={reply.status}"
        )
    if getattr(reply, "faces", None) is not None:
        for i, face in enumerate(reply.faces):
            print(
                f"  [{i}] {face.label or 'desconocido'} · "
                f"{face.match_score:.3f} · {face.bbox.x:.0f},{face.bbox.y:.0f} "
                f"{face.bbox.width:.0f}×{face.bbox.height:.0f}"
            )
    if getattr(reply, "patients", None) is None and hasattr(reply, "persons"):
        for p in reply.persons:
            print(f"  · {p.person_id} — {p.label} ({p.face_count} rostros)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face-service",
        description="Cliente CLI del servicio gRPC Sand.Es de reconocimiento facial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--host", default=config.settings.grpc_host,
        help="Dirección del servidor gRPC.",
    )
    p.add_argument(
        "--port", type=int, default=config.settings.grpc_port,
        help="Puerto del servidor gRPC.",
    )
    p.add_argument("--timeout", type=float, default=10.0, help="Timeout (s).")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="Enrola a una persona.")
    r.add_argument("person_id", type=str)
    r.add_argument("image", type=Path)
    r.add_argument("--label", default="", type=str)

    i = sub.add_parser("identify", help="Identifica rostros en una imagen.")
    i.add_argument("image", type=Path)
    i.add_argument("--min-conf", type=float)
    i.add_argument("--threshold", type=float)

    rm = sub.add_parser("remove", help="Elimina a una persona del índice.")
    rm.add_argument("person_id", type=str)

    v = sub.add_parser("verify", help="Verifica 1:1 (cédula ↔ imagen de cámara).")
    v.add_argument("person_id", type=str, help="Cédula (ej. 17818665 o 17818665.jpg).")
    v.add_argument("image", type=Path, help="Imagen capturada (nueva-imagen.jpg).")
    v.add_argument("--threshold", type=float, help="Umbral de match (coseno 0–1).")

    ls = sub.add_parser("list", help="Lista personas indexadas.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=config.settings.log_level)

    target = f"{args.host}:{args.port}"
    try:
        with grpc.insecure_channel(target) as channel:
            stub = _make_stub(channel)
            if args.command == "register":
                reply = register_face(
                    stub, args.person_id, args.image, args.label
                )
            elif args.command == "identify":
                reply = identify_image(
                    stub, args.image, args.min_conf, args.threshold
                )
            elif args.command == "remove":
                reply = remove_person(stub, args.person_id)
            elif args.command == "verify":
                reply = verify_face(
                    stub, args.person_id, args.image, threshold=args.threshold
                )
            else:
                reply = list_persons(stub)
            _print_reply(reply)
        return 0
    except grpc.RpcError as exc:
        logging.error("gRPC error: %s · %s", exc.code(), exc.details())
        return 1
    except Exception as exc:  # pragma: no cover
        logging.error("Error inesperado: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
