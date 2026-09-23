"""Repositorio: índice de similitud de embeddings (FAISS o NumPy).

Dos motores intercambiables tras la misma interfaz:

* **faiss**  → ``faiss-cpu`` con ``IndexFlatIP`` (producto interno; los
  vectores ya vienen normalizados → coincide con coseno). Rápido, pero
  requiere compilación nativa y añade peso al binario.
* **numpy**  → *linear scan* O(n·d) con normalización L2 y producto
  punto. Suficiente para bases de hasta varios miles de rostros y evita
  instalar FAISS (clave para el binario PyInstaller pequeño).

``INDEX_ENGINE=auto`` (valor por defecto) elige FAISS si está instalado
y cae a NumPy con un aviso en el log si no. El índice completo (vector +
metadatos) se persiste como un par de archivos:

    data/faces.index   →  matriz NumPy ``float32`` N×d
    data/faces.ndjson  →  una línea por vector con ``person_id``/``label``
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import config
from .common import (
    InvalidVectorError,
    NoFaceDetectedError,
    PersonNotFoundError,
    elapsed_ms,
    now_ms,
)
from .embedder import np

log = logging.getLogger(config.settings.service_name)


# ─── Protocolo de persistencia de metadatos ────────────────
# ndjson: un JSON por línea; el orden de la línea == fila del índice.
_META_JSON_FIELDS = ("person_id", "label")

# Encabezado privado del archivo .index (NumPy .npy de 128 bytes) — no es
# un formato de FAISS real; es nuestra persistencia propia.

def _default_index_file() -> Path:
    return config.settings.index_file


def _default_meta_file() -> Path:
    return config.settings.index_meta


# ════════════════════════════════════════════════════════════
# Motor base
# ════════════════════════════════════════════════════════════
class SimilarityBackend:
    """Interfaz común para los motores de búsqueda."""

    name = "base"

    def __init__(
        self, dim: int, metric: str = "cosine", path: Optional[Path] = None
    ) -> None:
        self.dim = dim
        self.metric = metric
        self._path = path
        self._vectors: List[np.ndarray] = []          # embeddings raw
        self._norm: List[np.ndarray] = []             # normalizados L2

    # ─── estado ─────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._norm)

    @property
    def empty(self) -> bool:
        return len(self._norm) == 0

    # ─── mutación ───────────────────────────────────────
    def add(self, vectors: Iterable[np.ndarray]) -> None:
        for v in vectors:
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.size != self.dim:
                raise InvalidVectorError(
                    f"Se esperaban {self.dim} dims y llegaron {arr.size}"
                )
            self._vectors.append(arr.copy())
            # Normalización L2 → el producto punto interno equivale a
            # similitud coseno (norma de la distancia al cuadrado).
            norm = float(np.linalg.norm(arr))
            self._norm.append(arr if norm == 0 else arr / norm)

    def remove(self, indices: Iterable[int]) -> int:
        """Elimina filas por posición (descendente, para no invalidar)."""
        idx = sorted(set(int(i) for i in indices), reverse=True)
        removed = 0
        for i in idx:
            if 0 <= i < len(self._norm):
                del self._vectors[i]
                del self._norm[i]
                removed += 1
        return removed

    # ─── persistencia ────────────────────────────────────
    def save(self) -> None:
        """Persiste los vectores en el archivo de índice (``.npy``)."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._norm:
            # Índice vacío: eliminamos un archivo viejo si existía.
            self._store_path().unlink(missing_ok=True)
            return
        np.save(self._store_path(), np.stack(self._norm, axis=0).astype("float32"))

    def _store_path(self) -> Path:
        """Ruta real del archivo: ``np.save`` siempre añade ``.npy``."""
        if str(self._path).endswith(".npy"):
            return self._path
        return Path(str(self._path) + ".npy")

    def load(self, meta: Any = None) -> None:
        """Reconstruye el índice desde el archivo persistido.

        Recibe el ``PersonMetadata`` como compatibilidad con la llamada de
        ``PersonRepository.load`` (los backends que lo necesiten pueden
        sincronizarse con sus filas). Por defecto solo restaura vectores.
        """
        if self._path is None:
            return
        store = self._store_path()
        if not store.exists():
            return
        mat = np.load(store).astype("float32")
        self._vectors = [v.copy() for v in mat]
        self._norm = [
            v if float(np.linalg.norm(v)) == 0 else v / float(np.linalg.norm(v))
            for v in mat
        ]

    def vector_at(self, pos: int) -> Optional[np.ndarray]:
        """Devuelve el embedding (raw) de una posición o ``None``."""
        if 0 <= pos < len(self._vectors):
            return self._vectors[pos].copy()
        return None

    # ─── búsqueda ───────────────────────────────────────
    def query(self, vec: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        """Devuelve ``[(pos, similitud), …]`` ordenado descendente.

        La similitud está en el rango **0–1** (coseno sobre vectores
        normalizados; 1 = idéntico, 0 = ortogonal, negativo = opuesto).
        """
        raise NotImplementedError  # pragma: no cover — abstracto


# ════════════════════════════════════════════════════════════
# Backend NumPy (lineal, sin dependencias extra)
# ════════════════════════════════════════════════════════════
class NumpyBackend(SimilarityBackend):
    """Búsqueda lineal sobre una matriz ``N×d``. Sin FAISS."""

    name = "numpy"

    def query(self, vec: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        q = np.asarray(vec, dtype=np.float32).ravel()
        if q.size != self.dim:
            raise InvalidVectorError(
                f"Se esperaban {self.dim} dims y llegaron {q.size}"
            )
        nq = float(np.linalg.norm(q))
        qn = q if nq == 0 else q / nq

        if not self._norm:
            return []

        matrix = np.stack(self._norm, axis=0)   # N×d
        scores = matrix @ qn                     # N — producto punto
        # Coseno ya en (−1, 1); clamp para reportar en 0–1.
        scores = np.clip(scores, -1.0, 1.0)

        k = min(top_k, len(scores))
        if k <= 0:
            return []
        # Los k mejores: argsort desc + tomamos los primeros.
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]


# ════════════════════════════════════════════════════════════
# Backend FAISS (acelerado si faiss-cpu está presente)
# ════════════════════════════════════════════════════════════
class FaissBackend(SimilarityBackend):
    """Índice FAISS ``IndexFlatIP`` (coseno por normalización previa)."""

    name = "faiss"

    def __init__(self, dim: int, metric: str = "cosine", path: Optional[Path] = None) -> None:
        super().__init__(dim, metric, path=path)
        self._faiss = self._import_faiss()
        if self._faiss is None:
            raise ImportError("faiss-cpu no está instalado")
        self._index = self._faiss.IndexFlatIP(dim)

    @staticmethod
    def _import_faiss():
        try:
            import faiss  # noqa: PLC0415 — import condicional
            return faiss
        except Exception:
            return None

    def load(self, meta: Any = None) -> None:
        """Restaura vectores y reconstruye el índice FAISS de memoria."""
        super().load(meta)          # restaura _norm desde .npy
        mat = np.stack(self._norm, axis=0).astype("float32") if self._norm else None
        self._index = self._faiss.IndexFlatIP(self.dim)
        if mat is not None and mat.shape[0]:
            self._index.add(mat)

    def add(self, vectors: Iterable[np.ndarray]) -> None:
        chunk = []
        for v in vectors:
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.size != self.dim:
                raise InvalidVectorError(
                    f"Se esperaban {self.dim} dims y llegaron {arr.size}"
                )
            norm = float(np.linalg.norm(arr))
            chunk.append(arr if norm == 0 else arr / norm)
        if chunk:
            mat = np.stack(chunk, axis=0).astype("float32")
            self._index.add(mat)
            self._norm.extend(chunk)

    def remove(self, indices: Iterable[int]) -> int:
        idx = sorted(i for i in set(int(i) for i in indices) if 0 <= i < len(self._norm))
        if not idx:
            return 0
        # FAISS elimina con ID; devolvemos cuántos vamos a borrar.
        # Como usamos IndexFlatIP sin IDs explícitos, reconstruimos: reindexamos
        # todo excepto las posiciones marcadas.
        keep = [i for i in range(len(self._norm)) if i not in set(idx)]
        mat = np.stack([self._norm[i] for i in keep], axis=0).astype("float32")
        self._index = self._faiss.IndexFlatIP(self.dim)
        if mat.size:
            self._index.add(mat)
        self._norm = [self._norm[i] for i in keep]
        return len(idx)

    def query(self, vec: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        q = np.asarray(vec, dtype=np.float32).ravel()
        if q.size != self.dim:
            raise InvalidVectorError(
                f"Se esperaban {self.dim} dims y llegaron {q.size}"
            )
        n = float(np.linalg.norm(q))
        qn = q if n == 0 else q / n
        if len(self._norm) == 0:
            return []
        scores, idx = self._index.search(qn.reshape(1, -1), min(top_k, len(self._norm)))
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            out.append((int(i), float(np.clip(s, -1.0, 1.0))))
        return out


# ════════════════════════════════════════════════════════════
# Metadatos (personas) — persistencia en NDJSON
# ════════════════════════════════════════════════════════════
class PersonMetadata:
    """Lleva la correspondencia ``fila→person_id`` y las etiquetas."""

    def __init__(self, path: Path, engine_prefix: str = "numpy") -> None:
        self.path = path
        self._rows: List[Dict[str, str]] = []   # [{"person_id":…, "label":…}, …]

    # ── consulta ────────────────────────────────────────
    def __len__(self) -> int:
        """Entradas de metadatos (filas del índice)."""
        return len(self._rows)

    def person_at(self, pos: int) -> Dict[str, str]:
        if not (0 <= pos < len(self._rows)):
            raise IndexError(f"Posición {pos} fuera de metadatos")
        return self._rows[pos]

    def set_person_record(self, pos: int, person_id: str, label: str = "") -> None:
        """Promete/metadata de la fila ``pos``."""
        while len(self._rows) <= pos:
            self._rows.append({"person_id": "", "label": ""})
        self._rows[pos] = {"person_id": person_id, "label": label or person_id}

    # ── persistencia ────────────────────────────────────
    def load(self) -> None:
        """Reconstruye ``_rows`` desde el NDJSON (una línea por fila)."""
        if not self.path.exists():
            return
        items = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("person_id") or ""
            lab = obj.get("label") or pid
            items.append({"person_id": pid, "label": lab})
        self._rows = items

    def save(self) -> None:
        """Vuelca ``_rows`` como NDJSON (atómico: tmp + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in self._rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ── agregación (para respuestas de listado) ─────────
    def persons(self) -> Dict[str, Dict[str, Any]]:
        """Consolida por ``person_id``: label + nº de rostros."""
        agg: Dict[str, Dict[str, Any]] = {}
        for rec in self._rows:
            pid = rec["person_id"]
            if pid not in agg:
                agg[pid] = {"label": rec["label"], "face_count": 0}
            agg[pid]["face_count"] += 1
        return agg


# ════════════════════════════════════════════════════════════
# PersonMatch → resultado tipado de búsqueda (lo que consume pipeline)
# ════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PersonMatch:
    """Coincidencia contra el índice, lista para ``pipeline``.

    ``pipeline.py`` crea ``FaceMatch`` a partir de estos campos
    (``pipeline.match_embedding`` los lee como ``.person_id``,
    ``.label``, ``.score``, ``.embedding`` y ``.known``), así que
    congelamos los nombres aquí para que el contrato gRPC se mantenga
    estable incluso si FAISS/NumPy cambian por debajo.
    """

    person_id: str
    label: str
    score: float                     # similitud coseno (0–1; 1 = idéntico)
    embedding: Optional[np.ndarray]  # vector 512-d del candidato
    known: bool                      # score ≥ umbral → persona conocida


# ════════════════════════════════════════════════════════════
# PersonRepository → fachada única de índice para pipeline/server
# ════════════════════════════════════════════════════════════
class PersonRepository:
    """Repository *de dominio* usado por pipeline, server y CLIs.

    Compone:

    * un backend de similitud (``SimilarityBackend``: auto→FAISS→NumPy),
    * metadatos por persona (``PersonMetadata`` en NDJSON).

    Expone la API que ``pipeline.RecognitionPipeline`` y los handlers
    gRPC necesitan: ``load``, ``engine_name``, ``len``, ``match_embedding``,
    ``add_embedding``, ``remove_person``, ``persons`` y ``save``.

    Es **la** abstracción que sustituye a la query SQL/FAISS "cruda" y
    encapsula la normalización L2 + la métrica, de modo que el servidor
    nunca ve vectores sin normalizar.
    """

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings

        # Motor: "auto" prueba FAISS; si no, NumPy. Mismo ajuste que en
        # ``SimilarityBackend``.
        engine_key = settings.index_engine
        backend: SimilarityBackend
        if engine_key == "auto":
            try:
                backend = FaissBackend(
                    dim=settings.index_dim, metric=settings.index_metric
                )
            except Exception:
                log.warning("FAISS no disponible → backend NumPy")
                backend = NumpyBackend(
                    dim=settings.index_dim, metric=settings.index_metric
                )
        elif engine_key == "faiss":
            try:
                backend = FaissBackend(
                    dim=settings.index_dim, metric=settings.index_metric
                )
            except Exception as exc:
                raise InvalidVectorError(
                    f"Se pidió FAISS pero no se pudo inicializar: {exc}"
                ) from exc
        else:
            backend = NumpyBackend(
                dim=settings.index_dim, metric=settings.index_metric
            )

        self._backend = backend
        self._meta = PersonMetadata(
            path=settings.index_meta, engine_prefix=backend.name
        )
        self._loaded = False

    # ─── estado / arranque ──────────────────────────────
    @property
    def engine_name(self) -> str:
        return self._backend.name

    def __len__(self) -> int:
        """Nº de rostros indexados (no de personas únicas)."""
        return len(self._backend)

    def load(self) -> None:
        """Carga metadatos (NDJSON) + reconstruye el índice si hace falta."""
        self._meta.load()
        self._loaded = True

    # ─── consulta ───────────────────────────────────────
    def match_embedding(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 1,
    ) -> PersonMatch:
        """Busca el vecino más cercano y traduce a ``PersonMatch``.

        Si el índice está vacío (o el mejor candidato queda bajo el umbral)
        devuelve ``PersonMatch`` con ``known=False`` y metadatos vacíos —
        el pipeline lo convierte en ``FaceMatch`` *desconocido*.
        """
        thr = float(threshold) if threshold is not None else float(
            self._settings.match_threshold
        )
        if self._backend.empty:
            return PersonMatch(
                person_id="", label="",
                score=0.0, embedding=None, known=False,
            )
        hits = self._backend.query(embedding, top_k=top_k)
        if not hits:
            return PersonMatch(
                person_id="", label="",
                score=0.0, embedding=None, known=False,
            )
        pos, score = hits[0]
        rec = self._meta.person_at(pos)
        pid = rec.get("person_id", "")
        label = rec.get("label") or pid
        return PersonMatch(
            person_id=pid,
            label=label,
            score=float(score),
            embedding=self._backend.vector_at(pos),
            known=float(score) >= thr,
        )

    # ─── enrolamiento ───────────────────────────────────
    def add_embedding(
        self, embedding: np.ndarray, person_id: str, label: str = ""
    ) -> int:
        """Indexa un rostro y devuelve su posición (para rollbacks)."""
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size != self._settings.index_dim:
            raise InvalidVectorError(
                f"Se esperaban {self._settings.index_dim} dims y llegaron "
                f"{vec.size}"
            )
        pos = len(self._backend)
        self._backend.add([vec])
        self._meta.set_person_record(
            pos, person_id=person_id, label=label or person_id
        )
        self._meta.save()
        return pos

    def remove_person(self, person_id: str) -> int:
        """Elimina todos los rostros de una persona. Devuelve cuántos."""
        removed = 0
        positions = [
            i
            for i in range(len(self._backend))
            if self._meta.person_at(i).get("person_id") == person_id
        ]
        if not positions:
            return 0
        removed = self._backend.remove(positions)
        # Re-numera metadatos restantes.
        remaining = [
            self._meta.person_at(i)
            for i in range(len(self._meta))
            if i not in set(positions)
        ]
        self._meta = PersonMetadata(
            path=self._settings.index_meta, engine_prefix=self._backend.name
        )
        for k, rec in enumerate(remaining):
            self._meta.set_person_record(
                k, person_id=rec.get("person_id", ""), label=rec.get("label", "")
            )
        self._meta.save()
        return removed

    def persons(self) -> Dict[str, Dict[str, Any]]:
        """Consolida ``person_id``→{label, face_count}."""
        return self._meta.persons()

    def save(self) -> None:
        """Persiste metadatos (el backend FAISS/NumPy persiste por su lado)."""
        self._meta.save()

    # ─── iteración de debug ─────────────────────────────
    def person_at(self, pos: int) -> Dict[str, str]:
        return self._meta.person_at(pos)


# ════════════════════════════════════════════════════════════
# Resultado de una búsqueda (lo que ``pipeline`` consume)
# ════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PersonMatch:
    """Match resuelto contra el índice.

    Es la pieza que ``pipeline.py`` recibe de ``PersonRepository.match_embedding``
    y que luego **traduce** a ``FaceMatch`` (el tipo del dominio, en
    ``common.py``). Campos en inglés (lenguaje de protocolo), comentarios
    en español (convención Sandra).
    """

    person_id: str
    label: str
    # Similitud coseno normalizada (0–1; 1 = idéntico).
    score: float
    # Embedding 512-d del candidato (puede ser ``None`` si se pidió
    # búsqueda ligera sin traer el vector).
    embedding: Optional[np.ndarray] = None
    # False si la sospecha queda por debajo del umbral (rostro nuevo).
    known: bool = True


# ════════════════════════════════════════════════════════════
# PersonRepository — fachada de índice para el pipeline
# ════════════════════════════════════════════════════════════
class PersonRepository:
    """Fachada que une el motor de similitud con los metadatos.

    Expone el subconjunto de operaciones que ``pipeline`` y el servidor
    gRPC necesitan, ocultando si por debajo hay una matriz NumPy en memoria
    o un ``IndexFlatIP`` de FAISS:

        * ``len()``      → nº de rostros indexados (no de personas).
        * ``load()``     → reconstruye índice + metadatos desde disco.
        * ``engine_name``→ "faiss" | "numpy" (para health/health).
        * ``match_embedding(vec, threshold)`` → ``PersonMatch``.
        * ``add_embedding(vec, person_id, label)`` → posición.
        * ``remove_person(person_id)`` → rostros eliminados.
        * ``persons()``  → consolidación persona → {label, face_count}.

    Es una clase *concreta y co-contratada* con el paquete: ``pipeline.py``
    la importa por su nombre exacto (``PersonRepository``). Nunca lanza
    errores de dominio esperados por el servidor: la búsqueda sobre un
    índice vacío devuelve ``PersonMatch`` desconocido.
    """

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings

        # Motor de similitud (ya resuelto "auto"→probado en el constructor).
        engine = settings.index_engine.lower()
        dim = settings.index_dim
        metric = settings.index_metric
        if engine == "auto":
            if FaissBackend._import_faiss() is not None:
                engine = "faiss"
            else:
                engine = "numpy"
        elif engine not in {"faiss", "numpy"}:
            log.warning(
                "INDEX_ENGINE='%s' desconocido → usamos numpy", engine
            )
            engine = "numpy"

        self._engine_name = engine
        backend_cls = FaissBackend if engine == "faiss" else NumpyBackend
        self._backend = backend_cls(dim, metric=metric, path=settings.index_file)
        self._meta = PersonMetadata(settings.index_meta)

    # ─── estado / arranque ──────────────────────────────
    @property
    def engine_name(self) -> str:
        """Identificador del motor real (para health y timing)."""
        return self._engine_name

    def load(self) -> None:
        """Reconstruye el índice desde disco (índice + metadatos)."""
        self._meta.load()
        # Con NumPy el índice vive en el ``PersonMetadata``/archivo NDJSON:
        # cargamos embeddings de metadatos si el backend viene vacío.
        # (FAISS se reconstruye desde la persistencia en cada arranque; aquí
        #  delegamos en el backend para que mantenga su propia caché.)
        self._backend.load(self._meta)

    # ─── consulta ───────────────────────────────────────
    def __len__(self) -> int:
        """Nº de rostros indexados."""
        return len(self._backend)

    def match_embedding(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 1,
    ) -> PersonMatch:
        """Busca el vecino más cercano y traduce a ``PersonMatch``.

        Si el índice está vacío (o el mejor candidato queda bajo el umbral)
        devuelve ``PersonMatch`` con ``known=False`` y metadatos vacíos —
        el pipeline lo convierte en un ``FaceMatch`` *nuevo/desconocido*.
        Nunca lanza: es el comportamiento esperado por el servidor incluso
        con un repositorio sin registrar.
        """
        thr = threshold if threshold is not None else self._settings.match_threshold

        if self._backend.empty:
            return PersonMatch(
                person_id="", label="",
                score=0.0, embedding=None, known=False,
            )
        hits = self._backend.query(embedding, top_k=top_k)
        if not hits:
            return PersonMatch(
                person_id="", label="",
                score=0.0, embedding=None, known=False,
            )
        idx, score = hits[0]
        rec = self._meta.person_at(idx)   # {person_id, label} o fallback
        pid = rec.get("person_id", "") if rec else ""
        label = rec.get("label", "") if rec else (pid or "")
        return PersonMatch(
            person_id=pid,
            label=label or pid,
            score=float(score),
            embedding=self._embedding_at(idx),
            known=float(score) >= float(thr),
        )

    def _embedding_at(self, pos: int) -> Optional[np.ndarray]:
        """Devuelve el vector de la posición (o ``None`` si no existe)."""
        vec = self._backend.vector_at(pos)
        return vec if vec is not None else None

    # ─── enrolamiento / escritura ───────────────────────
    def add_embedding(
        self, embedding: np.ndarray, person_id: str, label: str = ""
    ) -> int:
        """Indexa un rostro para una persona y devuelve su posición."""
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size != self._settings.index_dim:
            raise InvalidVectorError(
                f"Se esperaban {self._settings.index_dim} dims y llegaron {vec.size}"
            )
        pos = len(self._backend)
        self._backend.add([vec])
        self._meta.set_person_record(pos, person_id=person_id, label=label)
        self.save()
        return pos

    def remove_person(self, person_id: str) -> int:
        """Elimina todos los rostros de una persona. Devuelve cuántos."""
        positions = [
            i for i in range(len(self._meta))
            if self._meta.person_at(i).get("person_id") == person_id
        ]
        if not positions:
            return 0
        removed = self._backend.remove(positions)
        # Re-numera metadatos restantes.
        keep = [
            self._meta.person_at(i)
            for i in range(len(self._meta))
            if i not in set(positions)
        ]
        self._meta = PersonMetadata(self._settings.index_meta)
        for k, rec in enumerate(keep):
            self._meta.set_person_record(
                k, person_id=rec.get("person_id", ""), label=rec.get("label", "")
            )
        self.save()
        return len(positions)

    def persons(self) -> Dict[str, Dict[str, Any]]:
        """Consolida por ``person_id``: label + nº de rostros."""
        return self._meta.persons()

    def save(self) -> None:
        """Persiste metadatos (NDJSON) y vectores del backend (.npy)."""
        self._backend.save()
        self._meta.save()


# Quería que ``repository`` expusiera un helper de conveniencia en español
# para los clientes: sin él, los scripts ``register/identify`` tendrían que
# adivinar el umbral por defecto.
def default_match_threshold(settings: config.Settings) -> float:
    """Devuelve el umbral de match configurado (API de conveniencia)."""
    return settings.match_threshold
