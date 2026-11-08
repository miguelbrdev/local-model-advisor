"""Benchmark snapshot: validated, content-addressed, atomically written.

A snapshot is one JSON file holding the scores of every mapped family
plus the provenance of each source.  Unlike the catalogue there is no
manifest and no history: ``runtime/benchmarks/current.json`` is replaced
atomically only after the new content is fully validated, so a failed
synchronisation leaves the previous file untouched.

Invariants (enforced by the model validators):

* ``hash_contenido`` is the SHA-256 of the scores plus the source
  versions, and ``snapshot_id == "bench-" + hash[:16]``.
* ``familias_con_datos`` equals the number of families with at least one
  score; ``fuentes`` names every ``fuente`` that appears in the scores.
* ``fecha_sincronizacion`` is timezone-aware.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.benchmarks.fuentes import FuenteMeta
from src.models.schemas import EvidenciaBenchmark

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

_RAIZ = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNTIME_BENCHMARKS_PATH = _RAIZ / "runtime" / "benchmarks" / "current.json"
DEFAULT_BENCHMARKS_BOOTSTRAP_PATH = _RAIZ / "data" / "benchmarks_bootstrap.json"


def compute_content_hash(
    puntuaciones: dict[str, list[EvidenciaBenchmark]], fuentes: list[FuenteMeta]
) -> str:
    """Deterministic SHA-256 over the scores and the source versions.

    Download timestamps, record counts and file paths are excluded so the
    same data always yields the same ``snapshot_id``.
    """
    payload = {
        "fuentes": sorted((f.nombre, f.version) for f in fuentes),
        "puntuaciones": {
            familia: sorted(
                (e.model_dump(mode="json") for e in evidencias),
                key=lambda d: (d["fuente"], d["benchmark"], d["modelo_fuente"]),
            )
            for familia, evidencias in sorted(puntuaciones.items())
        },
    }
    canonico = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def snapshot_id_para(hash_contenido: str) -> str:
    return f"bench-{hash_contenido[:16]}"


class BenchmarksMeta(BaseModel):
    """Reproducibility header of a benchmark snapshot.

    Attributes:
        schema_version: Layout version of the file.
        snapshot_id: ``"bench-" + hash_contenido[:16]``.
        fecha_sincronizacion: UTC instant the snapshot was built.
        hash_contenido: See :func:`compute_content_hash`.
        fuentes: Provenance of every source that contributed scores.
        mapeo_version: Version of the mapping table used.
        mapeo_snapshot_id: Catalogue snapshot the mapping was reviewed for.
        familias_mapeadas: Families present in the mapping table.
        familias_con_datos: Families that received at least one score.
        sin_coincidencia: Pinned names not found in the downloaded data,
            as ``"<canirun_id>: <fuente> '<nombre>'"``; reviewed by a
            human after each synchronisation.
    """

    schema_version: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    fecha_sincronizacion: datetime
    hash_contenido: str = Field(pattern=r"^[0-9a-f]{64}$")
    fuentes: list[FuenteMeta]
    mapeo_version: str = Field(min_length=1)
    mapeo_snapshot_id: str = Field(min_length=1)
    familias_mapeadas: int = Field(ge=0)
    familias_con_datos: int = Field(ge=0)
    sin_coincidencia: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invariantes(self) -> BenchmarksMeta:
        if self.fecha_sincronizacion.tzinfo is None:
            raise ValueError("fecha_sincronizacion must be timezone-aware (UTC)")
        nombres = [f.nombre for f in self.fuentes]
        if len(nombres) != len(set(nombres)):
            raise ValueError("duplicate source in fuentes")
        if self.snapshot_id != snapshot_id_para(self.hash_contenido):
            raise ValueError("snapshot_id does not match content hash prefix")
        return self


class BenchmarksSnapshot(BaseModel):
    """Scores per ``canirun_id`` plus the header; see module docstring."""

    meta: BenchmarksMeta
    puntuaciones: dict[str, list[EvidenciaBenchmark]]

    @model_validator(mode="after")
    def _contenido(self) -> BenchmarksSnapshot:
        con_datos = sum(1 for e in self.puntuaciones.values() if e)
        if self.meta.familias_con_datos != con_datos:
            raise ValueError(
                f"familias_con_datos ({self.meta.familias_con_datos}) does not match "
                f"families with scores ({con_datos})"
            )
        declaradas = {f.nombre for f in self.meta.fuentes}
        usadas = {e.fuente for evidencias in self.puntuaciones.values() for e in evidencias}
        if not usadas <= declaradas:
            raise ValueError(f"scores from undeclared sources: {sorted(usadas - declaradas)}")
        if self.meta.hash_contenido != compute_content_hash(self.puntuaciones, self.meta.fuentes):
            raise ValueError("hash_contenido does not match snapshot content")
        return self

    @property
    def atribuciones(self) -> list[str]:
        return [f.atribucion for f in self.meta.fuentes]

    def solo_redistribuible(self, fecha: datetime | None = None) -> BenchmarksSnapshot:
        """Copy keeping only sources whose data may be committed.

        Families left without scores are dropped; the header is rebuilt
        (new hash and id).  ``fecha`` defaults to the original timestamp
        so the bootstrap says when its data was synchronised.
        """
        fuentes = [f for f in self.meta.fuentes if f.redistribuible]
        permitidas = {f.nombre for f in fuentes}
        puntuaciones = {
            familia: [e for e in evidencias if e.fuente in permitidas]
            for familia, evidencias in self.puntuaciones.items()
        }
        puntuaciones = {k: v for k, v in puntuaciones.items() if v}
        return construir_snapshot(
            puntuaciones,
            fuentes,
            fecha=fecha or self.meta.fecha_sincronizacion,
            mapeo_version=self.meta.mapeo_version,
            mapeo_snapshot_id=self.meta.mapeo_snapshot_id,
            familias_mapeadas=self.meta.familias_mapeadas,
            sin_coincidencia=[
                s for s in self.meta.sin_coincidencia
                if any(f": {nombre} " in s for nombre in permitidas)
            ],
        )


def construir_snapshot(
    puntuaciones: dict[str, list[EvidenciaBenchmark]],
    fuentes: list[FuenteMeta],
    *,
    fecha: datetime,
    mapeo_version: str,
    mapeo_snapshot_id: str,
    familias_mapeadas: int,
    sin_coincidencia: list[str] | None = None,
) -> BenchmarksSnapshot:
    """Assemble a validated snapshot, computing hash and id."""
    hash_contenido = compute_content_hash(puntuaciones, fuentes)
    meta = BenchmarksMeta(
        schema_version=SCHEMA_VERSION,
        snapshot_id=snapshot_id_para(hash_contenido),
        fecha_sincronizacion=fecha,
        hash_contenido=hash_contenido,
        fuentes=fuentes,
        mapeo_version=mapeo_version,
        mapeo_snapshot_id=mapeo_snapshot_id,
        familias_mapeadas=familias_mapeadas,
        familias_con_datos=sum(1 for e in puntuaciones.values() if e),
        sin_coincidencia=list(sin_coincidencia or []),
    )
    return BenchmarksSnapshot(meta=meta, puntuaciones=puntuaciones)


def write_snapshot(snapshot: BenchmarksSnapshot, path: Path) -> Path:
    """Atomically write ``snapshot`` to ``path`` (temp file, round-trip check, ``os.replace``).

    Raises:
        OSError: On filesystem failure; no partial file is left behind.
        ValidationError: If the serialised file does not re-validate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_name(path.name + ".tmp")
    try:
        temporal.write_text(
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        BenchmarksSnapshot.model_validate_json(temporal.read_text(encoding="utf-8"))
        os.replace(temporal, path)
    finally:
        if temporal.exists():
            temporal.unlink(missing_ok=True)
    return path


def read_snapshot(path: Path) -> BenchmarksSnapshot | None:
    """Load and fully validate the snapshot at ``path``; ``None`` if missing or corrupt."""
    if not path.is_file():
        return None
    try:
        return BenchmarksSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        logger.warning("Benchmark snapshot %s rejected: %s", path, exc)
        return None
