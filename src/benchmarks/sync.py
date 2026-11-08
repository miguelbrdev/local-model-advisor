"""One benchmark synchronisation cycle: download → map → validate → atomic write.

Policy:

* A failing **mandatory** source (Epoch AI) aborts the run; the previous
  snapshot stays in place.
* A failing **optional** source (Artificial Analysis) is recorded in the
  result and the run continues with the remaining sources, so an operator
  without a key, or with the daily quota exhausted, still gets the base
  data.
* Scores reach families only through the mapping table; a pinned name
  missing from the downloaded data is listed in
  ``meta.sin_coincidencia`` for human review, never guessed.
* An identical snapshot (same content hash) is not rewritten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.benchmarks.fuentes import (
    DatosFuente,
    FuenteBenchmarks,
    FuenteBenchmarksError,
    FuenteMeta,
)
from src.benchmarks.mapping import MapeoBenchmarks
from src.benchmarks.snapshot import (
    BenchmarksSnapshot,
    compute_content_hash,
    construir_snapshot,
    read_snapshot,
    write_snapshot,
)
from src.models.schemas import EvidenciaBenchmark

logger = logging.getLogger(__name__)


@dataclass
class ResultadoSyncBenchmarks:
    """Outcome of :func:`sync_benchmarks`.

    Attributes:
        ok: A valid snapshot is in place (new or unchanged).
        snapshot_id: Id of that snapshot, when known.
        error: Failure reason when ``ok`` is false.
        fuentes_ok: Sources that contributed data.
        fuentes_fallidas: Optional sources that failed, with the reason.
        familias_con_datos: Families that received at least one score.
        sin_coincidencia: Pinned names not found in the data.
        unchanged: The identical snapshot already existed.
    """

    ok: bool
    snapshot_id: str | None = None
    error: str | None = None
    fuentes_ok: list[str] = field(default_factory=list)
    fuentes_fallidas: dict[str, str] = field(default_factory=dict)
    familias_con_datos: int = 0
    sin_coincidencia: list[str] = field(default_factory=list)
    unchanged: bool = False


def sync_benchmarks(
    destino: Path,
    fuentes: list[FuenteBenchmarks],
    mapeo: MapeoBenchmarks,
    *,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
    ahora: datetime | None = None,
) -> ResultadoSyncBenchmarks:
    """Run one cycle and write ``destino`` atomically on success.

    Args:
        destino: Snapshot file to (re)write, normally
            ``runtime/benchmarks/current.json``.
        fuentes: Sources to download, in attribution order.
        mapeo: Reviewed mapping table.
        timeout: HTTP timeout per request.
        transport: Injected transport for tests.
        ahora: Snapshot timestamp (tests pin it); defaults to now (UTC).

    Returns:
        :class:`ResultadoSyncBenchmarks`; on failure ``destino`` is untouched.
    """
    descargas: list[DatosFuente] = []
    resultado = ResultadoSyncBenchmarks(ok=False)
    for fuente in fuentes:
        try:
            descargas.append(fuente.descargar(timeout=timeout, transport=transport))
            resultado.fuentes_ok.append(fuente.nombre)
        except FuenteBenchmarksError as exc:
            if fuente.obligatoria:
                logger.error("Mandatory source %s failed: %s", fuente.nombre, exc)
                resultado.error = f"fuente_obligatoria_fallida: {exc}"
                return resultado
            logger.warning("Optional source %s skipped: %s", fuente.nombre, exc)
            resultado.fuentes_fallidas[fuente.nombre] = str(exc)

    if not descargas:
        resultado.error = "sin_fuentes"
        return resultado

    puntuaciones, sin_coincidencia = aplicar_mapeo(mapeo, descargas)
    resultado.sin_coincidencia = sin_coincidencia
    resultado.familias_con_datos = sum(1 for e in puntuaciones.values() if e)
    if resultado.familias_con_datos == 0:
        resultado.error = "sin_datos: no mapped family received a score"
        return resultado

    metas: list[FuenteMeta] = [d.meta for d in descargas]
    anterior = read_snapshot(destino)
    hash_nuevo = compute_content_hash(puntuaciones, metas)
    if anterior is not None and anterior.meta.hash_contenido == hash_nuevo:
        logger.info("Benchmark snapshot unchanged (%s).", anterior.meta.snapshot_id)
        resultado.ok = True
        resultado.unchanged = True
        resultado.snapshot_id = anterior.meta.snapshot_id
        return resultado

    try:
        snapshot = construir_snapshot(
            puntuaciones,
            metas,
            fecha=ahora or datetime.now(UTC),
            mapeo_version=mapeo.version,
            mapeo_snapshot_id=mapeo.snapshot_id,
            familias_mapeadas=len(mapeo.familias),
            sin_coincidencia=sin_coincidencia,
        )
        write_snapshot(snapshot, destino)
    except (OSError, ValidationError) as exc:
        logger.error("Benchmark snapshot write failed: %s", exc)
        resultado.error = f"escritura_fallida: {exc}"
        return resultado

    resultado.ok = True
    resultado.snapshot_id = snapshot.meta.snapshot_id
    return resultado


def aplicar_mapeo(
    mapeo: MapeoBenchmarks, descargas: list[DatosFuente]
) -> tuple[dict[str, list[EvidenciaBenchmark]], list[str]]:
    """Attach downloaded scores to families by exact pinned name.

    For Epoch AI the per-benchmark name only pulls benchmark rows and the
    index name only pulls the ``eci`` row, decided by *which field* the
    name came from (the two namespaces often share a spelling, e.g.
    ``Qwen3-32B``), so a reviewer pinning one never silently receives the
    other.

    Returns:
        ``(scores per canirun_id, names not found)``.  Families whose
        pinned names found nothing get an empty list.
    """
    puntuaciones: dict[str, list[EvidenciaBenchmark]] = {}
    sin_coincidencia: list[str] = []
    for canirun_id, entrada in mapeo.familias.items():
        encontradas: list[EvidenciaBenchmark] = []
        for datos in descargas:
            for nombre, solo_indice in entrada.consultas(datos.meta.nombre):
                filas = datos.puntuaciones.get(nombre, [])
                if solo_indice is not None:
                    filas = [f for f in filas if (f.benchmark == "eci") == solo_indice]
                if filas:
                    encontradas.extend(filas)
                else:
                    sin_coincidencia.append(
                        f"{canirun_id}: {datos.meta.nombre} {nombre!r} not found"
                    )
        puntuaciones[canirun_id] = encontradas
    return puntuaciones, sin_coincidencia


__all__ = ["BenchmarksSnapshot", "ResultadoSyncBenchmarks", "aplicar_mapeo", "sync_benchmarks"]
