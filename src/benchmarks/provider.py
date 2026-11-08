"""Read side of the benchmark data: what the pipeline is injected with.

Mirrors the catalogue design — one Protocol, one production adapter with
a fallback chain, in-memory doubles for tests — with one deliberate
difference: an exhausted chain does **not** abort the boot.  Benchmarks
are a quality signal, not the catalogue; without them every candidate
scores the neutral value and the response says so in ``limitaciones``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.benchmarks.snapshot import (
    DEFAULT_BENCHMARKS_BOOTSTRAP_PATH,
    DEFAULT_RUNTIME_BENCHMARKS_PATH,
    BenchmarksSnapshot,
    read_snapshot,
)
from src.models.schemas import EvidenciaBenchmark

logger = logging.getLogger(__name__)


@runtime_checkable
class ProveedorBenchmarks(Protocol):
    """Measured scores per ``canirun_id`` for the ranking.

    Attributes:
        snapshot_id: Id of the benchmark snapshot served, or ``None`` when
            no data is loaded.  Persisted alongside the catalogue id so a
            recommendation can be reproduced.
        fecha_sincronizacion: When that snapshot was built, if known.
        atribuciones: Attribution lines of every source behind the data;
            shown verbatim in each recommendation (a licence condition).
    """

    @property
    def snapshot_id(self) -> str | None: ...

    @property
    def fecha_sincronizacion(self) -> datetime | None: ...

    @property
    def atribuciones(self) -> list[str]: ...

    def puntuaciones_de(self, canirun_id: str) -> list[EvidenciaBenchmark]:
        """Scores for one family (empty when unmapped or unknown).  Never raises."""
        ...


class BenchmarksLocales:
    """Snapshot from ``runtime/benchmarks/current.json`` with bootstrap fallback.

    Resolution order:

    1. The runtime file written by ``scripts/sync_benchmarks.py`` (may
       include non-redistributable sources).
    2. The committed ``data/benchmarks_bootstrap.json`` (Epoch AI only).
    3. Nothing: every lookup returns ``[]`` and ``snapshot_id`` is ``None``.

    Construction performs no I/O; the first access resolves and memoizes
    the snapshot for the lifetime of the instance (one snapshot per
    process, like the catalogue).
    """

    def __init__(
        self, runtime_path: Path | None = None, bootstrap_path: Path | None = None
    ) -> None:
        self._runtime_path = runtime_path or DEFAULT_RUNTIME_BENCHMARKS_PATH
        self._bootstrap_path = bootstrap_path or DEFAULT_BENCHMARKS_BOOTSTRAP_PATH
        self._resuelto = False
        self._snapshot: BenchmarksSnapshot | None = None

    def _resolver(self) -> BenchmarksSnapshot | None:
        if not self._resuelto:
            self._resuelto = True
            for ruta in (self._runtime_path, self._bootstrap_path):
                snapshot = read_snapshot(ruta)
                if snapshot is not None:
                    logger.info(
                        "Benchmark snapshot %s loaded from %s (%d families with data).",
                        snapshot.meta.snapshot_id, ruta, snapshot.meta.familias_con_datos,
                    )
                    self._snapshot = snapshot
                    break
            else:
                logger.warning(
                    "No benchmark snapshot found (%s, %s); capacidad_medida will be neutral.",
                    self._runtime_path, self._bootstrap_path,
                )
        return self._snapshot

    @property
    def disponible(self) -> bool:
        return self._resolver() is not None

    @property
    def snapshot_id(self) -> str | None:
        snapshot = self._resolver()
        return snapshot.meta.snapshot_id if snapshot else None

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        snapshot = self._resolver()
        return snapshot.meta.fecha_sincronizacion if snapshot else None

    @property
    def atribuciones(self) -> list[str]:
        snapshot = self._resolver()
        return snapshot.atribuciones if snapshot else []

    @property
    def fuentes(self) -> list[str]:
        snapshot = self._resolver()
        return [f.nombre for f in snapshot.meta.fuentes] if snapshot else []

    def puntuaciones_de(self, canirun_id: str) -> list[EvidenciaBenchmark]:
        snapshot = self._resolver()
        if snapshot is None:
            return []
        return list(snapshot.puntuaciones.get(canirun_id, []))


class BenchmarksEnMemoria:
    """Test double holding scores in memory."""

    def __init__(
        self,
        puntuaciones: dict[str, list[EvidenciaBenchmark]] | None = None,
        *,
        snapshot_id: str | None = "bench-test",
        fecha: datetime | None = None,
        atribuciones: list[str] | None = None,
    ) -> None:
        self._puntuaciones = dict(puntuaciones or {})
        self._snapshot_id = snapshot_id
        self._fecha = fecha
        self._atribuciones = list(atribuciones or [])

    @property
    def snapshot_id(self) -> str | None:
        return self._snapshot_id

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        return self._fecha

    @property
    def atribuciones(self) -> list[str]:
        return list(self._atribuciones)

    def puntuaciones_de(self, canirun_id: str) -> list[EvidenciaBenchmark]:
        return list(self._puntuaciones.get(canirun_id, []))
