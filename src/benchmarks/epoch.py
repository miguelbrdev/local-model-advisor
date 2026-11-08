"""Epoch AI Benchmarking Hub adapter (mandatory base source).

Contract observed on 2026-11-08 at ``https://epoch.ai/data/benchmark_data.zip``
(CC BY 4.0, no account):

* One CSV per benchmark with the columns ``Model version``,
  ``mean_score``, ``Best score (across scorers)``, ``Release date``,
  ``Organization``, ``stderr``, ``id``, …  Model names carry a reasoning
  effort suffix (``_none`` = no reasoning, ``_low`` … ``_max``); the
  reviewed mapping pins which exact name represents a family.
* ``benchmark_metadata.csv`` with ``source_file``, ``random_baseline`` and
  ``score_ceiling`` per benchmark, used to anchor accuracies on 0–1.
* ``epoch_capabilities_index/eci_scores.csv`` with ``Model``, ``eci``,
  ``date`` and ``Accessibility group``: the Epoch Capabilities Index, a
  general capability score used for ``chat`` queries.

Only the benchmarks in :data:`BENCHMARKS_EPOCH` plus the index are read;
the rest of the ZIP is ignored.  When a model appears twice in one CSV
(the SWE-bench file does this) the highest best score is kept, so the
value never depends on row order.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import UTC, datetime

import httpx

from src.benchmarks.fuentes import (
    DatosFuente,
    FuenteBenchmarksError,
    FuenteMeta,
    normalizar,
)
from src.models.schemas import EvidenciaBenchmark
from src.tools.tls import contexto_tls

logger = logging.getLogger(__name__)

EPOCH_ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"

BENCHMARKS_EPOCH: dict[str, str] = {
    "gpqa_diamond": "gpqa_diamond.csv",
    "math_level_5": "math_level_5.csv",
    "swe_bench_verified": "swe_bench_verified.csv",
}
"""Internal benchmark key → CSV inside the ZIP."""

ECI_CSV = "epoch_capabilities_index/eci_scores.csv"
METADATA_CSV = "benchmark_metadata.csv"

COLUMNA_MODELO = "Model version"
COLUMNA_PUNTUACION = "Best score (across scorers)"
COLUMNA_FECHA = "Release date"

ECI_ANCLA_MIN = 100.0
ECI_ANCLA_MAX = 160.0
"""Fixed anchors mapping the Epoch Capabilities Index onto 0–1.  Open-weight
models ranged 50–159 in November 2026; 100 is roughly a 2023 7B model and
160 the frontier, so local models spread over the scale instead of
crowding near 1.0.  Fixed (not min–max of the snapshot) so a new model in
the index never moves the others."""

LICENCIA = "CC BY 4.0"
ATRIBUCION = (
    "Benchmark scores and Capabilities Index: Epoch AI, \"Capabilities & benchmarking\", "
    "https://epoch.ai/benchmarks (CC BY 4.0)."
)


class FuenteEpoch:
    """Download the Epoch ZIP and return every score keyed by source model name."""

    nombre = "epoch_ai"
    obligatoria = True

    def __init__(self, url: str = EPOCH_ZIP_URL) -> None:
        self._url = url

    def descargar(
        self, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
    ) -> DatosFuente:
        try:
            with httpx.Client(
                transport=transport, timeout=timeout, verify=contexto_tls()
            ) as client:
                response = client.get(self._url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FuenteBenchmarksError(f"epoch_ai: download failed: {exc}") from exc
        return self.parsear(response.content, descargado_en=datetime.now(UTC), url=self._url)

    @classmethod
    def parsear(
        cls, contenido_zip: bytes, *, descargado_en: datetime, url: str = EPOCH_ZIP_URL
    ) -> DatosFuente:
        """Parse the ZIP bytes (pure; tests build small archives in memory).

        Raises:
            FuenteBenchmarksError: If the archive is invalid, a required
                CSV is missing or no score could be read at all.
        """
        try:
            archivo = zipfile.ZipFile(io.BytesIO(contenido_zip))
        except zipfile.BadZipFile as exc:
            raise FuenteBenchmarksError(f"epoch_ai: not a ZIP archive: {exc}") from exc

        with archivo:
            nombres = set(archivo.namelist())
            faltan = [
                n for n in (*BENCHMARKS_EPOCH.values(), ECI_CSV, METADATA_CSV) if n not in nombres
            ]
            if faltan:
                raise FuenteBenchmarksError(f"epoch_ai: files missing from the archive: {faltan}")

            limites = _leer_limites(_filas(archivo, METADATA_CSV))
            puntuaciones: dict[str, list[EvidenciaBenchmark]] = {}
            for benchmark, csv_name in BENCHMARKS_EPOCH.items():
                minimo, maximo = limites.get(csv_name, (0.0, 1.0))
                for nombre, valor, fecha in _mejores_por_modelo(_filas(archivo, csv_name)):
                    puntuaciones.setdefault(nombre, []).append(
                        EvidenciaBenchmark(
                            fuente="epoch_ai",
                            benchmark=benchmark,
                            metrica=COLUMNA_PUNTUACION,
                            valor=round(valor, 4),
                            valor_normalizado=normalizar(valor, minimo, maximo),
                            modelo_fuente=nombre,
                            fecha_modelo=fecha or None,
                            nota=f"Anchored on random baseline {minimo:g} and ceiling {maximo:g}.",
                        )
                    )
            for fila in _filas(archivo, ECI_CSV):
                nombre = (fila.get("Model") or "").strip()
                valor = _flotante(fila.get("eci"))
                if not nombre or valor is None:
                    continue
                puntuaciones.setdefault(nombre, []).append(
                    EvidenciaBenchmark(
                        fuente="epoch_ai",
                        benchmark="eci",
                        metrica="eci",
                        valor=round(valor, 2),
                        valor_normalizado=normalizar(valor, ECI_ANCLA_MIN, ECI_ANCLA_MAX),
                        modelo_fuente=nombre,
                        fecha_modelo=(fila.get("date") or "").strip() or None,
                        nota=(
                            f"Epoch Capabilities Index, anchored on {ECI_ANCLA_MIN:g}-"
                            f"{ECI_ANCLA_MAX:g}."
                        ),
                    )
                )
            version = _version_del_archivo(archivo)

        if not puntuaciones:
            raise FuenteBenchmarksError("epoch_ai: no score could be read from the archive")

        meta = FuenteMeta(
            nombre="epoch_ai",
            version=version,
            url=url,
            licencia=LICENCIA,
            atribucion=ATRIBUCION,
            redistribuible=True,
            registros=len(puntuaciones),
            descargado_en=descargado_en,
        )
        logger.info("epoch_ai: %d source models, data version %s", len(puntuaciones), version)
        return DatosFuente(meta=meta, puntuaciones=puntuaciones)


def _filas(archivo: zipfile.ZipFile, nombre: str) -> list[dict[str, str]]:
    try:
        with archivo.open(nombre) as fh:
            return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise FuenteBenchmarksError(f"epoch_ai: cannot read {nombre}: {exc}") from exc


def _leer_limites(filas: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """``source_file`` → (random baseline, ceiling) from ``benchmark_metadata.csv``."""
    limites: dict[str, tuple[float, float]] = {}
    for fila in filas:
        origen = (fila.get("source_file") or "").strip()
        minimo = _flotante(fila.get("random_baseline"))
        maximo = _flotante(fila.get("score_ceiling"))
        if origen and minimo is not None and maximo is not None and maximo > minimo:
            limites[origen] = (minimo, maximo)
    return limites


def _mejores_por_modelo(filas: list[dict[str, str]]) -> list[tuple[str, float, str]]:
    """(model, best score, release date) per model, keeping the highest score."""
    mejores: dict[str, tuple[float, str]] = {}
    for fila in filas:
        nombre = (fila.get(COLUMNA_MODELO) or "").strip()
        valor = _flotante(fila.get(COLUMNA_PUNTUACION))
        if not nombre or valor is None:
            continue
        fecha = (fila.get(COLUMNA_FECHA) or "").strip()
        if nombre not in mejores or valor > mejores[nombre][0]:
            mejores[nombre] = (valor, fecha)
    return [(nombre, valor, fecha) for nombre, (valor, fecha) in mejores.items()]


def _flotante(texto: str | None) -> float | None:
    if texto is None or not texto.strip():
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def _version_del_archivo(archivo: zipfile.ZipFile) -> str:
    """Newest member timestamp as ``YYYY-MM-DD`` (the ZIP carries no other version)."""
    fechas = [info.date_time[:3] for info in archivo.infolist()]
    if not fechas:
        return "unknown"
    anio, mes, dia = max(fechas)
    return f"{anio:04d}-{mes:02d}-{dia:02d}"
