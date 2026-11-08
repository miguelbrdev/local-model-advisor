"""Contract every benchmark source adapter implements.

A *source* downloads its public data once and returns it keyed by the
exact model name the source uses, already converted to
:class:`~src.models.schemas.EvidenciaBenchmark` with a normalised 0–1
value.  Matching those names to catalogue families is **not** the
source's job: that happens in :mod:`src.benchmarks.sync` through the
reviewed mapping table, so adapters stay free of any heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field, model_validator

from src.models.schemas import EvidenciaBenchmark

NombreFuente = Literal["epoch_ai", "artificial_analysis"]


class FuenteBenchmarksError(Exception):
    """A source could not be downloaded or parsed.

    Raised instead of returning partial data: a half-parsed benchmark
    table would silently change rankings.  The message names the source
    and the step that failed.
    """


class FuenteMeta(BaseModel):
    """Provenance of one source inside a snapshot.

    Attributes:
        nombre: Source identifier (also the ``fuente`` of every score).
        version: Source-side version of the data (Epoch: date of the
            published files; Artificial Analysis: Intelligence Index
            version).  Recorded so a snapshot can be reproduced.
        url: Endpoint the data was downloaded from.
        licencia: Licence or terms the data is used under.
        atribucion: Attribution line shown to users in every
            recommendation built with this data.
        redistribuible: Whether the data may be committed to the
            repository (CC BY) or must stay in ``runtime/`` only.
        registros: Source models with at least one score.
        descargado_en: Timezone-aware download instant.
    """

    nombre: NombreFuente
    version: str = Field(min_length=1)
    url: str = Field(min_length=1)
    licencia: str = Field(min_length=1)
    atribucion: str = Field(min_length=1)
    redistribuible: bool
    registros: int = Field(ge=0)
    descargado_en: datetime

    @model_validator(mode="after")
    def _tz(self) -> FuenteMeta:
        if self.descargado_en.tzinfo is None:
            raise ValueError("descargado_en must be timezone-aware (UTC)")
        return self


@dataclass
class DatosFuente:
    """Everything a source returns from one download.

    Attributes:
        meta: Provenance of the download.
        puntuaciones: Scores keyed by the exact source model name.  A
            name may carry several scores (one per benchmark).
    """

    meta: FuenteMeta
    puntuaciones: dict[str, list[EvidenciaBenchmark]]


@runtime_checkable
class FuenteBenchmarks(Protocol):
    """Downloadable benchmark source.

    Attributes:
        nombre: Source identifier, matching ``FuenteMeta.nombre``.
        obligatoria: Whether a failed download aborts the whole
            synchronisation (the base source) or is only recorded as a
            missing source (an optional complement).
    """

    nombre: NombreFuente
    obligatoria: bool

    def descargar(
        self, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
    ) -> DatosFuente:
        """Download and parse the source.

        Args:
            timeout: HTTP timeout in seconds.
            transport: Injected HTTP transport; tests pass
                ``httpx.MockTransport`` so no real connection is opened.

        Raises:
            FuenteBenchmarksError: On any transport, HTTP or parsing
                failure.  Never returns partial data.
        """
        ...


def normalizar(valor: float, minimo: float, maximo: float) -> float:
    """Map ``valor`` linearly from ``[minimo, maximo]`` onto ``[0, 1]``, clamped.

    Used for every score so the scorer compares like with like: a
    benchmark accuracy is anchored on its random baseline and ceiling, an
    index on documented fixed anchors.  ``maximo`` must exceed ``minimo``.
    """
    if maximo <= minimo:
        raise ValueError(f"invalid normalisation bounds: [{minimo}, {maximo}]")
    return round(min(1.0, max(0.0, (valor - minimo) / (maximo - minimo))), 4)
