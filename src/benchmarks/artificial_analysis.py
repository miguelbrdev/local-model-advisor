"""Artificial Analysis adapter (optional complement, per-operator key).

Contract observed on 2026-11-08 at
``GET https://artificialanalysis.ai/api/v2/language/models/free?page=N``
with header ``x-api-key`` (free tier, 100 requests per 24 h):

* Body ``{"tier": "free", "intelligence_index_version": 5,
  "pagination": {"page", "page_size", "total_pages", "has_more"},
  "data": [...]}``; 200 models per page, four pages in total.
* Each model: ``id``, ``name``, ``slug``, ``release_date``,
  ``evaluations.artificial_analysis_intelligence_index`` (``null`` for
  most entries on the free tier) plus pricing and performance fields
  that are ignored here.

Only models with a non-null Intelligence Index are kept, keyed by
``slug``.  The index is published on a nominal 0–100 scale, which is the
normalisation used.  The data may not be redistributed, so the adapter
marks it ``redistribuible=False`` and the bootstrap promotion drops it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.benchmarks.fuentes import (
    DatosFuente,
    FuenteBenchmarksError,
    FuenteMeta,
    normalizar,
)
from src.models.schemas import EvidenciaBenchmark

logger = logging.getLogger(__name__)

AA_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
VARIABLE_CLAVE = "ARTIFICIAL_ANALYSIS_API_KEY"
CAMPO_INDICE = "artificial_analysis_intelligence_index"
INDICE_MAX = 100.0
"""Nominal upper bound of the Intelligence Index."""
MAX_PAGINAS = 20
"""Hard stop on pagination so a misbehaving ``has_more`` cannot burn the daily quota."""

LICENCIA = "Artificial Analysis terms of use (attribution required, no redistribution)"
ATRIBUCION_PLANTILLA = (
    "Intelligence Index (version {version}): Artificial Analysis, "
    "https://artificialanalysis.ai (used under its terms; not redistributed)."
)


class _ModeloAA(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slug: str = Field(min_length=1)
    name: str = ""
    release_date: str | None = None
    evaluations: dict[str, float | None] = Field(default_factory=dict)


class _Paginacion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    has_more: bool = False


class _RespuestaAA(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intelligence_index_version: int | None = None
    pagination: _Paginacion = Field(default_factory=_Paginacion)
    data: list[_ModeloAA] = Field(default_factory=list)


class FuenteArtificialAnalysis:
    """Page through the free endpoint and keep every model with an index."""

    nombre = "artificial_analysis"
    obligatoria = False

    def __init__(self, api_key: str, url: str = AA_URL) -> None:
        if not api_key.strip():
            raise ValueError("Artificial Analysis API key must not be empty")
        self._api_key = api_key.strip()
        self._url = url

    @classmethod
    def desde_entorno(
        cls, env: Mapping[str, str] | None = None
    ) -> FuenteArtificialAnalysis | None:
        """Build the source from :data:`VARIABLE_CLAVE`, or ``None`` when unset.

        The key is never read from a file by this package: the operator
        exports it (or Docker Compose substitutes it from ``.env``).
        """
        clave = (env if env is not None else os.environ).get(VARIABLE_CLAVE, "").strip()
        return cls(clave) if clave else None

    def descargar(
        self, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
    ) -> DatosFuente:
        puntuaciones: dict[str, list[EvidenciaBenchmark]] = {}
        version: int | None = None
        try:
            with httpx.Client(
                transport=transport, timeout=timeout, headers={"x-api-key": self._api_key}
            ) as client:
                for pagina in range(1, MAX_PAGINAS + 1):
                    respuesta = client.get(self._url, params={"page": pagina})
                    if respuesta.status_code in (401, 403):
                        raise FuenteBenchmarksError(
                            f"artificial_analysis: API key rejected (HTTP {respuesta.status_code})"
                        )
                    respuesta.raise_for_status()
                    cuerpo = _RespuestaAA.model_validate_json(respuesta.content)
                    version = cuerpo.intelligence_index_version or version
                    for modelo in cuerpo.data:
                        valor = modelo.evaluations.get(CAMPO_INDICE)
                        if valor is None:
                            continue
                        puntuaciones[modelo.slug] = [
                            EvidenciaBenchmark(
                                fuente="artificial_analysis",
                                benchmark="intelligence_index",
                                metrica=CAMPO_INDICE,
                                valor=round(valor, 2),
                                valor_normalizado=normalizar(valor, 0.0, INDICE_MAX),
                                modelo_fuente=modelo.slug,
                                fecha_modelo=modelo.release_date,
                                nota=f"{modelo.name}; nominal 0-{INDICE_MAX:g} scale.".strip(),
                            )
                        ]
                    if not cuerpo.pagination.has_more:
                        break
                else:
                    logger.warning(
                        "artificial_analysis: stopped after %d pages with has_more still true",
                        MAX_PAGINAS,
                    )
        except httpx.HTTPError as exc:
            raise FuenteBenchmarksError(f"artificial_analysis: download failed: {exc}") from exc
        except ValidationError as exc:
            raise FuenteBenchmarksError(
                f"artificial_analysis: unexpected response: {exc}"
            ) from exc

        if not puntuaciones:
            raise FuenteBenchmarksError("artificial_analysis: no model carries an index")

        texto_version = f"v{version}" if version is not None else "unknown"
        meta = FuenteMeta(
            nombre="artificial_analysis",
            version=texto_version,
            url=self._url,
            licencia=LICENCIA,
            atribucion=ATRIBUCION_PLANTILLA.format(version=texto_version),
            redistribuible=False,
            registros=len(puntuaciones),
            descargado_en=datetime.now(UTC),
        )
        logger.info(
            "artificial_analysis: %d models with an index (%s)", len(puntuaciones), texto_version
        )
        return DatosFuente(meta=meta, puntuaciones=puntuaciones)
