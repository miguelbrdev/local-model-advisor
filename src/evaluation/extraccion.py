"""Golden evaluation of free-text criteria extraction.

The extractor is the one component of the system driven by a language
model, so its accuracy cannot be reasoned about; it has to be measured.
A suite (``data/golden_extraccion.yaml``) pairs messages with the criteria
a reviewer expects.  Only the listed fields are judged, and a ``null``
expectation pins that the field must stay empty — the "never invent"
rule: no VRAM figure from a GPU name, no task from an unrelated request.

Running the suite needs the local Ollama model, so it is a script
(``scripts/evaluate_extraction.py``), not a pytest test; the evaluation
logic itself is tested with a scripted extractor.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from src.chains.extraccion_criterios import (
    ExtraccionNoDisponibleError,
    ExtractorCriterios,
    MensajeNoEstructuradoError,
    normalizar_idiomas,
)
from src.models.schemas import CriteriosBusqueda

DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "golden_extraccion.yaml"
)

CAMPOS_EVALUABLES = frozenset(
    {"tarea", "vram_gb", "idiomas_requeridos", "runtime_preferido", "requiere_licencia_comercial"}
)


class CasoExtraccion(BaseModel):
    """One message with the fields a reviewer expects (``null`` = must be empty)."""

    id: str = Field(min_length=1)
    mensaje: str = Field(min_length=1)
    esperado: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _campos_conocidos(self) -> CasoExtraccion:
        desconocidos = set(self.esperado) - CAMPOS_EVALUABLES
        if desconocidos:
            raise ValueError(f"case {self.id!r}: unknown fields {sorted(desconocidos)}")
        return self


class SuiteExtraccion(BaseModel):
    version: str = Field(min_length=1)
    descripcion: str = ""
    casos: list[CasoExtraccion] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unicos(self) -> SuiteExtraccion:
        ids = [c.id for c in self.casos]
        duplicados = sorted({i for i in ids if ids.count(i) > 1})
        if duplicados:
            raise ValueError(f"duplicate case ids: {duplicados}")
        return self


class ResultadoCasoExtraccion(BaseModel):
    id: str
    aprobado: bool
    campos_fallidos: dict[str, dict[str, Any]] = Field(default_factory=dict)
    obtenido: dict[str, Any] | None = None
    error: str | None = None
    segundos: float


class InformeExtraccion(BaseModel):
    version: str
    total: int
    aprobados: int
    tasa_aprobacion: float
    exactitud_por_campo: dict[str, float]
    errores_backend: int
    segundos_medio: float
    resultados: list[ResultadoCasoExtraccion]


def cargar_suite(path: Path | None = None) -> SuiteExtraccion:
    """Load and validate the suite (raises ``OSError``/``ValidationError``)."""
    with open(path or DEFAULT_SUITE_PATH, encoding="utf-8") as fh:
        return SuiteExtraccion.model_validate(yaml.safe_load(fh))


def evaluar_extraccion(suite: SuiteExtraccion, extractor: ExtractorCriterios) -> InformeExtraccion:
    """Run every case through ``extractor`` and aggregate field accuracy."""
    resultados = [evaluar_caso(caso, extractor) for caso in suite.casos]
    aciertos: dict[str, int] = {}
    totales: dict[str, int] = {}
    for caso, resultado in zip(suite.casos, resultados, strict=True):
        for campo in caso.esperado:
            totales[campo] = totales.get(campo, 0) + 1
            if resultado.error is None and campo not in resultado.campos_fallidos:
                aciertos[campo] = aciertos.get(campo, 0) + 1
    aprobados = sum(r.aprobado for r in resultados)
    return InformeExtraccion(
        version=suite.version,
        total=len(resultados),
        aprobados=aprobados,
        tasa_aprobacion=aprobados / len(resultados),
        exactitud_por_campo={c: aciertos.get(c, 0) / n for c, n in sorted(totales.items())},
        errores_backend=sum(r.error is not None for r in resultados),
        segundos_medio=sum(r.segundos for r in resultados) / len(resultados),
        resultados=resultados,
    )


def evaluar_caso(caso: CasoExtraccion, extractor: ExtractorCriterios) -> ResultadoCasoExtraccion:
    inicio = time.perf_counter()
    try:
        criterios = extractor.extraer(caso.mensaje)
    except (ExtraccionNoDisponibleError, MensajeNoEstructuradoError) as exc:
        return ResultadoCasoExtraccion(
            id=caso.id, aprobado=False, error=f"{type(exc).__name__}: {exc}",
            segundos=time.perf_counter() - inicio,
        )
    segundos = time.perf_counter() - inicio
    obtenido = criterios.model_dump()
    fallidos = {
        campo: {"esperado": esperado, "obtenido": obtenido.get(campo)}
        for campo, esperado in caso.esperado.items()
        if not coincide(campo, esperado, obtenido.get(campo))
    }
    return ResultadoCasoExtraccion(
        id=caso.id, aprobado=not fallidos, campos_fallidos=fallidos, obtenido=obtenido,
        segundos=segundos,
    )


def coincide(campo: str, esperado: Any, obtenido: Any) -> bool:
    """Field-aware comparison between an expectation and an extracted value."""
    if esperado is None:
        return obtenido in (None, [], False)
    if campo == "idiomas_requeridos":
        return set(normalizar_idiomas(list(esperado))) == set(obtenido or [])
    if campo == "vram_gb":
        return obtenido is not None and math.isclose(
            float(esperado), float(obtenido), abs_tol=0.01
        )
    if campo == "runtime_preferido":
        return obtenido is not None and " ".join(str(esperado).lower().split()) == obtenido
    if campo == "requiere_licencia_comercial":
        return bool(obtenido) is bool(esperado)
    return obtenido == esperado


def criterios_a_texto(criterios: CriteriosBusqueda) -> str:
    """Compact one-line rendering for reports."""
    partes = {k: v for k, v in criterios.model_dump().items() if v not in (None, [], False)}
    return ", ".join(f"{k}={v}" for k, v in partes.items()) or "(nada)"
