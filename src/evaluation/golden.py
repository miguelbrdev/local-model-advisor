"""Golden-query evaluation: pin the ranking behaviour with reviewed cases.

A *golden suite* (``data/golden_queries.yaml``) is a versioned list of
query cases, each with the criteria a user would send and the outcome a
reviewer expects from a given catalogue snapshot: families that must be in
the top N, families that must be filtered out entirely, optionally the
expected first result, and a minimum number of candidates so an
ambiguous empty answer is itself a failure.

The evaluation is a regression net for the deterministic part of the
pipeline (filter + scorer).  It therefore requires a deterministic
compatibility oracle — in practice
:class:`src.compatibility.heuristica.HeuristicaParametrosProvider` — and
**does not** exercise the real canirun.ai answers.  A suite is written
against one catalogue ``snapshot_id`` and, since 1.3.0, one benchmark
snapshot (``benchmarks_snapshot_id``, the committed Epoch-only bootstrap);
running it against other data is allowed but flagged in the report,
because expectations may legitimately differ when the data changes.

Metrics reported:

* ``tasa_aprobacion``: cases with every expectation met / total cases.
* ``hit_rate``: expected inclusions actually found in the top N / expected
  inclusions (``None`` if the suite expects none).
* ``tasa_exclusion``: expected exclusions actually absent / expected
  exclusions (``None`` if none).
* ``aciertos_primero`` / ``esperados_primero``: cases whose first result
  matched the expectation.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from src.benchmarks.provider import ProveedorBenchmarks
from src.catalog.protocol import CatalogoModelos
from src.chains.recommend_pipeline import ejecutar_pipeline
from src.compatibility.provider import VRAMCompatibilityProvider
from src.models.schemas import CriteriosBusqueda

DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "golden_queries.yaml"
)

# Large enough to return every eligible family of any realistic catalogue,
# so exclusions can be checked against the full ranking, not just the top N.
_TODOS_LOS_CANDIDATOS = 10_000


class CasoGolden(BaseModel):
    """One reviewed query with its expected outcome.

    Attributes:
        id: Unique, stable identifier used in reports.
        descripcion: Why this case exists (what behaviour it pins).
        criterios: Exactly what a client would send in ``mensaje``.
        top_n: Size of the window ``debe_incluir`` and ``primero`` refer to.
        debe_incluir: Family ids that must appear within the top ``top_n``.
        debe_excluir: Family ids that must not survive the filter at all.
        debe_quedar_fuera_del_top: Family ids that may be candidates but
            must not rank within the top ``top_n`` (pins "no longer
            recommended first" without forbidding the model).
        primero: Family id expected at rank 1, if the case pins it.
        min_candidatos: Minimum number of eligible candidates; the default
            of 1 makes an empty answer a failure unless a case opts out.
    """

    id: str = Field(min_length=1)
    descripcion: str = ""
    criterios: CriteriosBusqueda
    top_n: int = Field(default=3, ge=1)
    debe_incluir: list[str] = Field(default_factory=list)
    debe_excluir: list[str] = Field(default_factory=list)
    debe_quedar_fuera_del_top: list[str] = Field(default_factory=list)
    primero: str | None = None
    min_candidatos: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _coherente(self) -> CasoGolden:
        solapados = set(self.debe_incluir) & set(self.debe_excluir)
        if solapados:
            raise ValueError(f"case {self.id!r}: families both included and excluded: {solapados}")
        fuera = set(self.debe_quedar_fuera_del_top)
        if fuera & set(self.debe_incluir):
            raise ValueError(
                f"case {self.id!r}: families both included and kept out of the top: "
                f"{fuera & set(self.debe_incluir)}"
            )
        if self.primero is not None and self.primero in set(self.debe_excluir) | fuera:
            raise ValueError(
                f"case {self.id!r}: 'primero' {self.primero!r} is also excluded or kept out"
            )
        if self.criterios.campos_faltantes():
            raise ValueError(
                f"case {self.id!r}: criteria missing mandatory fields "
                f"{self.criterios.campos_faltantes()}"
            )
        return self


class SuiteGolden(BaseModel):
    """Versioned set of golden cases written against one catalogue snapshot
    and (optionally) one benchmark snapshot."""

    version: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    benchmarks_snapshot_id: str | None = None
    descripcion: str = ""
    casos: list[CasoGolden] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unicos(self) -> SuiteGolden:
        ids = [c.id for c in self.casos]
        duplicados = {i for i in ids if ids.count(i) > 1}
        if duplicados:
            raise ValueError(f"duplicate case ids: {sorted(duplicados)}")
        return self


class ResultadoCaso(BaseModel):
    """Outcome of one case: what was expected, what came back, what failed."""

    id: str
    aprobado: bool
    fallos: list[str] = Field(default_factory=list)
    top: list[str] = Field(default_factory=list)
    candidatos_totales: int
    incluidos_encontrados: int
    incluidos_esperados: int
    excluidos_respetados: int
    excluidos_esperados: int
    primero_ok: bool | None = None


class InformeGolden(BaseModel):
    """Aggregate report of one suite run."""

    version: str
    snapshot_id_suite: str
    snapshot_id_catalogo: str | None
    snapshot_coincide: bool
    benchmarks_id_suite: str | None = None
    benchmarks_id_usado: str | None = None
    benchmarks_coinciden: bool = True
    total: int
    aprobados: int
    tasa_aprobacion: float
    hit_rate: float | None
    tasa_exclusion: float | None
    aciertos_primero: int
    esperados_primero: int
    resultados: list[ResultadoCaso]


def cargar_suite(path: Path | None = None) -> SuiteGolden:
    """Load and validate a golden suite from YAML.

    Raises:
        OSError: If the file cannot be read.
        pydantic.ValidationError: If the content violates the schema or a
            case is internally inconsistent.
    """
    ruta = path or DEFAULT_SUITE_PATH
    with open(ruta, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return SuiteGolden.model_validate(raw)


def evaluar_suite(
    suite: SuiteGolden,
    *,
    catalogo: CatalogoModelos,
    vram_provider: VRAMCompatibilityProvider,
    benchmarks: ProveedorBenchmarks | None = None,
) -> InformeGolden:
    """Run every case of ``suite`` through the pipeline and aggregate metrics.

    Args:
        suite: Validated golden suite.
        catalogo: Catalogue to rank over.  Its ``snapshot_id`` is compared
            with the suite's and the mismatch (if any) is reported, never
            treated as an error here.
        vram_provider: Must be deterministic and offline for results to be
            reproducible; callers are expected to pass the heuristic.
        benchmarks: Benchmark scores injected into the pipeline; ``None``
            evaluates with neutral ``capacidad_medida``.  Its snapshot id
            is compared with the suite's ``benchmarks_snapshot_id`` and a
            mismatch is reported the same way.

    Returns:
        The :class:`InformeGolden` with one :class:`ResultadoCaso` per case,
        in suite order.
    """
    resultados = [
        evaluar_caso(
            caso, catalogo=catalogo, vram_provider=vram_provider, benchmarks=benchmarks
        )
        for caso in suite.casos
    ]

    incluidos_esperados = sum(r.incluidos_esperados for r in resultados)
    incluidos_encontrados = sum(r.incluidos_encontrados for r in resultados)
    excluidos_esperados = sum(r.excluidos_esperados for r in resultados)
    excluidos_respetados = sum(r.excluidos_respetados for r in resultados)
    con_primero = [r for r in resultados if r.primero_ok is not None]
    aprobados = sum(r.aprobado for r in resultados)
    snapshot_catalogo = catalogo.snapshot_id
    benchmarks_usado = benchmarks.snapshot_id if benchmarks is not None else None

    return InformeGolden(
        version=suite.version,
        snapshot_id_suite=suite.snapshot_id,
        snapshot_id_catalogo=snapshot_catalogo,
        snapshot_coincide=snapshot_catalogo == suite.snapshot_id,
        benchmarks_id_suite=suite.benchmarks_snapshot_id,
        benchmarks_id_usado=benchmarks_usado,
        benchmarks_coinciden=benchmarks_usado == suite.benchmarks_snapshot_id,
        total=len(resultados),
        aprobados=aprobados,
        tasa_aprobacion=aprobados / len(resultados),
        hit_rate=_ratio(incluidos_encontrados, incluidos_esperados),
        tasa_exclusion=_ratio(excluidos_respetados, excluidos_esperados),
        aciertos_primero=sum(r.primero_ok is True for r in con_primero),
        esperados_primero=len(con_primero),
        resultados=resultados,
    )


def evaluar_caso(
    caso: CasoGolden,
    *,
    catalogo: CatalogoModelos,
    vram_provider: VRAMCompatibilityProvider,
    benchmarks: ProveedorBenchmarks | None = None,
) -> ResultadoCaso:
    """Evaluate one case against the full ranking.

    The pipeline is asked for every eligible candidate so ``debe_excluir``
    is checked against the whole filtered set, and the top ``top_n`` slice
    is what ``debe_incluir`` and ``primero`` are judged on.
    """
    ranking = ejecutar_pipeline(
        caso.criterios,
        catalogo=catalogo,
        top_n=_TODOS_LOS_CANDIDATOS,
        vram_provider=vram_provider,
        benchmarks=benchmarks,
    )
    todos = [c.familia.familia_id for c in ranking]
    top = todos[: caso.top_n]
    fallos: list[str] = []

    if len(todos) < caso.min_candidatos:
        fallos.append(f"expected at least {caso.min_candidatos} candidates, got {len(todos)}")

    faltan = [fid for fid in caso.debe_incluir if fid not in top]
    if faltan:
        fallos.append(f"missing from top {caso.top_n}: {faltan}")

    presentes = [fid for fid in caso.debe_excluir if fid in todos]
    if presentes:
        fallos.append(f"should have been filtered out: {presentes}")

    colados = [fid for fid in caso.debe_quedar_fuera_del_top if fid in top]
    if colados:
        fallos.append(f"should not rank in top {caso.top_n}: {colados}")

    primero_ok: bool | None = None
    if caso.primero is not None:
        primero_ok = bool(top) and top[0] == caso.primero
        if not primero_ok:
            fallos.append(f"expected first {caso.primero!r}, got {top[0] if top else None!r}")

    return ResultadoCaso(
        id=caso.id,
        aprobado=not fallos,
        fallos=fallos,
        top=top,
        candidatos_totales=len(todos),
        incluidos_encontrados=len(caso.debe_incluir) - len(faltan),
        incluidos_esperados=len(caso.debe_incluir),
        excluidos_respetados=len(caso.debe_excluir) - len(presentes),
        excluidos_esperados=len(caso.debe_excluir),
        primero_ok=primero_ok,
    )


def _ratio(numerador: int, denominador: int) -> float | None:
    return None if denominador == 0 else numerador / denominador
