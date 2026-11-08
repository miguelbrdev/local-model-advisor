"""Deterministic structured filter for model candidates.

Filters the catalogue by task, license, language, runtime and VRAM
compatibility — no semantic search, no LLM calls.  Same input always
produces the same output.

The task filter reads ``FamiliaModelo.tareas_soportadas``, the full list
of internal tasks a model qualifies for, never ``pipeline_tag`` (which is
only the first of those tasks).  57 of the 91 bootstrap families declare
more than one task, so filtering on ``pipeline_tag`` silently dropped
42 of 51 ``codigo`` models and 50 of 59 ``razonamiento`` models.

VRAM checks run concurrently: the compatibility provider may call
canirun.ai (about a second per model), and a request can have twenty
eligible families.  Results are collected in catalogue order so the
output is identical to a sequential run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.compatibility.provider import VRAMCompatibilityProvider
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.tools.licencias import es_licencia_permisiva
from src.tools.runtimes import normalizar_runtime

MAX_WORKERS_VRAM = 8
"""Concurrent compatibility checks per request.  Bounded so an outage costs
at most this many simultaneous timeouts and canirun.ai is not flooded."""


def filtrar_candidatos(
    familias: list[FamiliaModelo],
    criterios: CriteriosBusqueda,
    vram_provider: VRAMCompatibilityProvider,
) -> list[CandidatoRankeado]:
    """Filter families by task, license, language, runtime and VRAM compatibility.

    The static criteria (task, license, language, runtime) are evaluated first so
    the VRAM provider — the only step that may perform I/O — is consulted
    only for families that could actually be recommended, and those
    checks run in parallel (see :data:`MAX_WORKERS_VRAM`).

    Args:
        familias: Catalogue families to evaluate, in catalogue order.
        criterios: User criteria.  ``runtime_preferido`` filters families
            that declare runtimes and is also forwarded to the provider
            (no provider reads it today).
        vram_provider: Compatibility checker; must never raise and must be
            safe to call from several threads.

    Returns:
        One ``CandidatoRankeado`` per eligible family, in input order,
        with the VRAM result attached.  Families whose compatibility is
        unknown (``compatible is None``) are kept: an unavailable provider
        must not empty the result set.
    """
    aptas = [f for f in familias if _cumple_criterios_estaticos(f, criterios)]
    resultados = _comprobar_vram(aptas, criterios, vram_provider)
    return [
        CandidatoRankeado(
            familia=familia,
            variante_recomendada=resultado.variante,
            compatibilidad_vram=resultado,
        )
        for familia, resultado in zip(aptas, resultados, strict=True)
        if resultado.compatible is not False
    ]


def _comprobar_vram(
    familias: list[FamiliaModelo],
    criterios: CriteriosBusqueda,
    vram_provider: VRAMCompatibilityProvider,
) -> list[CompatibilityResult]:
    """Check every family, concurrently when there is more than one."""
    vram_gb = criterios.vram_gb or 0.0

    def comprobar(familia: FamiliaModelo) -> CompatibilityResult:
        return vram_provider.check(familia, vram_gb, criterios.runtime_preferido)

    if len(familias) <= 1:
        return [comprobar(f) for f in familias]
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS_VRAM, len(familias))) as pool:
        return list(pool.map(comprobar, familias))


def soporta_tarea(familia: FamiliaModelo, tarea: str) -> bool:
    """Return whether ``familia`` declares ``tarea`` in ``tareas_soportadas``."""
    return tarea in familia.tareas_soportadas


def _cumple_criterios_estaticos(familia: FamiliaModelo, criterios: CriteriosBusqueda) -> bool:
    """Apply the I/O-free criteria: task, commercial license, languages, runtime."""
    if criterios.tarea and not soporta_tarea(familia, criterios.tarea):
        return False

    if criterios.requiere_licencia_comercial and not es_licencia_permisiva(
        familia.license_declarada
    ):
        return False

    return _cumple_idiomas(familia, criterios) and _cumple_runtime(familia, criterios)


def _cumple_idiomas(familia: FamiliaModelo, criterios: CriteriosBusqueda) -> bool:
    """Every required language must be among the family's declared ones.

    Only enforced when the family declares languages (the Hugging Face
    card of 30 of the 91 bootstrap families does).  A family without a
    declaration passes by design: unknown languages are a limitation to
    report, not a reason to exclude.
    """
    if not criterios.idiomas_requeridos or not familia.idiomas_detectados:
        return True
    return set(criterios.idiomas_requeridos) <= set(familia.idiomas_detectados)


def _cumple_runtime(familia: FamiliaModelo, criterios: CriteriosBusqueda) -> bool:
    """The preferred runtime must be among the family's known runtimes.

    Same policy as languages: only enforced when the family declares
    runtimes (``runtimes_disponibles`` from the enrichment step); an
    unknown runtime set never excludes.  The user's value is normalised
    with the same aliases the catalogue uses.
    """
    runtime = normalizar_runtime(criterios.runtime_preferido)
    if runtime is None or not familia.runtimes_disponibles:
        return True
    return runtime in familia.runtimes_disponibles
