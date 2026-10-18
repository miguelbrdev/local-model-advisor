"""Deterministic structured filter for model candidates.

Filters the catalogue by task, license, language and VRAM compatibility —
no semantic search, no LLM calls.  Same input always produces the same
output.

The task filter reads ``FamiliaModelo.tareas_soportadas``, the full list
of internal tasks a model qualifies for, never ``pipeline_tag`` (which is
only the first of those tasks).  57 of the 91 bootstrap families declare
more than one task, so filtering on ``pipeline_tag`` silently dropped
42 of 51 ``codigo`` models and 50 of 59 ``razonamiento`` models.
"""

from __future__ import annotations

from src.compatibility.provider import VRAMCompatibilityProvider
from src.models.schemas import (
    CandidatoRankeado,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.tools.licencias import es_licencia_permisiva


def filtrar_candidatos(
    familias: list[FamiliaModelo],
    criterios: CriteriosBusqueda,
    vram_provider: VRAMCompatibilityProvider,
) -> list[CandidatoRankeado]:
    """Filter families by task, license, language and VRAM compatibility.

    The static criteria (task, license, language) are evaluated first so
    the VRAM provider — the only step that may perform I/O — is consulted
    only for families that could actually be recommended.

    Args:
        familias: Catalogue families to evaluate, in catalogue order.
        criterios: User criteria.  ``runtime_preferido`` is forwarded to
            the provider but no provider reads it today.
        vram_provider: Compatibility checker; must never raise.

    Returns:
        One ``CandidatoRankeado`` per eligible family, in input order,
        with the VRAM result attached.  Families whose compatibility is
        unknown (``compatible is None``) are kept: an unavailable provider
        must not empty the result set.
    """
    elegibles: list[CandidatoRankeado] = []

    for familia in familias:
        if not _cumple_criterios_estaticos(familia, criterios):
            continue

        vram_result = vram_provider.check(
            familia, criterios.vram_gb or 0.0, criterios.runtime_preferido
        )
        if vram_result.compatible is False:
            continue

        elegibles.append(
            CandidatoRankeado(
                familia=familia,
                variante_recomendada=vram_result.variante,
                compatibilidad_vram=vram_result,
            )
        )

    return elegibles


def soporta_tarea(familia: FamiliaModelo, tarea: str) -> bool:
    """Return whether ``familia`` declares ``tarea`` in ``tareas_soportadas``."""
    return tarea in familia.tareas_soportadas


def _cumple_criterios_estaticos(familia: FamiliaModelo, criterios: CriteriosBusqueda) -> bool:
    """Apply the I/O-free criteria: task, commercial license, languages."""
    if criterios.tarea and not soporta_tarea(familia, criterios.tarea):
        return False

    if criterios.requiere_licencia_comercial and not es_licencia_permisiva(
        familia.license_declarada
    ):
        return False

    return _cumple_idiomas(familia, criterios)


def _cumple_idiomas(familia: FamiliaModelo, criterios: CriteriosBusqueda) -> bool:
    """Every required language must be among the family's declared ones.

    Only enforced when the family declares languages.  The synced source
    exposes none (``idiomas_detectados`` is ``None`` for every synced
    family), so the check passes there by design: unknown languages are a
    limitation to report, not a reason to exclude.
    """
    if not criterios.idiomas_requeridos or not familia.idiomas_detectados:
        return True
    return set(criterios.idiomas_requeridos) <= set(familia.idiomas_detectados)
