"""Deterministic structured filter for model candidates.

Filters the curated catalogue by task, VRAM compatibility, language,
and license — no semantic search, no LLM calls. Same input always
produces the same output.
"""

from __future__ import annotations

from src.compatibility.provider import VRAMCompatibilityProvider
from src.models.schemas import (
    CandidatoRankeado,
    CriteriosBusqueda,
    FamiliaModelo,
)


def filtrar_candidatos(
    familias: list[FamiliaModelo],
    criterios: CriteriosBusqueda,
    vram_provider: VRAMCompatibilityProvider,
) -> list[CandidatoRankeado]:
    """Filter families by task, VRAM, language and license.

    Returns a list of ``CandidatoRankeado`` with VRAM compatibility
    already evaluated. Families that don't match are silently excluded.
    """
    elegibles: list[CandidatoRankeado] = []

    for familia in familias:
        # 1. Task filter: pipeline_tag must match the requested task
        if criterios.tarea and not _tarea_match(familia.pipeline_tag, criterios.tarea):
            continue

        # 2. License filter
        if criterios.requiere_licencia_comercial and not _es_licencia_comercial(
            familia.license_declarada
        ):
            continue

        # 3. Language filter
        if criterios.idiomas_requeridos and familia.idiomas_detectados:
            if not set(criterios.idiomas_requeridos).issubset(
                set(familia.idiomas_detectados)
            ):
                continue

        # 4. VRAM compatibility check
        vram_result = vram_provider.check(
            familia, criterios.vram_gb or 0.0, criterios.runtime_preferido
        )

        # Exclude models that definitely don't fit
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


def _tarea_match(pipeline_tag: str, tarea: str) -> bool:
    """Map pipeline_tag to the 5 supported task categories."""
    mapping = {
        "chat": {"text-generation", "conversational", "chat"},
        "codigo": {"code", "text-generation"},
        "razonamiento": {"text-generation"},
        "embeddings": {"sentence-similarity", "feature-extraction", "retrieval"},
        "rerankers": {"text-classification"},
    }
    tags = mapping.get(tarea, set())
    return pipeline_tag.lower() in tags or pipeline_tag.lower() == tarea


def _es_licencia_comercial(licencia: str | None) -> bool:
    """Check if a license allows commercial use."""
    if not licencia:
        return False
    permissive = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0"}
    normalized = licencia.lower().replace(" ", "-")
    return normalized in permissive
