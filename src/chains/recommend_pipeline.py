"""Pipeline orchestrator for the Semana 0 vertical slice.

Coordinates: load catalogue → filter → VRAM check → score → return top N.
No Qdrant, no reranking, no LLM — purely deterministic.
"""

from __future__ import annotations

from src.compatibility.provider import elegir_provider
from src.models.schemas import (
    CandidatoRankeado,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.ranking.scorer import calcular_puntuacion, ordenar_por_puntuacion
from src.tools.seed_loader import load_seed
from src.tools.structured_filter import filtrar_candidatos


def ejecutar_pipeline(
    criterios: CriteriosBusqueda,
    familias: list[FamiliaModelo] | None = None,
    top_n: int = 3,
) -> list[CandidatoRankeado]:
    """Run the full filter → score → rank pipeline.

    Args:
        criterios: User search criteria.
        familias: Override catalogue (uses seed YAML if ``None``).
        top_n: Number of top candidates to return.

    Returns:
        Top N ranked candidates sorted by score descending.
    """
    if familias is None:
        familias = load_seed()

    # 1. Choose VRAM provider based on task
    provider = elegir_provider(criterios.tarea or "codigo")

    # 2. Deterministic filter
    candidatos = filtrar_candidatos(familias, criterios, provider)

    # 3. Score each candidate
    for c in candidatos:
        calcular_puntuacion(c)

    # 4. Sort and return top N
    ordenados = ordenar_por_puntuacion(candidatos)
    return ordenados[:top_n]
