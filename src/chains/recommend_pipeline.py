"""Pipeline orchestrator for the Semana 0 vertical slice.

Coordinates: load catalogue → filter → VRAM check → score → return top N.
No Qdrant, no reranking, no LLM — purely deterministic.
"""

from __future__ import annotations

from src.catalog.protocol import CatalogoModelos
from src.compatibility.provider import elegir_provider
from src.models.schemas import (
    CandidatoRankeado,
    CriteriosBusqueda,
)
from src.ranking.scorer import calcular_puntuacion, ordenar_por_puntuacion
from src.tools.structured_filter import filtrar_candidatos


def ejecutar_pipeline(
    criterios: CriteriosBusqueda,
    *,
    catalogo: CatalogoModelos,
    top_n: int = 3,
) -> list[CandidatoRankeado]:
    """Run the full load → filter → score → rank pipeline.

    The families come exclusively from the injected ``catalogo``: the
    pipeline opens no files itself (dependency inversion), so the API can
    mount one process-wide snapshot behind the Protocol and tests can
    inject an in-memory double.  A ``load_seed``-style fallback would
    reintroduce a second source of truth for the same question.

    Args:
        criterios: User search criteria.
        catalogo: Source of the model families to rank.  ``cargar()`` is
            called at most once per invocation, after the ``top_n``
            short-circuits, so ``top_n=0`` never touches the catalogue.
        top_n: Number of top candidates to return. Must be >= 0.
            When ``top_n`` exceeds the number of eligible candidates, all
            candidates are returned. When ``top_n`` is 0, an empty list is
            returned.

    Raises:
        ValueError: If ``top_n`` is negative.
        CatalogUnavailableError: Propagated untouched from
            ``catalogo.cargar()`` when no catalogue source resolves.  The
            API resolves the catalogue during startup precisely so this
            cannot surface mid-request in production.

    Returns:
        Top N ranked candidates sorted by score descending.
    """
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    if top_n == 0:
        return []

    familias = catalogo.cargar()

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
