"""Pipeline orchestrator: load catalogue → filter → VRAM check → score → top N.

No Qdrant, no reranking, no LLM — purely deterministic.
"""

from __future__ import annotations

from src.catalog.protocol import CatalogoModelos
from src.compatibility.provider import VRAMCompatibilityProvider, elegir_provider
from src.models.schemas import CandidatoRankeado, CriteriosBusqueda
from src.ranking.scorer import calcular_puntuacion, ordenar_por_puntuacion
from src.tools.structured_filter import filtrar_candidatos


def ejecutar_pipeline(
    criterios: CriteriosBusqueda,
    *,
    catalogo: CatalogoModelos,
    top_n: int = 3,
    vram_provider: VRAMCompatibilityProvider | None = None,
) -> list[CandidatoRankeado]:
    """Run the full load → filter → score → rank pipeline.

    The families come exclusively from the injected ``catalogo``: the
    pipeline opens no files itself, so the API can mount one process-wide
    snapshot behind the Protocol and tests can inject an in-memory double.

    Args:
        criterios: User search criteria.  ``vram_gb`` feeds both the
            compatibility check and the ``ajuste_vram`` sub-score.
        catalogo: Source of the model families to rank.  ``cargar()`` is
            called at most once per invocation, after the ``top_n``
            short-circuit, so ``top_n=0`` never touches the catalogue.
        top_n: Number of top candidates to return.  Must be >= 0.  When it
            exceeds the number of eligible candidates, all are returned.
        vram_provider: Compatibility oracle.  ``None`` (production) routes
            by task through :func:`elegir_provider`; the golden-query
            evaluation injects a deterministic offline heuristic instead.

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
    provider = vram_provider or elegir_provider(criterios.tarea or "codigo")

    candidatos = filtrar_candidatos(familias, criterios, provider)
    for candidato in candidatos:
        calcular_puntuacion(candidato, vram_disponible_gb=criterios.vram_gb)

    return ordenar_por_puntuacion(candidatos)[:top_n]
