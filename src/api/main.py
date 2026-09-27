"""FastAPI application — Semana 0 vertical slice.

Endpoints:
- ``POST /chat`` — accept criteria (hardcoded for now), run pipeline,
  return top candidates.
- ``GET /health`` — basic health check.
- ``GET /families`` — list curated families.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI

from src.catalog.synced import CatalogoSincronizado
from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    ClarificacionNecesaria,
    CriteriosBusqueda,
    PeticionChat,
    RespuestaChat,
    RespuestaRecomendacion,
)
from src.state import conversation_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Mount one catalogue per process, resolving it eagerly at startup.

    Builds a single :class:`CatalogoSincronizado` (construction does no
    I/O) and forces its lazy resolution with ``cargar()`` *before* the
    first request.  When both sources are exhausted
    (``CatalogUnavailableError``), the exception propagates out of this
    generator on purpose: **the app refuses to boot loudly** rather than
    starting in a degraded state.  A degraded mode was rejected because
    every alternative either served an ambiguous empty catalogue (which
    this project forbids) or required inventing a new error response
    body, changing the HTTP contract.  A fresh clone always boots: the
    bootstrap file is committed, so only a corrupted deployment fails.

    One instance per process is a deliberate design point: the resolved
    snapshot is memoized on the instance, so a single ``snapshot_id``
    backs every conversation, and constructing a catalogue per request
    would re-read disk and forfeit that guarantee.

    Args:
        app: The application whose ``state`` receives ``catalogo``.
            Handlers read ``app.state.catalogo``, which therefore only
            exists after this lifespan has run (tests must enter the
            TestClient context manager).

    Yields:
        Control to the running application.  On shutdown the state
        attribute is left in place; a re-entered lifespan replaces it.
    """
    catalogo = CatalogoSincronizado()
    catalogo.cargar()  # resolve + memoize now, or raise and abort boot
    app.state.catalogo = catalogo
    yield


app = FastAPI(title="Local Model Advisor", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/families")
def list_families():
    """List every family of the mounted catalogue snapshot.

    Returns:
        One plain-dict serialisation per ``FamiliaModelo``, identical in
        shape to the legacy seed endpoint (no ``response_model``: the
        shape is pinned by tests against ``FamiliaModelo`` itself).
    """
    return [f.model_dump() for f in app.state.catalogo.cargar()]


@app.post("/chat", response_model=RespuestaChat)
def chat(peticion: PeticionChat) -> RespuestaChat:
    """Process a chat message. For the vertical slice, criteria are
    provided directly in the first message (no LLM extraction yet).
    """
    # New conversation: create conversation_id
    if peticion.conversation_id is None:
        conversation_id = uuid4()
        criterios = CriteriosBusqueda.model_validate_json(peticion.mensaje)
    else:
        conversation_id = peticion.conversation_id
        existing = conversation_store.cargar(conversation_id)
        if existing is None:
            # Treat as new conversation
            conversation_id = uuid4()
            criterios = CriteriosBusqueda.model_validate_json(peticion.mensaje)
        else:
            criterios = existing.criterios

    # Validate mandatory fields
    faltantes = criterios.campos_faltantes()
    if faltantes:
        return ClarificacionNecesaria(
            conversation_id=conversation_id,
            pregunta=f"Faltan campos obligatorios: {', '.join(faltantes)}. "
            "Proporcione tarea y VRAM en GB.",
            campo_faltante=faltantes[0],
        )

    # Run the pipeline against the process-wide catalogue (mounted by lifespan)
    candidatos = ejecutar_pipeline(criterios, catalogo=app.state.catalogo, top_n=3)

    # Build response
    respuesta = RespuestaRecomendacion(
        conversation_id=conversation_id,
        criterios_interpretados=criterios,
        candidatos=candidatos,
        limitaciones=[
            "Benchmark evidence not yet available (OpenEvals pending).",
            "Explanatory evidence (Qdrant) not included in this slice.",
        ],
        datos_actualizados_hace="just now",
    )

    # Persist conversation state
    from src.models.schemas import ConversationRun, TurnoConversacion

    run = ConversationRun(
        conversation_id=conversation_id,
        criterios=criterios,
        candidatos_elegibles=candidatos,
        historial=[
            TurnoConversacion(rol="usuario", contenido=peticion.mensaje),
            TurnoConversacion(rol="sistema", contenido=respuesta.model_dump_json()),
        ],
    )
    conversation_store.guardar(run)

    return respuesta
