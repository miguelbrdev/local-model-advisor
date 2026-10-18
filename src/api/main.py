"""FastAPI application.

Endpoints:
- ``POST /chat`` — accept structured criteria as JSON in ``mensaje`` (no
  natural-language interpretation yet), run the pipeline, return the top
  candidates or a clarification request.
- ``GET /health`` — basic health check.
- ``GET /families`` — list the families of the mounted catalogue snapshot.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

from src.catalog.synced import CatalogoSincronizado
from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    ClarificacionNecesaria,
    ConversationRun,
    CriteriosBusqueda,
    PeticionChat,
    RespuestaChat,
    RespuestaRecomendacion,
    TurnoConversacion,
)
from src.state import conversation_store

TOP_N_RECOMENDACIONES = 3

LIMITACIONES_ACTUALES = [
    "Benchmark evidence not yet available (OpenEvals pending).",
    "Explanatory evidence (Qdrant) not included in this slice.",
    "The catalogue source exposes no language data; language requirements "
    "are not enforced on synced models.",
    "runtime_preferido is accepted but not applied yet.",
]
"""Limitations reported with every recommendation.  They describe what the
pipeline cannot do today so an empty or short result is never ambiguous."""


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
    """Process a chat message and return a recommendation or a clarification.

    The first message of a conversation carries the criteria as a JSON
    ``CriteriosBusqueda`` payload in ``mensaje``.  Follow-ups reuse the
    criteria stored server-side under ``conversation_id``; the client is
    never the source of truth for criteria, candidates or scores.

    Raises:
        HTTPException: ``422`` when a new conversation's ``mensaje`` is not
            a valid ``CriteriosBusqueda`` JSON document.  The detail lists
            the validation errors so the client can fix its payload.

    Side effects:
        Persists the ``ConversationRun`` — cumulative history plus the
        ``snapshot_id`` of the mounted catalogue, for reproducibility — in
        SQLite.
    """
    conversation_id, criterios, historial = _resolver_conversacion(peticion)

    faltantes = criterios.campos_faltantes()
    if faltantes:
        return ClarificacionNecesaria(
            conversation_id=conversation_id,
            pregunta=f"Faltan campos obligatorios: {', '.join(faltantes)}. "
            "Proporcione tarea y VRAM en GB.",
            campo_faltante=faltantes[0],
        )

    catalogo = app.state.catalogo
    candidatos = ejecutar_pipeline(criterios, catalogo=catalogo, top_n=TOP_N_RECOMENDACIONES)

    respuesta = RespuestaRecomendacion(
        conversation_id=conversation_id,
        criterios_interpretados=criterios,
        candidatos=candidatos,
        limitaciones=list(LIMITACIONES_ACTUALES),
        datos_actualizados_hace=describir_antiguedad(catalogo.fecha_sincronizacion),
    )

    conversation_store.guardar(
        ConversationRun(
            conversation_id=conversation_id,
            snapshot_id=catalogo.snapshot_id or "",
            criterios=criterios,
            candidatos_elegibles=candidatos,
            historial=[
                *historial,
                TurnoConversacion(rol="usuario", contenido=peticion.mensaje),
                TurnoConversacion(rol="sistema", contenido=respuesta.model_dump_json()),
            ],
        )
    )
    return respuesta


def describir_antiguedad(fecha: datetime | None, ahora: datetime | None = None) -> str:
    """Describe how old the catalogue data is, in Spanish, for the response.

    Args:
        fecha: Timezone-aware synchronisation instant of the served
            snapshot, or ``None`` when the catalogue has no date.
        ahora: Reference instant; defaults to the current UTC time.  Tests
            inject it to keep the output deterministic.

    Returns:
        ``""`` when the date is unknown (the field's documented "no
        information" value), otherwise e.g. ``"hace 26 días (sincronizado
        el 2026-09-22)"``.  Negative ages (clock skew) are reported as
        ``hace 0 días``.
    """
    if fecha is None:
        return ""
    referencia = ahora or datetime.now(UTC)
    dias = max(0, (referencia - fecha).days)
    unidad = "día" if dias == 1 else "días"
    return f"hace {dias} {unidad} (sincronizado el {fecha.date().isoformat()})"


def _resolver_conversacion(
    peticion: PeticionChat,
) -> tuple[UUID, CriteriosBusqueda, list[TurnoConversacion]]:
    """Return the conversation id, criteria and prior history for a request.

    An unknown or expired ``conversation_id`` is treated as a new
    conversation: a fresh id is issued, the criteria are parsed from
    ``mensaje`` and the history starts empty, exactly as for a first
    message.

    Raises:
        HTTPException: ``422`` if a new conversation's ``mensaje`` is not a
            valid ``CriteriosBusqueda`` JSON document.
    """
    if peticion.conversation_id is not None:
        existente = conversation_store.cargar(peticion.conversation_id)
        if existente is not None:
            return peticion.conversation_id, existente.criterios, list(existente.historial)
    return uuid4(), _parsear_criterios(peticion.mensaje), []


def _parsear_criterios(mensaje: str) -> CriteriosBusqueda:
    """Parse the first message of a conversation as ``CriteriosBusqueda`` JSON.

    Malformed input is a client error, not a server fault: it is reported
    as ``422`` with FastAPI's usual ``loc``/``msg``/``type`` error items
    instead of leaking a ``500``.
    """
    try:
        return CriteriosBusqueda.model_validate_json(mensaje)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[
                {"loc": ["body", "mensaje", *error["loc"]], "msg": error["msg"],
                 "type": error["type"]}
                for error in exc.errors(include_url=False)
            ],
        ) from exc
