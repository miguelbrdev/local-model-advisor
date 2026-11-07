"""FastAPI application.

Endpoints:
- ``POST /chat`` — accept criteria as JSON or as free text in ``mensaje``,
  run the pipeline, return the top candidates or a clarification request.
- ``GET /health`` — basic health check.
- ``GET /families`` — list the families of the mounted catalogue snapshot.

Configuration (environment variables, read once at startup):
- ``LMA_EXTRACCION_LLM``: ``"0"``/``"false"`` disables free-text input;
  only criteria JSON is then accepted (default: enabled).
- ``LMA_OLLAMA_MODEL``: Ollama model tag (default ``qwen2.5:3b``).
- ``LMA_OLLAMA_BASE_URL``: Ollama server (default ``http://127.0.0.1:11434``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status

from src.catalog.synced import CatalogoSincronizado
from src.chains.extraccion_criterios import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    ExtraccionNoDisponibleError,
    ExtractorCompuesto,
    ExtractorCriterios,
    ExtractorJson,
    ExtractorOllama,
    MensajeNoEstructuradoError,
)
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


def construir_extractor(env: dict[str, str] | None = None) -> ExtractorCompuesto:
    """Build the message extractor from environment configuration.

    Args:
        env: Mapping to read instead of ``os.environ`` (tests).

    Returns:
        JSON-first extractor, with the Ollama free-text extractor attached
        unless ``LMA_EXTRACCION_LLM`` disables it.  Building performs no
        I/O: Ollama is only contacted when a non-JSON message arrives.
    """
    fuente = os.environ if env is None else env
    habilitado = fuente.get("LMA_EXTRACCION_LLM", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    llm = (
        ExtractorOllama(
            modelo=fuente.get("LMA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=fuente.get("LMA_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )
        if habilitado
        else None
    )
    return ExtractorCompuesto(ExtractorJson(), llm)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Mount one catalogue and one extractor per process at startup.

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

    The extractor (``app.state.extractor``) is built from the environment;
    an unreachable Ollama does not block startup because structured JSON
    input keeps working without it.

    Args:
        app: The application whose ``state`` receives ``catalogo`` and
            ``extractor``.  Both only exist after this lifespan has run
            (tests must enter the TestClient context manager).

    Yields:
        Control to the running application.
    """
    catalogo = CatalogoSincronizado()
    catalogo.cargar()  # resolve + memoize now, or raise and abort boot
    app.state.catalogo = catalogo
    app.state.extractor = construir_extractor()
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

    ``mensaje`` may be a JSON ``CriteriosBusqueda`` document or free text.
    Criteria accumulate per conversation: a first message that lacks the
    VRAM gets a clarification, and the follow-up ("12 GB") is merged with
    what is already stored.  Once the stored criteria are complete,
    follow-ups reuse them.  The client is never the source of truth for
    criteria, candidates or scores.

    Raises:
        HTTPException: ``422`` when free-text input is disabled and
            ``mensaje`` is not valid criteria JSON (detail lists the
            validation errors); ``503`` when the message needs the
            language model and Ollama is unavailable or returned invalid
            output — the client can retry with JSON.

    Side effects:
        Persists the ``ConversationRun`` — criteria so far, cumulative
        history and the catalogue ``snapshot_id`` — in SQLite, also on
        clarifications so the follow-up can complete them.
    """
    conversation_id, criterios, historial = _resolver_conversacion(peticion)
    catalogo = app.state.catalogo

    respuesta: RespuestaChat
    candidatos = []
    faltantes = criterios.campos_faltantes()
    if faltantes:
        respuesta = ClarificacionNecesaria(
            conversation_id=conversation_id,
            pregunta=f"Faltan campos obligatorios: {', '.join(faltantes)}. "
            "Proporcione tarea y VRAM en GB.",
            campo_faltante=faltantes[0],
        )
    else:
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

    * Unknown or expired ``conversation_id`` → new conversation: fresh id,
      criteria extracted from ``mensaje``, empty history.
    * Known conversation with complete criteria → reuse them untouched.
    * Known conversation with missing fields → extract from ``mensaje`` and
      merge over the stored criteria (clarification follow-up).

    Raises:
        HTTPException: ``422`` or ``503`` as described in :func:`chat`.
    """
    if peticion.conversation_id is not None:
        existente = conversation_store.cargar(peticion.conversation_id)
        if existente is not None:
            historial = list(existente.historial)
            if not existente.criterios.campos_faltantes():
                return peticion.conversation_id, existente.criterios, historial
            criterios = _extraer(peticion.mensaje, base=existente.criterios)
            return peticion.conversation_id, criterios, historial
    return uuid4(), _extraer(peticion.mensaje, base=None), []


def _extraer(mensaje: str, *, base: CriteriosBusqueda | None) -> CriteriosBusqueda:
    """Run the mounted extractor, mapping its failures to HTTP errors."""
    extractor: ExtractorCriterios = app.state.extractor
    try:
        return extractor.extraer(mensaje, base)
    except MensajeNoEstructuradoError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[
                {"loc": ["body", "mensaje", *e["loc"]], "msg": e["msg"], "type": e["type"]}
                for e in exc.errores
            ],
        ) from exc
    except ExtraccionNoDisponibleError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Free-text extraction is unavailable (local Ollama model not reachable "
                f"or returned invalid output: {exc}). Send the criteria as JSON instead."
            ),
        ) from exc
