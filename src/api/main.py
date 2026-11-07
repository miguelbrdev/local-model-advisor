"""FastAPI application.

Endpoints:
- ``POST /chat`` — accept criteria as JSON or as free text in ``mensaje``,
  run the pipeline, return the top candidates or a clarification request.
- ``GET /health`` — basic health check.
- ``GET /families`` — list the families of the mounted catalogue snapshot.

Configuration (environment variables, read once at startup):
- ``LMA_EXTRACCION_LLM``: ``"0"``/``"false"`` disables free-text input;
  only criteria JSON is then accepted (default: enabled).
- ``LMA_OLLAMA_MODEL``: Ollama model tag (default ``gemma4:e4b``).
- ``LMA_OLLAMA_BASE_URL``: Ollama server (default ``http://127.0.0.1:11434``).
- ``LMA_EVIDENCIA``: ``"0"``/``"false"`` disables explanatory evidence
  (default: enabled).
- ``LMA_QDRANT_URL``: Qdrant server (default ``http://127.0.0.1:6333``).
- ``LMA_EMBEDDING_MODEL``: Ollama embedding model (default ``embeddinggemma``).
- ``LMA_CALENTAR_MODELOS``: ``"0"`` skips loading the Ollama models in the
  background at startup (default: enabled).
- ``LMA_API_KEY``: when set, ``/chat`` and ``/families`` require the same
  value in the ``X-API-Key`` header (default: open, for localhost use).
- ``LMA_RATE_LIMIT_POR_MINUTO``: ``/chat`` requests per client address per
  minute, ``0`` disables (default 60).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status

from src.api.limites import RATE_LIMIT_POR_DEFECTO, LimitadorTasa, api_key_valida
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
from src.evidence.embeddings import DEFAULT_EMBEDDING_MODEL, crear_embedder_ollama
from src.evidence.enriquecer import (
    BuscadorEvidencia,
    ResultadoEnriquecimiento,
    consulta_para,
    enriquecer_candidatos,
)
from src.evidence.qdrant_index import DEFAULT_QDRANT_URL, IndiceEvidencia
from src.models.schemas import (
    CandidatoRankeado,
    ClarificacionNecesaria,
    ConversationRun,
    CriteriosBusqueda,
    PeticionChat,
    RespuestaChat,
    RespuestaRecomendacion,
    TurnoConversacion,
)
from src.state import conversation_store

logger = logging.getLogger(__name__)

TOP_N_RECOMENDACIONES = 3

DIAS_OBSOLESCENCIA_CATALOGO = 28
"""A snapshot older than this (21-day sync cadence plus a week of slack) is
reported as stale in ``/health`` and in every recommendation."""

CONVERSACIONES_ENTRE_LIMPIEZAS = 100
"""Expired conversations are purged at startup and every this many chats."""

ETIQUETAS_CAMPOS = {"tarea": "la tarea (chat, codigo o razonamiento)", "vram_gb": "la VRAM en GB"}


def pregunta_aclaracion(criterios: CriteriosBusqueda, faltantes: list[str]) -> str:
    """Deterministic clarification text: what was understood, what is missing.

    Saying what *was* understood lets the user answer with just the missing
    value ("12 GB") instead of repeating the whole request, which is exactly
    what the follow-up merge supports.
    """
    entendido = []
    if criterios.tarea:
        entendido.append(f"tarea={criterios.tarea}")
    if criterios.vram_gb is not None:
        entendido.append(f"vram_gb={criterios.vram_gb:g}")
    if criterios.idiomas_requeridos:
        entendido.append("idiomas=" + ",".join(criterios.idiomas_requeridos))
    prefijo = f"Entendido: {', '.join(entendido)}. " if entendido else ""
    pedidos = " y ".join(ETIQUETAS_CAMPOS.get(campo, campo) for campo in faltantes)
    return f"{prefijo}Falta indicar {pedidos}."

LIMITACIONES_BASE = [
    "Benchmark evidence not yet available (OpenEvals pending).",
]
"""Limitations that apply to every recommendation today."""


def limitaciones_para(
    criterios: CriteriosBusqueda,
    candidatos: list[CandidatoRankeado] | None = None,
    evidencia: ResultadoEnriquecimiento | None = None,
) -> list[str]:
    """Limitations to report for *this* request.

    Always the base list, plus one line for each criterion the user did
    state that the pipeline cannot honour yet — so the response says
    exactly which of the user's wishes were not applied instead of a
    generic disclaimer.  The catalogue source exposes no language or
    runtime data, so ``idiomas_requeridos`` cannot be enforced on synced
    models and ``runtime_preferido`` is not used at all.  When some
    candidates carry a locally estimated VRAM verdict (canirun.ai down or
    without an entry), a line says how many, so measured and estimated
    results are never confused.  ``evidencia`` describes the explanatory
    evidence lookup: ``None`` means it is disabled, an error means the
    index was unreachable, and candidates without fragments (gated or
    unindexed documentation) are counted.
    """
    limitaciones = list(LIMITACIONES_BASE)
    dias = dias_desde(getattr(app.state, "catalogo", None))
    if dias is not None and dias > DIAS_OBSOLESCENCIA_CATALOGO:
        limitaciones.append(
            f"Catalogue snapshot is {dias} days old (synchronisation expected every 21 days); "
            "newer models may be missing."
        )
    if evidencia is None:
        limitaciones.append("Explanatory evidence is disabled (LMA_EVIDENCIA=0).")
    elif evidencia.error is not None:
        limitaciones.append(
            f"Explanatory evidence unavailable: {evidencia.error}. Candidates are shown "
            "without documentation fragments."
        )
    elif evidencia.sin_evidencia:
        limitaciones.append(
            f"{evidencia.sin_evidencia} of {evidencia.con_evidencia + evidencia.sin_evidencia} "
            "candidates have no documentation fragments (README gated, missing or not indexed)."
        )
    estimados = sum(
        1 for c in candidatos or [] if c.compatibilidad_vram.fuente == "heuristica_propia"
    )
    if estimados:
        limitaciones.append(
            f"VRAM compatibility of {estimados} of {len(candidatos or [])} candidates was "
            "estimated locally from parameter counts (canirun.ai unavailable or without an "
            "entry); see compatibilidad_vram.fuente per candidate."
        )
    if criterios.idiomas_requeridos:
        limitaciones.append(
            f"Languages {criterios.idiomas_requeridos} could not be verified: the catalogue "
            "source exposes no language data, so every candidate is shown."
        )
    if criterios.runtime_preferido:
        limitaciones.append(
            f"runtime_preferido {criterios.runtime_preferido!r} was not applied: the catalogue "
            "source exposes no runtime data."
        )
    return limitaciones


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


def construir_buscador_evidencia(env: dict[str, str] | None = None) -> BuscadorEvidencia | None:
    """Build the explanatory-evidence index client from the environment.

    Returns ``None`` when ``LMA_EVIDENCIA`` disables it.  Construction
    performs no I/O: Qdrant and the embedding model are contacted only when
    a recommendation is produced, and failures then become a limitation
    line, never an error response.
    """
    fuente = os.environ if env is None else env
    habilitado = fuente.get("LMA_EVIDENCIA", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    if not habilitado:
        return None
    from qdrant_client import QdrantClient

    cliente = QdrantClient(
        url=fuente.get("LMA_QDRANT_URL", DEFAULT_QDRANT_URL),
        timeout=5,
        check_compatibility=False,  # no version probe at startup: construction stays I/O-free
    )
    embedder = crear_embedder_ollama(
        fuente.get("LMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        fuente.get("LMA_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    return IndiceEvidencia(cliente, embedder)


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
    app.state.evidencia = construir_buscador_evidencia()
    app.state.limitador = LimitadorTasa(
        _entero_env("LMA_RATE_LIMIT_POR_MINUTO", RATE_LIMIT_POR_DEFECTO)
    )
    app.state.api_key = os.environ.get("LMA_API_KEY") or None
    app.state.chats_desde_limpieza = 0
    app.state.modelos_listos = None
    _limpiar_conversaciones("at startup")
    calentar = os.environ.get("LMA_CALENTAR_MODELOS", "1").strip().lower()
    if calentar not in {"0", "false", "no", "off"}:
        threading.Thread(target=_calentar_en_segundo_plano, name="lma-warmup", daemon=True).start()
    yield


def _calentar_en_segundo_plano() -> None:
    app.state.modelos_listos = calentar_modelos(app.state.extractor, app.state.evidencia)


def _entero_env(nombre: str, por_defecto: int) -> int:
    try:
        return int(os.environ.get(nombre, por_defecto))
    except ValueError:
        logger.warning("Ignoring non-integer %s; using %d.", nombre, por_defecto)
        return por_defecto


def _limpiar_conversaciones(momento: str) -> None:
    try:
        borradas = conversation_store.limpiar_expiradas()
        if borradas:
            logger.info("Removed %d expired conversations %s.", borradas, momento)
    except Exception as exc:  # a broken store must not block serving
        logger.warning("Could not clean expired conversations: %s", exc)


def exigir_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Reject the request when ``LMA_API_KEY`` is set and the header does not match."""
    if not api_key_valida(getattr(app.state, "api_key", None), x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key"
        )


def limitar_tasa(request: Request, response: Response) -> None:
    """Per-client fixed-window limit on ``/chat``; 429 with ``Retry-After`` when exceeded."""
    limitador: LimitadorTasa | None = getattr(app.state, "limitador", None)
    if limitador is None:
        return
    clave = request.client.host if request.client else "unknown"
    admitida, espera = limitador.admitir(clave)
    if not admitida:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for this client; retry later.",
            headers={"Retry-After": str(int(espera) + 1)},
        )


app = FastAPI(title="Local Model Advisor", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness plus what this process is serving with.

    ``status`` stays ``"ok"`` for existing probes; the other keys say which
    catalogue snapshot is mounted and whether free-text input and
    explanatory evidence are enabled (not whether their backends are up:
    that is only known per request and reported in ``limitaciones``).
    """
    catalogo = getattr(app.state, "catalogo", None)
    extractor = getattr(app.state, "extractor", None)
    fecha = catalogo.fecha_sincronizacion if catalogo is not None else None
    dias = dias_desde(catalogo)
    return {
        "status": "ok",
        "snapshot_id": catalogo.snapshot_id if catalogo is not None else None,
        "catalogo_sincronizado": fecha.isoformat() if fecha is not None else None,
        "catalogo_dias": dias,
        "catalogo_obsoleto": dias is not None and dias > DIAS_OBSOLESCENCIA_CATALOGO,
        "texto_libre": bool(extractor is not None and extractor.acepta_texto_libre),
        "evidencia": getattr(app.state, "evidencia", None) is not None,
        "modelos_listos": getattr(app.state, "modelos_listos", None),
        "api_key_requerida": bool(getattr(app.state, "api_key", None)),
    }


@app.get("/families", dependencies=[Depends(exigir_api_key)])
def list_families():
    """List every family of the mounted catalogue snapshot.

    Returns:
        One plain-dict serialisation per ``FamiliaModelo``, identical in
        shape to the legacy seed endpoint (no ``response_model``: the
        shape is pinned by tests against ``FamiliaModelo`` itself).
    """
    return [f.model_dump() for f in app.state.catalogo.cargar()]


@app.post(
    "/chat",
    response_model=RespuestaChat,
    dependencies=[Depends(exigir_api_key), Depends(limitar_tasa)],
)
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
    app.state.chats_desde_limpieza += 1
    if app.state.chats_desde_limpieza >= CONVERSACIONES_ENTRE_LIMPIEZAS:
        app.state.chats_desde_limpieza = 0
        _limpiar_conversaciones("periodically")

    conversation_id, criterios, historial = _resolver_conversacion(peticion)
    catalogo = app.state.catalogo

    respuesta: RespuestaChat
    candidatos = []
    faltantes = criterios.campos_faltantes()
    if faltantes:
        respuesta = ClarificacionNecesaria(
            conversation_id=conversation_id,
            pregunta=pregunta_aclaracion(criterios, faltantes),
            campo_faltante=faltantes[0],
        )
    else:
        candidatos = ejecutar_pipeline(criterios, catalogo=catalogo, top_n=TOP_N_RECOMENDACIONES)
        buscador: BuscadorEvidencia | None = app.state.evidencia
        evidencia = None
        if buscador is not None:
            # After ranking, on purpose: evidence explains, it never scores.
            evidencia = enriquecer_candidatos(
                candidatos, consulta_para(criterios, peticion.mensaje), buscador
            )
        respuesta = RespuestaRecomendacion(
            conversation_id=conversation_id,
            criterios_interpretados=criterios,
            candidatos=candidatos,
            limitaciones=limitaciones_para(criterios, candidatos, evidencia),
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


def dias_desde(catalogo, ahora: datetime | None = None) -> int | None:
    """Age in days of the mounted catalogue snapshot, or ``None`` if unknown."""
    fecha = getattr(catalogo, "fecha_sincronizacion", None) if catalogo is not None else None
    if fecha is None:
        return None
    return max(0, ((ahora or datetime.now(UTC)) - fecha).days)


def calentar_modelos(extractor, evidencia) -> dict[str, bool | None]:
    """Load the local models so the first user request is fast.

    Runs the chat model first (the big one), then the embedding model.
    Each step is independent and never raises; the result says what is
    ready.  Intended to run in a background thread from ``lifespan``.
    """
    resultado: dict[str, bool | None] = {"extractor": None, "evidencia": None}
    for nombre, componente in (("extractor", extractor), ("evidencia", evidencia)):
        calentar = getattr(componente, "calentar", None)
        if not callable(calentar):
            continue
        inicio = time.perf_counter()
        try:
            resultado[nombre] = bool(calentar())
        except Exception as exc:  # a warm-up must never take the API down
            logger.warning("Warm-up of %s raised: %s", nombre, exc)
            resultado[nombre] = False
        logger.info(
            "Warm-up %s: %s in %.1f s", nombre,
            "ready" if resultado[nombre] else "not ready", time.perf_counter() - inicio,
        )
    return resultado


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
