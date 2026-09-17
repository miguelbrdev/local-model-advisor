"""FastAPI application — Semana 0 vertical slice.

Endpoints:
- ``POST /chat`` — accept criteria (hardcoded for now), run pipeline,
  return top candidates.
- ``GET /health`` — basic health check.
- ``GET /families`` — list curated families.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI

from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    ClarificacionNecesaria,
    CriteriosBusqueda,
    PeticionChat,
    RespuestaChat,
    RespuestaRecomendacion,
)
from src.state import conversation_store
from src.tools.seed_loader import load_seed

app = FastAPI(title="Local Model Advisor", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/families")
def list_families():
    return [f.model_dump() for f in load_seed()]


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

    # Run the pipeline
    candidatos = ejecutar_pipeline(criterios, top_n=3)

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
