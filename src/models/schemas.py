"""Core domain models for Local Model Advisor.

All models use Pydantic v2. The ranking is 100% deterministic and
structured — no semantic evidence ever affects scores.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

class FamiliaModelo(BaseModel):
    """A curated model family with its variants and metadata.

    Fields marked *sync* are populated during catalogue synchronisation
    from canIRun.ai.  Fields left as ``None``/``[]`` are not available
    in the external source and may be enriched later.
    """

    familia_id: str
    variantes: list[str]
    pipeline_tag: str
    license_declarada: str | None = None
    license_url: str | None = None
    idiomas_detectados: list[str] | None = None
    formatos_disponibles: list[str] = Field(default_factory=list)
    canirun_id: str | None = None  # mapping to canirun.ai model id
    # --- extended fields (populated by sync normaliser) ---
    tareas_soportadas: list[str] = Field(default_factory=list)
    """Internal task identifiers this model qualifies for (e.g. ``["chat", "code"]``)."""
    tags_informativos: list[str] = Field(default_factory=list)
    """Non-task tags preserved from the external source (e.g. ``["multilingual"]``)."""
    nombre_mostrado: str | None = None
    """Human-readable display name from the external source."""
    proveedor: str | None = None
    """Model provider or author."""
    familia_origen: str | None = None
    """External family grouping (e.g. ``"Qwen"``)."""
    parametros_b: float | None = None
    """Parameter count in billions."""
    parametros_texto: str | None = None
    """Parameter count as a human-readable string (e.g. ``"7B"``)."""
    arquitectura: str | None = None
    """Model architecture (``"dense"`` or ``"moe"``)."""
    fecha_lanzamiento: str | None = None
    """Release date in ``YYYY-MM`` format."""
    contexto_maximo: int | None = None
    """Maximum context length in tokens."""


# ---------------------------------------------------------------------------
# VRAM compatibility
# ---------------------------------------------------------------------------

class CompatibilityResult(BaseModel):
    """Result of a VRAM compatibility check against a specific hardware profile."""

    modelo_id: str
    variante: str
    compatible: bool | None = None
    grado: str | None = None  # e.g. "comfortable", "tight"
    fuente: Literal["canirun_ai", "heuristica_propia", "no_disponible"] = "no_disponible"
    nota: str = ""
    tokens_per_second: float | None = None
    vram_required_gb: float | None = None


# ---------------------------------------------------------------------------
# Benchmark evidence
# ---------------------------------------------------------------------------

class EvidenciaBenchmark(BaseModel):
    """Structured benchmark data for a specific checkpoint.

    Only matched by exact HF model ID — never extrapolated between
    checkpoints, quantisations, base/instruct variants, or repos.
    """

    fuente: Literal["openevals", "model_index"]
    benchmark: str
    metrica: str
    valor: float
    cobertura_percent: float | None = None
    nota: str = ""


# ---------------------------------------------------------------------------
# Search criteria
# ---------------------------------------------------------------------------

class CriteriosBusqueda(BaseModel):
    """Criteria extracted from the user's request.

    ``tarea`` and ``vram_gb`` are mandatory for the filter to be useful;
    the rest has sensible defaults.
    """

    tarea: Literal["chat", "codigo", "razonamiento"] | None = (
        None
    )
    vram_gb: float | None = None
    idiomas_requeridos: list[str] = Field(default_factory=list)
    runtime_preferido: str | None = None
    requiere_licencia_comercial: bool = False

    def campos_faltantes(self) -> list[str]:
        """Return names of mandatory fields that are still missing."""
        faltan: list[str] = []
        if self.tarea is None:
            faltan.append("tarea")
        if self.vram_gb is None:
            faltan.append("vram_gb")
        return faltan


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class SubPuntuacion(BaseModel):
    """One dimension of the multi-criteria score."""

    criterio: str
    valor: float = Field(ge=0, le=1)
    peso: float = Field(ge=0, le=1)
    justificacion: str


class CandidatoRankeado(BaseModel):
    """A ranked candidate returned by the pipeline."""

    familia: FamiliaModelo
    variante_recomendada: str
    compatibilidad_vram: CompatibilityResult
    evidencia_benchmark: EvidenciaBenchmark | None = None
    subpuntuaciones: list[SubPuntuacion] = Field(default_factory=list)
    puntuacion_total: float = 0.0


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

class TurnoConversacion(BaseModel):
    rol: Literal["usuario", "sistema"]
    contenido: str


class ConversationRun(BaseModel):
    """Server-side conversation state. The client only ever sends a
    ``conversation_id`` — never criteria, candidates, or scores.
    """

    conversation_id: UUID = Field(default_factory=uuid4)
    snapshot_id: str = ""
    criterios: CriteriosBusqueda = Field(default_factory=CriteriosBusqueda)
    candidatos_elegibles: list[CandidatoRankeado] = Field(default_factory=list)
    historial: list[TurnoConversacion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# API request / response types
# ---------------------------------------------------------------------------

class PeticionChat(BaseModel):
    """Incoming chat request. ``conversation_id`` is ``None`` only for the
    first message of a new conversation.
    """

    conversation_id: UUID | None = None
    mensaje: str


class ClarificacionNecesaria(BaseModel):
    tipo: Literal["clarificacion"] = "clarificacion"
    conversation_id: UUID
    pregunta: str
    campo_faltante: str


class RespuestaRecomendacion(BaseModel):
    tipo: Literal["recomendacion"] = "recomendacion"
    conversation_id: UUID
    criterios_interpretados: CriteriosBusqueda
    candidatos: list[CandidatoRankeado]
    limitaciones: list[str] = Field(default_factory=list)
    datos_actualizados_hace: str = ""


class RespuestaSeguimiento(BaseModel):
    tipo: Literal["seguimiento"] = "seguimiento"
    conversation_id: UUID
    respuesta: str
    datos_estructurados: list[str] = Field(default_factory=list)
    evidencia_explicativa: list[str] = Field(default_factory=list)
    confianza: Literal["alta", "media", "baja"] = "alta"


RespuestaChat = ClarificacionNecesaria | RespuestaRecomendacion | RespuestaSeguimiento
