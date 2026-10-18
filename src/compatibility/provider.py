"""VRAM compatibility provider interface and implementations.

The ``VRAMCompatibilityProvider`` protocol defines the contract.
``CanIRunProvider`` calls the canirun.ai public API for chat/code/reasoning
tasks.  If the service is unreachable, it returns a controlled
``CompatibilityResult`` with ``fuente="no_disponible"`` — never raises.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from src.models.schemas import CompatibilityResult, FamiliaModelo


@runtime_checkable
class VRAMCompatibilityProvider(Protocol):
    """Protocol for VRAM compatibility checkers."""

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult: ...


CANIRUN_API_BASE = "https://canirun.ai/api"
CANIRUN_TIMEOUT = 10.0


def _variante_por_defecto(familia: FamiliaModelo, *, ultima: bool = False) -> str:
    """Label of the variant a provider reports when it cannot choose one.

    Synced families carry no nested ``variantes`` (each canIRun.ai record
    is a single checkpoint), so the parameter-count text (``"7B"``) is the
    only meaningful label left; ``"unknown"`` is the last resort.
    """
    if familia.variantes:
        return familia.variantes[-1] if ultima else familia.variantes[0]
    return familia.parametros_texto or "unknown"


class CanIRunProvider:
    """Checks VRAM compatibility via the canirun.ai public API.

    Used for chat, code, and reasoning tasks.  Maps ``familia.canirun_id``
    to the canirun.ai model identifier.  On any failure returns a
    controlled result with ``fuente="no_disponible"``.  ``runtime`` is
    accepted for Protocol compatibility but not sent: the compatibility
    endpoint has no runtime parameter in the observed contract.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        canirun_id = familia.canirun_id
        if not canirun_id:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=_variante_por_defecto(familia),
                compatible=None,
                fuente="no_disponible",
                nota="No canirun_id mapping configured for this family.",
            )

        variante = _variante_por_defecto(familia, ultima=True)
        try:
            response = httpx.post(
                f"{CANIRUN_API_BASE}/compatibility",
                json={
                    "hardware": {"ramGb": int(vram_gb * 2), "gpu": {"name": "Custom"}},
                    "modelId": canirun_id,
                    "quantization": "Q4_K_M",
                },
                timeout=CANIRUN_TIMEOUT,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
            estimado = data.get("estimated", {})

            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=data.get("compatible"),
                grado=data.get("status"),
                fuente="canirun_ai",
                nota="; ".join(data.get("notes", [])),
                tokens_per_second=estimado.get("tokensPerSecond"),
                vram_required_gb=estimado.get("vramRequiredGb"),
            )
        except Exception as exc:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=None,
                fuente="no_disponible",
                nota=f"canirun.ai API error: {exc}",
            )


class TamanoDirectoProvider:
    """Fallback heuristic for small models (embeddings/rerankers).

    These models are small (hundreds of MB to ~2 GB) and rarely VRAM-bound,
    so they are reported as compatible without an external call.  Not
    reachable from the v1 API, whose tasks all route to ``CanIRunProvider``.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=_variante_por_defecto(familia),
            compatible=True,
            grado="runs_well",
            fuente="heuristica_propia",
            nota="Size-based heuristic for small models (embeddings/rerankers).",
        )


def elegir_provider(tarea: str) -> VRAMCompatibilityProvider:
    """Route to the right VRAM provider based on task category."""
    if tarea in ("chat", "codigo", "razonamiento"):
        return CanIRunProvider()
    return TamanoDirectoProvider()
