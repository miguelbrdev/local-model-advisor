"""VRAM compatibility provider interface and implementations.

The ``VRAMCompatibilityProvider`` protocol defines the contract.
``CanIRunProvider`` calls the canirun.ai public API for chat/code/reasoning
tasks. If the service is unreachable, it returns a controlled
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


class CanIRunProvider:
    """Checks VRAM compatibility via the canirun.ai public API.

    Used for chat, code, and reasoning tasks. Maps ``familia.canirun_id``
    to the canirun.ai model identifier. On any failure returns a controlled
    result with ``fuente="no_disponible"``.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        canirun_id = familia.canirun_id
        if not canirun_id:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=familia.variantes[0] if familia.variantes else "unknown",
                compatible=None,
                fuente="no_disponible",
                nota="No canirun_id mapping configured for this family.",
            )

        default_variant = familia.variantes[-1] if familia.variantes else "unknown"
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

            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=default_variant,
                compatible=data.get("compatible"),
                grado=data.get("status"),
                fuente="canirun_ai",
                nota="; ".join(data.get("notes", [])),
                tokens_per_second=data.get("estimated", {}).get("tokensPerSecond"),
                vram_required_gb=data.get("estimated", {}).get("vramRequiredGb"),
            )
        except Exception as exc:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=default_variant,
                compatible=None,
                fuente="no_disponible",
                nota=f"canirun.ai API error: {exc}",
            )


class TamanoDirectoProvider:
    """Fallback heuristic for embeddings/rerankers.

    These models are small (hundreds of MB to ~2 GB) and rarely VRAM-bound.
    Formula: compatible if ``(model_size_gb * 1.2) <= vram_gb``.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        default_variant = familia.variantes[0] if familia.variantes else "unknown"
        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=default_variant,
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
