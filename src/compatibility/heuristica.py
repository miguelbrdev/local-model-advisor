"""Deterministic, offline VRAM heuristic based on parameter count.

``HeuristicaParametrosProvider`` estimates the memory a model needs from
``FamiliaModelo.parametros_b`` alone, with no network access and no
randomness, so the same catalogue and criteria always yield the same
compatibility verdicts.  It exists for two uses:

* **Golden-query evaluation** (``scripts/evaluate_golden.py``): the
  deterministic part of the pipeline (filter + ranking) can only be pinned
  with a compatibility oracle that never changes between runs.
* **Future fallback** when canirun.ai is unreachable, as anticipated in the
  README; it is *not* wired into production routing yet.

Every result is labelled ``fuente="heuristica_propia"`` so a consumer can
always tell an estimate from a measured canirun.ai answer.

Formula (documented, not tuned to any benchmark)::

    vram_required_gb = parametros_b * GB_POR_B_PARAMETROS_Q4 + SOBRECARGA_GB

``GB_POR_B_PARAMETROS_Q4`` approximates a 4-bit quantisation (Q4_K_M
averages ~4.5–4.8 bits per weight, i.e. ~0.56–0.6 GB per billion
parameters); ``SOBRECARGA_GB`` covers KV cache and runtime buffers at a
modest context length.  Mixture-of-experts models count *all* parameters
because every expert has to be resident in memory even if only a few are
active per token.
"""

from __future__ import annotations

from src.compatibility.vram import FRACCION_VRAM_UTIL
from src.models.schemas import CompatibilityResult, FamiliaModelo

GB_POR_B_PARAMETROS_Q4 = 0.6
"""Estimated GB of weights per billion parameters at ~4-bit quantisation."""

SOBRECARGA_GB = 1.0
"""Fixed allowance for KV cache and runtime buffers."""

UMBRAL_HOLGADO = FRACCION_VRAM_UTIL
"""A model using at most this share of the available VRAM is ``comfortable``;
above it (but still fitting) it is ``tight``.  Shared with the scorer."""


def estimar_vram_gb(parametros_b: float) -> float:
    """Estimated VRAM in GB for a model with ``parametros_b`` billion parameters."""
    return round(parametros_b * GB_POR_B_PARAMETROS_Q4 + SOBRECARGA_GB, 1)


class HeuristicaParametrosProvider:
    """Parameter-count VRAM heuristic satisfying ``VRAMCompatibilityProvider``.

    Never raises.  ``runtime`` is accepted for Protocol compatibility and
    ignored: the heuristic has no runtime-specific data.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        variante = familia.parametros_texto or "unknown"
        if familia.parametros_b is None or familia.parametros_b <= 0:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=None,
                fuente="heuristica_propia",
                nota="Parameter count unknown; VRAM cannot be estimated.",
            )

        requerida = estimar_vram_gb(familia.parametros_b)
        if vram_gb <= 0:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=None,
                fuente="heuristica_propia",
                nota="Available VRAM not provided.",
                vram_required_gb=requerida,
            )

        if requerida <= vram_gb * UMBRAL_HOLGADO:
            compatible, grado = True, "comfortable"
        elif requerida <= vram_gb:
            compatible, grado = True, "tight"
        else:
            compatible, grado = False, "insufficient"

        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=variante,
            compatible=compatible,
            grado=grado,
            fuente="heuristica_propia",
            nota=(
                f"Estimated from {familia.parametros_b:g}B parameters at ~4-bit: "
                f"{requerida} GB of {vram_gb:g} GB available."
            ),
            vram_required_gb=requerida,
        )
