"""Offline VRAM verdict from the per-quantisation table synced from canIRun.ai.

The canIRun.ai detail record lists, for every model, the memory each
quantisation needs (``quants[]``: ``Q2_K`` … ``F16`` with ``vramGB``).  The
catalogue synchronisation stores that table in
``FamiliaModelo.cuantizaciones``, so the compatibility check no longer
needs a network call per candidate: the verdict is read from the snapshot,
is identical on every run (golden evaluation included) and comes from the
same source the live endpoint used.

The verdict is taken at :data:`CUANTIZACION_REFERENCIA` (``Q4_K_M``), the
quantisation the live check always asked for, so ``vram_required_gb`` keeps
the meaning the scorer relies on.  The note also names the best
quantisation that still fits comfortably, which is what a user should
actually download.

Families without a table (not yet re-synced, or absent from the detail
endpoint) get ``no_disponible`` so the caller falls back to another source.
"""

from __future__ import annotations

from src.compatibility.vram import FRACCION_VRAM_UTIL
from src.models.schemas import CompatibilityResult, Cuantizacion, FamiliaModelo

CUANTIZACION_REFERENCIA = "Q4_K_M"
"""Quantisation the verdict is computed at (the one the live canIRun.ai
check requested); its ``vramGB`` becomes ``vram_required_gb``."""

FUENTE = "canirun_snapshot"


class CuantizacionesProvider:
    """``VRAMCompatibilityProvider`` backed by the synced quantisation table.

    Never raises and performs no I/O.  ``runtime`` is accepted for the
    Protocol and ignored: the table does not depend on the runtime.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        variante = familia.parametros_texto or "unknown"
        referencia = _buscar(familia.cuantizaciones, CUANTIZACION_REFERENCIA)
        if referencia is None:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=None,
                fuente="no_disponible",
                nota=f"No {CUANTIZACION_REFERENCIA} entry in the synced quantisation table.",
            )
        requerida = referencia.vram_gb
        if vram_gb <= 0:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=variante,
                compatible=None,
                fuente=FUENTE,
                nota="Available VRAM not provided.",
                vram_required_gb=requerida,
            )

        if requerida <= vram_gb * FRACCION_VRAM_UTIL:
            compatible, grado = True, "comfortable"
        elif requerida <= vram_gb:
            compatible, grado = True, "tight"
        else:
            compatible, grado = False, "insufficient"

        nota = (
            f"{CUANTIZACION_REFERENCIA} needs {requerida:g} GB of {vram_gb:g} GB available "
            f"(canIRun.ai quantisation table)."
        )
        recomendada = recomendar_cuantizacion(familia.cuantizaciones, vram_gb)
        if recomendada is not None and recomendada.nombre != CUANTIZACION_REFERENCIA:
            nota += (
                f" Best comfortable fit: {recomendada.nombre} ({recomendada.vram_gb:g} GB, "
                f"{recomendada.calidad or 'quality unknown'})."
            )
        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=variante,
            compatible=compatible,
            grado=grado,
            fuente=FUENTE,
            nota=nota,
            vram_required_gb=requerida,
        )


def recomendar_cuantizacion(
    cuantizaciones: list[Cuantizacion], vram_gb: float
) -> Cuantizacion | None:
    """Highest-precision quantisation that fits within the comfortable budget.

    Ties on bits are broken by the larger memory footprint (the less
    aggressive variant).  ``None`` when nothing fits comfortably.
    """
    presupuesto = vram_gb * FRACCION_VRAM_UTIL
    aptas = [c for c in cuantizaciones if c.vram_gb is not None and c.vram_gb <= presupuesto]
    if not aptas:
        return None
    return max(aptas, key=lambda c: (c.bits or 0, c.vram_gb or 0))


def _buscar(cuantizaciones: list[Cuantizacion], nombre: str) -> Cuantizacion | None:
    for c in cuantizaciones:
        if c.nombre == nombre and c.vram_gb is not None and c.vram_gb > 0:
            return c
    return None
