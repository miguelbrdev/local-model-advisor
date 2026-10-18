"""Deterministic multi-criteria ranking scorer.

Computes a weighted sum of normalised sub-scores.  Every score comes from
structured data only — no semantic evidence, no LLM calls.  Weights are
explicit and documented.  Same input always produces the same ranking.

Inputs read today (and nothing else): ``tareas_soportadas``,
``license_declarada`` and ``idiomas_detectados`` from the family, plus
``vram_required_gb`` from the compatibility result and the user's
available VRAM.  ``documentacion`` remains a placeholder until a real
documentation source is approved.

VRAM policy: *the most capable model that still fits comfortably*.
``ajuste_vram`` is a safety margin (full score up to
``FRACCION_VRAM_UTIL`` of the available memory, falling to zero at 100 %)
and ``capacidad`` rewards using that usable budget, with the model's
estimated VRAM as the capability proxy.  Together they peak around 80 %
usage; the margin weighs more than the capacity so a model at 85 % does
not outrank one at 60 %.  The previous pair (headroom + absolute size)
rewarded being small twice, so a 0.6B model topped every chat query
regardless of the GPU.
"""

from __future__ import annotations

from src.compatibility.vram import FRACCION_VRAM_UTIL
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    FamiliaModelo,
    SubPuntuacion,
)
from src.tools.licencias import es_licencia_permisiva

# Default weights — documented, never hidden.  They sum to 1.0.
PESOS: dict[str, float] = {
    "ajuste_vram": 0.30,
    "capacidad": 0.20,
    "evidencia_codigo": 0.25,
    "documentacion": 0.15,
    "licencia": 0.10,
}

VALOR_NEUTRO = 0.5
"""Sub-score used when the data needed to judge a criterion is missing.
Neutral rather than zero so an unavailable provider never sinks a model."""


def calcular_puntuacion(
    candidato: CandidatoRankeado,
    vram_disponible_gb: float | None = None,
    pesos: dict[str, float] | None = None,
) -> CandidatoRankeado:
    """Compute the total score for a candidate and attach its sub-scores.

    Args:
        candidato: Candidate with its VRAM compatibility already resolved.
            Mutated in place: ``subpuntuaciones`` and ``puntuacion_total``
            are overwritten.
        vram_disponible_gb: VRAM the user has available.  Needed by
            ``ajuste_vram``; when ``None`` or non-positive that criterion
            falls back to :data:`VALOR_NEUTRO`.
        pesos: Weight per criterion; defaults to :data:`PESOS`.  Must
            contain every key of ``PESOS``.

    Returns:
        The same ``candidato`` object, for chaining.
    """
    pesos = pesos or PESOS
    familia = candidato.familia
    vram = candidato.compatibilidad_vram

    subpuntuaciones = [
        _puntuar_ajuste_vram(vram, vram_disponible_gb, pesos["ajuste_vram"]),
        _puntuar_capacidad(vram, vram_disponible_gb, pesos["capacidad"]),
        _puntuar_evidencia_codigo(familia, pesos["evidencia_codigo"]),
        _puntuar_documentacion(familia, pesos["documentacion"]),
        _puntuar_licencia(familia, pesos["licencia"]),
    ]

    candidato.subpuntuaciones = subpuntuaciones
    candidato.puntuacion_total = round(sum(s.valor * s.peso for s in subpuntuaciones), 4)
    return candidato


def ordenar_por_puntuacion(
    candidatos: list[CandidatoRankeado],
) -> list[CandidatoRankeado]:
    """Sort candidates by total score descending (stable for ties)."""
    return sorted(candidatos, key=lambda c: c.puntuacion_total, reverse=True)


# ---------------------------------------------------------------------------
# Sub-criteria
# ---------------------------------------------------------------------------


def _ratio_uso(vram: CompatibilityResult, vram_disponible_gb: float | None) -> float | None:
    """Share of the available VRAM the model needs, or ``None`` if unknown."""
    requerida = vram.vram_required_gb
    if requerida and requerida > 0 and vram_disponible_gb and vram_disponible_gb > 0:
        return requerida / vram_disponible_gb
    return None


def _puntuar_ajuste_vram(
    vram: CompatibilityResult, vram_disponible_gb: float | None, peso: float
) -> SubPuntuacion:
    """Safety margin on the user's GPU.

    Full score while the model uses at most ``FRACCION_VRAM_UTIL`` of the
    available VRAM; from there it falls linearly to 0.0 at 100 %.  Being
    smaller than the comfortable threshold earns nothing extra — that is
    what ``capacidad`` is for.  Neutral when either figure is unknown.
    """
    ratio = _ratio_uso(vram, vram_disponible_gb)
    if ratio is None:
        valor = VALOR_NEUTRO
    elif ratio <= FRACCION_VRAM_UTIL:
        valor = 1.0
    else:
        valor = max(0.0, 1.0 - (ratio - FRACCION_VRAM_UTIL) / (1.0 - FRACCION_VRAM_UTIL))
    return SubPuntuacion(
        criterio="ajuste_vram",
        valor=valor,
        peso=peso,
        justificacion=_justificar_vram(vram, vram_disponible_gb, ratio),
    )


def _puntuar_capacidad(
    vram: CompatibilityResult, vram_disponible_gb: float | None, peso: float
) -> SubPuntuacion:
    """Reward using the usable budget: the closer a model gets to
    ``FRACCION_VRAM_UTIL`` of the available VRAM, the more capable it is
    assumed to be.

    The model's estimated VRAM is the capability proxy — the only
    size-related figure every provider returns.  It is a proxy, not a
    benchmark: a mixture-of-experts model needs memory for all its experts
    while computing with a few, so it reads as larger than it behaves.
    Clamped to 1.0 above the threshold; neutral when unknown.
    """
    ratio = _ratio_uso(vram, vram_disponible_gb)
    if ratio is None:
        valor = VALOR_NEUTRO
        justificacion = f"Estimated VRAM: {vram.vram_required_gb or 'unknown'} GB"
    else:
        valor = min(ratio / FRACCION_VRAM_UTIL, 1.0)
        justificacion = (
            f"Uses {valor:.0%} of the usable budget "
            f"({FRACCION_VRAM_UTIL:.0%} of {vram_disponible_gb:g} GB)"
        )
    return SubPuntuacion(criterio="capacidad", valor=valor, peso=peso, justificacion=justificacion)


def _puntuar_evidencia_codigo(familia: FamiliaModelo, peso: float) -> SubPuntuacion:
    """Reward models the source declares as *dedicated* to code.

    A model is dedicated when ``tareas_soportadas == ["codigo"]`` — mere
    presence of ``codigo`` would give 1.0 to every candidate of a
    ``codigo`` query (all of them declare it after filtering) and make
    the criterion a constant.
    """
    dedicado = familia.tareas_soportadas == ["codigo"]
    if dedicado:
        justificacion = "Dedicated coding model"
    elif "codigo" in familia.tareas_soportadas:
        justificacion = "General-purpose model with code support"
    else:
        justificacion = "No code specialisation declared"
    return SubPuntuacion(
        criterio="evidencia_codigo",
        valor=1.0 if dedicado else 0.3,
        peso=peso,
        justificacion=justificacion,
    )


def _puntuar_documentacion(familia: FamiliaModelo, peso: float) -> SubPuntuacion:
    """Placeholder for documentation quality.

    No documentation source is approved yet, so this only rewards the
    presence of a declared license and a rich language list.  Synced
    families have no languages, so today it is effectively a license
    presence check.
    """
    idiomas = familia.idiomas_detectados or []
    valor = VALOR_NEUTRO
    if familia.license_declarada:
        valor += 0.3
    if len(idiomas) > 3:
        valor += 0.2
    return SubPuntuacion(
        criterio="documentacion",
        valor=min(valor, 1.0),
        peso=peso,
        justificacion=(
            f"License: {familia.license_declarada or 'unknown'}, "
            f"{len(idiomas)} languages detected"
        ),
    )


def _puntuar_licencia(familia: FamiliaModelo, peso: float) -> SubPuntuacion:
    """Permissive license → 1.0, any other declared license → 0.5, none → 0.0."""
    if es_licencia_permisiva(familia.license_declarada):
        valor = 1.0
    elif familia.license_declarada:
        valor = 0.5
    else:
        valor = 0.0
    return SubPuntuacion(
        criterio="licencia",
        valor=valor,
        peso=peso,
        justificacion=f"License: {familia.license_declarada or 'unknown'}",
    )


def _justificar_vram(
    vram: CompatibilityResult, vram_disponible_gb: float | None, ratio: float | None
) -> str:
    disponible = f"{vram_disponible_gb:g} GB" if vram_disponible_gb else "unknown"
    margen = ""
    if ratio is not None:
        holgura = "comfortable" if ratio <= FRACCION_VRAM_UTIL else "tight"
        margen = f" Uses {ratio:.0%} of available VRAM ({holgura} margin)."
    if vram.compatible is True:
        return (
            f"VRAM compatible ({vram.grado}). Requires {vram.vram_required_gb or '?'} GB "
            f"of {disponible} available.{margen}"
        )
    if vram.compatible is False:
        return f"VRAM incompatible: {vram.nota}"
    return f"VRAM status unknown: {vram.nota}{margen}"
