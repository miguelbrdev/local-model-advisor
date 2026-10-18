"""Deterministic multi-criteria ranking scorer.

Computes a weighted sum of normalised sub-scores.  Every score comes from
structured data only — no semantic evidence, no LLM calls.  Weights are
explicit and documented.  Same input always produces the same ranking.

Inputs read today (and nothing else): ``tareas_soportadas``,
``license_declarada`` and ``idiomas_detectados`` from the family, plus
``vram_required_gb`` from the compatibility result and the user's
available VRAM.  ``documentacion`` remains a placeholder until a real
documentation source is approved.
"""

from __future__ import annotations

from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    FamiliaModelo,
    SubPuntuacion,
)
from src.tools.licencias import es_licencia_permisiva

# Default weights — documented, never hidden.  They sum to 1.0.
PESOS: dict[str, float] = {
    "ajuste_vram": 0.35,
    "evidencia_codigo": 0.25,
    "documentacion": 0.15,
    "tamanio": 0.15,
    "licencia": 0.10,
}

VALOR_NEUTRO = 0.5
"""Sub-score used when the data needed to judge a criterion is missing.
Neutral rather than zero so an unavailable provider never sinks a model."""

VRAM_REFERENCIA_GB = 32.0
"""Upper bound of the ``tamanio`` scale: a model needing this much VRAM or
more scores 0.0 on size efficiency.  Chosen as the largest consumer GPU
memory commonly found in local setups."""


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
        _puntuar_evidencia_codigo(familia, pesos["evidencia_codigo"]),
        _puntuar_documentacion(familia, pesos["documentacion"]),
        _puntuar_tamanio(vram, pesos["tamanio"]),
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


def _puntuar_ajuste_vram(
    vram: CompatibilityResult, vram_disponible_gb: float | None, peso: float
) -> SubPuntuacion:
    """Headroom left on the user's GPU: the smaller the share of available
    VRAM the model needs, the higher the score.

    ``1 - required / available``, clamped to ``[0, 1]``.  Neutral when
    either figure is unknown — the previous formula divided GB by
    tokens-per-second, which mixed units and rewarded fast models for
    being fast rather than for fitting.
    """
    requerida = vram.vram_required_gb
    if requerida and requerida > 0 and vram_disponible_gb and vram_disponible_gb > 0:
        valor = max(0.0, 1.0 - min(requerida / vram_disponible_gb, 1.0))
    else:
        valor = VALOR_NEUTRO
    return SubPuntuacion(
        criterio="ajuste_vram",
        valor=valor,
        peso=peso,
        justificacion=_justificar_vram(vram, vram_disponible_gb),
    )


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


def _puntuar_tamanio(vram: CompatibilityResult, peso: float) -> SubPuntuacion:
    """Size efficiency on an absolute scale: smaller models are preferred
    for local use regardless of the user's GPU.  Derived from the VRAM
    the provider estimates, not from the parameter count."""
    requerida = vram.vram_required_gb
    if requerida and requerida > 0:
        valor = max(0.0, 1.0 - requerida / VRAM_REFERENCIA_GB)
    else:
        valor = VALOR_NEUTRO
    return SubPuntuacion(
        criterio="tamanio",
        valor=valor,
        peso=peso,
        justificacion=f"Estimated VRAM: {requerida or 'unknown'} GB",
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


def _justificar_vram(vram: CompatibilityResult, vram_disponible_gb: float | None) -> str:
    disponible = f"{vram_disponible_gb:g} GB" if vram_disponible_gb else "unknown"
    if vram.compatible is True:
        return (
            f"VRAM compatible ({vram.grado}). Requires {vram.vram_required_gb or '?'} GB "
            f"of {disponible} available. Est. {vram.tokens_per_second or '?'} tok/s."
        )
    if vram.compatible is False:
        return f"VRAM incompatible: {vram.nota}"
    return f"VRAM status unknown: {vram.nota}"
