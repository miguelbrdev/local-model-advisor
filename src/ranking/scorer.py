"""Deterministic multi-criteria ranking scorer.

Computes a weighted sum of normalised sub-scores.  Every score comes from
structured data only — no semantic evidence, no LLM calls.  Weights are
explicit and documented.  Same input always produces the same ranking.

Inputs read today (and nothing else): ``tareas_soportadas``,
``license_declarada``, ``fecha_lanzamiento`` and ``contexto_maximo`` from
the family, ``vram_required_gb`` from the compatibility result, and the
user's available VRAM and requested task.  All of them come from the
synced source (canIRun.ai exposes ``releaseDate`` and ``contextLength``);
nothing is inferred from model names or benchmarks.

Criteria:

* ``ajuste_vram`` / ``capacidad`` — the VRAM policy (below).
* ``especializacion`` — task-aware: for ``codigo`` and ``razonamiento`` a
  model dedicated to that task scores 1.0, a generalist 0.3; for ``chat``
  there is no meaningful specialisation signal (chat-only models are the
  small generalists), so every candidate gets the neutral value.  It
  replaces the former ``evidencia_codigo``, which was a constant for every
  non-code query and wasted a quarter of the weight there.
* ``recencia`` — months since release relative to the newest model in the
  catalogue, linear to 0 at 24 months.  Newer generations dominate older
  ones at equal size, and the source dates every record.
* ``contexto`` — log scale of the context window from 4k (0.0) to 1M
  (1.0): 32k ≈ 0.38, 128k ≈ 0.63, 256k ≈ 0.75.
* ``licencia`` — permissive / other / none.

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

import math

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
    "especializacion": 0.20,
    "recencia": 0.12,
    "contexto": 0.08,
    "licencia": 0.10,
}

MESES_RECENCIA = 24
"""Age at which ``recencia`` reaches 0.0."""

CONTEXTO_MINIMO = 4_096
CONTEXTO_MAXIMO = 1_048_576
"""Context-window bounds of the ``contexto`` log scale."""

TAREAS_CON_ESPECIALIZACION = frozenset({"codigo", "razonamiento"})

VALOR_NEUTRO = 0.5
"""Sub-score used when the data needed to judge a criterion is missing.
Neutral rather than zero so an unavailable provider never sinks a model."""


def calcular_puntuacion(
    candidato: CandidatoRankeado,
    vram_disponible_gb: float | None = None,
    pesos: dict[str, float] | None = None,
    *,
    tarea: str | None = None,
    fecha_referencia: str | None = None,
) -> CandidatoRankeado:
    """Compute the total score for a candidate and attach its sub-scores.

    Args:
        candidato: Candidate with its VRAM compatibility already resolved.
            Mutated in place: ``subpuntuaciones`` and ``puntuacion_total``
            are overwritten.
        vram_disponible_gb: VRAM the user has available.  Needed by
            ``ajuste_vram`` and ``capacidad``; when ``None`` or
            non-positive they fall back to :data:`VALOR_NEUTRO`.
        pesos: Weight per criterion; defaults to :data:`PESOS`.  Must
            contain every key of ``PESOS``.
        tarea: Requested task, for ``especializacion``.  ``None`` or
            ``chat`` yields the neutral value.
        fecha_referencia: ``YYYY-MM`` of the newest model in the catalogue,
            the zero point of ``recencia``; ``None`` makes it neutral.

    Returns:
        The same ``candidato`` object, for chaining.
    """
    pesos = pesos or PESOS
    familia = candidato.familia
    vram = candidato.compatibilidad_vram

    subpuntuaciones = [
        _puntuar_ajuste_vram(vram, vram_disponible_gb, pesos["ajuste_vram"]),
        _puntuar_capacidad(vram, vram_disponible_gb, pesos["capacidad"]),
        _puntuar_especializacion(familia, tarea, pesos["especializacion"]),
        _puntuar_recencia(familia, fecha_referencia, pesos["recencia"]),
        _puntuar_contexto(familia, pesos["contexto"]),
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


def _puntuar_especializacion(
    familia: FamiliaModelo, tarea: str | None, peso: float
) -> SubPuntuacion:
    """Reward models the source declares as *dedicated* to the requested task.

    Dedicated means ``tareas_soportadas == [tarea]``; mere presence of the
    task would give 1.0 to every candidate (all of them declare it after
    filtering) and make the criterion a constant.  Only ``codigo`` and
    ``razonamiento`` carry a meaningful signal; for ``chat`` (or no task)
    every candidate gets :data:`VALOR_NEUTRO`.
    """
    if tarea not in TAREAS_CON_ESPECIALIZACION:
        return SubPuntuacion(
            criterio="especializacion",
            valor=VALOR_NEUTRO,
            peso=peso,
            justificacion=f"No specialisation signal for task {tarea or 'unknown'!r}",
        )
    dedicado = familia.tareas_soportadas == [tarea]
    return SubPuntuacion(
        criterio="especializacion",
        valor=1.0 if dedicado else 0.3,
        peso=peso,
        justificacion=(
            f"Dedicated {tarea} model" if dedicado
            else f"General-purpose model that also supports {tarea}"
        ),
    )


def meses_entre(desde: str, hasta: str) -> int | None:
    """Whole months from ``desde`` to ``hasta`` (``YYYY-MM``), or ``None`` if unparsable."""
    try:
        a1, m1 = (int(x) for x in desde.split("-")[:2])
        a2, m2 = (int(x) for x in hasta.split("-")[:2])
    except (ValueError, AttributeError):
        return None
    return (a2 - a1) * 12 + (m2 - m1)


def _puntuar_recencia(
    familia: FamiliaModelo, fecha_referencia: str | None, peso: float
) -> SubPuntuacion:
    """Linear decay from 1.0 (released with the newest model) to 0.0 at
    :data:`MESES_RECENCIA` months older.  Neutral without dates."""
    meses = (
        meses_entre(familia.fecha_lanzamiento, fecha_referencia)
        if familia.fecha_lanzamiento and fecha_referencia
        else None
    )
    if meses is None:
        valor = VALOR_NEUTRO
        justificacion = "Release date unknown"
    else:
        valor = max(0.0, min(1.0, 1.0 - max(meses, 0) / MESES_RECENCIA))
        justificacion = (
            f"Released {familia.fecha_lanzamiento}, {max(meses, 0)} months after the "
            f"newest model ({fecha_referencia})"
            if meses > 0 else f"Released {familia.fecha_lanzamiento} (newest generation)"
        )
    return SubPuntuacion(
        criterio="recencia", valor=valor, peso=peso, justificacion=justificacion
    )


def _puntuar_contexto(familia: FamiliaModelo, peso: float) -> SubPuntuacion:
    """Log scale of the context window between :data:`CONTEXTO_MINIMO` and
    :data:`CONTEXTO_MAXIMO`.  Neutral when unknown."""
    contexto = familia.contexto_maximo
    if not contexto or contexto <= 0:
        return SubPuntuacion(
            criterio="contexto", valor=VALOR_NEUTRO, peso=peso,
            justificacion="Context length unknown",
        )
    acotado = min(max(contexto, CONTEXTO_MINIMO), CONTEXTO_MAXIMO)
    valor = math.log2(acotado / CONTEXTO_MINIMO) / math.log2(CONTEXTO_MAXIMO / CONTEXTO_MINIMO)
    return SubPuntuacion(
        criterio="contexto",
        valor=round(valor, 4),
        peso=peso,
        justificacion=f"Context window {contexto:,} tokens",
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
