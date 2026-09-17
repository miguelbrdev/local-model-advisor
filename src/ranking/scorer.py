"""Deterministic multi-criteria ranking scorer.

Computes a weighted sum of normalised sub-scores. Every score comes from
structured data only — no semantic evidence, no LLM calls. Weights are
explicit and documented. Same input always produces the same ranking.
"""

from __future__ import annotations

from src.models.schemas import CandidatoRankeado, SubPuntuacion

# Default weights — documented, never hidden
PESOS = {
    "ajuste_vram": 0.35,
    "evidencia_codigo": 0.25,
    "documentacion": 0.15,
    "tamanio": 0.15,
    "licencia": 0.10,
}


def calcular_puntuacion(
    candidato: CandidatoRankeado,
    pesos: dict[str, float] | None = None,
) -> CandidatoRankeado:
    """Compute the total score for a candidate and attach sub-scores.

    Mutates and returns the same ``candidato`` object.
    """
    pesos = pesos or PESOS
    subpuntuaciones: list[SubPuntuacion] = []

    # --- VRAM fit (lower vram_required / available = better, capped at 1) ---
    vram = candidato.compatibilidad_vram
    if vram.vram_required_gb and vram.vram_required_gb > 0:
        ratio = min(vram.vram_required_gb / max(vram.tokens_per_second or 1, 1), 1.0)
        ajuste = max(0.0, 1.0 - ratio)
    else:
        ajuste = 0.5  # unknown → neutral
    subpuntuaciones.append(
        SubPuntuacion(
            criterio="ajuste_vram",
            valor=ajuste,
            peso=pesos["ajuste_vram"],
            justificacion=_justificar_vram(vram),
        )
    )

    # --- Code evidence (is this a dedicated coding model?) ---
    es_codigo = candidato.familia.pipeline_tag.lower() == "code"
    evidencia_codigo = 1.0 if es_codigo else 0.3
    subpuntuaciones.append(
        SubPuntuacion(
            criterio="evidencia_codigo",
            valor=evidencia_codigo,
            peso=pesos["evidencia_codigo"],
            justificacion=(
                "Dedicated coding model"
                if es_codigo
                else "General-purpose model with code support"
            ),
        )
    )

    # --- Documentation quality (placeholder: check license presence) ---
    doc_score = 0.5
    if candidato.familia.license_declarada:
        doc_score += 0.3
    if candidato.familia.idiomas_detectados and len(candidato.familia.idiomas_detectados) > 3:
        doc_score += 0.2
    doc_score = min(doc_score, 1.0)
    subpuntuaciones.append(
        SubPuntuacion(
            criterio="documentacion",
            valor=doc_score,
            peso=pesos["documentacion"],
            justificacion=f"License: {candidato.familia.license_declarada or 'unknown'}, "
            f"{len(candidato.familia.idiomas_detectados or [])} languages detected",
        )
    )

    # --- Size efficiency (smaller models preferred for local use) ---
    if vram.vram_required_gb and vram.vram_required_gb > 0:
        # Normalise: 0 GB → 1.0, 32 GB → ~0.0
        tamanio = max(0.0, 1.0 - (vram.vram_required_gb / 32.0))
    else:
        tamanio = 0.5
    subpuntuaciones.append(
        SubPuntuacion(
            criterio="tamanio",
            valor=tamanio,
            peso=pesos["tamanio"],
            justificacion=f"Estimated VRAM: {vram.vram_required_gb or 'unknown'} GB",
        )
    )

    # --- License permissiveness ---
    lic = (candidato.familia.license_declarada or "").lower()
    if lic in ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"):
        lic_score = 1.0
    elif lic:
        lic_score = 0.5
    else:
        lic_score = 0.0
    subpuntuaciones.append(
        SubPuntuacion(
            criterio="licencia",
            valor=lic_score,
            peso=pesos["licencia"],
            justificacion=f"License: {candidato.familia.license_declarada or 'unknown'}",
        )
    )

    # --- Weighted sum ---
    total = sum(s.valor * s.peso for s in subpuntuaciones)
    candidato.subpuntuaciones = subpuntuaciones
    candidato.puntuacion_total = round(total, 4)
    return candidato


def ordenar_por_puntuacion(
    candidatos: list[CandidatoRankeado],
) -> list[CandidatoRankeado]:
    """Sort candidates by total score descending."""
    return sorted(candidatos, key=lambda c: c.puntuacion_total, reverse=True)


def _justificar_vram(vram) -> str:
    if vram.compatible is True:
        return (
            f"VRAM compatible ({vram.grado}). "
            f"Est. {vram.tokens_per_second or '?'} tok/s. "
            f"Requires {vram.vram_required_gb or '?'} GB."
        )
    if vram.compatible is False:
        return f"VRAM incompatible: {vram.nota}"
    return f"VRAM status unknown: {vram.nota}"
