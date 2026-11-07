"""Attach explanatory evidence to already-ranked candidates.

Invariant (AGENTS.md): evidence explains, it never scores.  This module
runs *after* :func:`src.chains.recommend_pipeline.ejecutar_pipeline`,
receives the final list and only fills ``evidencia_explicativa`` on each
candidate.  It reads neither ``puntuacion_total`` nor ``subpuntuaciones``
and never reorders the list; the tests pin both.

Failure handling: the index is an optional dependency.  When it cannot be
reached the remaining candidates are left without evidence and the error
is returned so the API can report it as a limitation, never as a 500.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.evidence.qdrant_index import Fragmento, IndiceNoDisponibleError
from src.models.schemas import CandidatoRankeado, CriteriosBusqueda, EvidenciaExplicativa

FRAGMENTOS_POR_CANDIDATO = 2


@runtime_checkable
class BuscadorEvidencia(Protocol):
    """What the API needs from the index: per-family fragment search."""

    def buscar(self, consulta: str, familia_id: str, k: int = 2) -> list[Fragmento]: ...


@dataclass(frozen=True)
class ResultadoEnriquecimiento:
    """What happened while attaching evidence.

    Attributes:
        con_evidencia: Candidates that received at least one fragment.
        sin_evidencia: Candidates left without fragments (no documentation
            indexed, or not queried after a backend failure).
        error: Backend failure message, or ``None``.
    """

    con_evidencia: int
    sin_evidencia: int
    error: str | None


def enriquecer_candidatos(
    candidatos: list[CandidatoRankeado],
    consulta: str,
    buscador: BuscadorEvidencia,
    *,
    k: int = FRAGMENTOS_POR_CANDIDATO,
) -> ResultadoEnriquecimiento:
    """Fill ``evidencia_explicativa`` on each candidate, in place.

    Args:
        candidatos: Final ranked list; order and scores are never touched.
        consulta: Text to search with (see :func:`consulta_para`).
        buscador: Index to query, filtered by each candidate's family.
        k: Fragments per candidate.

    Returns:
        Counts and the first backend error, if any.  After an error the
        remaining candidates are not queried.
    """
    con_evidencia = 0
    error: str | None = None
    for candidato in candidatos:
        if error is not None:
            break
        try:
            fragmentos = buscador.buscar(consulta, candidato.familia.familia_id, k=k)
        except IndiceNoDisponibleError as exc:
            error = str(exc)
            break
        candidato.evidencia_explicativa = [
            EvidenciaExplicativa(
                fuente_url=f.url,
                fragmento=f.texto,
                relevancia=max(0.0, min(1.0, f.score)),
            )
            for f in fragmentos
        ]
        if fragmentos:
            con_evidencia += 1
    return ResultadoEnriquecimiento(
        con_evidencia=con_evidencia,
        sin_evidencia=len(candidatos) - con_evidencia,
        error=error,
    )


def consulta_para(criterios: CriteriosBusqueda, mensaje: str) -> str:
    """Text used to search each candidate's documentation.

    The user's own words are the best query when the message was free
    text.  When it was criteria JSON (or empty, on a follow-up), a short
    description built from the criteria is used instead so the search
    still targets the task.
    """
    texto = mensaje.strip()
    if texto and not _parece_json(texto):
        return texto
    partes = [f"modelo para {criterios.tarea}" if criterios.tarea else "modelo local"]
    if criterios.idiomas_requeridos:
        partes.append("idiomas " + ", ".join(criterios.idiomas_requeridos))
    if criterios.requiere_licencia_comercial:
        partes.append("uso comercial")
    return "; ".join(partes)


def _parece_json(texto: str) -> bool:
    try:
        json.loads(texto)
    except ValueError:
        return False
    return True
