"""Tests for src.evidence.enriquecer — evidence is attached after ranking and
can never change scores or order (AGENTS.md invariant)."""

from __future__ import annotations

import pytest

from src.evidence.enriquecer import (
    BuscadorEvidencia,
    ResultadoEnriquecimiento,
    consulta_para,
    enriquecer_candidatos,
)
from src.evidence.qdrant_index import Fragmento, IndiceNoDisponibleError
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    EvidenciaExplicativa,
    FamiliaModelo,
)


def _candidato(fid: str, score: float) -> CandidatoRankeado:
    return CandidatoRankeado(
        familia=FamiliaModelo(
            familia_id=fid, variantes=[], pipeline_tag="codigo", tareas_soportadas=["codigo"]
        ),
        variante_recomendada="7B",
        compatibilidad_vram=CompatibilityResult(modelo_id=fid, variante="7B", compatible=True),
        puntuacion_total=score,
    )


class _Buscador:
    def __init__(self, por_familia: dict[str, list[Fragmento]], fallo_en: str | None = None):
        self._por_familia = por_familia
        self._fallo_en = fallo_en
        self.consultas: list[tuple[str, str, int]] = []

    def buscar(self, consulta: str, familia_id: str, k: int = 2) -> list[Fragmento]:
        self.consultas.append((consulta, familia_id, k))
        if familia_id == self._fallo_en:
            raise IndiceNoDisponibleError("qdrant down")
        return self._por_familia.get(familia_id, [])


def _frag(fid: str, texto: str, score: float = 0.8, indice: int = 0) -> Fragmento:
    return Fragmento(
        familia_id=fid, texto=texto, score=score, url="https://hf.co/x", indice=indice
    )


class TestEnriquecer:
    def test_attaches_fragments_without_touching_ranking(self) -> None:
        candidatos = [_candidato("a", 0.9), _candidato("b", 0.7), _candidato("c", 0.5)]
        buscador = _Buscador({"a": [_frag("a", "A is great at code")], "c": [_frag("c", "C")]})

        resultado = enriquecer_candidatos(candidatos, "code model", buscador)

        assert [c.familia.familia_id for c in candidatos] == ["a", "b", "c"]
        assert [c.puntuacion_total for c in candidatos] == [0.9, 0.7, 0.5]
        assert candidatos[0].evidencia_explicativa == [
            EvidenciaExplicativa(fuente_url="https://hf.co/x", fragmento="A is great at code",
                                 relevancia=0.8)
        ]
        assert candidatos[1].evidencia_explicativa == []
        assert resultado == ResultadoEnriquecimiento(con_evidencia=2, sin_evidencia=1, error=None)
        assert [q[1] for q in buscador.consultas] == ["a", "b", "c"]
        assert all(q[0] == "code model" and q[2] == 2 for q in buscador.consultas)

    def test_backend_failure_stops_and_is_reported(self) -> None:
        candidatos = [_candidato("a", 0.9), _candidato("b", 0.7), _candidato("c", 0.5)]
        buscador = _Buscador({"a": [_frag("a", "A")], "c": [_frag("c", "C")]}, fallo_en="b")

        resultado = enriquecer_candidatos(candidatos, "q", buscador)

        assert resultado.error is not None and "qdrant down" in resultado.error
        assert resultado.con_evidencia == 1
        assert resultado.sin_evidencia == 2
        assert candidatos[2].evidencia_explicativa == []  # not queried after the failure
        assert [q[1] for q in buscador.consultas] == ["a", "b"]

    def test_empty_candidates(self) -> None:
        buscador = _Buscador({})
        assert enriquecer_candidatos([], "q", buscador) == ResultadoEnriquecimiento(0, 0, None)
        assert buscador.consultas == []

    def test_k_is_forwarded_and_fragments_kept_in_score_order(self) -> None:
        candidatos = [_candidato("a", 0.9)]
        buscador = _Buscador({"a": [_frag("a", "first", 0.9, 0), _frag("a", "second", 0.4, 3)]})
        enriquecer_candidatos(candidatos, "q", buscador, k=5)
        assert buscador.consultas[0][2] == 5
        assert [e.relevancia for e in candidatos[0].evidencia_explicativa] == [0.9, 0.4]

    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(_Buscador({}), BuscadorEvidencia)


class TestConsultaPara:
    def test_free_text_is_used_verbatim(self) -> None:
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=12)
        assert consulta_para(criterios, "quiero programar en Python") == (
            "quiero programar en Python"
        )

    def test_json_message_is_replaced_by_a_description(self) -> None:
        criterios = CriteriosBusqueda(tarea="razonamiento", vram_gb=16, idiomas_requeridos=["es"])
        consulta = consulta_para(criterios, '{"tarea": "razonamiento", "vram_gb": 16}')
        assert "razonamiento" in consulta
        assert "es" in consulta
        assert "{" not in consulta

    def test_follow_up_without_text_falls_back_to_description(self) -> None:
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=8)
        assert "chat" in consulta_para(criterios, "")


@pytest.mark.parametrize("valor", [-0.1, 1.5])
def test_relevancia_is_bounded(valor: float) -> None:
    with pytest.raises(ValueError):
        EvidenciaExplicativa(fuente_url=None, fragmento="x", relevancia=valor)
