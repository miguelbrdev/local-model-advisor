"""Tests for the ``capacidad_medida`` criterion and its injection by the pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.benchmarks.provider import BenchmarksEnMemoria
from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    EvidenciaBenchmark,
    FamiliaModelo,
)
from src.ranking.scorer import PESOS, VALOR_NEUTRO, calcular_puntuacion


def _ev(benchmark: str, normalizado: float, fuente: str = "epoch_ai") -> EvidenciaBenchmark:
    return EvidenciaBenchmark(
        fuente=fuente, benchmark=benchmark, metrica="m", valor=normalizado * 100,
        valor_normalizado=normalizado, modelo_fuente=f"src-{benchmark}",
    )


def _candidato(
    evidencias: list[EvidenciaBenchmark], canirun_id: str | None = "fam"
) -> CandidatoRankeado:
    return CandidatoRankeado(
        familia=FamiliaModelo(
            familia_id="fam", variantes=["7B"], pipeline_tag="chat",
            tareas_soportadas=["chat", "codigo", "razonamiento"], canirun_id=canirun_id,
        ),
        variante_recomendada="7B",
        compatibilidad_vram=CompatibilityResult(
            modelo_id="fam", variante="7B", compatible=True, fuente="canirun_ai",
            vram_required_gb=5.0,
        ),
        evidencia_benchmark=evidencias,
    )


def _sub(c: CandidatoRankeado):
    return next(s for s in c.subpuntuaciones if s.criterio == "capacidad_medida")


class TestCapacidadMedida:
    def test_weight_present_and_total_sums_to_one(self) -> None:
        assert PESOS["capacidad_medida"] == pytest.approx(0.15)
        assert sum(PESOS.values()) == pytest.approx(1.0)

    def test_neutral_without_evidence(self) -> None:
        s = _sub(calcular_puntuacion(_candidato([]), tarea="codigo"))
        assert s.valor == VALOR_NEUTRO
        assert "No benchmark data" in s.justificacion

    def test_task_specific_benchmark_wins_over_general_index(self) -> None:
        c = _candidato([_ev("eci", 0.9), _ev("swe_bench_verified", 0.3)])
        s = _sub(calcular_puntuacion(c, tarea="codigo"))
        assert s.valor == pytest.approx(0.3)
        assert "SWE-bench Verified" in s.justificacion and "Epoch AI" in s.justificacion

    def test_reasoning_averages_its_benchmarks(self) -> None:
        c = _candidato([_ev("gpqa_diamond", 0.4), _ev("math_level_5", 0.8), _ev("eci", 0.1)])
        s = _sub(calcular_puntuacion(c, tarea="razonamiento"))
        assert s.valor == pytest.approx(0.6)
        assert "GPQA Diamond" in s.justificacion and "MATH level 5" in s.justificacion

    def test_chat_uses_general_index_epoch_first(self) -> None:
        c = _candidato([
            _ev("intelligence_index", 0.2, fuente="artificial_analysis"),
            _ev("eci", 0.7),
            _ev("gpqa_diamond", 0.95),
        ])
        s = _sub(calcular_puntuacion(c, tarea="chat"))
        assert s.valor == pytest.approx(0.7)
        assert "Epoch Capabilities Index" in s.justificacion

    def test_artificial_analysis_index_is_the_fallback(self) -> None:
        c = _candidato([_ev("intelligence_index", 0.2, fuente="artificial_analysis")])
        s = _sub(calcular_puntuacion(c, tarea="codigo"))
        assert s.valor == pytest.approx(0.2)
        assert "Artificial Analysis" in s.justificacion

    def test_unknown_task_falls_back_to_general_index(self) -> None:
        c = _candidato([_ev("eci", 0.6), _ev("gpqa_diamond", 0.1)])
        assert _sub(calcular_puntuacion(c, tarea=None)).valor == pytest.approx(0.6)

    def test_measured_candidate_outranks_identical_unmeasured_one(self) -> None:
        medido = calcular_puntuacion(
            _candidato([_ev("eci", 0.9)]), vram_disponible_gb=16, tarea="chat"
        )
        neutro = calcular_puntuacion(_candidato([]), vram_disponible_gb=16, tarea="chat")
        assert medido.puntuacion_total > neutro.puntuacion_total
        diferencia = medido.puntuacion_total - neutro.puntuacion_total
        assert diferencia == pytest.approx(0.4 * 0.15, abs=1e-4)

    def test_deterministic(self) -> None:
        c = _candidato([_ev("eci", 0.6)])
        a = calcular_puntuacion(c, tarea="chat").puntuacion_total
        b = calcular_puntuacion(c, tarea="chat").puntuacion_total
        assert a == b


class _Provider:
    def check(self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None):
        return CompatibilityResult(
            modelo_id=familia.familia_id, variante="7B", compatible=True,
            fuente="canirun_ai", vram_required_gb=5.0,
        )


class _Catalogo:
    def __init__(self, familias: list[FamiliaModelo]) -> None:
        self._familias = familias

    snapshot_id = "snap-test"
    fecha_sincronizacion = None

    def cargar(self) -> list[FamiliaModelo]:
        return list(self._familias)


def _fam(fid: str, canirun_id: str | None) -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid, variantes=["7B"], pipeline_tag="chat", tareas_soportadas=["chat"],
        license_declarada="MIT", canirun_id=canirun_id,
    )


class TestPipelineInjection:
    def test_scores_attached_by_canirun_id_before_scoring(self) -> None:
        benchmarks = BenchmarksEnMemoria({"b-id": [_ev("eci", 0.9)]})
        with patch("src.chains.recommend_pipeline.elegir_provider", return_value=_Provider()):
            ranking = ejecutar_pipeline(
                CriteriosBusqueda(tarea="chat", vram_gb=16),
                catalogo=_Catalogo([_fam("a", "a-id"), _fam("b", "b-id"), _fam("c", None)]),
                top_n=3, benchmarks=benchmarks,
            )
        por_id = {c.familia.familia_id: c for c in ranking}
        assert ranking[0].familia.familia_id == "b"
        assert [e.benchmark for e in por_id["b"].evidencia_benchmark] == ["eci"]
        assert por_id["a"].evidencia_benchmark == []
        assert por_id["c"].evidencia_benchmark == []
        assert _sub(por_id["b"]).valor == pytest.approx(0.9)
        assert _sub(por_id["a"]).valor == VALOR_NEUTRO

    def test_without_provider_everything_is_neutral(self) -> None:
        with patch("src.chains.recommend_pipeline.elegir_provider", return_value=_Provider()):
            ranking = ejecutar_pipeline(
                CriteriosBusqueda(tarea="chat", vram_gb=16),
                catalogo=_Catalogo([_fam("a", "a-id")]), top_n=3,
            )
        assert ranking[0].evidencia_benchmark == []
        assert _sub(ranking[0]).valor == VALOR_NEUTRO
