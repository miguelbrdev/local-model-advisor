"""Tests for src.ranking.scorer.

Covers: determinism (same input → same output), sub-score breakdown,
weighted-sum correctness, and edge cases (unknown VRAM, no license).
"""

from __future__ import annotations

import pytest

from src.compatibility.vram import FRACCION_VRAM_UTIL
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    FamiliaModelo,
    SubPuntuacion,
)
from src.ranking.scorer import (
    PESOS,
    VALOR_NEUTRO,
    calcular_puntuacion,
    ordenar_por_puntuacion,
)


def _make_candidate(
    familia_id: str = "test-model",
    tareas: list[str] | None = None,
    license_: str | None = "MIT",
    idiomas: list[str] | None = None,
    compatible: bool | None = True,
    vram_required: float | None = 5.0,
    tokens_per_second: float | None = 40.0,
    grado: str | None = "comfortable",
) -> CandidatoRankeado:
    """Helper to build a CandidatoRankeado with controlled fields."""
    tareas = ["codigo"] if tareas is None else tareas
    return CandidatoRankeado(
        familia=FamiliaModelo(
            familia_id=familia_id,
            variantes=["7B"],
            pipeline_tag=tareas[0] if tareas else "",
            tareas_soportadas=tareas,
            license_declarada=license_,
            idiomas_detectados=idiomas or ["en"],
            formatos_disponibles=["gguf"],
        ),
        variante_recomendada="7B",
        compatibilidad_vram=CompatibilityResult(
            modelo_id=familia_id,
            variante="7B",
            compatible=compatible,
            grado=grado,
            fuente="canirun_ai",
            tokens_per_second=tokens_per_second,
            vram_required_gb=vram_required,
        ),
    )


def _sub(candidato: CandidatoRankeado, criterio: str) -> SubPuntuacion:
    return next(s for s in candidato.subpuntuaciones if s.criterio == criterio)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminismo:
    def test_same_input_same_output(self) -> None:
        c1 = calcular_puntuacion(_make_candidate(), vram_disponible_gb=16.0)
        c2 = calcular_puntuacion(_make_candidate(), vram_disponible_gb=16.0)
        assert c1.puntuacion_total == c2.puntuacion_total
        assert c1.subpuntuaciones == c2.subpuntuaciones

    def test_deterministic_across_multiple_calls(self) -> None:
        scores = {
            calcular_puntuacion(
                _make_candidate(vram_required=8.0), vram_disponible_gb=16.0
            ).puntuacion_total
            for _ in range(10)
        }
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# Sub-score breakdown
# ---------------------------------------------------------------------------

class TestSubpuntuaciones:
    def test_dimensions_in_weight_order(self) -> None:
        c = calcular_puntuacion(_make_candidate())
        assert [s.criterio for s in c.subpuntuaciones] == list(PESOS)

    def test_weights_sum_to_one(self) -> None:
        assert sum(PESOS.values()) == pytest.approx(1.0)

    def test_all_values_between_0_and_1(self) -> None:
        c = calcular_puntuacion(_make_candidate(), vram_disponible_gb=16.0)
        for s in c.subpuntuaciones:
            assert 0 <= s.valor <= 1, f"{s.criterio} value {s.valor} out of range"
            assert 0 <= s.peso <= 1, f"{s.criterio} weight {s.peso} out of range"

    def test_peso_matches_config(self) -> None:
        c = calcular_puntuacion(_make_candidate())
        for s in c.subpuntuaciones:
            assert s.peso == PESOS[s.criterio]

    def test_justificacion_not_empty(self) -> None:
        c = calcular_puntuacion(_make_candidate())
        for s in c.subpuntuaciones:
            assert s.justificacion.strip()


# ---------------------------------------------------------------------------
# Individual sub-score logic
# ---------------------------------------------------------------------------

class TestAjusteVRAM:
    """Safety margin: full score up to FRACCION_VRAM_UTIL, zero at 100 %."""

    @pytest.mark.parametrize("required", [1.0, 4.0, 12.8])
    def test_comfortable_fit_scores_full_regardless_of_size(self, required: float) -> None:
        """Being smaller than the threshold earns nothing extra (that was the bias)."""
        c = calcular_puntuacion(_make_candidate(vram_required=required), vram_disponible_gb=16.0)
        assert _sub(c, "ajuste_vram").valor == 1.0

    def test_tight_fit_decays_linearly(self) -> None:
        """90 % of 16 GB → halfway between the 80 % threshold and 100 % → 0.5."""
        c = calcular_puntuacion(_make_candidate(vram_required=14.4), vram_disponible_gb=16.0)
        assert _sub(c, "ajuste_vram").valor == pytest.approx(0.5)

    def test_full_budget_scores_zero(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=32.0), vram_disponible_gb=32.0)
        assert _sub(c, "ajuste_vram").valor == pytest.approx(0.0)

    def test_exceeding_available_clamps_to_zero(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=24.0), vram_disponible_gb=8.0)
        assert _sub(c, "ajuste_vram").valor == 0.0

    def test_threshold_is_the_shared_constant(self) -> None:
        at = calcular_puntuacion(
            _make_candidate(vram_required=16.0 * FRACCION_VRAM_UTIL), vram_disponible_gb=16.0
        )
        above = calcular_puntuacion(
            _make_candidate(vram_required=16.0 * FRACCION_VRAM_UTIL + 0.1),
            vram_disponible_gb=16.0,
        )
        assert _sub(at, "ajuste_vram").valor == 1.0
        assert _sub(above, "ajuste_vram").valor < 1.0

    def test_justification_names_margin(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=14.4), vram_disponible_gb=16.0)
        assert "tight margin" in _sub(c, "ajuste_vram").justificacion
        c = calcular_puntuacion(_make_candidate(vram_required=4.0), vram_disponible_gb=16.0)
        assert "comfortable margin" in _sub(c, "ajuste_vram").justificacion

    def test_tokens_per_second_does_not_affect_fit(self) -> None:
        """Regression: the old formula divided GB by tok/s."""
        slow = calcular_puntuacion(
            _make_candidate(vram_required=4.0, tokens_per_second=2.0), vram_disponible_gb=16.0
        )
        fast = calcular_puntuacion(
            _make_candidate(vram_required=4.0, tokens_per_second=200.0), vram_disponible_gb=16.0
        )
        assert _sub(slow, "ajuste_vram").valor == _sub(fast, "ajuste_vram").valor

    def test_unknown_required_vram_neutral(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=None), vram_disponible_gb=16.0)
        assert _sub(c, "ajuste_vram").valor == VALOR_NEUTRO

    def test_unknown_available_vram_neutral(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=4.0))
        assert _sub(c, "ajuste_vram").valor == VALOR_NEUTRO

    def test_justification_names_available_vram(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=4.0), vram_disponible_gb=16.0)
        assert "16 GB" in _sub(c, "ajuste_vram").justificacion


class TestEvidenciaCodigo:
    def test_dedicated_coder_scores_1(self) -> None:
        c = calcular_puntuacion(_make_candidate(tareas=["codigo"]))
        ev = _sub(c, "evidencia_codigo")
        assert ev.valor == 1.0
        assert "Dedicated" in ev.justificacion

    def test_multitask_with_code_scores_03(self) -> None:
        """Presence of ``codigo`` is not enough: only dedication scores 1.0,
        otherwise the criterion is a constant for every ``codigo`` query."""
        c = calcular_puntuacion(_make_candidate(tareas=["chat", "codigo"]))
        ev = _sub(c, "evidencia_codigo")
        assert ev.valor == 0.3
        assert "code support" in ev.justificacion

    def test_no_code_scores_03(self) -> None:
        c = calcular_puntuacion(_make_candidate(tareas=["chat"]))
        ev = _sub(c, "evidencia_codigo")
        assert ev.valor == 0.3
        assert "No code" in ev.justificacion

    def test_reads_tareas_soportadas_not_pipeline_tag(self) -> None:
        c = _make_candidate(tareas=["chat", "codigo"])
        c.familia = c.familia.model_copy(update={"pipeline_tag": "codigo"})
        calcular_puntuacion(c)
        assert _sub(c, "evidencia_codigo").valor == 0.3


class TestSinDocumentacion:
    def test_documentacion_criterion_is_gone(self) -> None:
        """Retired: it only re-counted the declared licence."""
        c = calcular_puntuacion(_make_candidate())
        assert "documentacion" not in {s.criterio for s in c.subpuntuaciones}
        assert "documentacion" not in PESOS

    def test_languages_do_not_affect_the_score(self) -> None:
        many = calcular_puntuacion(_make_candidate(idiomas=["en", "es", "fr", "de"]))
        few = calcular_puntuacion(_make_candidate(idiomas=["en"]))
        assert many.puntuacion_total == few.puntuacion_total


class TestCapacidad:
    """Use of the usable budget as capability proxy: bigger is better up to the threshold."""

    def test_proportional_to_usable_budget(self) -> None:
        """4 GB of a 12.8 GB usable budget (80 % of 16) → 0.3125."""
        c = calcular_puntuacion(_make_candidate(vram_required=4.0), vram_disponible_gb=16.0)
        assert _sub(c, "capacidad").valor == pytest.approx(0.3125)

    def test_at_threshold_scores_full(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=12.8), vram_disponible_gb=16.0)
        assert _sub(c, "capacidad").valor == pytest.approx(1.0)

    def test_above_threshold_clamps_to_one(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=15.0), vram_disponible_gb=16.0)
        assert _sub(c, "capacidad").valor == 1.0

    def test_bigger_model_scores_higher_when_both_fit(self) -> None:
        small = calcular_puntuacion(_make_candidate(vram_required=2.0), vram_disponible_gb=16.0)
        big = calcular_puntuacion(_make_candidate(vram_required=10.0), vram_disponible_gb=16.0)
        assert _sub(big, "capacidad").valor > _sub(small, "capacidad").valor
        assert big.puntuacion_total > small.puntuacion_total

    def test_unknown_required_vram_neutral(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=None), vram_disponible_gb=16.0)
        assert _sub(c, "capacidad").valor == VALOR_NEUTRO

    def test_unknown_available_vram_neutral(self) -> None:
        c = calcular_puntuacion(_make_candidate(vram_required=4.0))
        assert _sub(c, "capacidad").valor == VALOR_NEUTRO


class TestPoliticaVRAM:
    """The two criteria together: most capable that fits comfortably."""

    def test_comfortable_large_beats_tight_larger(self) -> None:
        """14B at 59 % must outrank a 20B at 85 % (the gpt-oss-20b case)."""
        comfortable = calcular_puntuacion(
            _make_candidate(vram_required=9.4), vram_disponible_gb=16.0
        )
        tight = calcular_puntuacion(_make_candidate(vram_required=13.6), vram_disponible_gb=16.0)
        assert comfortable.puntuacion_total > tight.puntuacion_total

    def test_tiny_model_no_longer_wins_on_a_big_gpu(self) -> None:
        tiny = calcular_puntuacion(_make_candidate(vram_required=1.4), vram_disponible_gb=48.0)
        mid = calcular_puntuacion(_make_candidate(vram_required=20.2), vram_disponible_gb=48.0)
        assert mid.puntuacion_total > tiny.puntuacion_total


class TestLicencia:
    @pytest.mark.parametrize("lic", ["MIT", "Apache-2.0", "Apache 2.0", "BSD-2-Clause", "ISC"])
    def test_permissive_scores_1(self, lic: str) -> None:
        c = calcular_puntuacion(_make_candidate(license_=lic))
        assert _sub(c, "licencia").valor == 1.0

    def test_source_spelling_apache_scores_1(self) -> None:
        """Regression: ``"Apache 2.0"`` (48 of 91 synced families) scored 0.5."""
        c = calcular_puntuacion(_make_candidate(license_="Apache 2.0"))
        assert _sub(c, "licencia").valor == 1.0

    def test_other_license_scores_05(self) -> None:
        c = calcular_puntuacion(_make_candidate(license_="Llama 3.1 Community"))
        assert _sub(c, "licencia").valor == 0.5

    def test_no_license_scores_0(self) -> None:
        c = calcular_puntuacion(_make_candidate(license_=None))
        assert _sub(c, "licencia").valor == 0.0


# ---------------------------------------------------------------------------
# Weighted sum
# ---------------------------------------------------------------------------

class TestPuntuacionTotal:
    def test_is_weighted_sum(self) -> None:
        c = calcular_puntuacion(_make_candidate(), vram_disponible_gb=16.0)
        expected = sum(s.valor * s.peso for s in c.subpuntuaciones)
        assert c.puntuacion_total == pytest.approx(expected, abs=0.001)

    def test_custom_weights(self) -> None:
        c1 = calcular_puntuacion(_make_candidate(vram_required=2.0), vram_disponible_gb=16.0)
        c2 = calcular_puntuacion(
            _make_candidate(vram_required=2.0),
            vram_disponible_gb=16.0,
            pesos={k: 0.2 for k in PESOS},
        )
        assert c1.puntuacion_total != c2.puntuacion_total

    def test_returns_same_object(self) -> None:
        c = _make_candidate()
        assert calcular_puntuacion(c) is c


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

class TestOrdenarPorPuntuacion:
    def test_descending_order(self) -> None:
        c_low = calcular_puntuacion(
            _make_candidate("low", vram_required=31.0, license_=None), vram_disponible_gb=32.0
        )
        c_high = calcular_puntuacion(
            _make_candidate("high", vram_required=20.0, license_="MIT"), vram_disponible_gb=32.0
        )
        resultado = ordenar_por_puntuacion([c_low, c_high])
        assert [r.familia.familia_id for r in resultado] == ["high", "low"]

    def test_stable_for_ties(self) -> None:
        a = calcular_puntuacion(_make_candidate("a"))
        b = calcular_puntuacion(_make_candidate("b"))
        assert a.puntuacion_total == b.puntuacion_total
        assert [r.familia.familia_id for r in ordenar_por_puntuacion([a, b])] == ["a", "b"]

    def test_empty_list(self) -> None:
        assert ordenar_por_puntuacion([]) == []
