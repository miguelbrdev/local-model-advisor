"""Tests for src.ranking.scorer.

Covers: determinism (same input → same output), sub-score breakdown,
weighted-sum correctness, and edge cases (unknown VRAM, no license).
"""

from __future__ import annotations

import pytest

from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    FamiliaModelo,
)
from src.ranking.scorer import PESOS, calcular_puntuacion, ordenar_por_puntuacion


def _make_candidate(
    familia_id: str = "test-model",
    pipeline_tag: str = "code",
    license_: str | None = "MIT",
    idiomas: list[str] | None = None,
    compatible: bool | None = True,
    vram_required: float | None = 5.0,
    tokens_per_second: float | None = 40.0,
    grado: str | None = "comfortable",
) -> CandidatoRankeado:
    """Helper to build a CandidatoRankeado with controlled fields."""
    return CandidatoRankeado(
        familia=FamiliaModelo(
            familia_id=familia_id,
            variantes=["7B"],
            pipeline_tag=pipeline_tag,
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


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminismo:
    def test_same_input_same_output(self) -> None:
        c1 = _make_candidate()
        c2 = _make_candidate()
        calcular_puntuacion(c1)
        calcular_puntuacion(c2)
        assert c1.puntuacion_total == c2.puntuacion_total
        assert c1.subpuntuaciones == c2.subpuntuaciones

    def test_deterministic_across_multiple_calls(self) -> None:
        scores = []
        for _ in range(10):
            c = _make_candidate(vram_required=8.0, tokens_per_second=30.0)
            calcular_puntuacion(c)
            scores.append(c.puntuacion_total)
        assert len(set(scores)) == 1


# ---------------------------------------------------------------------------
# Sub-score breakdown
# ---------------------------------------------------------------------------

class TestSubpuntuaciones:
    def test_five_dimensions(self) -> None:
        c = _make_candidate()
        calcular_puntuacion(c)
        criterios = [s.criterio for s in c.subpuntuaciones]
        expected = [
            "ajuste_vram", "evidencia_codigo", "documentacion",
            "tamanio", "licencia",
        ]
        assert criterios == expected

    def test_all_values_between_0_and_1(self) -> None:
        c = _make_candidate()
        calcular_puntuacion(c)
        for s in c.subpuntuaciones:
            assert 0 <= s.valor <= 1, f"{s.criterio} value {s.valor} out of range"
            assert 0 <= s.peso <= 1, f"{s.criterio} weight {s.peso} out of range"

    def test_peso_matches_config(self) -> None:
        c = _make_candidate()
        calcular_puntuacion(c)
        for s in c.subpuntuaciones:
            assert s.peso == PESOS[s.criterio]

    def test_justificacion_not_empty(self) -> None:
        c = _make_candidate()
        calcular_puntuacion(c)
        for s in c.subpuntuaciones:
            assert s.justificacion.strip()


# ---------------------------------------------------------------------------
# Individual sub-score logic
# ---------------------------------------------------------------------------

class TestAjusteVRAM:
    def test_high_vram_low_score(self) -> None:
        """Model requiring 30 GB on a 32 GB budget → low ajuste_vram."""
        c = _make_candidate(vram_required=30.0, tokens_per_second=10.0)
        calcular_puntuacion(c)
        ajuste = next(s for s in c.subpuntuaciones if s.criterio == "ajuste_vram")
        assert ajuste.valor < 0.5

    def test_low_vram_high_score(self) -> None:
        """Model requiring 2 GB → high ajuste_vram."""
        c = _make_candidate(vram_required=2.0, tokens_per_second=50.0)
        calcular_puntuacion(c)
        ajuste = next(s for s in c.subpuntuaciones if s.criterio == "ajuste_vram")
        assert ajuste.valor > 0.5

    def test_unknown_vram_neutral(self) -> None:
        c = _make_candidate(vram_required=None, tokens_per_second=None)
        calcular_puntuacion(c)
        ajuste = next(s for s in c.subpuntuaciones if s.criterio == "ajuste_vram")
        assert ajuste.valor == 0.5


class TestEvidenciaCodigo:
    def test_code_pipeline_scores_1(self) -> None:
        c = _make_candidate(pipeline_tag="code")
        calcular_puntuacion(c)
        ev = next(s for s in c.subpuntuaciones if s.criterio == "evidencia_codigo")
        assert ev.valor == 1.0

    def test_non_code_pipeline_scores_03(self) -> None:
        c = _make_candidate(pipeline_tag="text-generation")
        calcular_puntuacion(c)
        ev = next(s for s in c.subpuntuaciones if s.criterio == "evidencia_codigo")
        assert ev.valor == 0.3


class TestDocumentacion:
    def test_with_license_and_many_languages(self) -> None:
        c = _make_candidate(license_="MIT", idiomas=["en", "es", "fr", "de"])
        calcular_puntuacion(c)
        doc = next(s for s in c.subpuntuaciones if s.criterio == "documentacion")
        assert doc.valor == 1.0  # 0.5 + 0.3 + 0.2

    def test_without_license(self) -> None:
        c = _make_candidate(license_=None, idiomas=["en"])
        calcular_puntuacion(c)
        doc = next(s for s in c.subpuntuaciones if s.criterio == "documentacion")
        assert doc.valor == 0.5  # base only

    def test_with_license_few_languages(self) -> None:
        c = _make_candidate(license_="MIT", idiomas=["en"])
        calcular_puntuacion(c)
        doc = next(s for s in c.subpuntuaciones if s.criterio == "documentacion")
        assert doc.valor == pytest.approx(0.8)  # 0.5 + 0.3


class TestTamanio:
    def test_small_model_high_score(self) -> None:
        c = _make_candidate(vram_required=1.0)
        calcular_puntuacion(c)
        tam = next(s for s in c.subpuntuaciones if s.criterio == "tamanio")
        assert tam.valor > 0.9

    def test_large_model_low_score(self) -> None:
        c = _make_candidate(vram_required=28.0)
        calcular_puntuacion(c)
        tam = next(s for s in c.subpuntuaciones if s.criterio == "tamanio")
        assert tam.valor < 0.2

    def test_unknown_vram_neutral(self) -> None:
        c = _make_candidate(vram_required=None)
        calcular_puntuacion(c)
        tam = next(s for s in c.subpuntuaciones if s.criterio == "tamanio")
        assert tam.valor == 0.5


class TestLicencia:
    @pytest.mark.parametrize("lic", ["MIT", "Apache-2.0", "BSD-2-Clause", "ISC"])
    def test_permissive_scores_1(self, lic: str) -> None:
        c = _make_candidate(license_=lic)
        calcular_puntuacion(c)
        lic_s = next(s for s in c.subpuntuaciones if s.criterio == "licencia")
        assert lic_s.valor == 1.0

    def test_other_license_scores_05(self) -> None:
        c = _make_candidate(license_="Llama 3.1 Community")
        calcular_puntuacion(c)
        lic_s = next(s for s in c.subpuntuaciones if s.criterio == "licencia")
        assert lic_s.valor == 0.5

    def test_no_license_scores_0(self) -> None:
        c = _make_candidate(license_=None)
        calcular_puntuacion(c)
        lic_s = next(s for s in c.subpuntuaciones if s.criterio == "licencia")
        assert lic_s.valor == 0.0


# ---------------------------------------------------------------------------
# Weighted sum
# ---------------------------------------------------------------------------

class TestPuntuacionTotal:
    def test_is_weighted_sum(self) -> None:
        c = _make_candidate()
        calcular_puntuacion(c)
        expected = sum(s.valor * s.peso for s in c.subpuntuaciones)
        assert c.puntuacion_total == pytest.approx(expected, abs=0.001)

    def test_custom_weights(self) -> None:
        c1 = _make_candidate(vram_required=2.0)
        c2 = _make_candidate(vram_required=2.0)
        custom_pesos = {k: 0.2 for k in PESOS}
        calcular_puntuacion(c1)
        calcular_puntuacion(c2, pesos=custom_pesos)
        # Different weights → different total
        assert c1.puntuacion_total != c2.puntuacion_total


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------

class TestOrdenarPorPuntuacion:
    def test_descending_order(self) -> None:
        c_low = _make_candidate("low", vram_required=30.0, license_=None)
        c_high = _make_candidate("high", vram_required=2.0, license_="MIT")
        calcular_puntuacion(c_low)
        calcular_puntuacion(c_high)
        resultado = ordenar_por_puntuacion([c_low, c_high])
        assert resultado[0].familia.familia_id == "high"
        assert resultado[1].familia.familia_id == "low"

    def test_empty_list(self) -> None:
        assert ordenar_por_puntuacion([]) == []
