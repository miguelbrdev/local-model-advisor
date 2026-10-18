"""Tests for src.compatibility.heuristica — offline parameter-count VRAM oracle."""

from __future__ import annotations

import pytest

from src.compatibility.heuristica import (
    GB_POR_B_PARAMETROS_Q4,
    SOBRECARGA_GB,
    HeuristicaParametrosProvider,
    estimar_vram_gb,
)
from src.compatibility.provider import VRAMCompatibilityProvider
from src.models.schemas import FamiliaModelo


def _fam(parametros_b: float | None, texto: str | None = "7B") -> FamiliaModelo:
    return FamiliaModelo(
        familia_id="m",
        variantes=[],
        pipeline_tag="chat",
        tareas_soportadas=["chat"],
        parametros_b=parametros_b,
        parametros_texto=texto,
    )


class TestEstimacion:
    def test_formula_is_the_documented_one(self) -> None:
        assert estimar_vram_gb(7.0) == round(7.0 * GB_POR_B_PARAMETROS_Q4 + SOBRECARGA_GB, 1)

    def test_rounded_to_one_decimal(self) -> None:
        assert estimar_vram_gb(1.5) == 1.9


class TestVeredictos:
    provider = HeuristicaParametrosProvider()

    def test_satisfies_protocol(self) -> None:
        assert isinstance(self.provider, VRAMCompatibilityProvider)

    def test_comfortable_when_under_80_percent(self) -> None:
        r = self.provider.check(_fam(7.0), vram_gb=16.0)  # 5.2 GB of 16
        assert r.compatible is True
        assert r.grado == "comfortable"
        assert r.vram_required_gb == 5.2
        assert r.fuente == "heuristica_propia"

    def test_tight_when_between_80_and_100_percent(self) -> None:
        r = self.provider.check(_fam(7.0), vram_gb=6.0)  # 5.2 GB of 6 → 87 %
        assert r.compatible is True
        assert r.grado == "tight"

    def test_insufficient_when_over_budget(self) -> None:
        r = self.provider.check(_fam(70.0), vram_gb=24.0)  # 43 GB of 24
        assert r.compatible is False
        assert r.grado == "insufficient"

    def test_boundary_exactly_fits(self) -> None:
        r = self.provider.check(_fam(5.0), vram_gb=4.0)  # exactly 4.0 GB
        assert r.compatible is True
        assert r.grado == "tight"

    @pytest.mark.parametrize("parametros", [None, 0.0, -1.0])
    def test_unknown_parameters_are_not_a_verdict(self, parametros: float | None) -> None:
        r = self.provider.check(_fam(parametros), vram_gb=16.0)
        assert r.compatible is None
        assert r.vram_required_gb is None
        assert "unknown" in r.nota

    def test_missing_vram_is_not_a_verdict_but_estimate_is_reported(self) -> None:
        r = self.provider.check(_fam(7.0), vram_gb=0.0)
        assert r.compatible is None
        assert r.vram_required_gb == 5.2

    def test_variant_label_is_parameter_text(self) -> None:
        assert self.provider.check(_fam(7.0, "7B"), vram_gb=16.0).variante == "7B"
        assert self.provider.check(_fam(7.0, None), vram_gb=16.0).variante == "unknown"

    def test_no_tokens_per_second_invented(self) -> None:
        assert self.provider.check(_fam(7.0), vram_gb=16.0).tokens_per_second is None

    def test_deterministic(self) -> None:
        a = self.provider.check(_fam(7.0), vram_gb=16.0)
        b = self.provider.check(_fam(7.0), vram_gb=16.0)
        assert a == b

    def test_runtime_is_ignored(self) -> None:
        a = self.provider.check(_fam(7.0), vram_gb=16.0, runtime="ollama")
        b = self.provider.check(_fam(7.0), vram_gb=16.0, runtime=None)
        assert a == b
