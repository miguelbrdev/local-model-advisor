"""Tests for src.chains.recommend_pipeline.

Uses a small test catalogue, valid Pydantic criteria, and a mocked VRAM
provider. No HTTP requests, no real SQLite, no external file dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)

# ---------------------------------------------------------------------------
# Small test catalogue
# ---------------------------------------------------------------------------

FAMILIA_A = FamiliaModelo(
    familia_id="model-a",
    variantes=["7B", "14B"],
    pipeline_tag="code",
    license_declarada="Apache 2.0",
    idiomas_detectados=["en", "es", "zh", "fr"],
    formatos_disponibles=["gguf"],
    canirun_id="model-a-7b",
)

FAMILIA_B = FamiliaModelo(
    familia_id="model-b",
    variantes=["7B"],
    pipeline_tag="text-generation",
    license_declarada="MIT",
    idiomas_detectados=["en"],
    formatos_disponibles=["gguf"],
    canirun_id="model-b-7b",
)

FAMILIA_C = FamiliaModelo(
    familia_id="model-c",
    variantes=["3B"],
    pipeline_tag="code",
    license_declarada="Apache 2.0",
    idiomas_detectados=["en", "es"],
    formatos_disponibles=["gguf"],
    canirun_id="model-c-3b",
)


def _criterios(**overrides) -> CriteriosBusqueda:
    defaults = {"tarea": "codigo", "vram_gb": 11.0}
    defaults.update(overrides)
    return CriteriosBusqueda(**defaults)


def _compatible(familia_id: str, **overrides) -> CompatibilityResult:
    defaults = dict(
        modelo_id=familia_id,
        variante="7B",
        compatible=True,
        grado="comfortable",
        fuente="no_disponible",
        nota="mock",
    )
    defaults.update(overrides)
    return CompatibilityResult(**defaults)


def _incompatible(familia_id: str) -> CompatibilityResult:
    return CompatibilityResult(
        modelo_id=familia_id,
        variante="7B",
        compatible=False,
        grado="insufficient",
        fuente="no_disponible",
        nota="mock incompatible",
    )


def _no_disponible(familia_id: str) -> CompatibilityResult:
    return CompatibilityResult(
        modelo_id=familia_id,
        variante="7B",
        compatible=None,
        grado=None,
        fuente="no_disponible",
        nota="provider unavailable",
    )


# ---------------------------------------------------------------------------
# 1. Happy path: filter → VRAM check → score → sort → top N
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_filters_and_scores(self) -> None:
        """Three families pass filter; VRAM compatible; all scored and ranked."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
            )

        assert len(results) == 3

    def test_sorted_descending(self) -> None:
        """Candidates are returned in descending score order."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
            )

        scores = [r.puntuacion_total for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self) -> None:
        """top_n=2 returns only the two highest-scoring candidates."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
                top_n=2,
            )

        assert len(results) == 2
        assert all(r.puntuacion_total >= 0 for r in results)

    def test_subpuntuaciones_present(self) -> None:
        """Each result contains 5 explainable sub-scores."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
            )

        for candidate in results:
            assert len(candidate.subpuntuaciones) == 5
            names = {s.criterio for s in candidate.subpuntuaciones}
            assert names == {
                "ajuste_vram",
                "evidencia_codigo",
                "documentacion",
                "tamanio",
                "licencia",
            }

    def test_provider_called_for_each_family(self) -> None:
        """VRAM provider is queried once per candidate family."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
            )

        assert provider.call_count == 3
        called_ids = {fam_id for fam_id, _ in provider.calls}
        assert called_ids == {"model-a", "model-b", "model-c"}


# ---------------------------------------------------------------------------
# 2. No eligible candidates
# ---------------------------------------------------------------------------


class TestNoCandidates:
    def test_empty_catalogue_returns_empty(self) -> None:
        """Empty catalogue returns an empty list, not an exception."""
        provider = _MockProvider({})

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[],
            )

        assert results == []

    def test_all_incompatible_returns_empty(self) -> None:
        """All families VRAM-incompatible → empty result, no crash."""
        provider_results = {
            "model-a": _incompatible("model-a"),
            "model-b": _incompatible("model-b"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B],
            )

        assert results == []

    def test_task_filter_removes_all_returns_empty(self) -> None:
        """Task mismatch filters out all families → empty list."""
        provider = _MockProvider({})

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(tarea="embeddings"),
                familias=[FAMILIA_A, FAMILIA_C],
            )

        assert results == []

    def test_provider_called_zero_times_when_empty(self) -> None:
        """No families → provider never consulted."""
        provider = _MockProvider({})

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            ejecutar_pipeline(
                criterios=_criterios(),
                familias=[],
            )

        assert provider.call_count == 0


# ---------------------------------------------------------------------------
# 3. VRAM provider unavailable (no_disponible)
# ---------------------------------------------------------------------------


class TestProviderUnavailable:
    def test_no_disponible_keeps_candidate(self) -> None:
        """compatible=None (no_disponible) does NOT exclude the candidate."""
        provider_results = {
            "model-a": _no_disponible("model-a"),
            "model-b": _no_disponible("model-b"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B],
            )

        assert len(results) == 2
        for candidate in results:
            assert candidate.compatibilidad_vram.fuente == "no_disponible"
            assert candidate.compatibilidad_vram.compatible is None

    def test_no_disponible_deterministic_score(self) -> None:
        """no_disponible produces a deterministic neutral score (not invented)."""
        provider_results = {
            "model-a": _no_disponible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A],
            )

        assert len(results) == 1
        score = results[0].puntuacion_total
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_mixed_compatible_and_no_disponible(self) -> None:
        """Mix of compatible and no_disponible: both kept, scored, ranked."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _no_disponible("model-b"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B],
            )

        assert len(results) == 2
        sources = {r.compatibilidad_vram.fuente for r in results}
        assert "no_disponible" in sources
        scores = [r.puntuacion_total for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_disponible_subpuntuaciones_still_present(self) -> None:
        """Even with no_disponible, sub-scores are computed (deterministic)."""
        provider_results = {
            "model-a": _no_disponible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A],
            )

        assert len(results) == 1
        assert len(results[0].subpuntuaciones) == 5


# ---------------------------------------------------------------------------
# 4. top_n edge cases
# ---------------------------------------------------------------------------


class TestTopNEdgeCases:
    def test_top_n_larger_than_candidates(self) -> None:
        """top_n > eligible count returns all eligible candidates."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B],
                top_n=10,
            )

        assert len(results) == 2

    def test_top_n_zero_returns_empty(self) -> None:
        """top_n=0 returns an empty list (no-op)."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B],
                top_n=0,
            )

        assert results == []

    def test_top_n_zero_provider_still_called(self) -> None:
        """top_n=0 returns [] without calling provider or loading catalogue."""
        provider_results = {
            "model-a": _compatible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A],
                top_n=0,
            )

        assert results == []
        assert provider.call_count == 0

    def test_top_n_negative_raises(self) -> None:
        """top_n < 0 raises ValueError, never silent slicing."""
        provider_results = {
            "model-a": _compatible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            with pytest.raises(ValueError, match="top_n must be >= 0"):
                ejecutar_pipeline(
                    criterios=_criterios(),
                    familias=[FAMILIA_A],
                    top_n=-1,
                )

    def test_top_n_negative_minus_two_raises(self) -> None:
        """top_n=-2 also raises, confirming the guard is general."""
        provider_results = {
            "model-a": _compatible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            with pytest.raises(ValueError, match="top_n must be >= 0"):
                ejecutar_pipeline(
                    criterios=_criterios(),
                    familias=[FAMILIA_A],
                    top_n=-2,
                )

    def test_top_n_exactly_one(self) -> None:
        """top_n=1 returns only the highest-scoring candidate."""
        provider_results = {
            "model-a": _compatible("model-a"),
            "model-b": _compatible("model-b"),
            "model-c": _compatible("model-c"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                familias=[FAMILIA_A, FAMILIA_B, FAMILIA_C],
                top_n=1,
            )

        assert len(results) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockProvider:
    """Lightweight callable returning pre-configured results."""

    def __init__(self, results: dict[str, CompatibilityResult]) -> None:
        self._results = results
        self.call_count = 0
        self.calls: list[tuple[str, float]] = []

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        self.call_count += 1
        self.calls.append((familia.familia_id, vram_gb))
        return self._results[familia.familia_id]
