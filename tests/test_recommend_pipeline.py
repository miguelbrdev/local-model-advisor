"""Tests for src.chains.recommend_pipeline.

Uses a small test catalogue (served through an in-memory
``CatalogoModelos`` double), valid Pydantic criteria, and a mocked VRAM
provider. No HTTP requests, no real SQLite, no external file dependencies.
"""

from __future__ import annotations

import importlib.util
import inspect
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.catalog.protocol import CatalogUnavailableError
from src.chains import recommend_pipeline
from src.chains.recommend_pipeline import ejecutar_pipeline
from src.models.schemas import (
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)

# ---------------------------------------------------------------------------
# Small test catalogue
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAMILIA_A = FamiliaModelo(
    familia_id="model-a",
    variantes=["7B", "14B"],
    pipeline_tag="codigo",
    tareas_soportadas=["codigo"],
    license_declarada="Apache 2.0",
    idiomas_detectados=["en", "es", "zh", "fr"],
    formatos_disponibles=["gguf"],
    canirun_id="model-a-7b",
)

FAMILIA_B = FamiliaModelo(
    familia_id="model-b",
    variantes=["7B"],
    pipeline_tag="chat",
    tareas_soportadas=["chat", "codigo"],
    license_declarada="MIT",
    idiomas_detectados=["en"],
    formatos_disponibles=["gguf"],
    canirun_id="model-b-7b",
)

FAMILIA_C = FamiliaModelo(
    familia_id="model-c",
    variantes=["3B"],
    pipeline_tag="codigo",
    tareas_soportadas=["codigo"],
    license_declarada="Apache 2.0",
    idiomas_detectados=["en", "es"],
    formatos_disponibles=["gguf"],
    canirun_id="model-c-3b",
)


def _criterios(**overrides) -> CriteriosBusqueda:
    defaults = {"tarea": "codigo", "vram_gb": 11.0}
    defaults.update(overrides)
    return CriteriosBusqueda(**defaults)


class _Catalogo:
    """In-memory ``CatalogoModelos`` double.

    Mirrors ``tests/test_catalog_loader.py``'s double but also counts
    loads, so tests can pin *how often* the pipeline touches its
    dependency (e.g. never on the ``top_n=0`` short-circuit).
    """

    def __init__(self, familias: list[FamiliaModelo]) -> None:
        self._familias = list(familias)
        self.load_count = 0

    @property
    def snapshot_id(self) -> str | None:
        return "snap-test"

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def cargar(self) -> list[FamiliaModelo]:
        self.load_count += 1
        return list(self._familias)


def _catalogo(*familias: FamiliaModelo) -> _Catalogo:
    return _Catalogo(list(familias))


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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
                top_n=2,
            )

        assert len(results) == 2
        assert all(r.puntuacion_total >= 0 for r in results)

    def test_subpuntuaciones_present(self) -> None:
        """Each result contains 7 explainable sub-scores."""
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
            )

        for candidate in results:
            assert len(candidate.subpuntuaciones) == 7
            names = {s.criterio for s in candidate.subpuntuaciones}
            assert names == {
                "ajuste_vram", "capacidad", "capacidad_medida", "especializacion",
                "recencia", "contexto", "licencia",
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
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
                catalogo=_catalogo(),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B),
            )

        assert results == []

    def test_task_filter_removes_all_returns_empty(self) -> None:
        """Task mismatch filters out all families → empty list."""
        provider = _MockProvider({})

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(tarea="chat"),
                catalogo=_catalogo(FAMILIA_A, FAMILIA_C),
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
                catalogo=_catalogo(),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B),
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
                catalogo=_catalogo(FAMILIA_A),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B),
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
                catalogo=_catalogo(FAMILIA_A),
            )

        assert len(results) == 1
        assert len(results[0].subpuntuaciones) == 7


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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B),
                top_n=0,
            )

        assert results == []

    def test_top_n_zero_short_circuits_before_loading(self) -> None:
        """top_n=0 returns [] without calling provider or loading catalogue."""
        provider_results = {
            "model-a": _compatible("model-a"),
        }
        provider = _MockProvider(provider_results)
        cat = _catalogo(FAMILIA_A)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(),
                catalogo=cat,
                top_n=0,
            )

        assert results == []
        assert provider.call_count == 0
        assert cat.load_count == 0

    def test_top_n_negative_raises(self) -> None:
        """top_n < 0 raises ValueError, never silent slicing."""
        provider_results = {
            "model-a": _compatible("model-a"),
        }
        provider = _MockProvider(provider_results)

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ), pytest.raises(ValueError, match="top_n must be >= 0"):
            ejecutar_pipeline(
                criterios=_criterios(),
                catalogo=_catalogo(FAMILIA_A),
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
        ), pytest.raises(ValueError, match="top_n must be >= 0"):
            ejecutar_pipeline(
                criterios=_criterios(),
                catalogo=_catalogo(FAMILIA_A),
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
                catalogo=_catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C),
                top_n=1,
            )

        assert len(results) == 1


# ---------------------------------------------------------------------------
# 5. Catalogue injection (dependency inversion)
# ---------------------------------------------------------------------------


class TestCatalogueInjection:
    """The pipeline must reach families only through the injected Protocol."""

    def test_reads_families_from_injected_catalogue(self) -> None:
        """Families served by the double are the ones ranked."""
        cat = _catalogo(FAMILIA_A, FAMILIA_B, FAMILIA_C)
        provider = _MockProvider(
            {
                "model-a": _compatible("model-a"),
                "model-b": _compatible("model-b"),
                "model-c": _compatible("model-c"),
            }
        )

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(criterios=_criterios(), catalogo=cat)

        assert cat.load_count == 1
        assert {r.familia.familia_id for r in results} == {"model-a", "model-b", "model-c"}

    def test_legacy_seed_catalogue_is_gone(self) -> None:
        """The YAML seed and its loader were removed so the synced
        catalogue is the only source of truth; nothing may bring them back.
        """
        assert not hasattr(recommend_pipeline, "load_seed")
        assert importlib.util.find_spec("src.tools.seed_loader") is None
        assert not (PROJECT_ROOT / "data" / "families_seed.yaml").exists()

    def test_scorer_receives_the_users_vram(self) -> None:
        """``capacidad`` must be computed against ``criterios.vram_gb``:
        4 GB required of a 12.8 GB usable budget (80 % of 16) -> 0.3125."""
        provider = _MockProvider({"model-a": _compatible("model-a", vram_required_gb=4.0)})

        with patch(
            "src.chains.recommend_pipeline.elegir_provider", return_value=provider
        ):
            results = ejecutar_pipeline(
                criterios=_criterios(vram_gb=16.0), catalogo=_catalogo(FAMILIA_A)
            )

        capacidad = next(s for s in results[0].subpuntuaciones if s.criterio == "capacidad")
        assert capacidad.valor == pytest.approx(0.3125)

    def test_catalogue_failure_propagates_untouched(self) -> None:
        """An exhausted catalogue must surface as its typed error, never
        as an empty candidate list (the ambiguity this project forbids).
        """

        class _ExhaustedCatalogo:
            @property
            def snapshot_id(self) -> str | None:
                raise CatalogUnavailableError("no source")

            @property
            def fecha_sincronizacion(self) -> datetime | None:
                raise CatalogUnavailableError("no source")

            def cargar(self) -> list[FamiliaModelo]:
                raise CatalogUnavailableError("no source")

        with pytest.raises(CatalogUnavailableError):
            ejecutar_pipeline(criterios=_criterios(), catalogo=_ExhaustedCatalogo())

    def test_injected_vram_provider_replaces_task_routing(self) -> None:
        """When a provider is injected, ``elegir_provider`` must not be
        consulted at all — the evaluation relies on that to stay offline."""
        provider = _MockProvider({"model-a": _compatible("model-a")})

        with patch("src.chains.recommend_pipeline.elegir_provider") as routing:
            results = ejecutar_pipeline(
                criterios=_criterios(), catalogo=_catalogo(FAMILIA_A), vram_provider=provider
            )

        routing.assert_not_called()
        assert provider.call_count == 1
        assert [r.familia.familia_id for r in results] == ["model-a"]

    def test_scorer_receives_task_and_newest_release_date(self) -> None:
        """``especializacion`` must see the requested task and ``recencia``
        the newest date of the catalogue, not of the candidate."""
        viejo = FAMILIA_A.model_copy(update={"fecha_lanzamiento": "2024-08"})
        nuevo = FAMILIA_C.model_copy(update={"fecha_lanzamiento": "2026-08"})
        provider = _MockProvider(
            {"model-a": _compatible("model-a"), "model-c": _compatible("model-c")}
        )

        with patch("src.chains.recommend_pipeline.elegir_provider", return_value=provider):
            results = ejecutar_pipeline(criterios=_criterios(), catalogo=_catalogo(viejo, nuevo))

        por_id = {r.familia.familia_id: r for r in results}
        sub = lambda r, k: next(s for s in r.subpuntuaciones if s.criterio == k)  # noqa: E731
        assert sub(por_id["model-c"], "recencia").valor == 1.0
        assert sub(por_id["model-a"], "recencia").valor == 0.0
        assert sub(por_id["model-a"], "especializacion").valor == 1.0  # dedicated codigo

    def test_signature_requires_keyword_only_catalogue(self) -> None:
        """Pin design (a): no ``familias`` escape hatch, ``catalogo``
        keyword-only so it can never be confused with a positional
        familias list again, and *required* — omitting it must be a
        ``TypeError`` at the call site, never a ``None``-guarded empty
        result (the ambiguous empty-catalogue path this project forbids).
        """
        params = inspect.signature(ejecutar_pipeline).parameters
        assert "familias" not in params
        assert params["catalogo"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["catalogo"].default is inspect.Parameter.empty
        assert params["criterios"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


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
