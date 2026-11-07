"""Tests for src.compatibility.provider — CanIRunProvider.

All tests mock httpx to avoid real HTTP calls. Covers: successful response,
timeout, HTTP error, and missing canirun_id.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.compatibility.heuristica import HeuristicaParametrosProvider
from src.compatibility.provider import (
    CanIRunProvider,
    ProveedorConFallback,
    ProveedorExternoError,
    TamanoDirectoProvider,
    elegir_provider,
)
from src.models.schemas import CompatibilityResult, FamiliaModelo


def _familia_con_id() -> FamiliaModelo:
    return FamiliaModelo(
        familia_id="test-model",
        variantes=["7B", "14B"],
        pipeline_tag="code",
        canirun_id="test-model-7b",
    )


def _familia_sin_id() -> FamiliaModelo:
    return FamiliaModelo(
        familia_id="no-mapping",
        variantes=["7B"],
        pipeline_tag="code",
        canirun_id=None,
    )


# ---------------------------------------------------------------------------
# CanIRunProvider — missing canirun_id
# ---------------------------------------------------------------------------

class TestCanIRunSinMapping:
    def test_returns_no_disponible(self) -> None:
        provider = CanIRunProvider()
        result = provider.check(_familia_sin_id(), vram_gb=11.0)
        assert result.fuente == "no_disponible"
        assert result.compatible is None
        assert "No canirun_id" in result.nota


# ---------------------------------------------------------------------------
# CanIRunProvider — successful response
# ---------------------------------------------------------------------------

class TestCanIRunExito:
    @patch("src.compatibility.provider.httpx.post")
    def test_returns_compatible_on_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compatible": True,
            "status": "comfortable",
            "notes": ["Fits well in GPU memory."],
            "estimated": {
                "tokensPerSecond": 42.5,
                "vramRequiredGb": 4.1,
            },
        }
        mock_post.return_value = mock_response

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)

        assert result.fuente == "canirun_ai"
        assert result.compatible is True
        assert result.grado == "comfortable"
        assert result.tokens_per_second == 42.5
        assert result.vram_required_gb == 4.1
        assert "Fits well" in result.nota

    @patch("src.compatibility.provider.httpx.post")
    def test_uses_correct_api_url(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"compatible": True, "status": "ok", "notes": []}
        mock_post.return_value = mock_response

        provider = CanIRunProvider()
        provider.check(_familia_con_id(), vram_gb=11.0)

        call_args = mock_post.call_args
        assert "canirun.ai/api/compatibility" in call_args[0][0]

    @patch("src.compatibility.provider.httpx.post")
    def test_uses_last_variant(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"compatible": True, "status": "ok", "notes": []}
        mock_post.return_value = mock_response

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)
        assert result.variante == "14B"  # last variant


    @patch("src.compatibility.provider.httpx.post")
    def test_synced_family_without_variants_reports_parameter_text(
        self, mock_post: MagicMock
    ) -> None:
        """Synced records carry no ``variantes``; the parameter text is the
        only meaningful label left (never a bare ``"unknown"``)."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"compatible": True, "status": "ok", "notes": []}
        mock_post.return_value = mock_response
        familia = FamiliaModelo(
            familia_id="qwen3.5-0.8b",
            variantes=[],
            pipeline_tag="chat",
            canirun_id="qwen3.5-0.8b",
            parametros_texto="0.8B",
        )

        result = CanIRunProvider().check(familia, vram_gb=11.0)
        assert result.variante == "0.8B"

    def test_without_variants_or_parameter_text_is_unknown(self) -> None:
        familia = FamiliaModelo(familia_id="x", variantes=[], pipeline_tag="chat")
        result = CanIRunProvider().check(familia, vram_gb=11.0)
        assert result.variante == "unknown"


# ---------------------------------------------------------------------------
# CanIRunProvider — timeout
# ---------------------------------------------------------------------------

class TestCanIRunTimeout:
    @patch("src.compatibility.provider.httpx.post")
    def test_timeout_returns_no_disponible(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)

        assert result.fuente == "no_disponible"
        assert result.compatible is None
        assert "timeout" in result.nota.lower() or "timed" in result.nota.lower()


# ---------------------------------------------------------------------------
# CanIRunProvider — HTTP error
# ---------------------------------------------------------------------------

class TestCanIRunHTTPError:
    @patch("src.compatibility.provider.httpx.post")
    def test_500_returns_no_disponible(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_post.return_value = mock_response

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)

        assert result.fuente == "no_disponible"
        assert result.compatible is None

    @patch("src.compatibility.provider.httpx.post")
    def test_404_returns_no_disponible(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        mock_post.return_value = mock_response

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)

        assert result.fuente == "no_disponible"

    @patch("src.compatibility.provider.httpx.post")
    def test_generic_exception_returns_no_disponible(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RuntimeError("Unexpected failure")

        provider = CanIRunProvider()
        result = provider.check(_familia_con_id(), vram_gb=11.0)

        assert result.fuente == "no_disponible"
        assert "Unexpected failure" in result.nota


# ---------------------------------------------------------------------------
# CanIRunProvider — measured results are cached per (canirun_id, vram_gb)
# ---------------------------------------------------------------------------

class TestCanIRunCache:
    @patch("src.compatibility.provider.httpx.post")
    def test_second_call_is_served_from_cache(self, mock_post: MagicMock) -> None:
        from src.compatibility.cache import CacheCompatibilidad

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compatible": True, "status": "comfortable", "notes": [],
            "estimated": {"tokensPerSecond": 40.0, "vramRequiredGb": 4.1},
        }
        mock_post.return_value = mock_response
        provider = CanIRunProvider(cache=CacheCompatibilidad(reloj=lambda: 0.0))

        first = provider.check(_familia_con_id(), vram_gb=11.0)
        second = provider.check(_familia_con_id(), vram_gb=11.0)
        provider.check(_familia_con_id(), vram_gb=8.0)  # different budget → new call

        assert mock_post.call_count == 2
        assert first == second
        assert second.fuente == "canirun_ai"

    @patch("src.compatibility.provider.httpx.post")
    def test_failures_are_not_cached(self, mock_post: MagicMock) -> None:
        from src.compatibility.cache import CacheCompatibilidad

        mock_post.side_effect = httpx.TimeoutException("timed out")
        provider = CanIRunProvider(cache=CacheCompatibilidad(reloj=lambda: 0.0))
        provider.check(_familia_con_id(), vram_gb=11.0)
        provider.check(_familia_con_id(), vram_gb=11.0)
        assert mock_post.call_count == 2

    @patch("src.compatibility.provider.httpx.post")
    def test_no_cache_by_default(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"compatible": True, "status": "ok", "notes": []}
        mock_post.return_value = mock_response
        provider = CanIRunProvider()
        provider.check(_familia_con_id(), vram_gb=11.0)
        provider.check(_familia_con_id(), vram_gb=11.0)
        assert mock_post.call_count == 2

    def test_production_routing_uses_the_shared_cache(self) -> None:
        from src.compatibility.cache import CACHE_CANIRUN

        provider = elegir_provider("chat")
        assert provider._primario._cache is CACHE_CANIRUN


# ---------------------------------------------------------------------------
# CanIRunProvider.consultar — raises on failure (check() never does)
# ---------------------------------------------------------------------------

class TestConsultar:
    @patch("src.compatibility.provider.httpx.post")
    def test_transport_failure_raises(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(ProveedorExternoError, match="refused"):
            CanIRunProvider().consultar(_familia_con_id(), 11.0)

    def test_missing_id_is_a_gap_not_an_error(self) -> None:
        result = CanIRunProvider().consultar(_familia_sin_id(), 11.0)
        assert result.fuente == "no_disponible"
        assert result.compatible is None


# ---------------------------------------------------------------------------
# ProveedorConFallback — canirun.ai first, heuristic when it cannot answer
# ---------------------------------------------------------------------------

def _familia_7b(fid: str = "qwen-7b", canirun_id: str | None = "qwen-7b") -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid,
        variantes=[],
        pipeline_tag="chat",
        tareas_soportadas=["chat"],
        canirun_id=canirun_id,
        parametros_b=7.0,
        parametros_texto="7B",
    )


class _PrimarioScript:
    """CanIRunProvider double: a queue of results or exceptions per call."""

    def __init__(self, *respuestas) -> None:
        self._respuestas = list(respuestas)
        self.llamadas = 0

    def consultar(self, familia: FamiliaModelo, vram_gb: float):
        self.llamadas += 1
        r = self._respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _medido(fid: str, compatible: bool = True) -> CompatibilityResult:
    return CompatibilityResult(
        modelo_id=fid, variante="7B", compatible=compatible, grado="comfortable",
        fuente="canirun_ai", vram_required_gb=5.0,
    )


class TestProveedorConFallback:
    def test_measured_result_is_returned_untouched(self) -> None:
        primario = _PrimarioScript(_medido("qwen-7b"))
        provider = ProveedorConFallback(primario, HeuristicaParametrosProvider())
        result = provider.check(_familia_7b(), 16.0)
        assert result.fuente == "canirun_ai"
        assert result.vram_required_gb == 5.0
        assert provider.primario_caido is False

    def test_outage_falls_back_and_stops_calling_the_service(self) -> None:
        primario = _PrimarioScript(ProveedorExternoError("canirun.ai API error: timeout"))
        provider = ProveedorConFallback(primario, HeuristicaParametrosProvider())

        first = provider.check(_familia_7b("a"), 16.0)
        second = provider.check(_familia_7b("b"), 16.0)

        assert provider.primario_caido is True
        assert primario.llamadas == 1  # second candidate never hit the service
        for r in (first, second):
            assert r.fuente == "heuristica_propia"
            assert r.compatible is True
            assert r.vram_required_gb == 5.2
            assert r.nota.startswith("Estimated locally (canirun.ai API error: timeout)")

    def test_gap_falls_back_without_tripping_the_breaker(self) -> None:
        sin_id = CanIRunProvider().consultar(_familia_7b(canirun_id=None), 16.0)
        primario = _PrimarioScript(sin_id, _medido("b"))
        provider = ProveedorConFallback(primario, HeuristicaParametrosProvider())

        first = provider.check(_familia_7b("a", canirun_id=None), 16.0)
        second = provider.check(_familia_7b("b"), 16.0)

        assert first.fuente == "heuristica_propia"
        assert "No canirun_id" in first.nota
        assert second.fuente == "canirun_ai"
        assert provider.primario_caido is False
        assert primario.llamadas == 2

    def test_unknown_stays_unknown_when_heuristic_cannot_estimate(self) -> None:
        sin_params = _familia_7b(canirun_id=None).model_copy(update={"parametros_b": None})
        sin_id = CanIRunProvider().consultar(sin_params, 16.0)
        provider = ProveedorConFallback(_PrimarioScript(sin_id), HeuristicaParametrosProvider())
        result = provider.check(sin_params, 16.0)
        assert result.fuente == "no_disponible"
        assert result.compatible is None
        assert "No canirun_id" in result.nota

    def test_outage_and_no_parameters_yields_heuristic_unknown(self) -> None:
        sin_params = _familia_7b().model_copy(update={"parametros_b": None})
        primario = _PrimarioScript(ProveedorExternoError("down"))
        provider = ProveedorConFallback(primario, HeuristicaParametrosProvider())
        result = provider.check(sin_params, 16.0)
        assert result.compatible is None
        assert result.fuente == "heuristica_propia"

    def test_incompatible_verdicts_from_the_heuristic_are_kept(self) -> None:
        primario = _PrimarioScript(ProveedorExternoError("down"))
        provider = ProveedorConFallback(primario, HeuristicaParametrosProvider())
        result = provider.check(_familia_7b(), 4.0)  # 5.2 GB > 4 GB
        assert result.compatible is False
        assert result.fuente == "heuristica_propia"

    @patch("src.compatibility.provider.httpx.post")
    def test_real_primary_outage_end_to_end(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.TimeoutException("timed out")
        provider = ProveedorConFallback()
        a = provider.check(_familia_7b("a"), 16.0)
        b = provider.check(_familia_7b("b"), 16.0)
        assert mock_post.call_count == 1
        assert a.fuente == b.fuente == "heuristica_propia"

    def test_defaults_and_protocol(self) -> None:
        from src.compatibility.provider import VRAMCompatibilityProvider

        assert isinstance(ProveedorConFallback(), VRAMCompatibilityProvider)


# ---------------------------------------------------------------------------
# TamanoDirectoProvider
# ---------------------------------------------------------------------------

class TestTamanoDirecto:
    def test_returns_compatible(self) -> None:
        provider = TamanoDirectoProvider()
        familia = FamiliaModelo(
            familia_id="embed-model",
            variantes=["base"],
            pipeline_tag="sentence-similarity",
        )
        result = provider.check(familia, vram_gb=8.0)
        assert result.compatible is True
        assert result.fuente == "heuristica_propia"

    def test_uses_first_variant(self) -> None:
        provider = TamanoDirectoProvider()
        familia = FamiliaModelo(
            familia_id="embed-model",
            variantes=["small", "large"],
            pipeline_tag="sentence-similarity",
        )
        result = provider.check(familia, vram_gb=8.0)
        assert result.variante == "small"


# ---------------------------------------------------------------------------
# elegir_provider routing
# ---------------------------------------------------------------------------

class TestElegirProvider:
    @pytest.mark.parametrize("tarea", ["chat", "codigo", "razonamiento"])
    def test_v1_tasks_return_canirun_with_fallback(self, tarea: str) -> None:
        provider = elegir_provider(tarea)
        assert isinstance(provider, ProveedorConFallback)
        assert provider.primario_caido is False

    def test_each_call_gets_a_fresh_outage_flag(self) -> None:
        """The outage flag must be scoped to one pipeline run."""
        assert elegir_provider("chat") is not elegir_provider("chat")

    def test_embeddings_returns_tamano(self) -> None:
        assert isinstance(elegir_provider("embeddings"), TamanoDirectoProvider)

    def test_rerankers_returns_tamano(self) -> None:
        assert isinstance(elegir_provider("rerankers"), TamanoDirectoProvider)
