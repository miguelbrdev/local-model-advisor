"""Tests for src.compatibility.provider — CanIRunProvider.

All tests mock httpx to avoid real HTTP calls. Covers: successful response,
timeout, HTTP error, and missing canirun_id.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from src.compatibility.provider import (
    CanIRunProvider,
    TamanoDirectoProvider,
    elegir_provider,
)
from src.models.schemas import FamiliaModelo


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
    def test_chat_returns_canirun(self) -> None:
        assert isinstance(elegir_provider("chat"), CanIRunProvider)

    def test_codigo_returns_canirun(self) -> None:
        assert isinstance(elegir_provider("codigo"), CanIRunProvider)

    def test_razonamiento_returns_canirun(self) -> None:
        assert isinstance(elegir_provider("razonamiento"), CanIRunProvider)

    def test_embeddings_returns_tamano(self) -> None:
        assert isinstance(elegir_provider("embeddings"), TamanoDirectoProvider)

    def test_rerankers_returns_tamano(self) -> None:
        assert isinstance(elegir_provider("rerankers"), TamanoDirectoProvider)
