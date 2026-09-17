"""Tests for src.api.main — FastAPI endpoints.

Uses TestClient (synchronous) for /health and /families.
Uses mocked pipeline for /chat to avoid real canirun.ai calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _mock_pipeline_result() -> list[CandidatoRankeado]:
    return [
        CandidatoRankeado(
            familia=FamiliaModelo(
                familia_id="mock-model",
                variantes=["7B"],
                pipeline_tag="code",
                license_declarada="MIT",
                idiomas_detectados=["en"],
                formatos_disponibles=["gguf"],
            ),
            variante_recomendada="7B",
            compatibilidad_vram=CompatibilityResult(
                modelo_id="mock-model",
                variante="7B",
                compatible=True,
                grado="comfortable",
                fuente="canirun_ai",
                tokens_per_second=40.0,
                vram_required_gb=4.0,
            ),
            subpuntuaciones=[],
            puntuacion_total=0.8,
        )
    ]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /families
# ---------------------------------------------------------------------------

class TestFamilies:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/families")
        assert response.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        data = client.get("/families").json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_each_family_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/families").json()
        for fam in data:
            assert "familia_id" in fam
            assert "variantes" in fam
            assert "pipeline_tag" in fam


# ---------------------------------------------------------------------------
# POST /chat — new conversation
# ---------------------------------------------------------------------------

class TestChatNuevaConversacion:
    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_200(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200

    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_conversation_id(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert "conversation_id" in data
        uuid4()  # validate it's a valid UUID
        from uuid import UUID
        UUID(data["conversation_id"])  # raises if invalid

    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_candidatos(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "recomendacion"
        assert len(data["candidatos"]) == 1
        assert data["candidatos"][0]["familia"]["familia_id"] == "mock-model"

    @patch("src.api.main.ejecutar_pipeline")
    def test_calls_pipeline_with_criteria(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps(
                {"tarea": "codigo", "vram_gb": 11.0, "idiomas_requeridos": ["en"]}
            )
        }
        client.post("/chat", json=payload)
        call_args = mock_pipeline.call_args
        criterios: CriteriosBusqueda = call_args[0][0]
        assert criterios.tarea == "codigo"
        assert criterios.vram_gb == 11.0
        assert criterios.idiomas_requeridos == ["en"]

    @patch("src.api.main.ejecutar_pipeline")
    def test_persists_conversation(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        conv_id = data["conversation_id"]

        # Follow-up with same conversation_id should reuse existing criteria
        mock_pipeline.return_value = _mock_pipeline_result()
        followup = {
            "conversation_id": conv_id,
            "mensaje": "tell me more",
        }
        data2 = client.post("/chat", json=followup).json()
        assert data2["conversation_id"] == conv_id


# ---------------------------------------------------------------------------
# POST /chat — missing fields
# ---------------------------------------------------------------------------

class TestChatCamposFaltantes:
    def test_missing_tarea_returns_clarificacion(self, client: TestClient) -> None:
        payload = {
            "mensaje": json.dumps({"vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "clarificacion"
        assert data["campo_faltante"] == "tarea"

    def test_missing_vram_returns_clarificacion(self, client: TestClient) -> None:
        payload = {
            "mensaje": json.dumps({"tarea": "codigo"})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "clarificacion"
        assert data["campo_faltante"] == "vram_gb"


# ---------------------------------------------------------------------------
# POST /chat — invalid JSON in mensaje
# ---------------------------------------------------------------------------

class TestChatInvalidMessage:
    def test_invalid_json_returns_error(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as client:
            payload = {"mensaje": "not valid json {"}
            response = client.post("/chat", json=payload)
            assert response.status_code == 500
