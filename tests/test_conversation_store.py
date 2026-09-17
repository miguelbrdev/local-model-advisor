"""Tests for src.state.conversation_store.

Uses a temporary SQLite database to avoid polluting production data.
Covers: save/load, TTL expiry, delete, and cleanup.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    ConversationRun,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.state import conversation_store


def _make_run(conversation_id=None) -> ConversationRun:
    """Build a minimal ConversationRun for testing."""
    return ConversationRun(
        conversation_id=conversation_id or uuid4(),
        criterios=CriteriosBusqueda(tarea="codigo", vram_gb=11.0),
        candidatos_elegibles=[],
        historial=[],
        created_at=datetime.utcnow(),
    )


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: Path):
    """Redirect the store to a temporary database for every test."""
    db_path = tmp_path / "test_conversations.db"
    with patch.object(conversation_store, "_DB_PATH", db_path):
        yield


# ---------------------------------------------------------------------------
# guardar / cargar
# ---------------------------------------------------------------------------

class TestGuardarCargar:
    def test_round_trip(self) -> None:
        run = _make_run()
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is not None
        assert loaded.conversation_id == run.conversation_id
        assert loaded.criterios.tarea == "codigo"
        assert loaded.criterios.vram_gb == 11.0

    def test_cargar_nonexistent_returns_none(self) -> None:
        assert conversation_store.cargar(uuid4()) is None

    def test_upsert_overwrites(self) -> None:
        run = _make_run()
        conversation_store.guardar(run)
        # Modify and save again
        run.criterios.vram_gb = 24.0
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is not None
        assert loaded.criterios.vram_gb == 24.0

    def test_preserves_candidatos(self) -> None:
        run = _make_run()
        run.candidatos_elegibles = [
            CandidatoRankeado(
                familia=FamiliaModelo(
                    familia_id="m1", variantes=["7B"], pipeline_tag="code"
                ),
                variante_recomendada="7B",
                compatibilidad_vram=CompatibilityResult(
                    modelo_id="m1", variante="7B", compatible=True
                ),
            )
        ]
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is not None
        assert len(loaded.candidatos_elegibles) == 1
        assert loaded.candidatos_elegibles[0].familia.familia_id == "m1"


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestTTL:
    def test_expired_run_returns_none(self) -> None:
        run = _make_run()
        # Set created_at to 25 hours ago
        run.created_at = datetime.utcnow() - timedelta(hours=25)
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is None

    def test_fresh_run_returns_ok(self) -> None:
        run = _make_run()
        run.created_at = datetime.utcnow() - timedelta(hours=23)
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is not None

    def test_exactly_at_ttl_returns_none(self) -> None:
        run = _make_run()
        run.created_at = datetime.utcnow() - timedelta(hours=24, minutes=1)
        conversation_store.guardar(run)
        loaded = conversation_store.cargar(run.conversation_id)
        assert loaded is None


# ---------------------------------------------------------------------------
# eliminar
# ---------------------------------------------------------------------------

class TestEliminar:
    def test_delete_removes_run(self) -> None:
        run = _make_run()
        conversation_store.guardar(run)
        conversation_store.eliminar(run.conversation_id)
        assert conversation_store.cargar(run.conversation_id) is None

    def test_delete_nonexistent_is_noop(self) -> None:
        conversation_store.eliminar(uuid4())  # should not raise


# ---------------------------------------------------------------------------
# limpiar_expiradas
# ---------------------------------------------------------------------------

class TestLimpiarExpiradas:
    def test_deletes_expired(self) -> None:
        fresh = _make_run()
        fresh.created_at = datetime.utcnow() - timedelta(hours=1)
        expired = _make_run()
        expired.created_at = datetime.utcnow() - timedelta(hours=25)

        conversation_store.guardar(fresh)
        conversation_store.guardar(expired)

        deleted = conversation_store.limpiar_expiradas()
        assert deleted >= 1
        assert conversation_store.cargar(fresh.conversation_id) is not None
        assert conversation_store.cargar(expired.conversation_id) is None

    def test_returns_zero_when_nothing_expired(self) -> None:
        run = _make_run()
        run.created_at = datetime.utcnow()
        conversation_store.guardar(run)
        deleted = conversation_store.limpiar_expiradas()
        assert deleted == 0
