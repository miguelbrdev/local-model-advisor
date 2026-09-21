"""Tests for src.sync.canirun_schemas — external contract validation.

All tests use in-memory fixtures.  No HTTP calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sync.canirun_schemas import CanIRunModelRecord, CanIRunModelsResponse

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str = "canirun_models_response.json") -> dict:
    with open(_FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Root response validation
# ---------------------------------------------------------------------------


class TestModelsResponse:
    def test_valid_response_parses(self) -> None:
        raw = _load_fixture()
        resp = CanIRunModelsResponse.model_validate(raw)
        assert resp.count == 6
        assert len(resp.models) == 6

    def test_missing_count_rejected(self) -> None:
        raw = _load_fixture()
        del raw["count"]
        with pytest.raises(Exception):
            CanIRunModelsResponse.model_validate(raw)

    def test_missing_models_rejected(self) -> None:
        raw = _load_fixture()
        del raw["models"]
        with pytest.raises(Exception):
            CanIRunModelsResponse.model_validate(raw)

    def test_count_mismatch_does_not_crash(self) -> None:
        """count can differ from len(models) — we trust models list."""
        raw = _load_fixture()
        raw["count"] = 999
        resp = CanIRunModelsResponse.model_validate(raw)
        assert resp.count == 999
        assert len(resp.models) == 6

    def test_empty_models_list(self) -> None:
        resp = CanIRunModelsResponse(count=0, models=[])
        assert resp.models == []


# ---------------------------------------------------------------------------
# Individual record validation
# ---------------------------------------------------------------------------


class TestModelRecord:
    def test_valid_record(self) -> None:
        raw = _load_fixture()
        rec = CanIRunModelRecord.model_validate(raw["models"][0])
        assert rec.id == "qwen2.5-coder-7b"
        assert rec.name == "Qwen 2.5 Coder 7B"
        assert rec.useCase == ["code"]

    def test_all_required_fields_present(self) -> None:
        raw = _load_fixture()
        for m in raw["models"]:
            rec = CanIRunModelRecord.model_validate(m)
            assert rec.id
            assert rec.name

    def test_missing_id_rejected(self) -> None:
        data = {
            "name": "Test",
            "provider": "X",
            "family": "X",
            "params": "1B",
            "paramsBillions": 1,
        }
        with pytest.raises(Exception):
            CanIRunModelRecord.model_validate(data)

    def test_use_case_list_preserved(self) -> None:
        raw = _load_fixture()
        rec = CanIRunModelRecord.model_validate(raw["models"][3])
        assert rec.useCase == ["chat", "code", "reasoning", "multilingual"]

    def test_architecture_defaults_to_empty(self) -> None:
        rec = CanIRunModelRecord(
            id="x", name="X", provider="P", family="F",
            params="1B", paramsBillions=1,
        )
        assert rec.architecture == ""
        assert rec.releaseDate == ""
        assert rec.contextLength == 0
