"""Tests for src.sync.normalizer — external-to-internal record conversion.

All tests use in-memory fixtures and the real YAML policy.  No HTTP
calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.sync.canirun_schemas import CanIRunModelRecord, CanIRunModelsResponse
from src.sync.catalog_policy import CatalogPolicy
from src.sync.normalizer import (
    ExclusionReason,
    normalise_record,
    normalise_response,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_POLICY = CatalogPolicy()


def _load_response() -> CanIRunModelsResponse:
    with open(_FIXTURES / "canirun_models_response.json", encoding="utf-8") as fh:
        return CanIRunModelsResponse.model_validate(json.load(fh))


def _make_record(**overrides) -> CanIRunModelRecord:
    defaults = dict(
        id="test-model",
        name="Test Model",
        provider="TestProvider",
        family="TestFamily",
        params="7B",
        paramsBillions=7,
        architecture="dense",
        releaseDate="2025-01",
        contextLength=32768,
        useCase=["chat"],
        url="https://example.com",
        license="MIT",
    )
    defaults.update(overrides)
    return CanIRunModelRecord.model_validate(defaults)


# ---------------------------------------------------------------------------
# Single record normalisation
# ---------------------------------------------------------------------------


class TestNormaliseRecord:
    def test_code_model_gets_codigo_task(self) -> None:
        rec = _make_record(id="code-only", useCase=["code"])
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert result.familia_id == "code-only"
        assert result.canirun_id == "code-only"
        assert "code" not in result.tareas_soportadas  # external tag
        assert "codigo" in result.tareas_soportadas  # internal task
        assert result.pipeline_tag == "codigo"

    def test_multitarefa_conserva_all_tasks(self) -> None:
        rec = _make_record(
            id="multi",
            useCase=["chat", "code", "reasoning"],
        )
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert "chat" in result.tareas_soportadas
        assert "codigo" in result.tareas_soportadas
        assert "razonamiento" in result.tareas_soportadas
        # No duplicate records — single FamiliaModelo with all tasks
        assert result.pipeline_tag == "chat"  # first supported task

    def test_only_vision_excluded(self) -> None:
        rec = _make_record(id="vision-only", useCase=["vision"])
        result = normalise_record(rec, _POLICY)
        assert isinstance(result, ExclusionReason)
        assert result.reason == "no_supported_task"

    def test_only_image_excluded(self) -> None:
        rec = _make_record(id="image-only", useCase=["image"])
        result = normalise_record(rec, _POLICY)
        assert isinstance(result, ExclusionReason)
        assert result.reason == "no_supported_task"

    def test_only_video_excluded(self) -> None:
        rec = _make_record(id="video-only", useCase=["video"])
        result = normalise_record(rec, _POLICY)
        assert isinstance(result, ExclusionReason)
        assert result.reason == "no_supported_task"

    def test_informational_tags_preserved(self) -> None:
        rec = _make_record(
            id="multi-info",
            useCase=["chat", "multilingual", "edge", "rag"],
        )
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert "multilingual" in result.tags_informativos
        assert "edge" in result.tags_informativos
        assert "rag" in result.tags_informativos
        assert "chat" in result.tareas_soportadas

    def test_id_copied_to_familia_id_and_canirun_id(self) -> None:
        rec = _make_record(id="qwen2.5-coder-7b")
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert result.familia_id == "qwen2.5-coder-7b"
        assert result.canirun_id == "qwen2.5-coder-7b"

    def test_absent_optional_fields_default(self) -> None:
        rec = _make_record(
            id="minimal",
            architecture="",
            releaseDate="",
            contextLength=0,
            license="",
            url="",
        )
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert result.license_declarada is None
        assert result.license_url is None
        assert result.idiomas_detectados is None
        assert result.formatos_disponibles == []
        assert result.arquitectura is None
        assert result.fecha_lanzamiento is None
        assert result.contexto_maximo is None

    def test_empty_id_excluded(self) -> None:
        rec = _make_record(id="")
        result = normalise_record(rec, _POLICY)
        assert isinstance(result, ExclusionReason)
        assert result.reason == "empty_id"

    def test_no_supported_task_excluded(self) -> None:
        rec = _make_record(id="out-of-scope", useCase=["vision", "image"])
        result = normalise_record(rec, _POLICY)
        assert isinstance(result, ExclusionReason)
        assert result.reason == "no_supported_task"

    def test_tags_informativos_not_in_tareas(self) -> None:
        rec = _make_record(
            id="tagged",
            useCase=["chat", "multilingual", "rag"],
        )
        result = normalise_record(rec, _POLICY)
        from src.models.schemas import FamiliaModelo

        assert isinstance(result, FamiliaModelo)
        assert "multilingual" in result.tags_informativos
        assert "rag" in result.tags_informativos
        assert "multilingual" not in result.tareas_soportadas
        assert "rag" not in result.tareas_soportadas


# ---------------------------------------------------------------------------
# Full response normalisation
# ---------------------------------------------------------------------------


class TestNormaliseResponse:
    def test_fixture_normalises(self) -> None:
        resp = _load_response()
        result = normalise_response(resp, _POLICY)
        assert result.source_count == 6
        # qwen2.5-coder-7b (code), llama3.1-8b (chat,code,reasoning),
        # deepseek-r1-7b (reasoning), qwen3-8b (chat,code,reasoning,multilingual)
        # flux2-dev (image) -> excluded, wan2.1-t2v-1.3b (video) -> excluded
        assert result.included_count == 4
        assert len(result.exclusions) == 2

    def test_excluded_models_have_reasons(self) -> None:
        resp = _load_response()
        result = normalise_response(resp, _POLICY)
        excluded_ids = {e.record_id for e in result.exclusions}
        assert "flux2-dev" in excluded_ids
        assert "wan2.1-t2v-1.3b" in excluded_ids
        for exc in result.exclusions:
            assert exc.reason == "no_supported_task"

    def test_duplicate_id_detected(self) -> None:
        record = _make_record(id="dup-model", useCase=["chat"])
        dup_response = CanIRunModelsResponse(
            count=2,
            models=[record, record],
        )
        result = normalise_response(dup_response, _POLICY)
        assert result.included_count == 1
        assert len(result.exclusions) == 1
        assert result.exclusions[0].reason == "duplicate_id"

    def test_deterministic_output(self) -> None:
        resp = _load_response()
        r1 = normalise_response(resp, _POLICY)
        r2 = normalise_response(resp, _POLICY)
        assert [f.familia_id for f in r1.families] == [
            f.familia_id for f in r2.families
        ]
        assert r1.exclusions == r2.exclusions

    def test_no_http_calls(self) -> None:
        """Verify no httpx or urllib usage in normalizer module."""
        import inspect

        import src.sync.normalizer as mod

        source = inspect.getsource(mod)
        assert "httpx" not in source
        assert "urllib" not in source
        assert "requests" not in source

    def test_multitarefa_single_record_not_duplicated(self) -> None:
        resp = CanIRunModelsResponse(
            count=1,
            models=[_make_record(id="multi", useCase=["chat", "code", "reasoning"])],
        )
        result = normalise_response(resp, _POLICY)
        assert result.included_count == 1
        assert len(result.families) == 1
        fam = result.families[0]
        assert len(fam.tareas_soportadas) == 3

    def test_extended_fields_populated(self) -> None:
        resp = _load_response()
        result = normalise_response(resp, _POLICY)
        qwen = next(f for f in result.families if f.familia_id == "qwen2.5-coder-7b")
        assert qwen.nombre_mostrado == "Qwen 2.5 Coder 7B"
        assert qwen.proveedor == "Alibaba"
        assert qwen.familia_origen == "Qwen"
        assert qwen.parametros_b == 7
        assert qwen.parametros_texto == "7B"
        assert qwen.arquitectura == "dense"
        assert qwen.fecha_lanzamiento == "2024-11"
        assert qwen.contexto_maximo == 131072
        assert qwen.license_declarada == "Apache 2.0"
        assert qwen.license_url == "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"
        assert qwen.variantes == []
