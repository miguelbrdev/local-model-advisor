"""Tests for src.sync.catalog_sync — mocked HTTP synchronisation cycle.

Every test injects ``httpx.MockTransport`` and writes only under
``tmp_path``.  No real network, no canirun.ai, no persistent files
outside ``tmp_path``, and no reliance on wall-clock time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from src.sync.catalog_sync import SyncResult, sync_catalog
from src.sync.snapshot import load_active_snapshot, read_manifest, read_snapshot

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_DEFAULT_ENDPOINT = "https://www.canirun.ai/api/models"


def _fixture_payload() -> dict[str, Any]:
    return json.loads(
        (_FIXTURES / "canirun_models_response.json").read_text(encoding="utf-8")
    )


def _record(rid: str, use_case: list[str]) -> dict[str, Any]:
    return {
        "id": rid,
        "name": rid.title(),
        "provider": "TestProvider",
        "family": "TestFamily",
        "params": "7B",
        "paramsBillions": 7,
        "architecture": "dense",
        "releaseDate": "2024-01",
        "contextLength": 4096,
        "useCase": use_case,
        "url": "https://example.com/model",
        "license": "MIT",
    }


def _json_transport(payload: dict[str, Any], status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


def _text_transport(text: str, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=text)

    return httpx.MockTransport(handler)


def _timeout_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout")

    return httpx.MockTransport(handler)


def _server_error_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    return httpx.MockTransport(handler)


def _assert_disk_clean(catalog_dir: Path) -> None:
    assert not (catalog_dir / "current.json").exists()
    assert list(catalog_dir.glob("snap-*.json")) == []
    assert list(catalog_dir.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Successful synchronisation
# ---------------------------------------------------------------------------


class TestSyncSuccess:
    def test_valid_response_writes_snapshot_and_manifest(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert result.ok
        assert result.error is None
        assert result.snapshot_id is not None
        assert result.snapshot_id.startswith("snap-")
        assert (tmp_path / f"{result.snapshot_id}.json").exists()
        manifest = read_manifest(tmp_path)
        assert manifest is not None
        assert manifest.active_snapshot_id == result.snapshot_id
        assert manifest.previous_snapshot_id is None
        assert list(tmp_path.glob("*.tmp")) == []

    def test_families_converted(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert result.ok
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        assert snap is not None
        assert len(snap.familias) == 4
        qwen = next(f for f in snap.familias if f.familia_id == "qwen2.5-coder-7b")
        assert qwen.nombre_mostrado == "Qwen 2.5 Coder 7B"
        assert qwen.tareas_soportadas == ["codigo"]
        assert qwen.license_declarada == "Apache 2.0"
        assert snap.meta.source_url == _DEFAULT_ENDPOINT
        assert snap.meta.politica_version == "1.0.0"
        assert snap.meta.modelos_origen == 6
        assert snap.meta.modelos_incluidos == 4

    def test_out_of_scope_exclusions_recorded(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert result.ok
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        assert snap is not None
        reasons = {e.record_id: e.reason for e in snap.meta.motivos_exclusion}
        assert reasons.get("flux2-dev") == "out_of_scope"
        assert reasons.get("wan2.1-t2v-1.3b") == "out_of_scope"

    def test_deduplication(self, tmp_path: Path) -> None:
        rec = _record("dup-model", ["chat"])
        payload = {"count": 2, "models": [rec, rec]}
        result = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert result.ok
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        assert snap is not None
        assert len(snap.familias) == 1
        assert snap.meta.modelos_origen == 2
        assert snap.meta.modelos_incluidos == 1

    def test_unexpected_extra_field_ignored(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        payload["unexpected_top_level"] = {"foo": "bar"}
        result = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert result.ok

    def test_custom_endpoint_used(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json=_fixture_payload())

        transport = httpx.MockTransport(handler)
        result = sync_catalog(
            tmp_path,
            endpoint="https://example.test/api/models",
            transport=transport,
        )
        assert result.ok
        assert seen == ["https://example.test/api/models"]
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        assert snap is not None
        assert snap.meta.source_url == "https://example.test/api/models"

    def test_load_active_snapshot_after_sync(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert result.ok
        loaded = load_active_snapshot(tmp_path)
        assert loaded is not None
        assert loaded.meta.snapshot_id == result.snapshot_id


# ---------------------------------------------------------------------------
# Failures — disk must stay untouched
# ---------------------------------------------------------------------------


class TestSyncFailures:
    def test_timeout_keeps_disk_intact(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_timeout_transport())
        assert not result.ok
        assert result.error is not None
        assert result.error.startswith("http_error")
        _assert_disk_clean(tmp_path)

    def test_http_500_keeps_disk_intact(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_server_error_transport())
        assert not result.ok
        assert result.error is not None
        assert result.error.startswith("http_error")
        _assert_disk_clean(tmp_path)

    def test_invalid_json(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_text_transport("not-json{{{"))
        assert not result.ok
        assert result.error is not None
        assert result.error.startswith("invalid_external_response")
        _assert_disk_clean(tmp_path)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        result = sync_catalog(tmp_path, transport=_json_transport({"count": 1}))
        assert not result.ok
        assert result.error is not None
        assert result.error.startswith("invalid_external_response")
        _assert_disk_clean(tmp_path)

    def test_empty_models_list_skipped(self, tmp_path: Path) -> None:
        result = sync_catalog(
            tmp_path, transport=_json_transport({"count": 0, "models": []})
        )
        assert not result.ok
        assert result.skipped_empty
        assert result.error == "empty_catalog"
        _assert_disk_clean(tmp_path)

    def test_all_excluded_skipped(self, tmp_path: Path) -> None:
        payload = {
            "count": 1,
            "models": [_record("vision-only", ["vision"])],
        }
        result = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert not result.ok
        assert result.skipped_empty
        assert result.error == "empty_catalog"
        _assert_disk_clean(tmp_path)


# ---------------------------------------------------------------------------
# Atomicity — failure must preserve previous snapshot + manifest
# ---------------------------------------------------------------------------


class TestSyncAtomicity:
    def test_timeout_preserves_previous_snapshot(self, tmp_path: Path) -> None:
        first = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert first.ok
        manifest_before = (tmp_path / "current.json").read_text(encoding="utf-8")
        snapshot_before = (
            tmp_path / f"{first.snapshot_id}.json"
        ).read_text(encoding="utf-8")

        second = sync_catalog(tmp_path, transport=_timeout_transport())
        assert not second.ok

        assert (tmp_path / "current.json").read_text(encoding="utf-8") == manifest_before
        assert (
            tmp_path / f"{first.snapshot_id}.json"
        ).read_text(encoding="utf-8") == snapshot_before
        assert read_snapshot(tmp_path, first.snapshot_id or "") is not None

    def test_failed_validation_preserves_previous(self, tmp_path: Path) -> None:
        first = sync_catalog(tmp_path, transport=_json_transport(_fixture_payload()))
        assert first.ok
        manifest_before = (tmp_path / "current.json").read_text(encoding="utf-8")

        second = sync_catalog(tmp_path, transport=_text_transport("garbage"))
        assert not second.ok
        assert (tmp_path / "current.json").read_text(encoding="utf-8") == manifest_before
        assert read_snapshot(tmp_path, first.snapshot_id or "") is not None

    def test_successful_resync_rotates_manifest(self, tmp_path: Path) -> None:
        payload_a = {
            "count": 1,
            "models": [_record("model-a", ["chat"])],
        }
        payload_b = {
            "count": 1,
            "models": [_record("model-b", ["reasoning"])],
        }
        first = sync_catalog(tmp_path, transport=_json_transport(payload_a))
        assert first.ok
        second = sync_catalog(tmp_path, transport=_json_transport(payload_b))
        assert second.ok
        assert second.snapshot_id != first.snapshot_id

        manifest = read_manifest(tmp_path)
        assert manifest is not None
        assert manifest.active_snapshot_id == second.snapshot_id
        assert manifest.previous_snapshot_id == first.snapshot_id
        assert read_snapshot(tmp_path, first.snapshot_id or "") is not None
        assert read_snapshot(tmp_path, second.snapshot_id or "") is not None


# ---------------------------------------------------------------------------
# Idempotency / immutability of identical content
# ---------------------------------------------------------------------------


class TestSyncIdempotency:
    def test_second_sync_unchanged(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        first = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert first.ok
        assert not first.unchanged

        second = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert second.ok
        assert second.unchanged
        assert second.snapshot_id == first.snapshot_id
        assert len(list(tmp_path.glob("snap-*.json"))) == 1

    def test_unchanged_leaves_manifest_bytes_intact(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        first = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert first.ok
        manifest_before = (tmp_path / "current.json").read_text(encoding="utf-8")

        second = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert second.ok and second.unchanged
        assert (tmp_path / "current.json").read_text(encoding="utf-8") == manifest_before

    def test_unchanged_recreates_missing_manifest(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        first = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert first.ok
        (tmp_path / "current.json").unlink()

        second = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert second.ok
        assert second.unchanged
        manifest = read_manifest(tmp_path)
        assert manifest is not None
        assert manifest.active_snapshot_id == first.snapshot_id

    def test_corrupt_existing_same_id_refused(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        first = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert first.ok
        snap_path = tmp_path / f"{first.snapshot_id}.json"
        snap_path.write_text("{corrupted-on-purpose", encoding="utf-8")

        second = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert not second.ok
        assert second.error == "existing_snapshot_corrupt"
        assert snap_path.read_text(encoding="utf-8") == "{corrupted-on-purpose"

    def test_manifest_points_elsewhere_same_content(self, tmp_path: Path) -> None:
        payload = _fixture_payload()
        first = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert first.ok
        update_dir = tmp_path
        from src.sync.snapshot import update_manifest

        update_manifest("snap-ffffffffffffffff", first.snapshot_id, update_dir)

        second = sync_catalog(tmp_path, transport=_json_transport(payload))
        assert second.ok
        assert second.unchanged
        manifest = read_manifest(tmp_path)
        assert manifest is not None
        assert manifest.active_snapshot_id == first.snapshot_id
        assert manifest.previous_snapshot_id == "snap-ffffffffffffffff"


# ---------------------------------------------------------------------------
# Missing endpoint (policy without source.endpoint)
# ---------------------------------------------------------------------------


class TestMissingEndpoint:
    def test_returns_error_without_touching_disk(self, tmp_path: Path) -> None:
        from src.sync.catalog_policy import CatalogPolicy

        class _NoEndpoint(CatalogPolicy):
            def __init__(self) -> None:
                self.version = "1.0.0"
                self.description = ""
                self.source_endpoint = None
                self._use_case_mapping = {"chat": ["chat"]}
                self.informational_tags = []
                self.out_of_scope_tags = []
                self.require_non_empty_id = True
                self.require_unique_id = True
                self.require_at_least_one_supported_task = True

        result = sync_catalog(tmp_path, policy=_NoEndpoint())
        assert not result.ok
        assert result.error == "missing_endpoint"
        _assert_disk_clean(tmp_path)

    def test_sync_result_defaults(self) -> None:
        r = SyncResult(ok=False)
        assert r.snapshot_id is None
        assert r.error is None
        assert r.normalization is None
        assert not r.unchanged
        assert not r.skipped_empty
