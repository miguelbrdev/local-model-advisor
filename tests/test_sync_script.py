"""Tests for scripts.sync_catalog — the scheduled synchronisation entry point.

Every run injects ``httpx.MockTransport`` and writes only under
``tmp_path``.  No network, no real ``runtime/``, no real bootstrap file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import sync_catalog as script
from src.sync.bootstrap import load_bootstrap
from src.sync.snapshot import load_active_snapshot, read_manifest

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _payload() -> dict[str, Any]:
    return json.loads((_FIXTURES / "canirun_models_response.json").read_text(encoding="utf-8"))


def _json_transport(payload: dict[str, Any], status_code: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status_code, json=payload))


def _failing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout", request=request)

    return httpx.MockTransport(handler)


def _run(tmp_path: Path, *extra: str, transport: httpx.MockTransport) -> int:
    return script.main(["--catalog-dir", str(tmp_path / "catalog"), *extra], transport=transport)


class TestParser:
    def test_defaults_point_at_project_paths(self) -> None:
        args = script.build_parser().parse_args([])
        assert args.catalog_dir.name == "catalog"
        assert args.catalog_dir.parent.name == "runtime"
        assert args.bootstrap_path.name == "catalog_bootstrap.json"
        assert args.policy is None
        assert args.promote_bootstrap is False
        assert args.retain_days == script.DEFAULT_RETAIN_DAYS
        assert args.timeout == 30.0

    def test_exit_codes_are_distinct(self) -> None:
        assert len({script.EXIT_OK, script.EXIT_SYNC_FAILED, script.EXIT_POST_STEP_FAILED}) == 3
        assert script.EXIT_OK == 0


class TestSuccessfulRun:
    def test_writes_snapshot_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(tmp_path, transport=_json_transport(_payload()))

        assert code == script.EXIT_OK
        catalog_dir = tmp_path / "catalog"
        snapshot = load_active_snapshot(catalog_dir)
        assert snapshot is not None
        assert read_manifest(catalog_dir).active_snapshot_id == snapshot.meta.snapshot_id
        out = capsys.readouterr().out
        assert f"snapshot={snapshot.meta.snapshot_id}" in out
        assert "unchanged=False" in out
        assert f"included={snapshot.meta.modelos_incluidos}" in out

    def test_second_run_is_idempotent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        transport = _json_transport(_payload())
        assert _run(tmp_path, transport=transport) == script.EXIT_OK
        manifest_before = (tmp_path / "catalog" / "current.json").read_bytes()

        assert _run(tmp_path, transport=transport) == script.EXIT_OK

        assert "unchanged=True" in capsys.readouterr().out
        assert (tmp_path / "catalog" / "current.json").read_bytes() == manifest_before
        assert len(list((tmp_path / "catalog").glob("snap-*.json"))) == 1

    def test_does_not_touch_bootstrap_by_default(self, tmp_path: Path) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        code = _run(
            tmp_path, "--bootstrap-path", str(bootstrap), transport=_json_transport(_payload())
        )
        assert code == script.EXIT_OK
        assert not bootstrap.exists()

    def test_promote_bootstrap_writes_active_snapshot(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        code = _run(
            tmp_path,
            "--promote-bootstrap",
            "--bootstrap-path",
            str(bootstrap),
            transport=_json_transport(_payload()),
        )

        assert code == script.EXIT_OK
        promoted = load_bootstrap(bootstrap)
        active = load_active_snapshot(tmp_path / "catalog")
        assert promoted is not None and active is not None
        assert promoted.meta.snapshot_id == active.meta.snapshot_id
        assert f"bootstrap={bootstrap}" in capsys.readouterr().out

    def test_promote_overwrites_existing_bootstrap(self, tmp_path: Path) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        bootstrap.parent.mkdir(parents=True)
        bootstrap.write_text("stale", encoding="utf-8")

        code = _run(
            tmp_path,
            "--promote-bootstrap",
            "--bootstrap-path",
            str(bootstrap),
            transport=_json_transport(_payload()),
        )

        assert code == script.EXIT_OK
        assert load_bootstrap(bootstrap) is not None


class TestFailedRun:
    def test_http_failure_exits_one_and_leaves_directory_empty(self, tmp_path: Path) -> None:
        code = _run(tmp_path, transport=_failing_transport())
        assert code == script.EXIT_SYNC_FAILED
        catalog_dir = tmp_path / "catalog"
        assert not catalog_dir.exists() or list(catalog_dir.iterdir()) == []

    def test_http_failure_preserves_previous_snapshot(self, tmp_path: Path) -> None:
        assert _run(tmp_path, transport=_json_transport(_payload())) == script.EXIT_OK
        catalog_dir = tmp_path / "catalog"
        before = {p.name: p.read_bytes() for p in catalog_dir.iterdir()}

        code = _run(tmp_path, transport=_failing_transport())

        assert code == script.EXIT_SYNC_FAILED
        assert {p.name: p.read_bytes() for p in catalog_dir.iterdir()} == before

    def test_invalid_payload_exits_one(self, tmp_path: Path) -> None:
        code = _run(tmp_path, transport=_json_transport({"unexpected": True}))
        assert code == script.EXIT_SYNC_FAILED

    def test_failed_sync_never_promotes_bootstrap(self, tmp_path: Path) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        code = _run(
            tmp_path,
            "--promote-bootstrap",
            "--bootstrap-path",
            str(bootstrap),
            transport=_failing_transport(),
        )
        assert code == script.EXIT_SYNC_FAILED
        assert not bootstrap.exists()

    def test_promotion_failure_exits_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args, **kwargs):
            raise script.BootstrapError("disk full")

        monkeypatch.setattr(script, "promote_snapshot_to_bootstrap", _boom)
        code = _run(
            tmp_path,
            "--promote-bootstrap",
            "--bootstrap-path",
            str(tmp_path / "b.json"),
            transport=_json_transport(_payload()),
        )
        assert code == script.EXIT_POST_STEP_FAILED
        # the sync itself succeeded and must remain on disk
        assert load_active_snapshot(tmp_path / "catalog") is not None


class TestRetentionIntegration:
    def test_prunes_expired_unreferenced_snapshot(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        catalog_dir = tmp_path / "catalog"
        # Three syncs with different *included* content, so the first
        # snapshot ends up unreferenced, the second "previous", the third
        # active.  Dropping included models (not out-of-scope ones) is what
        # changes the content hash.
        payload_a = _payload()
        payload_b = _payload()
        payload_b["models"] = payload_b["models"][1:]
        payload_b["count"] = len(payload_b["models"])
        payload_c = _payload()
        payload_c["models"] = payload_c["models"][2:]
        payload_c["count"] = len(payload_c["models"])

        assert _run(tmp_path, transport=_json_transport(payload_a)) == script.EXIT_OK
        assert _run(tmp_path, transport=_json_transport(payload_b)) == script.EXIT_OK
        assert _run(tmp_path, transport=_json_transport(payload_c)) == script.EXIT_OK
        assert len(list(catalog_dir.glob("snap-*.json"))) == 3

        # retain nothing older than 0 days: snapshots were just written, so
        # only an artificially aged one can expire.
        manifest = read_manifest(catalog_dir)
        unreferenced = [
            p
            for p in catalog_dir.glob("snap-*.json")
            if p.stem not in {manifest.active_snapshot_id, manifest.previous_snapshot_id}
        ]
        assert len(unreferenced) == 1
        data = json.loads(unreferenced[0].read_text(encoding="utf-8"))
        data["meta"]["fecha_sincronizacion"] = "2020-01-01T00:00:00Z"
        unreferenced[0].write_text(json.dumps(data), encoding="utf-8")

        capsys.readouterr()
        assert (
            _run(tmp_path, "--retain-days", "30", transport=_json_transport(payload_c))
            == script.EXIT_OK
        )

        assert "pruned=1" in capsys.readouterr().out
        assert len(list(catalog_dir.glob("snap-*.json"))) == 2
        assert load_active_snapshot(catalog_dir) is not None
