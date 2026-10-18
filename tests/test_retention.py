"""Tests for src.sync.retention — pruning of historical snapshots.

All files live under ``tmp_path``; the reference instant is injected so no
test depends on the wall clock.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.models.schemas import FamiliaModelo
from src.sync.retention import DEFAULT_RETAIN_DAYS, prune_snapshots
from src.sync.snapshot import (
    CatalogSnapshot,
    SnapshotMeta,
    compute_content_hash,
    update_manifest,
    write_snapshot,
)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _snapshot(fid: str, synced_at: datetime) -> CatalogSnapshot:
    """One-family snapshot; ``fid`` makes the content (and id) unique."""
    familias = [
        FamiliaModelo(
            familia_id=fid, variantes=[], pipeline_tag="chat", canirun_id=fid,
            tareas_soportadas=["chat"],
        )
    ]
    h = compute_content_hash(familias, "1.0.0")
    meta = SnapshotMeta(
        schema_version="1.0",
        snapshot_id=f"snap-{h[:16]}",
        fecha_sincronizacion=synced_at,
        source_url="https://www.canirun.ai/api/models",
        hash_contenido=h,
        modelos_origen=1,
        modelos_incluidos=1,
        motivos_exclusion=[],
        politica_version="1.0.0",
    )
    return CatalogSnapshot(meta=meta, familias=familias)


def _write(catalog_dir: Path, fid: str, age_days: int) -> str:
    snapshot = _snapshot(fid, NOW - timedelta(days=age_days))
    write_snapshot(snapshot, catalog_dir)
    return snapshot.meta.snapshot_id


def _ids(catalog_dir: Path) -> set[str]:
    return {p.stem for p in catalog_dir.glob("snap-*.json")}


class TestSafetyInvariants:
    def test_no_manifest_prunes_nothing(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old", age_days=400)
        result = prune_snapshots(tmp_path, retain_days=1, now=NOW)
        assert result.skipped_reason == "no_manifest"
        assert result.deleted == []
        assert _ids(tmp_path) == {old}

    def test_corrupt_manifest_prunes_nothing(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old", age_days=400)
        (tmp_path / "current.json").write_text("{not json", encoding="utf-8")
        result = prune_snapshots(tmp_path, retain_days=1, now=NOW)
        assert result.skipped_reason == "no_manifest"
        assert _ids(tmp_path) == {old}

    def test_active_and_previous_are_never_deleted(self, tmp_path: Path) -> None:
        previous = _write(tmp_path, "previous", age_days=1000)
        active = _write(tmp_path, "active", age_days=900)
        update_manifest(active, previous, tmp_path)

        result = prune_snapshots(tmp_path, retain_days=0, now=NOW)

        assert sorted(result.protected) == sorted([active, previous])
        assert result.deleted == []
        assert _ids(tmp_path) == {active, previous}

    def test_negative_retention_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="retain_days"):
            prune_snapshots(tmp_path, retain_days=-1, now=NOW)

    def test_manifest_and_temp_files_untouched(self, tmp_path: Path) -> None:
        active = _write(tmp_path, "active", age_days=0)
        update_manifest(active, None, tmp_path)
        leftover = tmp_path / "snap-deadbeef.json.tmp"
        leftover.write_text("partial", encoding="utf-8")

        prune_snapshots(tmp_path, retain_days=0, now=NOW)

        assert (tmp_path / "current.json").exists()
        assert leftover.exists()


class TestRetentionWindow:
    def test_expired_unreferenced_deleted_recent_kept(self, tmp_path: Path) -> None:
        active = _write(tmp_path, "active", age_days=0)
        recent = _write(tmp_path, "recent", age_days=30)
        expired = _write(tmp_path, "expired", age_days=120)
        update_manifest(active, None, tmp_path)

        result = prune_snapshots(tmp_path, retain_days=DEFAULT_RETAIN_DAYS, now=NOW)

        assert result.deleted == [expired]
        assert result.kept == [recent]
        assert result.protected == [active]
        assert _ids(tmp_path) == {active, recent}

    def test_boundary_exactly_at_cutoff_is_kept(self, tmp_path: Path) -> None:
        active = _write(tmp_path, "active", age_days=0)
        boundary = _write(tmp_path, "boundary", age_days=90)
        update_manifest(active, None, tmp_path)

        result = prune_snapshots(tmp_path, retain_days=90, now=NOW)

        assert result.deleted == []
        assert boundary in result.kept

    def test_uses_embedded_sync_date_not_mtime(self, tmp_path: Path) -> None:
        """A freshly written file whose embedded date is old must still expire."""
        active = _write(tmp_path, "active", age_days=0)
        expired = _write(tmp_path, "expired", age_days=500)  # mtime is "now"
        update_manifest(active, None, tmp_path)

        result = prune_snapshots(tmp_path, retain_days=90, now=NOW)

        assert result.deleted == [expired]

    def test_corrupt_unreferenced_file_falls_back_to_mtime(self, tmp_path: Path) -> None:
        active = _write(tmp_path, "active", age_days=0)
        update_manifest(active, None, tmp_path)
        corrupt = tmp_path / "snap-0000000000000000.json"
        corrupt.write_text("{broken", encoding="utf-8")
        old_mtime = (NOW - timedelta(days=200)).timestamp()
        os.utime(corrupt, (old_mtime, old_mtime))

        result = prune_snapshots(tmp_path, retain_days=90, now=NOW)

        assert result.deleted == ["snap-0000000000000000"]
        assert not corrupt.exists()

    def test_corrupt_but_recent_file_is_kept(self, tmp_path: Path) -> None:
        active = _write(tmp_path, "active", age_days=0)
        update_manifest(active, None, tmp_path)
        corrupt = tmp_path / "snap-0000000000000000.json"
        corrupt.write_text("{broken", encoding="utf-8")
        fresh = NOW.timestamp()
        os.utime(corrupt, (fresh, fresh))

        result = prune_snapshots(tmp_path, retain_days=90, now=NOW)

        assert result.deleted == []
        assert corrupt.exists()

    def test_defaults_to_current_time(self, tmp_path: Path) -> None:
        """Without ``now`` the function must still run (uses real UTC clock)
        and never delete the active snapshot."""
        active = _write(tmp_path, "active", age_days=0)
        update_manifest(active, None, tmp_path)
        result = prune_snapshots(tmp_path, retain_days=0)
        assert result.protected == [active]
        assert result.deleted == []
