"""Tests for src.sync.snapshot — hash, invariants, atomic writes, manifest.

All filesystem tests use ``tmp_path``.  No HTTP calls are made and no
real clock assertions are performed (fixed timestamps in fixtures).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.schemas import FamiliaModelo
from src.sync.normalizer import ExclusionReason
from src.sync.snapshot import (
    CatalogSnapshot,
    Manifest,
    SnapshotMeta,
    compute_content_hash,
    load_active_snapshot,
    read_manifest,
    read_snapshot,
    update_manifest,
    write_snapshot,
)


def _fam(fid: str, **overrides) -> FamiliaModelo:
    base = dict(
        familia_id=fid,
        variantes=[],
        pipeline_tag="chat",
        canirun_id=fid,
        tareas_soportadas=["chat"],
    )
    base.update(overrides)
    return FamiliaModelo(**base)


def _meta(familias: list[FamiliaModelo], **overrides) -> SnapshotMeta:
    exclusions = overrides.pop("motivos_exclusion", [])
    h = compute_content_hash(familias, "1.0.0")
    data = dict(
        schema_version="1.0",
        snapshot_id=f"snap-{h[:16]}",
        fecha_sincronizacion=datetime(2026, 1, 1, tzinfo=UTC),
        source_url="https://www.canirun.ai/api/models",
        hash_contenido=h,
        modelos_origen=len(familias) + len(exclusions),
        modelos_incluidos=len(familias),
        motivos_exclusion=list(exclusions),
        politica_version="1.0.0",
    )
    data.update(overrides)
    return SnapshotMeta(**data)


def _snapshot(familias: list[FamiliaModelo] | None = None, **meta_over) -> CatalogSnapshot:
    fams = [_fam("m1")] if familias is None else familias
    return CatalogSnapshot(meta=_meta(fams, **meta_over), familias=fams)


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_deterministic(self) -> None:
        fams = [_fam("a"), _fam("b")]
        assert compute_content_hash(fams, "1.0.0") == compute_content_hash(fams, "1.0.0")

    def test_order_independent(self) -> None:
        fams = [_fam("a"), _fam("b"), _fam("c")]
        assert compute_content_hash(fams, "1.0.0") == compute_content_hash(
            list(reversed(fams)), "1.0.0"
        )

    def test_changes_with_familia(self) -> None:
        h1 = compute_content_hash([_fam("a")], "1.0.0")
        h2 = compute_content_hash([_fam("a", nombre_mostrado="Other")], "1.0.0")
        assert h1 != h2

    def test_changes_with_policy_version(self) -> None:
        fams = [_fam("a")]
        assert compute_content_hash(fams, "1.0.0") != compute_content_hash(fams, "2.0.0")

    def test_sha256_hex_format(self) -> None:
        h = compute_content_hash([_fam("a")], "1.0.0")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# SnapshotMeta invariants
# ---------------------------------------------------------------------------


class TestSnapshotMetaInvariants:
    def test_valid_meta(self) -> None:
        meta = _meta([_fam("a")])
        assert meta.modelos_incluidos == 1
        assert meta.modelos_origen == 1
        assert meta.fecha_sincronizacion.tzinfo is not None

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            _meta([_fam("a")], fecha_sincronizacion=datetime(2026, 1, 1))

    def test_rejects_bad_hash(self) -> None:
        with pytest.raises(ValidationError, match="hash_contenido"):
            _meta([_fam("a")], hash_contenido="not-a-hash")

    def test_rejects_non_http_url(self) -> None:
        with pytest.raises(ValidationError, match="source_url"):
            _meta([_fam("a")], source_url="ftp://example.com/models")

    def test_rejects_empty_schema_version(self) -> None:
        with pytest.raises(ValidationError, match="schema_version"):
            _meta([_fam("a")], schema_version="  ")

    def test_rejects_empty_snapshot_id(self) -> None:
        with pytest.raises(ValidationError, match="snapshot_id"):
            _meta([_fam("a")], snapshot_id="")

    def test_rejects_empty_politica_version(self) -> None:
        with pytest.raises(ValidationError, match="politica_version"):
            _meta([_fam("a")], politica_version="")

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValidationError):
            _meta([_fam("a")], modelos_incluidos=-1)

    def test_rejects_unbalanced_origin(self) -> None:
        with pytest.raises(ValidationError, match="modelos_origen"):
            _meta([_fam("a")], modelos_origen=5)

    def test_rejects_unbalanced_with_exclusions(self) -> None:
        excl = [ExclusionReason(record_id="x", reason="out_of_scope")]
        with pytest.raises(ValidationError, match="modelos_origen"):
            _meta([_fam("a")], motivos_exclusion=excl, modelos_origen=1)


# ---------------------------------------------------------------------------
# CatalogSnapshot invariants
# ---------------------------------------------------------------------------


class TestCatalogSnapshotInvariants:
    def test_valid_snapshot(self) -> None:
        snap = _snapshot([_fam("a"), _fam("b")])
        assert len(snap.familias) == 2

    def test_rejects_count_mismatch(self) -> None:
        fams = [_fam("a")]
        meta = _meta(fams).model_copy(
            update={"modelos_incluidos": 3, "modelos_origen": 3}
        )
        with pytest.raises(ValidationError, match="modelos_incluidos"):
            CatalogSnapshot(meta=meta, familias=fams)

    def test_rejects_duplicate_familia_id(self) -> None:
        fams = [_fam("dup"), _fam("dup")]
        with pytest.raises(ValidationError, match="familia_id"):
            _snapshot(fams)

    def test_rejects_duplicate_canirun_id(self) -> None:
        fams = [
            _fam("id1", canirun_id="same"),
            _fam("id2", canirun_id="same"),
        ]
        with pytest.raises(ValidationError, match="canirun_id"):
            _snapshot(fams)

    def test_allows_null_canirun_id_duplicates(self) -> None:
        fams = [_fam("x", canirun_id=None), _fam("y", canirun_id=None)]
        snap = _snapshot(fams)
        assert len(snap.familias) == 2

    def test_rejects_hash_content_mismatch(self) -> None:
        fams = [_fam("a")]
        with pytest.raises(ValidationError, match="hash_contenido"):
            _snapshot(fams, hash_contenido="b" * 64)

    def test_rejects_snapshot_id_prefix_mismatch(self) -> None:
        fams = [_fam("a")]
        with pytest.raises(ValidationError, match="snapshot_id"):
            _snapshot(fams, snapshot_id="snap-0000000000000000")


# ---------------------------------------------------------------------------
# write_snapshot / read_snapshot
# ---------------------------------------------------------------------------


class TestWriteSnapshot:
    def test_round_trip(self, tmp_path: Path) -> None:
        snap = _snapshot([_fam("a"), _fam("b")])
        path = write_snapshot(snap, tmp_path)
        assert path.exists()
        loaded = read_snapshot(tmp_path, snap.meta.snapshot_id)
        assert loaded is not None
        assert loaded.meta.snapshot_id == snap.meta.snapshot_id
        assert [f.familia_id for f in loaded.familias] == ["a", "b"]

    def test_no_temp_leftover(self, tmp_path: Path) -> None:
        write_snapshot(_snapshot(), tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        snap = _snapshot()
        path = write_snapshot(snap, tmp_path)
        path.write_text("GARBAGE-NOT-A-SNAPSHOT", encoding="utf-8")
        write_snapshot(snap, tmp_path)
        assert path.read_text(encoding="utf-8") == "GARBAGE-NOT-A-SNAPSHOT"

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "catalog"
        path = write_snapshot(_snapshot(), target)
        assert path.exists()


class TestReadSnapshot:
    def test_valid_returns_model(self, tmp_path: Path) -> None:
        snap = _snapshot()
        write_snapshot(snap, tmp_path)
        assert read_snapshot(tmp_path, snap.meta.snapshot_id) is not None

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_snapshot(tmp_path, "snap-missing") is None

    def test_corrupt_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "snap-bad.json").write_text("{not json", encoding="utf-8")
        assert read_snapshot(tmp_path, "snap-bad") is None

    def test_empty_id_returns_none(self, tmp_path: Path) -> None:
        assert read_snapshot(tmp_path, "") is None

    def test_tampered_content_returns_none(self, tmp_path: Path) -> None:
        snap = _snapshot([_fam("a")])
        path = write_snapshot(snap, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["familias"][0]["nombre_mostrado"] = "Tampered"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert read_snapshot(tmp_path, snap.meta.snapshot_id) is None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_update_and_read(self, tmp_path: Path) -> None:
        update_manifest("snap-a", None, tmp_path)
        m = read_manifest(tmp_path)
        assert m is not None
        assert m.active_snapshot_id == "snap-a"
        assert m.previous_snapshot_id is None
        assert m.updated_at is not None
        assert m.updated_at.tzinfo is not None

    def test_rotate_keeps_previous(self, tmp_path: Path) -> None:
        update_manifest("snap-a", None, tmp_path)
        update_manifest("snap-b", "snap-a", tmp_path)
        m = read_manifest(tmp_path)
        assert m is not None
        assert m.active_snapshot_id == "snap-b"
        assert m.previous_snapshot_id == "snap-a"

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_manifest(tmp_path) is None

    def test_read_corrupt_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "current.json").write_text("{broken", encoding="utf-8")
        assert read_manifest(tmp_path) is None

    def test_read_non_object_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "current.json").write_text("[1,2,3]", encoding="utf-8")
        assert read_manifest(tmp_path) is None

    def test_no_temp_leftover(self, tmp_path: Path) -> None:
        update_manifest("snap-a", None, tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_missing_fields_degrade_to_none(self, tmp_path: Path) -> None:
        (tmp_path / "current.json").write_text("{}", encoding="utf-8")
        m = read_manifest(tmp_path)
        assert m == Manifest(None, None, None)


# ---------------------------------------------------------------------------
# load_active_snapshot (full validation, not just ids)
# ---------------------------------------------------------------------------


class TestLoadActiveSnapshot:
    def _write_ok(self, tmp_path: Path, fid: str = "model") -> str:
        snap = _snapshot([_fam(fid)])
        write_snapshot(snap, tmp_path)
        return snap.meta.snapshot_id

    def test_loads_active(self, tmp_path: Path) -> None:
        sid = self._write_ok(tmp_path)
        update_manifest(sid, None, tmp_path)
        loaded = load_active_snapshot(tmp_path)
        assert loaded is not None
        assert loaded.meta.snapshot_id == sid

    def test_falls_back_to_previous_when_active_corrupt(self, tmp_path: Path) -> None:
        good_id = self._write_ok(tmp_path, "good")
        (tmp_path / "snap-active.json").write_text("{corrupt", encoding="utf-8")
        update_manifest("snap-active", good_id, tmp_path)
        loaded = load_active_snapshot(tmp_path)
        assert loaded is not None
        assert loaded.meta.snapshot_id == good_id

    def test_falls_back_when_active_missing(self, tmp_path: Path) -> None:
        good_id = self._write_ok(tmp_path, "good")
        update_manifest("snap-missing", good_id, tmp_path)
        loaded = load_active_snapshot(tmp_path)
        assert loaded is not None
        assert loaded.meta.snapshot_id == good_id

    def test_returns_none_when_both_missing(self, tmp_path: Path) -> None:
        update_manifest("snap-x", "snap-y", tmp_path)
        assert load_active_snapshot(tmp_path) is None

    def test_returns_none_when_both_corrupt(self, tmp_path: Path) -> None:
        (tmp_path / "snap-x.json").write_text("{bad", encoding="utf-8")
        (tmp_path / "snap-y.json").write_text("also bad", encoding="utf-8")
        update_manifest("snap-x", "snap-y", tmp_path)
        assert load_active_snapshot(tmp_path) is None

    def test_returns_none_without_manifest(self, tmp_path: Path) -> None:
        self._write_ok(tmp_path)
        assert load_active_snapshot(tmp_path) is None

    def test_returns_none_when_manifest_corrupt(self, tmp_path: Path) -> None:
        sid = self._write_ok(tmp_path)
        (tmp_path / "current.json").write_text("{nope", encoding="utf-8")
        assert load_active_snapshot(tmp_path) is None
        assert sid

    def test_rejects_id_mismatch_file(self, tmp_path: Path) -> None:
        snap = _snapshot([_fam("a")])
        write_snapshot(snap, tmp_path)
        renamed = tmp_path / "snap-ffffffffffffffff.json"
        (tmp_path / f"{snap.meta.snapshot_id}.json").rename(renamed)
        update_manifest("snap-ffffffffffffffff", None, tmp_path)
        assert load_active_snapshot(tmp_path) is None
