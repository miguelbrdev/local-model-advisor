"""Tests for src.sync.bootstrap — promote a snapshot to data/catalog_bootstrap.json.

All filesystem tests use ``tmp_path``.  No HTTP calls are made.  A guard
test asserts that promoting a snapshot never modifies ``runtime/catalog/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.models.schemas import FamiliaModelo
from src.sync.bootstrap import (
    BootstrapError,
    load_bootstrap,
    promote_snapshot_to_bootstrap,
)
from src.sync.snapshot import CatalogSnapshot, SnapshotMeta, compute_content_hash, write_snapshot

_RUNTIME_CATALOG = Path(__file__).resolve().parent.parent / "runtime" / "catalog"


def _fam(fid: str) -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid,
        variantes=[],
        pipeline_tag="chat",
        canirun_id=fid,
        tareas_soportadas=["chat"],
        nombre_mostrado=fid.title(),
    )


def _snapshot(familias: list[FamiliaModelo] | None = None) -> CatalogSnapshot:
    fams = [_fam("m1")] if familias is None else familias
    h = compute_content_hash(fams, "1.0.0")
    meta = SnapshotMeta(
        schema_version="1.0",
        snapshot_id=f"snap-{h[:16]}",
        fecha_sincronizacion=datetime(2026, 1, 1, tzinfo=UTC),
        source_url="https://www.canirun.ai/api/models",
        hash_contenido=h,
        modelos_origen=len(fams),
        modelos_incluidos=len(fams),
        motivos_exclusion=[],
        politica_version="1.0.0",
    )
    return CatalogSnapshot(meta=meta, familias=fams)


def _write_source(tmp_path: Path, snapshot: CatalogSnapshot | None = None) -> Path:
    snap = snapshot or _snapshot()
    src_dir = tmp_path / "catalog"
    return write_snapshot(snap, src_dir)


# ---------------------------------------------------------------------------
# Successful promotion
# ---------------------------------------------------------------------------


class TestPromoteSuccess:
    def test_promotion_writes_bootstrap(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        target = tmp_path / "data" / "catalog_bootstrap.json"
        result = promote_snapshot_to_bootstrap(source, target)
        assert result == target
        assert target.exists()
        assert list(target.parent.glob("*.tmp")) == []

    def test_bootstrap_round_trip(self, tmp_path: Path) -> None:
        snap = _snapshot([_fam("alpha"), _fam("beta")])
        source = _write_source(tmp_path, snap)
        target = tmp_path / "catalog_bootstrap.json"
        promote_snapshot_to_bootstrap(source, target)

        loaded = load_bootstrap(target)
        assert loaded is not None
        assert loaded.meta.snapshot_id == snap.meta.snapshot_id
        assert loaded.meta.hash_contenido == snap.meta.hash_contenido
        assert [f.familia_id for f in loaded.familias] == ["alpha", "beta"]
        assert loaded.meta.politica_version == "1.0.0"

    def test_bootstrap_serializable_to_dict(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        target = tmp_path / "catalog_bootstrap.json"
        promote_snapshot_to_bootstrap(source, target)

        data = json.loads(target.read_text(encoding="utf-8"))
        assert "meta" in data
        assert "familias" in data
        assert data["meta"]["snapshot_id"].startswith("snap-")
        assert len(data["meta"]["hash_contenido"]) == 64

    def test_source_left_untouched(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        before = source.read_bytes()
        promote_snapshot_to_bootstrap(source, tmp_path / "bootstrap.json")
        assert source.read_bytes() == before

    def test_overwrite_explicit(self, tmp_path: Path) -> None:
        source_a = _write_source(tmp_path, _snapshot([_fam("a")]))
        target = tmp_path / "bootstrap.json"
        promote_snapshot_to_bootstrap(source_a, target)
        first = target.read_text(encoding="utf-8")

        snap_b = _snapshot([_fam("b"), _fam("c")])
        source_b_dir = tmp_path / "other"
        source_b = write_snapshot(snap_b, source_b_dir)
        promote_snapshot_to_bootstrap(source_b, target, overwrite=True)

        loaded = load_bootstrap(target)
        assert loaded is not None
        assert [f.familia_id for f in loaded.familias] == ["b", "c"]
        assert target.read_text(encoding="utf-8") != first


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


class TestPromoteRejections:
    def test_bootstrap_invalid_content_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "catalog_bootstrap.json"
        target.write_text("{not-json", encoding="utf-8")
        source = _write_source(tmp_path)
        with pytest.raises(BootstrapError, match="already exists"):
            promote_snapshot_to_bootstrap(source, target)

        # With overwrite, invalid existing content is replaced atomically.
        promote_snapshot_to_bootstrap(source, target, overwrite=True)
        assert load_bootstrap(target) is not None

    def test_source_missing_rejected(self, tmp_path: Path) -> None:
        missing = tmp_path / "snap-0000000000000000.json"
        target = tmp_path / "bootstrap.json"
        with pytest.raises(BootstrapError, match="not found"):
            promote_snapshot_to_bootstrap(missing, target)
        assert not target.exists()

    def test_source_corrupt_rejected(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "catalog"
        src_dir.mkdir(parents=True)
        corrupt = src_dir / "snap-176d40242fd0157a.json"
        corrupt.write_text("{corrupted", encoding="utf-8")
        target = tmp_path / "bootstrap.json"
        with pytest.raises(BootstrapError, match="invalid"):
            promote_snapshot_to_bootstrap(corrupt, target)
        assert not target.exists()
        assert corrupt.read_text(encoding="utf-8") == "{corrupted"

    def test_existing_bootstrap_without_overwrite_rejected(self, tmp_path: Path) -> None:
        source = _write_source(tmp_path)
        target = tmp_path / "bootstrap.json"
        target.write_text('{"existing": true}', encoding="utf-8")
        with pytest.raises(BootstrapError, match="already exists"):
            promote_snapshot_to_bootstrap(source, target, overwrite=False)
        assert target.read_text(encoding="utf-8") == '{"existing": true}'

    def test_filename_mismatch_rejected(self, tmp_path: Path) -> None:
        snap = _snapshot()
        source = _write_source(tmp_path, snap)
        renamed = tmp_path / "catalog" / "snap-ffffffffffffffff.json"
        source.rename(renamed)
        target = tmp_path / "bootstrap.json"
        with pytest.raises(BootstrapError, match="does not match"):
            promote_snapshot_to_bootstrap(renamed, target)
        assert not target.exists()

    def test_tampered_source_rejected(self, tmp_path: Path) -> None:
        snap = _snapshot([_fam("a")])
        source = _write_source(tmp_path, snap)
        data = json.loads(source.read_text(encoding="utf-8"))
        data["familias"][0]["nombre_mostrado"] = "Tampered"
        source.write_text(json.dumps(data), encoding="utf-8")
        target = tmp_path / "bootstrap.json"
        with pytest.raises(BootstrapError, match="invalid"):
            promote_snapshot_to_bootstrap(source, target)
        assert not target.exists()


# ---------------------------------------------------------------------------
# runtime/catalog must stay untouched during tests
# ---------------------------------------------------------------------------


class TestRuntimeUntouched:
    def test_promotion_does_not_modify_runtime_catalog(self, tmp_path: Path) -> None:
        if not _RUNTIME_CATALOG.is_dir():
            pytest.skip("runtime/catalog does not exist in this environment")

        def _state() -> dict[str, tuple[int, int]]:
            state: dict[str, tuple[int, int]] = {}
            for p in sorted(_RUNTIME_CATALOG.iterdir()):
                if p.is_file():
                    st = p.stat()
                    state[p.name] = (st.st_size, st.st_mtime_ns)
            return state

        before = _state()
        source = _write_source(tmp_path)
        promote_snapshot_to_bootstrap(source, tmp_path / "bootstrap.json")
        assert _state() == before

    def test_default_bootstrap_path_constant(self) -> None:
        from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH

        assert DEFAULT_BOOTSTRAP_PATH.name == "catalog_bootstrap.json"
        assert DEFAULT_BOOTSTRAP_PATH.parent.name == "data"


# ---------------------------------------------------------------------------
# load_bootstrap
# ---------------------------------------------------------------------------


class TestLoadBootstrap:
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_bootstrap(tmp_path / "absent.json") is None

    def test_corrupt_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog_bootstrap.json"
        path.write_text("nope", encoding="utf-8")
        assert load_bootstrap(path) is None
