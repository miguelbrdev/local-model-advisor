"""Tests for src.catalog — catalogue abstraction and synchronized adapter.

Covers the three fallback levels (runtime directory → bootstrap file
→ explicit typed error), ``snapshot_id`` exposure, memoization, and
Protocol conformance of both the shipped adapter and an in-memory double.

Project rules honoured here: no network, no ``canirun.ai``, no
wall-clock dependence, and no writes outside ``tmp_path``.  The only
read of repository data is the committed ``data/catalog_bootstrap.json``,
used as a fixture.
"""

from __future__ import annotations

import json
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.catalog import synced
from src.catalog.protocol import CatalogoModelos, CatalogUnavailableError
from src.catalog.synced import DEFAULT_RUNTIME_CATALOG_DIR, CatalogoSincronizado
from src.models.schemas import FamiliaModelo
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH
from src.sync.snapshot import (
    CatalogSnapshot,
    SnapshotMeta,
    compute_content_hash,
    update_manifest,
    write_snapshot,
)

# Regression sentinels for the committed bootstrap: if this file is ever
# re-promoted, these tests must fail loudly instead of silently shifting.
BOOTSTRAP_SNAPSHOT_ID = "snap-176d40242fd0157a"
BOOTSTRAP_FAMILIA_COUNT = 91


# ---------------------------------------------------------------------------
# Builders (all output goes under tmp_path)
# ---------------------------------------------------------------------------


def _fam(fid: str) -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid,
        variantes=["7B"],
        pipeline_tag="chat",
        canirun_id=fid,
        tareas_soportadas=["chat"],
        nombre_mostrado=fid.title(),
    )


def _snapshot(familias: list[FamiliaModelo]) -> CatalogSnapshot:
    hashcontenido = compute_content_hash(familias, "1.0.0")
    meta = SnapshotMeta(
        schema_version="1.0",
        snapshot_id=f"snap-{hashcontenido[:16]}",
        fecha_sincronizacion=datetime(2026, 1, 1, tzinfo=UTC),
        source_url="https://www.canirun.ai/api/models",
        hash_contenido=hashcontenido,
        modelos_origen=len(familias),
        modelos_incluidos=len(familias),
        motivos_exclusion=[],
        politica_version="1.0.0",
    )
    return CatalogSnapshot(meta=meta, familias=familias)


def _write_runtime(runtime_dir: Path, snapshot: CatalogSnapshot) -> None:
    """Materialise a valid level-1 source: snapshot file + manifest."""
    write_snapshot(snapshot, runtime_dir)
    update_manifest(snapshot.meta.snapshot_id, None, runtime_dir)


# ---------------------------------------------------------------------------
# Level 1 — runtime directory
# ---------------------------------------------------------------------------


class TestRuntimeSource:
    def test_runtime_snapshot_wins_over_existing_bootstrap(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime" / "catalog"
        snap = _snapshot([_fam("alpha"), _fam("beta")])
        _write_runtime(runtime, snap)

        # Level 2 points at the real bootstrap (a different snapshot id),
        # so a wrong fallback order would be detected by the assertion.
        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=DEFAULT_BOOTSTRAP_PATH
        )
        familias = catalogo.cargar()

        assert catalogo.snapshot_id == snap.meta.snapshot_id
        assert catalogo.snapshot_id != BOOTSTRAP_SNAPSHOT_ID
        assert len(familias) == 2
        assert all(isinstance(f, FamiliaModelo) for f in familias)
        assert [f.familia_id for f in familias] == ["alpha", "beta"]

    def test_snapshot_id_available_before_first_load(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime" / "catalog"
        snap = _snapshot([_fam("solo")])
        _write_runtime(runtime, snap)

        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=tmp_path / "absent.json"
        )
        # Property access resolves lazily, so callers can record the id
        # without having to call cargar() first.
        assert catalogo.snapshot_id == snap.meta.snapshot_id
        assert [f.familia_id for f in catalogo.cargar()] == ["solo"]

    def test_resolution_is_memoized(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime" / "catalog"
        snap = _snapshot([_fam("keep")])
        _write_runtime(runtime, snap)
        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=tmp_path / "absent.json"
        )
        first = catalogo.cargar()

        # Destroy every source: a non-memoized implementation would now
        # raise, so a successful second call proves the snapshot is cached.
        for path in runtime.iterdir():
            path.unlink()
        second = catalogo.cargar()

        assert [f.familia_id for f in second] == [f.familia_id for f in first]
        assert catalogo.snapshot_id == snap.meta.snapshot_id

    def test_cargar_returns_a_fresh_list(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime" / "catalog"
        snap = _snapshot([_fam("one"), _fam("two")])
        _write_runtime(runtime, snap)
        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=tmp_path / "absent.json"
        )

        first = catalogo.cargar()
        first.clear()

        # Contract: the returned list is a fresh copy per call, so a
        # caller mutating it must never reach the memoized snapshot.
        second = catalogo.cargar()
        assert second is not first
        assert [f.familia_id for f in second] == ["one", "two"]


# ---------------------------------------------------------------------------
# Level 2 — committed bootstrap file
# ---------------------------------------------------------------------------


class TestBootstrapFallback:
    def test_falls_back_when_runtime_absent(self, tmp_path: Path) -> None:
        catalogo = CatalogoSincronizado(
            runtime_dir=tmp_path / "fresh-clone" / "catalog",
            bootstrap_path=DEFAULT_BOOTSTRAP_PATH,
        )
        familias = catalogo.cargar()

        assert catalogo.snapshot_id == BOOTSTRAP_SNAPSHOT_ID
        assert len(familias) == BOOTSTRAP_FAMILIA_COUNT
        assert all(isinstance(f, FamiliaModelo) for f in familias)

    def test_falls_back_when_manifest_is_corrupt(self, tmp_path: Path) -> None:
        runtime = tmp_path / "catalog"
        runtime.mkdir(parents=True)
        (runtime / "current.json").write_text("{not-json", encoding="utf-8")

        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=DEFAULT_BOOTSTRAP_PATH
        )
        assert catalogo.snapshot_id == BOOTSTRAP_SNAPSHOT_ID
        assert len(catalogo.cargar()) == BOOTSTRAP_FAMILIA_COUNT

    def test_falls_back_when_manifest_points_to_missing_snapshot(
        self, tmp_path: Path
    ) -> None:
        runtime = tmp_path / "catalog"
        runtime.mkdir(parents=True)
        (runtime / "current.json").write_text(
            json.dumps(
                {"active_snapshot_id": "snap-0000000000000000", "previous_snapshot_id": None}
            ),
            encoding="utf-8",
        )

        catalogo = CatalogoSincronizado(
            runtime_dir=runtime, bootstrap_path=DEFAULT_BOOTSTRAP_PATH
        )
        assert catalogo.snapshot_id == BOOTSTRAP_SNAPSHOT_ID
        assert len(catalogo.cargar()) == BOOTSTRAP_FAMILIA_COUNT


# ---------------------------------------------------------------------------
# Level 3 — both sources exhausted
# ---------------------------------------------------------------------------


class TestSourcesExhausted:
    def test_raises_typed_error_naming_both_sources(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime" / "catalog"
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        catalogo = CatalogoSincronizado(runtime_dir=runtime, bootstrap_path=bootstrap)

        with pytest.raises(CatalogUnavailableError) as excinfo:
            catalogo.cargar()

        message = str(excinfo.value)
        assert str(runtime) in message
        assert str(bootstrap) in message

    def test_raises_when_both_sources_are_invalid(self, tmp_path: Path) -> None:
        runtime = tmp_path / "catalog"
        runtime.mkdir(parents=True)
        (runtime / "current.json").write_text("{broken", encoding="utf-8")
        bootstrap = tmp_path / "catalog_bootstrap.json"
        bootstrap.write_text("{broken", encoding="utf-8")

        catalogo = CatalogoSincronizado(runtime_dir=runtime, bootstrap_path=bootstrap)
        with pytest.raises(CatalogUnavailableError) as excinfo:
            catalogo.cargar()

        message = str(excinfo.value)
        assert str(runtime) in message
        assert str(bootstrap) in message

    def test_snapshot_id_raises_the_same_typed_error(self, tmp_path: Path) -> None:
        catalogo = CatalogoSincronizado(
            runtime_dir=tmp_path / "absent",
            bootstrap_path=tmp_path / "absent.json",
        )
        # Never a silent None: a missing snapshot id would silently break
        # reproducibility, so the property fails exactly like cargar().
        with pytest.raises(CatalogUnavailableError):
            _ = catalogo.snapshot_id

    def test_bootstrap_level_requires_a_file_not_a_directory(
        self, tmp_path: Path
    ) -> None:
        # Pin the shape decision: level 2 is a single CatalogSnapshot
        # file. A directory (even a valid runtime-shaped one) must not
        # satisfy it — load_active_snapshot is level-1 only.
        runtime = tmp_path / "catalog"
        _write_runtime(runtime, _snapshot([_fam("dir-shaped")]))

        catalogo = CatalogoSincronizado(
            runtime_dir=tmp_path / "absent", bootstrap_path=runtime
        )
        with pytest.raises(CatalogUnavailableError):
            catalogo.cargar()


# ---------------------------------------------------------------------------
# Default paths and zero-argument construction
# ---------------------------------------------------------------------------


class TestDefaultPaths:
    def test_default_runtime_dir_resolves_to_repo_root(self) -> None:
        # Pure path math, expected root derived from *this* file
        # (tests/ -> repo root) instead of re-deriving it from the
        # production expression: a wrong parents[N] index in synced.py
        # must not be able to agree with itself.  The poisoned value this
        # guards against is src/runtime/catalog, which would silently
        # leave production bootstrap-only forever.
        repo_root = Path(__file__).resolve().parents[1]
        assert DEFAULT_RUNTIME_CATALOG_DIR == repo_root / "runtime" / "catalog"
        assert not DEFAULT_RUNTIME_CATALOG_DIR.is_relative_to(repo_root / "src")

        # Zero-argument construction performs no I/O (documented on the
        # class), so the default branch is genuinely reached here without
        # touching the filesystem.  _runtime_dir is private, but it is the
        # only observable evidence that the branch selected the constant.
        catalogo = CatalogoSincronizado()
        assert catalogo._runtime_dir == DEFAULT_RUNTIME_CATALOG_DIR

    def test_zero_arg_constructor_reads_the_module_constants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end proof that the no-argument branch consults the
        # module-level defaults at call time: point both at tmp_path,
        # build with no arguments, and resolve a snapshot written only
        # under tmp_path.  The bootstrap default is redirected too, so a
        # broken level 1 raises instead of silently succeeding against
        # the committed bootstrap.
        runtime = tmp_path / "runtime" / "catalog"
        snap = _snapshot([_fam("defaulted")])
        _write_runtime(runtime, snap)
        monkeypatch.setattr(synced, "DEFAULT_RUNTIME_CATALOG_DIR", runtime)
        monkeypatch.setattr(synced, "DEFAULT_BOOTSTRAP_PATH", tmp_path / "absent.json")

        catalogo = CatalogoSincronizado()

        assert catalogo.snapshot_id == snap.meta.snapshot_id
        assert catalogo.snapshot_id != BOOTSTRAP_SNAPSHOT_ID
        assert [f.familia_id for f in catalogo.cargar()] == ["defaulted"]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class _InMemoryCatalogo:
    """Minimal in-memory double proving the Protocol is structurally satisfied."""

    def __init__(self, familias: list[FamiliaModelo]) -> None:
        self._familias = list(familias)

    @property
    def snapshot_id(self) -> str | None:
        return "snap-in-memory"

    def cargar(self) -> list[FamiliaModelo]:
        return list(self._familias)


class _PartialCatalogo:
    """Double genuinely missing ``snapshot_id`` — must never conform.

    ``runtime_checkable`` isinstance checks member presence only, so a
    negative case needs an object that actually lacks a member to be
    meaningful.
    """

    def cargar(self) -> list[FamiliaModelo]:
        return []


def _declared_members() -> set[str]:
    """Declared member surface of :class:`CatalogoModelos`.

    Uses :func:`typing.get_protocol_members` (Python 3.12+) when the
    interpreter provides it, falling back to the class's public
    attributes so the assertion still holds on the 3.11 floor declared
    in ``pyproject.toml`` and exercised by CI.

    Returns:
        The names the protocol declares, excluding dunders.
    """
    if hasattr(typing, "get_protocol_members"):
        return set(typing.get_protocol_members(CatalogoModelos))
    return {name for name in dir(CatalogoModelos) if not name.startswith("_")}


class TestProtocolConformance:
    def test_adapter_satisfies_protocol(self, tmp_path: Path) -> None:
        catalogo = CatalogoSincronizado(
            runtime_dir=tmp_path / "absent", bootstrap_path=tmp_path / "absent.json"
        )
        assert isinstance(catalogo, CatalogoModelos)

    def test_in_memory_double_satisfies_protocol(self) -> None:
        assert isinstance(_InMemoryCatalogo([_fam("m1")]), CatalogoModelos)

    def test_non_conforming_objects_are_rejected(self) -> None:
        assert not isinstance(object(), CatalogoModelos)
        assert not isinstance(_PartialCatalogo(), CatalogoModelos)

    def test_protocol_declares_both_required_members(self) -> None:
        # Pin the declarations themselves: the positive conformance tests
        # only prove the implementations carry the members the protocol
        # currently declares, so nothing else fails if those declarations
        # are deleted.
        assert _declared_members() == {"cargar", "snapshot_id"}
