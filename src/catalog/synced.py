"""Synchronized catalogue adapter: runtime snapshot with bootstrap fallback.

Resolves the catalogue through an ordered fallback chain:

1. ``runtime/catalog/`` — the live, gitignored synchronisation output,
   absent in a fresh clone.  Resolved with
   :func:`src.sync.snapshot.load_active_snapshot`, which already applies
   the manifest's active → previous fallback *inside* that directory;
   this adapter only decides when to leave level 1.
2. ``data/catalog_bootstrap.json`` — the committed, versioned bootstrap.
   Shape mismatch handled explicitly: the bootstrap is a single
   ``CatalogSnapshot`` **file**, not a directory holding a
   ``current.json`` manifest, so ``load_active_snapshot`` does not apply
   and the file is validated with :func:`src.sync.bootstrap.load_bootstrap`.
3. Both levels exhausted → raise :class:`CatalogUnavailableError` naming
   both sources.

The legacy YAML seed catalogue is intentionally *not* a level: task T6
deletes it, and keeping it as a fallback would preserve two sources of
truth for the same question.

Default paths resolve from this file's location, so behaviour does not
depend on the process working directory.  No HTTP is performed here —
catalogue synchronisation is an external process.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.catalog.protocol import CatalogUnavailableError
from src.models.schemas import FamiliaModelo
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH, load_bootstrap
from src.sync.snapshot import CatalogSnapshot, load_active_snapshot

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_CATALOG_DIR = Path(__file__).resolve().parents[2] / "runtime" / "catalog"


class CatalogoSincronizado:
    """Catalogue backed by ``runtime/catalog`` with bootstrap fallback.

    Construction performs no I/O — instances are cheap to build (inside
    ``lifespan``, a test fixture, or anywhere else).  The first
    ``snapshot_id`` access or :meth:`cargar` call resolves the source
    lazily.

    The resolved snapshot is **memoized for the lifetime of the
    instance**: later calls never re-read disk, so ``snapshot_id`` and
    the returned familias can never drift apart, and a synchronisation
    landing mid-process is picked up only after restart — which is how
    this design guarantees one snapshot id per process.  Build a new
    instance to force a re-read.

    Resolution is expected to happen once during startup, so no locking
    is implemented.

    Args:
        runtime_dir: Level-1 directory holding ``current.json`` plus
            snapshot files.  Defaults to ``runtime/catalog/`` at the
            project root.
        bootstrap_path: Level-2 single-file snapshot.  Defaults to
            ``data/catalog_bootstrap.json`` (``DEFAULT_BOOTSTRAP_PATH``).
            It must be a *file*; passing a directory makes this level
            fail and the chain raise.
    """

    def __init__(
        self,
        runtime_dir: Path | None = None,
        bootstrap_path: Path | None = None,
    ) -> None:
        self._runtime_dir = (
            runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_CATALOG_DIR
        )
        self._bootstrap_path = (
            bootstrap_path if bootstrap_path is not None else DEFAULT_BOOTSTRAP_PATH
        )
        self._snapshot: CatalogSnapshot | None = None

    @property
    def snapshot_id(self) -> str | None:
        """Identifier of the resolved snapshot.

        Triggers lazy resolution when needed, so the id is available
        *before* the first :meth:`cargar` call.  This implementation
        never returns ``None``: with no source left it raises instead,
        because a silently missing id would break reproducibility.

        Returns:
            The ``"snap-…"`` id of the snapshot backing this instance.

        Raises:
            CatalogUnavailableError: If both the runtime directory and
                the bootstrap file are missing or invalid.
        """
        return self._resolver().meta.snapshot_id

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        """Timezone-aware instant the resolved snapshot was synchronised.

        Taken from ``SnapshotMeta.fecha_sincronizacion``; never ``None``
        for this implementation because every snapshot carries the field.

        Raises:
            CatalogUnavailableError: If both sources are missing or invalid.
        """
        return self._resolver().meta.fecha_sincronizacion

    def cargar(self) -> list[FamiliaModelo]:
        """Return the families of the resolved snapshot.

        Returns:
            Fresh list with every ``FamiliaModelo`` of the snapshot in
            snapshot order; the elements themselves are shared with the
            memoized snapshot and should be treated as read-only.

        Raises:
            CatalogUnavailableError: If both the runtime directory and
                the bootstrap file are missing or invalid.
        """
        return list(self._resolver().familias)

    def _resolver(self) -> CatalogSnapshot:
        """Return the resolved snapshot, resolving the fallback chain once.

        Memoized: after the first successful resolution the disk is
        never consulted again for this instance.

        Returns:
            The validated snapshot loaded from level 1 or level 2.

        Raises:
            CatalogUnavailableError: When both levels yield nothing.
        """
        if self._snapshot is None:
            self._snapshot = self._resolve_source()
        return self._snapshot

    def _resolve_source(self) -> CatalogSnapshot:
        """Apply the fallback chain: runtime directory → bootstrap → error.

        Each level uses the validator that matches its on-disk shape
        (directory with manifest vs. single snapshot file) and returns
        ``None`` on any inconsistency, which is what advances the chain.

        Returns:
            The first valid snapshot found.

        Raises:
            CatalogUnavailableError: Naming both exhausted sources.
        """
        snapshot = load_active_snapshot(self._runtime_dir)
        if snapshot is not None:
            logger.info(
                "Catalogue loaded from runtime directory %s (snapshot %s).",
                self._runtime_dir,
                snapshot.meta.snapshot_id,
            )
            return snapshot

        snapshot = load_bootstrap(self._bootstrap_path)
        if snapshot is not None:
            logger.warning(
                "Runtime catalogue %s unavailable; falling back to bootstrap %s "
                "(snapshot %s).",
                self._runtime_dir,
                self._bootstrap_path,
                snapshot.meta.snapshot_id,
            )
            return snapshot

        raise CatalogUnavailableError(
            "No valid catalogue snapshot available: runtime directory "
            f"{self._runtime_dir} (missing or invalid manifest/snapshot) and "
            f"bootstrap file {self._bootstrap_path} (missing or invalid content)."
        )
