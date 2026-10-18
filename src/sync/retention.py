"""Retention of historical catalogue snapshots.

Every synchronisation that changes the catalogue leaves a new immutable
``snap-*.json`` behind, so ``runtime/catalog/`` grows without bound unless
old files are pruned.  This module deletes snapshots that are both
*unreferenced* and *older than the retention window*, under two safety
invariants from the project contract:

* The snapshots named by ``current.json`` (``active_snapshot_id`` and
  ``previous_snapshot_id``) are never deleted, whatever their age.  They
  are the last valid snapshot and its fallback.
* When the manifest is missing or unreadable nothing is deleted: without
  it there is no way to know which file is the last valid snapshot.

A snapshot's age is taken from its own ``fecha_sincronizacion``; for a
file that no longer parses as a snapshot the filesystem mtime is used, so
corrupt leftovers are eventually cleaned up too.  ``current.json`` and
``*.tmp`` files are never considered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.sync.snapshot import read_manifest, read_snapshot

logger = logging.getLogger(__name__)

DEFAULT_RETAIN_DAYS = 90
"""Default retention window: four sync cycles at the 21-day cadence."""


@dataclass
class PruneResult:
    """Outcome of one :func:`prune_snapshots` run.

    Attributes:
        deleted: Snapshot ids whose files were removed.
        kept: Unreferenced snapshot ids kept because they are inside the
            retention window.
        protected: Snapshot ids referenced by the manifest, never deleted.
        skipped_reason: Why nothing was evaluated (e.g. ``"no_manifest"``),
            or ``None`` when the run evaluated the directory.
    """

    deleted: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


def prune_snapshots(
    catalog_dir: Path,
    *,
    retain_days: int = DEFAULT_RETAIN_DAYS,
    now: datetime | None = None,
) -> PruneResult:
    """Delete unreferenced snapshots older than ``retain_days``.

    Args:
        catalog_dir: Directory holding ``current.json`` and ``snap-*.json``.
        retain_days: Snapshots strictly older than this many days are
            eligible for deletion.  Must be >= 0.
        now: Reference instant (timezone-aware).  Defaults to the current
            UTC time; tests inject a fixed value.

    Returns:
        A :class:`PruneResult` listing what happened to every snapshot.

    Raises:
        ValueError: If ``retain_days`` is negative.

    Side effects:
        Removes files from ``catalog_dir``.  A file that cannot be deleted
        (``OSError``) is logged and reported as kept; the run continues.
    """
    if retain_days < 0:
        raise ValueError(f"retain_days must be >= 0, got {retain_days}")
    reference = now or datetime.now(UTC)
    cutoff = reference - timedelta(days=retain_days)

    manifest = read_manifest(catalog_dir)
    if manifest is None:
        logger.warning(
            "No readable manifest in %s; refusing to prune without knowing the active snapshot.",
            catalog_dir,
        )
        return PruneResult(skipped_reason="no_manifest")

    referenced = {
        sid for sid in (manifest.active_snapshot_id, manifest.previous_snapshot_id) if sid
    }
    result = PruneResult()

    for path in sorted(catalog_dir.glob("snap-*.json")):
        snapshot_id = path.stem
        if snapshot_id in referenced:
            result.protected.append(snapshot_id)
            continue

        if _snapshot_timestamp(catalog_dir, path) >= cutoff:
            result.kept.append(snapshot_id)
            continue

        try:
            path.unlink()
        except OSError as exc:
            logger.error("Could not delete expired snapshot %s: %s", path, exc)
            result.kept.append(snapshot_id)
            continue
        logger.info("Deleted expired snapshot %s.", snapshot_id)
        result.deleted.append(snapshot_id)

    return result


def _snapshot_timestamp(catalog_dir: Path, path: Path) -> datetime:
    """Return when the snapshot was synchronised, falling back to mtime.

    A file that no longer validates as a snapshot has no trustworthy
    embedded date, so its modification time stands in.
    """
    snapshot = read_snapshot(catalog_dir, path.stem)
    if snapshot is not None:
        return snapshot.meta.fecha_sincronizacion
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
