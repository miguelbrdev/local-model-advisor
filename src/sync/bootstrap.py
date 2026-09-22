"""Promote a validated on-disk ``CatalogSnapshot`` to the versioned bootstrap catalogue.

The bootstrap file (``data/catalog_bootstrap.json``) is the committed,
versioned copy of a real canIRun.ai snapshot.  It keeps the full
``CatalogSnapshot`` layout — ``SnapshotMeta`` header (hash,
``snapshot_id``, policy version, exclusions) plus familias — so it is
byte-compatible with snapshots under ``runtime/catalog/``.

Runtime policy:

* ``runtime/catalog/`` stays gitignored and keeps priority at run time;
  the bootstrap is only the fallback source for future pipeline work.
* Promotion never writes into ``runtime/catalog/`` — the source file is
  read-only input.
* The bootstrap is not overwritten unless ``overwrite=True`` is passed
  explicitly.

No HTTP is performed in this module.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from src.sync.snapshot import CatalogSnapshot

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "catalog_bootstrap.json"
)


class BootstrapError(Exception):
    """Raised when a snapshot cannot be safely promoted to the bootstrap file."""


def promote_snapshot_to_bootstrap(
    source_path: Path,
    bootstrap_path: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Promote *source_path* to the versioned ``data/catalog_bootstrap.json``.

    The source snapshot is fully validated as a ``CatalogSnapshot``
    (including content-hash recomputation and ``snapshot_id`` consistency
    with the filename stem).  The serialised bootstrap is round-trip
    validated before being moved into place with ``os.replace``.

    Args:
        source_path: Existing snapshot JSON file (typically under
            ``runtime/catalog/``).  Read-only — never modified.
        bootstrap_path: Destination path.  Defaults to
            ``data/catalog_bootstrap.json`` at the project root.
        overwrite: When ``False`` (default), refuses to replace an
            existing bootstrap file.

    Returns:
        Path of the written bootstrap file.

    Raises:
        BootstrapError: If the source is missing or corrupt, the
            destination already exists without ``overwrite=True``, or
            any filesystem/round-trip step fails.  The destination and
            the source directory are left unchanged on failure.
    """
    target = bootstrap_path or DEFAULT_BOOTSTRAP_PATH

    if not source_path.is_file():
        raise BootstrapError(f"source snapshot not found: {source_path}")

    try:
        raw = source_path.read_text(encoding="utf-8")
        snapshot = CatalogSnapshot.model_validate_json(raw)
    except (OSError, ValidationError) as exc:
        raise BootstrapError(f"source snapshot invalid: {source_path}: {exc}") from exc

    expected_stem = snapshot.meta.snapshot_id
    if source_path.stem != expected_stem:
        raise BootstrapError(
            f"source filename stem {source_path.stem!r} does not match "
            f"snapshot_id {expected_stem!r}"
        )

    if target.exists() and not overwrite:
        raise BootstrapError(
            f"bootstrap already exists: {target} (pass overwrite=True to replace it)"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(target.name + ".tmp")
    try:
        serialised = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        temp_path.write_text(serialised, encoding="utf-8")
        CatalogSnapshot.model_validate_json(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, target)
    except BootstrapError:
        raise
    except (OSError, ValidationError) as exc:
        raise BootstrapError(f"failed to write bootstrap {target}: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    logger.info(
        "Promoted snapshot %s -> %s (familias=%d, hash=%s)",
        expected_stem,
        target,
        len(snapshot.familias),
        snapshot.meta.hash_contenido,
    )
    return target


def load_bootstrap(bootstrap_path: Path | None = None) -> CatalogSnapshot | None:
    """Load and validate the versioned bootstrap catalogue if present.

    Args:
        bootstrap_path: Path to the bootstrap JSON.  Defaults to
            ``data/catalog_bootstrap.json``.

    Returns:
        The validated ``CatalogSnapshot``, or ``None`` if the file is
        missing or fails full validation (corrupt/tampered content).
    """
    path = bootstrap_path or DEFAULT_BOOTSTRAP_PATH
    if not path.is_file():
        return None
    try:
        return CatalogSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
