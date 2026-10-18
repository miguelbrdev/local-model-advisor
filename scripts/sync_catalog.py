"""Synchronise the canIRun.ai catalogue into ``runtime/catalog/``.

This is the entry point the operating-system scheduler runs every 21 days
(``schtasks`` on Windows, ``cron`` on Linux/macOS).  It is deliberately
*not* part of the FastAPI process: the API only ever reads the last valid
snapshot and never downloads the catalogue during a request.

What one run does, in order:

1. :func:`src.sync.catalog_sync.sync_catalog` fetches, validates,
   normalises and atomically persists a snapshot.  On any failure the
   previous snapshot and manifest are left untouched and the run exits 1.
2. :func:`src.sync.retention.prune_snapshots` deletes unreferenced
   snapshots older than the retention window, never the active or the
   previous one.
3. Optionally (``--promote-bootstrap``) the active snapshot is copied to
   ``data/catalog_bootstrap.json`` so a fresh clone boots from it.  That
   file is versioned: committing it is a human decision, which is why the
   flag is off by default.

Exit codes: ``0`` catalogue active (new or unchanged), ``1`` synchronisation
failed, ``2`` synchronisation succeeded but a post-step (bootstrap
promotion) failed.

Usage::

    python -m scripts.sync_catalog [--catalog-dir DIR] [--policy FILE]
        [--timeout SECONDS] [--retain-days N] [--promote-bootstrap]
        [--bootstrap-path FILE] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from src.catalog.synced import DEFAULT_RUNTIME_CATALOG_DIR
from src.sync.bootstrap import (
    DEFAULT_BOOTSTRAP_PATH,
    BootstrapError,
    promote_snapshot_to_bootstrap,
)
from src.sync.catalog_policy import CatalogPolicy
from src.sync.catalog_sync import sync_catalog
from src.sync.retention import DEFAULT_RETAIN_DAYS, prune_snapshots

logger = logging.getLogger("scripts.sync_catalog")

EXIT_OK = 0
EXIT_SYNC_FAILED = 1
EXIT_POST_STEP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate so tests can inspect defaults)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.sync_catalog",
        description="Fetch the canIRun.ai catalogue and persist an atomic local snapshot.",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_RUNTIME_CATALOG_DIR,
        help="Directory holding current.json and snap-*.json (default: runtime/catalog).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Versioned policy YAML (default: data/catalog_policy.yaml).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--retain-days",
        type=int,
        default=DEFAULT_RETAIN_DAYS,
        help=(
            "Delete unreferenced snapshots older than this many days "
            f"(default: {DEFAULT_RETAIN_DAYS}; the active and previous snapshots "
            "are never deleted)."
        ),
    )
    parser.add_argument(
        "--promote-bootstrap",
        action="store_true",
        help="After a successful sync, overwrite the versioned bootstrap with the active one.",
    )
    parser.add_argument(
        "--bootstrap-path",
        type=Path,
        default=DEFAULT_BOOTSTRAP_PATH,
        help="Destination of --promote-bootstrap (default: data/catalog_bootstrap.json).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG level instead of INFO."
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Run one synchronisation cycle and return the process exit code.

    Args:
        argv: Command-line arguments without the program name; ``None``
            reads ``sys.argv``.
        transport: Injected HTTP transport, used by tests to avoid the
            network.  Production callers leave it ``None``.

    Returns:
        ``EXIT_OK``, ``EXIT_SYNC_FAILED`` or ``EXIT_POST_STEP_FAILED``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    policy = CatalogPolicy(args.policy)
    result = sync_catalog(
        args.catalog_dir, policy, timeout=args.timeout, transport=transport
    )
    if not result.ok:
        logger.error(
            "Synchronisation failed (%s). The previous snapshot in %s is kept.",
            result.error,
            args.catalog_dir,
        )
        return EXIT_SYNC_FAILED

    prune = prune_snapshots(args.catalog_dir, retain_days=args.retain_days)

    normalisation = result.normalization
    print(
        f"snapshot={result.snapshot_id} unchanged={result.unchanged} "
        f"included={normalisation.included_count if normalisation else '?'} "
        f"excluded={len(normalisation.exclusions) if normalisation else '?'} "
        f"pruned={len(prune.deleted)}"
    )

    if args.promote_bootstrap:
        source = args.catalog_dir / f"{result.snapshot_id}.json"
        try:
            target = promote_snapshot_to_bootstrap(source, args.bootstrap_path, overwrite=True)
        except BootstrapError as exc:
            logger.error("Bootstrap promotion failed: %s", exc)
            return EXIT_POST_STEP_FAILED
        print(f"bootstrap={target}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
