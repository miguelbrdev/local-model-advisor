"""Fetch, validate, normalise and persist one canIRun.ai catalogue snapshot.

This is the Bloque A synchronisation orchestrator.  Tests inject
``httpx.MockTransport`` (or any ``httpx.BaseTransport``) so pytest never
opens a real network connection; production callers omit ``transport``
and the policy's source endpoint is used.

Atomicity contract
------------------
1. The external response is fully validated and normalised in memory.
2. An empty result (zero included models) aborts before any disk write.
3. The snapshot file is written via temp + round-trip validation +
   ``os.replace``.
4. ``current.json`` is rotated only after the snapshot file is durable.
5. Any failure leaves the previous snapshot and manifest untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.sync.canirun_schemas import CanIRunModelsResponse
from src.sync.catalog_policy import CatalogPolicy
from src.sync.normalizer import NormalizationResult, normalise_response
from src.sync.snapshot import (
    CatalogSnapshot,
    SnapshotMeta,
    compute_content_hash,
    read_manifest,
    read_snapshot,
    update_manifest,
    write_snapshot,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


@dataclass
class SyncResult:
    """Outcome of one :func:`sync_catalog` run.

    Attributes:
        ok: ``True`` when a snapshot is active (new or verified unchanged).
        snapshot_id: Deterministic snapshot id the run corresponds to, if known.
        error: Machine-readable failure reason, or ``None`` on success.
        normalization: Full normalisation report (families + exclusions), when available.
        unchanged: ``True`` when the identical snapshot already existed on disk.
        skipped_empty: ``True`` when zero models were included and nothing was written.
    """

    ok: bool
    snapshot_id: str | None = None
    error: str | None = None
    normalization: NormalizationResult | None = None
    unchanged: bool = False
    skipped_empty: bool = False


def sync_catalog(
    catalog_dir: Path,
    policy: CatalogPolicy | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> SyncResult:
    """Run one full synchronisation cycle.

    Args:
        catalog_dir: Directory holding snapshots and ``current.json``.
        policy: Catalog policy; defaults to ``data/catalog_policy.yaml``.
        endpoint: Overrides the policy source endpoint (used by tests).
        timeout: HTTP timeout in seconds.
        transport: Injected HTTP transport.  Tests pass
            ``httpx.MockTransport`` so no real connection is opened.

    Returns:
        :class:`SyncResult` describing success, idempotent no-op, or failure.

    Side effects:
        On success, writes ``{snapshot_id}.json`` (if new) and rotates
        ``current.json`` atomically.  On any failure the directory is
        left unchanged: no partial snapshot, no manifest rotation.

    Error values for ``SyncResult.error``:
        ``missing_endpoint``, ``http_error: ...``,
        ``invalid_external_response: ...``, ``empty_catalog``,
        ``existing_snapshot_corrupt``, ``snapshot_write_failed: ...``,
        ``manifest_update_failed: ...``.
    """
    active_policy = policy or CatalogPolicy()
    url = endpoint or active_policy.source_endpoint
    if url is None:
        return SyncResult(ok=False, error="missing_endpoint")

    # --- 1. fetch (memory only) ---
    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Catalog fetch failed for %s: %s", url, exc)
        return SyncResult(ok=False, error=f"http_error: {exc}")

    # --- 2. validate external contract ---
    try:
        external = CanIRunModelsResponse.model_validate_json(response.content)
    except ValidationError as exc:
        logger.warning("External catalogue response rejected: %s", exc)
        return SyncResult(ok=False, error=f"invalid_external_response: {exc}")

    # --- 3. normalise + filter ---
    normalisation = normalise_response(external, active_policy)
    if normalisation.included_count == 0:
        logger.warning(
            "Catalogue produced zero importable models (source=%d, excluded=%d); "
            "keeping previous snapshot.",
            len(external.models),
            len(normalisation.exclusions),
        )
        return SyncResult(
            ok=False,
            error="empty_catalog",
            normalization=normalisation,
            skipped_empty=True,
        )

    # --- 4. deterministic identity + snapshot model ---
    hash_contenido = compute_content_hash(normalisation.families, active_policy.version)
    snapshot_id = f"snap-{hash_contenido[:16]}"
    meta = SnapshotMeta(
        schema_version=SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        fecha_sincronizacion=datetime.now(UTC),
        source_url=url,
        hash_contenido=hash_contenido,
        modelos_origen=len(external.models),
        modelos_incluidos=normalisation.included_count,
        motivos_exclusion=normalisation.exclusions,
        politica_version=active_policy.version,
    )
    snapshot = CatalogSnapshot(meta=meta, familias=normalisation.families)

    # --- 5. snapshot file (immutable, atomic) ---
    snapshot_path = catalog_dir / f"{snapshot_id}.json"
    if snapshot_path.exists():
        existing = read_snapshot(catalog_dir, snapshot_id)
        if existing is None:
            logger.error(
                "Existing snapshot %s is corrupt or inconsistent; refusing to overwrite.",
                snapshot_id,
            )
            return SyncResult(
                ok=False,
                snapshot_id=snapshot_id,
                error="existing_snapshot_corrupt",
                normalization=normalisation,
            )
        unchanged = True
        logger.info(
            "Catalog unchanged: snapshot %s already present and valid.",
            snapshot_id,
        )
    else:
        try:
            write_snapshot(snapshot, catalog_dir)
        except OSError as exc:
            logger.error("Snapshot write failed: %s", exc)
            return SyncResult(
                ok=False,
                error=f"snapshot_write_failed: {exc}",
                normalization=normalisation,
            )
        unchanged = False

    # --- 6. manifest (only after the snapshot is durable) ---
    try:
        previous_manifest = read_manifest(catalog_dir)
        if previous_manifest is None or previous_manifest.active_snapshot_id != snapshot_id:
            previous_id = (
                previous_manifest.active_snapshot_id if previous_manifest else None
            )
            update_manifest(snapshot_id, previous_id, catalog_dir)
    except OSError as exc:
        logger.error("Manifest update failed: %s", exc)
        return SyncResult(
            ok=False,
            snapshot_id=snapshot_id,
            error=f"manifest_update_failed: {exc}",
            normalization=normalisation,
            unchanged=unchanged,
        )

    logger.info(
        "Catalog sync finished: snapshot=%s unchanged=%s models=%d excluded=%d hash=%s",
        snapshot_id,
        unchanged,
        normalisation.included_count,
        len(normalisation.exclusions),
        hash_contenido,
    )
    return SyncResult(
        ok=True,
        snapshot_id=snapshot_id,
        normalization=normalisation,
        unchanged=unchanged,
    )
