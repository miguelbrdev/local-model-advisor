"""Catalog snapshot models, deterministic hashing, atomic writes and manifest.

A snapshot is an immutable JSON file ``{snapshot_dir}/{snapshot_id}.json``
holding a ``SnapshotMeta`` header plus the normalised ``FamiliaModelo``
list.  ``runtime/catalog/current.json`` is a small mutable manifest that
points at the active (and previous) snapshot id.

Invariants enforced here
-------------------------
* ``SnapshotMeta.modelos_incluidos == len(CatalogSnapshot.familias)``.
* ``SnapshotMeta.modelos_origen == modelos_incluidos + len(motivos_exclusion)``.
* ``hash_contenido`` equals a recomputed SHA-256 over the familias and
  ``politica_version`` (content-addressed integrity check).
* ``snapshot_id == "snap-" + hash_contenido[:16]``.
* No duplicate ``familia_id``; no duplicate non-``None`` ``canirun_id``.
* ``fecha_sincronizacion`` is timezone-aware (UTC).

No HTTP is performed in this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.models.schemas import FamiliaModelo
from src.sync.normalizer import ExclusionReason

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"^https?://")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def compute_content_hash(
    familias: list[FamiliaModelo],
    politica_version: str,
) -> str:
    """Return the deterministic SHA-256 hex digest of *familias* + *politica_version*.

    The digest deliberately excludes timestamps, snapshot ids, file
    paths and input ordering: families are sorted by ``canirun_id``
    (falling back to ``familia_id``) and serialised as canonical JSON
    so the same catalogue content always yields the same snapshot id.

    Args:
        familias: Normalised model families that form the snapshot content.
        politica_version: Version of the catalog policy applied during normalisation.

    Returns:
        Lowercase 64-character SHA-256 hex digest.
    """
    ordered = sorted(familias, key=lambda f: f.canirun_id or f.familia_id)
    payload = {
        "politica_version": politica_version,
        "familias": [f.model_dump(mode="json") for f in ordered],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SnapshotMeta(BaseModel):
    """Reproducibility header stored alongside every snapshot.

    Attributes:
        schema_version: Version of the snapshot file layout.
        snapshot_id: Deterministic id, ``"snap-" + hash_contenido[:16]``.
        fecha_sincronizacion: UTC creation timestamp (must be tz-aware).
        source_url: HTTP(S) endpoint the catalogue was fetched from.
        hash_contenido: SHA-256 of the normalised content + policy version.
        modelos_origen: External records processed (``len(response.models)``).
        modelos_incluidos: Families included in this snapshot.
        motivos_exclusion: One entry per excluded external record.
        politica_version: Catalog policy version used for normalisation.

    Invariants (enforced by the model validator):
        * ``modelos_origen == modelos_incluidos + len(motivos_exclusion)``
          because every source record produces exactly one familia or one exclusion.
        * Non-empty ``schema_version``/``snapshot_id``/``politica_version``.
        * ``hash_contenido`` is 64 lowercase hex chars; ``source_url`` is HTTP(S).
        * ``fecha_sincronizacion`` is timezone-aware (naive datetimes rejected).
    """

    schema_version: str
    snapshot_id: str
    fecha_sincronizacion: datetime
    source_url: str
    hash_contenido: str
    modelos_origen: int = Field(ge=0)
    modelos_incluidos: int = Field(ge=0)
    motivos_exclusion: list[ExclusionReason] = Field(default_factory=list)
    politica_version: str

    @model_validator(mode="after")
    def _check_invariants(self) -> SnapshotMeta:
        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        if not self.politica_version.strip():
            raise ValueError("politica_version must not be empty")
        if self.fecha_sincronizacion.tzinfo is None:
            raise ValueError("fecha_sincronizacion must be timezone-aware (UTC)")
        if not _HASH_RE.match(self.hash_contenido):
            raise ValueError("hash_contenido must be a 64-char lowercase hex SHA-256")
        if not _URL_RE.match(self.source_url):
            raise ValueError("source_url must be an http(s) URL")
        if self.modelos_origen != self.modelos_incluidos + len(self.motivos_exclusion):
            raise ValueError(
                "modelos_origen must equal modelos_incluidos + len(motivos_exclusion)"
            )
        return self


class CatalogSnapshot(BaseModel):
    """A complete, immutable catalogue snapshot on disk.

    Attributes:
        meta: Reproducibility header (see :class:`SnapshotMeta`).
        familias: Normalised model families included in the snapshot.

    Invariants beyond ``SnapshotMeta``:
        * ``meta.modelos_incluidos == len(familias)``.
        * ``meta.hash_contenido`` matches a recomputation over ``familias``
          and ``meta.politica_version`` (detects tampering/corruption).
        * ``meta.snapshot_id == "snap-" + recomputed_hash[:16]``.
        * Unique ``familia_id``; unique non-``None`` ``canirun_id``.
    """

    meta: SnapshotMeta
    familias: list[FamiliaModelo]

    @model_validator(mode="after")
    def _check_familias(self) -> CatalogSnapshot:
        if self.meta.modelos_incluidos != len(self.familias):
            raise ValueError(
                f"modelos_incluidos ({self.meta.modelos_incluidos}) "
                f"does not match len(familias) ({len(self.familias)})"
            )
        familia_ids = [f.familia_id for f in self.familias]
        if len(familia_ids) != len(set(familia_ids)):
            raise ValueError("duplicate familia_id in snapshot")
        canirun_ids = [f.canirun_id for f in self.familias if f.canirun_id is not None]
        if len(canirun_ids) != len(set(canirun_ids)):
            raise ValueError("duplicate canirun_id in snapshot")
        expected_hash = compute_content_hash(self.familias, self.meta.politica_version)
        if self.meta.hash_contenido != expected_hash:
            raise ValueError("hash_contenido does not match snapshot content")
        if self.meta.snapshot_id != f"snap-{expected_hash[:16]}":
            raise ValueError("snapshot_id does not match content hash prefix")
        return self


@dataclass
class Manifest:
    """Contents of ``runtime/catalog/current.json``.

    Attributes:
        active_snapshot_id: Snapshot currently in use, or ``None`` if unknown.
        previous_snapshot_id: Snapshot rotated out, kept as fallback; ``None`` on first sync.
        updated_at: When the manifest was last rewritten, if parseable.
    """

    active_snapshot_id: str | None
    previous_snapshot_id: str | None
    updated_at: datetime | None


def write_snapshot(snapshot: CatalogSnapshot, catalog_dir: Path) -> Path:
    """Atomically write *snapshot* into *catalog_dir* if not already present.

    Snapshots are content-addressed and immutable: if
    ``{snapshot_id}.json`` already exists it is **not** overwritten and
    its path is returned unchanged (the caller is responsible for
    validating the pre-existing file).

    The write goes to a ``.tmp`` sibling, is round-trip validated with
    Pydantic, and only then moved into place with ``os.replace`` (atomic
    on Windows and POSIX).

    Args:
        snapshot: Fully validated snapshot to persist.
        catalog_dir: Target directory; created recursively if missing.

    Returns:
        Absolute-or-relative path of the snapshot file (new or pre-existing).

    Raises:
        ValidationError: If serialising and re-reading the snapshot fails
            (round-trip guard) — no partial file is left behind.
        OSError: If the filesystem operation fails — no partial file is
            left behind.
    """
    catalog_dir.mkdir(parents=True, exist_ok=True)
    final_path = catalog_dir / f"{snapshot.meta.snapshot_id}.json"
    if final_path.exists():
        logger.info(
            "Snapshot %s already exists — leaving immutable file untouched.",
            snapshot.meta.snapshot_id,
        )
        return final_path

    temp_path = catalog_dir / f"{snapshot.meta.snapshot_id}.json.tmp"
    try:
        serialised = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        temp_path.write_text(serialised, encoding="utf-8")
        CatalogSnapshot.model_validate_json(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return final_path


def read_snapshot(catalog_dir: Path, snapshot_id: str) -> CatalogSnapshot | None:
    """Load and fully validate ``{snapshot_id}.json`` from *catalog_dir*.

    Full validation includes recomputing the content hash, so a
    tampered or corrupt file returns ``None`` instead of raising.

    Args:
        catalog_dir: Directory that holds snapshot files.
        snapshot_id: Snapshot id (without extension).

    Returns:
        The validated snapshot, or ``None`` if the file is missing,
        corrupt, inconsistent, or its embedded ``snapshot_id`` does not
        match the requested one.
    """
    if not snapshot_id:
        return None
    path = catalog_dir / f"{snapshot_id}.json"
    if not path.exists():
        return None
    try:
        snapshot = CatalogSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
    if snapshot.meta.snapshot_id != snapshot_id:
        return None
    return snapshot


def update_manifest(
    active_id: str,
    previous_id: str | None,
    catalog_dir: Path,
) -> Path:
    """Atomically rewrite ``current.json`` pointing at *active_id*.

    Args:
        active_id: Snapshot id to mark as active.
        previous_id: Snapshot id being rotated out, or ``None`` on the
            first synchronisation.
        catalog_dir: Directory that holds the manifest.

    Returns:
        Path of ``current.json``.

    Raises:
        OSError: If the filesystem operation fails — the previous
            manifest remains in place.
    """
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "current.json"
    temp_path = catalog_dir / "current.json.tmp"
    payload = {
        "active_snapshot_id": active_id,
        "previous_snapshot_id": previous_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json.loads(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return path


def read_manifest(catalog_dir: Path) -> Manifest | None:
    """Read ``current.json`` from *catalog_dir*.

    Args:
        catalog_dir: Directory that holds the manifest.

    Returns:
        Parsed :class:`Manifest`, or ``None`` if the file is missing,
        not valid JSON, or not a JSON object.  Non-string or unparsable
        id/timestamp fields degrade to ``None`` rather than raising.
    """
    path = catalog_dir / "current.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    active = data.get("active_snapshot_id")
    previous = data.get("previous_snapshot_id")
    updated_raw = data.get("updated_at")
    updated: datetime | None = None
    if isinstance(updated_raw, str):
        try:
            updated = datetime.fromisoformat(updated_raw)
        except ValueError:
            updated = None
    return Manifest(
        active_snapshot_id=active if isinstance(active, str) else None,
        previous_snapshot_id=previous if isinstance(previous, str) else None,
        updated_at=updated,
    )


def load_active_snapshot(catalog_dir: Path) -> CatalogSnapshot | None:
    """Resolve and validate the currently active catalogue snapshot.

    Fallback chain (per Bloque A contract):

    1. Read ``current.json``; if unreadable → ``None``.
    2. Load + validate the file for ``active_snapshot_id``.
    3. If that fails (missing/corrupt/mismatched), try
       ``previous_snapshot_id``.
    4. If both fail → ``None``.

    Merely returning ids from the manifest is not enough — the snapshot
    file itself must parse and pass every content invariant.

    Args:
        catalog_dir: Directory that holds snapshots and the manifest.

    Returns:
        The active (or fallback) validated snapshot, else ``None``.
    """
    manifest = read_manifest(catalog_dir)
    if manifest is None:
        return None
    for snapshot_id in (manifest.active_snapshot_id, manifest.previous_snapshot_id):
        if not snapshot_id:
            continue
        snapshot = read_snapshot(catalog_dir, snapshot_id)
        if snapshot is not None:
            if snapshot_id != manifest.active_snapshot_id:
                logger.warning(
                    "Active snapshot unavailable; falling back to previous snapshot %s.",
                    snapshot_id,
                )
            return snapshot
    return None
