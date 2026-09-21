"""Normalise canIRun.ai external records into internal ``FamiliaModelo``.

This module is the bridge between the external contract validated by
``canirun_schemas`` and the internal domain model used by the
recommendation pipeline.  It applies the local ``CatalogPolicy`` to
decide which records are importable.

No HTTP calls are made here — the caller is responsible for
downloading the raw response beforehand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.schemas import FamiliaModelo
from src.sync.canirun_schemas import CanIRunModelRecord, CanIRunModelsResponse
from src.sync.catalog_policy import CatalogPolicy


@dataclass(frozen=True)
class ExclusionReason:
    """A record that was excluded during normalisation with an explicit reason."""

    record_id: str
    reason: str


@dataclass
class NormalizationResult:
    """Result of normalising a full canIRun.ai response.

    Attributes
    ----------
    families:
        Successfully normalised ``FamiliaModelo`` instances.
    exclusions:
        Records that were excluded with an explicit reason.
    source_count:
        Number of model records in the original external response.
    included_count:
        Number of records that were successfully normalised.
    """

    families: list[FamiliaModelo] = field(default_factory=list)
    exclusions: list[ExclusionReason] = field(default_factory=list)
    source_count: int = 0
    included_count: int = 0


def _none_if_empty(value: str) -> str | None:
    """Return *value* as-is if non-empty, else ``None``."""
    return value if value else None


def normalise_record(
    record: CanIRunModelRecord,
    policy: CatalogPolicy,
) -> FamiliaModelo | ExclusionReason:
    """Normalise a single external record to a ``FamiliaModelo``.

    Returns ``FamiliaModelo`` on success, or ``ExclusionReason`` if
    the record is excluded by the policy.
    """
    # --- validity: non-empty id ---
    id_error = policy.validate_record_id(record.id)
    if id_error is not None:
        return ExclusionReason(record_id=record.id or "(empty)", reason=id_error)

    # --- task mapping ---
    tasks = policy.tasks_from_use_cases(record.useCase)
    if policy.require_at_least_one_supported_task and not tasks:
        return ExclusionReason(
            record_id=record.id,
            reason="no_supported_task",
        )

    # --- preserve informational tags ---
    informational = [t for t in record.useCase if policy.is_informational(t)]

    # --- build FamiliaModelo ---
    pipeline_tag = tasks[0] if tasks else ""

    return FamiliaModelo(
        familia_id=record.id,
        variantes=[],  # canIRun checkpoints are individual — no nested variants
        pipeline_tag=pipeline_tag,
        tareas_soportadas=tasks,
        tags_informativos=informational,
        license_declarada=_none_if_empty(record.license),
        license_url=_none_if_empty(record.url),
        idiomas_detectados=None,
        formatos_disponibles=[],
        canirun_id=record.id,
        nombre_mostrado=record.name,
        proveedor=record.provider,
        familia_origen=record.family,
        parametros_b=record.paramsBillions,
        parametros_texto=record.params,
        arquitectura=_none_if_empty(record.architecture),
        fecha_lanzamiento=_none_if_empty(record.releaseDate),
        contexto_maximo=record.contextLength or None,
    )


def normalise_response(
    response: CanIRunModelsResponse,
    policy: CatalogPolicy,
) -> NormalizationResult:
    """Normalise an entire canIRun.ai models response.

    Deduplicates by ``id``.  Records that fail validity checks or have
    no supported tasks are collected in ``exclusions``.
    """
    result = NormalizationResult(source_count=response.count)

    seen_ids: set[str] = set()

    for record in response.models:
        # --- deduplication ---
        if record.id in seen_ids:
            result.exclusions.append(
                ExclusionReason(record_id=record.id, reason="duplicate_id")
            )
            continue
        seen_ids.add(record.id)

        normalised = normalise_record(record, policy)
        if isinstance(normalised, ExclusionReason):
            result.exclusions.append(normalised)
        else:
            result.families.append(normalised)

    result.included_count = len(result.families)
    return result
