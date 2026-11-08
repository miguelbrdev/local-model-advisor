"""Pydantic schemas for the canIRun.ai ``GET /api/models`` response.

These models validate the **raw** external contract before any
normalisation to internal domain types.  They intentionally mirror
the external field names so that changes in the upstream API are
detected as early as possible.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CanIRunModelRecord(BaseModel):
    """A single model record as returned by canIRun.ai.

    All fields are required except ``architecture`` which may be
    absent in future API versions.
    """

    id: str
    name: str
    provider: str
    family: str
    params: str
    paramsBillions: float  # noqa: N815
    architecture: str = ""
    releaseDate: str = ""  # noqa: N815
    contextLength: int = 0  # noqa: N815
    useCase: list[str] = Field(default_factory=list)  # noqa: N815
    url: str = ""
    license: str = ""


class CanIRunQuant(BaseModel):
    """One row of ``quants`` in the detail record (not used by the pipeline yet)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    bits: int | None = None
    vramGB: float | None = None  # noqa: N815
    diskGB: float | None = None  # noqa: N815
    quality: str | None = None


class CanIRunModelDetail(CanIRunModelRecord):
    """``GET /api/models/{id}`` as observed on 2026-11-08.

    Extends the list record with runtime library ids, capability flags
    and a per-quantisation memory table.  Every extra field is optional
    and unknown fields are ignored, so the list contract stays strict
    while the detail one tolerates growth.
    """

    model_config = ConfigDict(extra="ignore")

    activeParams: float | str | None = None  # noqa: N815 — "3B active" in some records
    description: str = ""
    tools: bool | None = None
    thinking: bool | None = None
    minRamGB: float | None = None  # noqa: N815
    recommendedRamGB: float | None = None  # noqa: N815
    ollamaId: str | None = None  # noqa: N815
    lmStudioId: str | None = None  # noqa: N815
    lineage: str | None = None
    supersededBy: str | None = None  # noqa: N815
    intelligenceIndex: float | None = None  # noqa: N815
    quants: list[CanIRunQuant] = Field(default_factory=list)


class CanIRunModelsResponse(BaseModel):
    """Top-level response of ``GET /api/models``.

    The response must contain exactly two keys: ``count`` (int) and
    ``models`` (list of ``CanIRunModelRecord``).
    """

    count: int
    models: list[CanIRunModelRecord]
