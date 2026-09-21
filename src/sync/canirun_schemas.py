"""Pydantic schemas for the canIRun.ai ``GET /api/models`` response.

These models validate the **raw** external contract before any
normalisation to internal domain types.  They intentionally mirror
the external field names so that changes in the upstream API are
detected as early as possible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class CanIRunModelsResponse(BaseModel):
    """Top-level response of ``GET /api/models``.

    The response must contain exactly two keys: ``count`` (int) and
    ``models`` (list of ``CanIRunModelRecord``).
    """

    count: int
    models: list[CanIRunModelRecord]
