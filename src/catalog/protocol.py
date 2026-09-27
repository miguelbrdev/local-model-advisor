"""Catalogue contract consumed by the recommendation pipeline.

Task T1 of the catalog-loader migration: the pipeline must depend on a
catalogue *abstraction*, never on a concrete YAML/JSON path.  The API
layer mounts one concrete implementation per process, so a single
snapshot id backs the whole process, and injects it into the pipeline,
which only ever sees this Protocol.

``CatalogUnavailableError`` lives here rather than inside a concrete
adapter because it is part of the contract: callers catch a single type
regardless of which implementation exhausted its sources.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.models.schemas import FamiliaModelo


class CatalogUnavailableError(Exception):
    """Raised when a catalogue implementation has no source left to serve.

    Deliberately never swallowed and never converted into an empty
    catalogue: an exhausted fallback chain is a deployment fault, and an
    empty list would be indistinguishable from a legitimately empty
    catalogue (the exact ambiguity this project forbids).

    The exception message must name every source that was tried, so the
    operator knows which file or directory to fix.
    """


@runtime_checkable
class CatalogoModelos(Protocol):
    """Read-only catalogue of model families the pipeline ranks over.

    ``runtime_checkable`` is required (not decorative): the repository
    runs no type checker, so an ``isinstance`` assertion in the tests is
    the only executable proof that a test double or a new implementation
    actually satisfies the contract.

    Attributes:
        snapshot_id: Identifier of the snapshot the served data comes
            from (e.g. ``"snap-176d40242fd0157a"``), or ``None`` for
            implementations without snapshot semantics.  It must always
            describe exactly the data :meth:`cargar` returns, because it
            is persisted per conversation to keep recommendations
            reproducible.
    """

    @property
    def snapshot_id(self) -> str | None:
        """Snapshot backing the catalogue.

        Returns:
            The snapshot id, or ``None`` only for implementations that
            genuinely have no snapshot concept.

        Raises:
            CatalogUnavailableError: When resolution fails because every
                source was exhausted.  It must not degrade to a silent
                ``None`` in that case — a missing id would silently
                break reproducibility.
        """
        ...

    def cargar(self) -> list[FamiliaModelo]:
        """Load the full catalogue.

        Returns:
            All model families in snapshot order.  The list itself is a
            fresh copy per call; the ``FamiliaModelo`` elements are
            shared between calls and must be treated as read-only.

        Raises:
            CatalogUnavailableError: When every source was exhausted.
        """
        ...
