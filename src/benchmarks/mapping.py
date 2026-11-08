"""Reviewed mapping from catalogue families to source model names.

``data/benchmark_mapping.yaml`` is the only bridge between a
``canirun_id`` and the names Epoch AI and Artificial Analysis use.  It is
written and reviewed by a human, versioned, and tied to the catalogue
snapshot it was reviewed against.  Nothing in the code guesses a name:
a family absent from the table, or whose pinned name is absent from the
downloaded data, simply gets no score (and the scorer stays neutral).

Shape::

    version: "1.0.0"
    snapshot_id: "snap-176d40242fd0157a"
    familias:
      qwen3-8b:
        epoch_model_version: qwen3-8b     # name in the per-benchmark CSVs
        epoch_eci_model: Qwen3-8B         # name in eci_scores.csv
        artificial_analysis_slug: null    # slug in the Data API
        nota: optional reviewer remark
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_mapping.yaml"
)


class EntradaMapeo(BaseModel):
    """Source names pinned for one family; every field is optional."""

    model_config = ConfigDict(extra="forbid")

    epoch_model_version: str | None = None
    epoch_eci_model: str | None = None
    artificial_analysis_slug: str | None = None
    nota: str = ""

    def consultas(self, fuente: str) -> list[tuple[str, bool | None]]:
        """Exact names to look up in ``fuente`` as ``(name, solo_indice)``.

        ``solo_indice`` is ``True`` for the Epoch Capabilities Index name,
        ``False`` for the Epoch per-benchmark name and ``None`` for
        sources with a single namespace.  Empty when nothing is pinned.
        """
        if fuente == "epoch_ai":
            return [
                (nombre, solo_indice)
                for nombre, solo_indice in (
                    (self.epoch_model_version, False),
                    (self.epoch_eci_model, True),
                )
                if nombre
            ]
        if fuente == "artificial_analysis":
            return [(self.artificial_analysis_slug, None)] if self.artificial_analysis_slug else []
        return []


class MapeoBenchmarks(BaseModel):
    """The whole reviewed table.

    Attributes:
        version: Table version, bumped on every reviewed change.
        snapshot_id: Catalogue snapshot the table was reviewed against.
        familias: ``canirun_id`` → pinned names.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    familias: dict[str, EntradaMapeo] = Field(default_factory=dict)


class MapeoInvalidoError(Exception):
    """The mapping file is missing, unreadable or does not match the schema."""


def cargar_mapeo(path: Path | None = None) -> MapeoBenchmarks:
    """Load and validate the mapping table.

    Raises:
        MapeoInvalidoError: Naming the file and the underlying problem.
    """
    ruta = path or DEFAULT_MAPPING_PATH
    try:
        datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        return MapeoBenchmarks.model_validate(datos)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise MapeoInvalidoError(f"benchmark mapping unusable ({ruta}): {exc}") from exc
