"""Seed catalogue loader.

Reads ``data/families_seed.yaml`` and returns a list of ``FamiliaModelo``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.models.schemas import FamiliaModelo

_SEED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "families_seed.yaml"


def load_seed(path: Path | None = None) -> list[FamiliaModelo]:
    """Load the curated family catalogue from YAML."""
    path = path or _SEED_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [FamiliaModelo(**item) for item in data]
