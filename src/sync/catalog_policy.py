"""Local catalog policy loader and evaluator.

Loads ``data/catalog_policy.yaml`` and provides query helpers that
determine whether an external model record is eligible for import.

The policy is loaded once at construction time and is immutable
afterwards — it mirrors the versioned static policy that is
committed alongside the code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "catalog_policy.yaml"


class CatalogPolicy:
    """Loaded and evaluated local catalog policy.

    Parameters
    ----------
    policy_path:
        Path to the YAML policy file.  Defaults to
        ``data/catalog_policy.yaml`` at the project root.
    """

    def __init__(self, policy_path: Path | None = None) -> None:
        path = policy_path or _POLICY_PATH
        with open(path, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh)

        self.version: str = raw["version"]
        self.description: str = raw.get("description", "")
        source = raw.get("source") or {}
        self.source_endpoint: str | None = source.get("endpoint")

        # use_case_mapping: external tag -> list[str] internal tasks
        self._use_case_mapping: dict[str, list[str]] = raw.get("use_case_mapping", {})

        self.informational_tags: list[str] = raw.get("informational_tags", [])
        self.out_of_scope_tags: list[str] = raw.get("out_of_scope_tags", [])

        rules = raw.get("validity_rules", {})
        self.require_non_empty_id: bool = rules.get("require_non_empty_id", True)
        self.require_unique_id: bool = rules.get("require_unique_id", True)
        self.require_at_least_one_supported_task: bool = rules.get(
            "require_at_least_one_supported_task", True
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def supported_tasks(self) -> list[str]:
        """Return the list of internal task identifiers that are supported."""
        tasks: list[str] = []
        for internal_tasks in self._use_case_mapping.values():
            for task in internal_tasks:
                if task not in tasks:
                    tasks.append(task)
        return tasks

    def external_tags_for_task(self, internal_task: str) -> list[str]:
        """Return external useCase tags that map to *internal_task*."""
        result: list[str] = []
        for ext_tag, internal_tasks in self._use_case_mapping.items():
            if internal_task in internal_tasks:
                result.append(ext_tag)
        return result

    def tasks_from_use_cases(self, use_cases: list[str]) -> list[str]:
        """Map a list of external useCase tags to internal tasks.

        Returns only tasks that appear in the policy mapping.  Unknown
        tags are silently ignored.
        """
        tasks: list[str] = []
        for tag in use_cases:
            for internal_tasks in self._use_case_mapping.get(tag, []):
                if internal_tasks not in tasks:
                    tasks.append(internal_tasks)
        return tasks

    def is_supported_use_case(self, tag: str) -> bool:
        """Return ``True`` if *tag* maps to at least one internal task."""
        return tag in self._use_case_mapping

    def has_any_supported_task(self, use_cases: list[str]) -> bool:
        """Return ``True`` if at least one tag maps to an internal task."""
        return any(self.is_supported_use_case(t) for t in use_cases)

    def is_informational(self, tag: str) -> bool:
        """Return ``True`` if *tag* is an informational (non-task) tag."""
        return tag in self.informational_tags

    def is_out_of_scope(self, tag: str) -> bool:
        """Return ``True`` if *tag* is explicitly out of scope."""
        return tag in self.out_of_scope_tags

    def validate_record_id(self, record_id: str) -> str | None:
        """Return ``None`` if the id is valid, or a reason string if not."""
        if self.require_non_empty_id and not record_id.strip():
            return "empty_id"
        return None
