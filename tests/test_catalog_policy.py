"""Tests for src.sync.catalog_policy — local policy loader and evaluator.

All tests use the real YAML policy from data/catalog_policy.yaml.
No HTTP calls are made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sync.catalog_policy import CatalogPolicy


@pytest.fixture
def policy() -> CatalogPolicy:
    return CatalogPolicy()


class TestCatalogPolicy:
    def test_loads_default_policy(self, policy: CatalogPolicy) -> None:
        assert policy.version == "1.0.0"

    def test_supported_tasks(self, policy: CatalogPolicy) -> None:
        tasks = policy.supported_tasks()
        assert "chat" in tasks
        assert "codigo" in tasks
        assert "razonamiento" in tasks
        assert len(tasks) == 3

    def test_external_tags_for_chat(self, policy: CatalogPolicy) -> None:
        tags = policy.external_tags_for_task("chat")
        assert tags == ["chat"]

    def test_external_tags_for_codigo(self, policy: CatalogPolicy) -> None:
        tags = policy.external_tags_for_task("codigo")
        assert tags == ["code"]

    def test_external_tags_for_razonamiento(self, policy: CatalogPolicy) -> None:
        tags = policy.external_tags_for_task("razonamiento")
        assert tags == ["reasoning"]

    def test_tasks_from_use_cases_code(self, policy: CatalogPolicy) -> None:
        tasks = policy.tasks_from_use_cases(["code"])
        assert tasks == ["codigo"]

    def test_tasks_from_use_cases_multitask(self, policy: CatalogPolicy) -> None:
        tasks = policy.tasks_from_use_cases(["chat", "code", "reasoning"])
        assert "chat" in tasks
        assert "codigo" in tasks
        assert "razonamiento" in tasks

    def test_tasks_from_use_cases_unknown_ignored(self, policy: CatalogPolicy) -> None:
        tasks = policy.tasks_from_use_cases(["chat", "unknown_tag"])
        assert tasks == ["chat"]

    def test_is_supported_use_case(self, policy: CatalogPolicy) -> None:
        assert policy.is_supported_use_case("chat") is True
        assert policy.is_supported_use_case("code") is True
        assert policy.is_supported_use_case("reasoning") is True
        assert policy.is_supported_use_case("vision") is False
        assert policy.is_supported_use_case("image") is False

    def test_has_any_supported_task(self, policy: CatalogPolicy) -> None:
        assert policy.has_any_supported_task(["chat"]) is True
        assert policy.has_any_supported_task(["chat", "code"]) is True
        assert policy.has_any_supported_task(["vision", "image"]) is False
        assert policy.has_any_supported_task([]) is False

    def test_is_informational(self, policy: CatalogPolicy) -> None:
        assert policy.is_informational("multilingual") is True
        assert policy.is_informational("edge") is True
        assert policy.is_informational("rag") is True
        assert policy.is_informational("chat") is False

    def test_is_out_of_scope(self, policy: CatalogPolicy) -> None:
        assert policy.is_out_of_scope("vision") is True
        assert policy.is_out_of_scope("image") is True
        assert policy.is_out_of_scope("video") is True
        assert policy.is_out_of_scope("chat") is False

    def test_validate_record_id_valid(self, policy: CatalogPolicy) -> None:
        assert policy.validate_record_id("qwen2.5-coder-7b") is None

    def test_validate_record_id_empty(self, policy: CatalogPolicy) -> None:
        assert policy.validate_record_id("") == "empty_id"

    def test_validate_record_id_whitespace(self, policy: CatalogPolicy) -> None:
        assert policy.validate_record_id("   ") == "empty_id"

    def test_informational_tags_list(self, policy: CatalogPolicy) -> None:
        assert "multilingual" in policy.informational_tags
        assert "edge" in policy.informational_tags
        assert "rag" in policy.informational_tags

    def test_out_of_scope_tags_list(self, policy: CatalogPolicy) -> None:
        assert "vision" in policy.out_of_scope_tags
        assert "image" in policy.out_of_scope_tags
        assert "video" in policy.out_of_scope_tags

    def test_custom_policy_path(self, tmp_path: Path) -> None:
        policy_yaml = tmp_path / "policy.yaml"
        policy_yaml.write_text(
            'version: "2.0"\n'
            'use_case_mapping:\n'
            '  code:\n'
            '    - codigo\n'
            'informational_tags: []\n'
            'out_of_scope_tags: []\n'
            'validity_rules:\n'
            '  require_non_empty_id: true\n'
            '  require_unique_id: true\n'
            '  require_at_least_one_supported_task: true\n',
            encoding="utf-8",
        )
        policy = CatalogPolicy(policy_path=policy_yaml)
        assert policy.version == "2.0"
        assert policy.supported_tasks() == ["codigo"]
