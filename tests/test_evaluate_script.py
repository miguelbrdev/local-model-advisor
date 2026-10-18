"""Tests for scripts.evaluate_golden — the golden-suite command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import evaluate_golden as script
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH


def _write_suite(path: Path, *, snapshot_id: str = "snap-176d40242fd0157a", cases: str) -> Path:
    path.write_text(
        f"version: 'test'\nsnapshot_id: {snapshot_id}\ncasos:\n{cases}", encoding="utf-8"
    )
    return path


PASSING = (
    "  - id: ok\n    criterios: {tarea: codigo, vram_gb: 8}\n"
    "    debe_incluir: [qwen2.5-coder-7b]\n    debe_excluir: [qwen3-coder-480b]\n"
)
FAILING = (
    "  - id: ko\n    criterios: {tarea: codigo, vram_gb: 8}\n"
    "    debe_incluir: [qwen3-coder-480b]\n"
)


def _run(tmp_path: Path, suite: Path, *extra: str) -> int:
    return script.main(
        [
            "--suite", str(suite),
            "--catalog-dir", str(tmp_path / "absent"),
            "--bootstrap-path", str(DEFAULT_BOOTSTRAP_PATH),
            *extra,
        ]
    )


class TestParser:
    def test_defaults(self) -> None:
        args = script.build_parser().parse_args([])
        assert args.suite.name == "golden_queries.yaml"
        assert args.catalog_dir.name == "catalog"
        assert args.json is False and args.strict_snapshot is False


class TestExitCodes:
    def test_passing_suite_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite = _write_suite(tmp_path / "s.yaml", cases=PASSING)
        assert _run(tmp_path, suite) == script.EXIT_OK
        out = capsys.readouterr().out
        assert "[PASS] ok" in out
        assert "cases passed: 1/1" in out

    def test_failing_case_exits_one_and_names_the_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite = _write_suite(tmp_path / "s.yaml", cases=PASSING + FAILING)
        assert _run(tmp_path, suite) == script.EXIT_CASES_FAILED
        out = capsys.readouterr().out
        assert "[FAIL] ko" in out
        assert "missing from top 3" in out
        assert "cases passed: 1/2" in out

    def test_invalid_suite_exits_two(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 'x'\n", encoding="utf-8")
        assert _run(tmp_path, bad) == script.EXIT_SUITE_INVALID

    def test_missing_suite_exits_two(self, tmp_path: Path) -> None:
        assert _run(tmp_path, tmp_path / "nope.yaml") == script.EXIT_SUITE_INVALID

    def test_snapshot_mismatch_is_only_fatal_when_strict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite = _write_suite(tmp_path / "s.yaml", snapshot_id="snap-other", cases=PASSING)
        assert _run(tmp_path, suite) == script.EXIT_OK
        assert "SNAPSHOT MISMATCH" in capsys.readouterr().out
        assert _run(tmp_path, suite, "--strict-snapshot") == script.EXIT_CASES_FAILED


class TestJsonOutput:
    def test_json_report_is_parseable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite = _write_suite(tmp_path / "s.yaml", cases=PASSING)
        assert _run(tmp_path, suite, "--json") == script.EXIT_OK
        informe = json.loads(capsys.readouterr().out)
        assert informe["total"] == 1
        assert informe["resultados"][0]["id"] == "ok"
        assert informe["snapshot_id_catalogo"] == "snap-176d40242fd0157a"


class TestCommittedSuiteViaScript:
    def test_default_suite_passes(self, tmp_path: Path) -> None:
        code = script.main(
            ["--catalog-dir", str(tmp_path / "absent"),
             "--bootstrap-path", str(DEFAULT_BOOTSTRAP_PATH), "--strict-snapshot"]
        )
        assert code == script.EXIT_OK
