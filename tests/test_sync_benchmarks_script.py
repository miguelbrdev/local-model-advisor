"""Tests for scripts.sync_benchmarks — the scheduled benchmark entry point.

Both sources are served by one ``httpx.MockTransport`` that routes on the
host; everything is written under ``tmp_path``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from scripts import sync_benchmarks as script
from src.benchmarks.artificial_analysis import VARIABLE_CLAVE
from src.benchmarks.snapshot import read_snapshot

_COLS = (
    "Model version,mean_score,Best score (across scorers),Release date,Organization,stderr,id\n"
)


def _zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archivo:
        archivo.writestr(
            "gpqa_diamond.csv", _COLS + "qwen3-8b,0.56,0.5676,2025-04-28,Alibaba,0,a\n"
        )
        archivo.writestr("math_level_5.csv", _COLS)
        archivo.writestr("swe_bench_verified.csv", _COLS)
        archivo.writestr(
            "benchmark_metadata.csv",
            "benchmark,in_eci,source_file,score_column,scale,random_baseline,score_ceiling\n"
            "GPQA diamond,True,gpqa_diamond.csv,Best score (across scorers),1.0,0.25,0.97\n",
        )
        archivo.writestr(
            "epoch_capabilities_index/eci_scores.csv",
            "Model,eci,date,Accessibility group\nQwen3-8B,132.2,2025-04-28,Open weights\n",
        )
    return buffer.getvalue()


def _aa_page() -> dict:
    return {
        "intelligence_index_version": 5,
        "pagination": {"has_more": False},
        "data": [
            {"slug": "glm-5-3", "name": "GLM-5.3", "release_date": "2026-08-18",
             "evaluations": {"artificial_analysis_intelligence_index": 31.7}},
        ],
    }


class _Red:
    def __init__(self, *, epoch_status: int = 200, aa_status: int = 200) -> None:
        self.epoch_status = epoch_status
        self.aa_status = aa_status
        self.hosts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.hosts.append(request.url.host)
        if request.url.host == "epoch.ai":
            return httpx.Response(self.epoch_status, content=_zip())
        if request.url.host == "artificialanalysis.ai":
            return httpx.Response(self.aa_status, json=_aa_page())
        return httpx.Response(404)


def _mapping(tmp_path: Path) -> Path:
    ruta = tmp_path / "mapping.yaml"
    ruta.write_text(
        "version: 't'\nsnapshot_id: snap-test\nfamilias:\n"
        "  qwen3-8b:\n    epoch_model_version: qwen3-8b\n    epoch_eci_model: Qwen3-8B\n"
        "  glm-5.3:\n    artificial_analysis_slug: glm-5-3\n",
        encoding="utf-8",
    )
    return ruta


def _run(tmp_path: Path, *extra: str, red: _Red, env: dict[str, str] | None = None) -> int:
    return script.main(
        ["--output", str(tmp_path / "runtime" / "current.json"),
         "--mapping", str(_mapping(tmp_path)), *extra],
        transport=httpx.MockTransport(red),
        env=env if env is not None else {},
    )


class TestParser:
    def test_defaults(self) -> None:
        args = script.build_parser().parse_args([])
        assert args.output.name == "current.json" and args.output.parent.name == "benchmarks"
        assert args.mapping.name == "benchmark_mapping.yaml"
        assert args.bootstrap_path.name == "benchmarks_bootstrap.json"
        assert args.promote_bootstrap is False
        assert args.skip_artificial_analysis is False
        assert args.timeout == 60.0

    def test_exit_codes_distinct(self) -> None:
        assert len({script.EXIT_OK, script.EXIT_SYNC_FAILED, script.EXIT_POST_STEP_FAILED}) == 3


class TestSources:
    def test_without_key_only_epoch(self) -> None:
        fuentes = script.elegir_fuentes(skip_artificial_analysis=False, env={})
        assert [f.nombre for f in fuentes] == ["epoch_ai"]

    def test_with_key_both(self) -> None:
        fuentes = script.elegir_fuentes(skip_artificial_analysis=False, env={VARIABLE_CLAVE: "k"})
        assert [f.nombre for f in fuentes] == ["epoch_ai", "artificial_analysis"]

    def test_skip_flag_ignores_key(self) -> None:
        fuentes = script.elegir_fuentes(skip_artificial_analysis=True, env={VARIABLE_CLAVE: "k"})
        assert [f.nombre for f in fuentes] == ["epoch_ai"]


class TestRuns:
    def test_epoch_only_without_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        red = _Red()
        assert _run(tmp_path, red=red) == script.EXIT_OK
        assert red.hosts == ["epoch.ai"]
        snapshot = read_snapshot(tmp_path / "runtime" / "current.json")
        assert [f.nombre for f in snapshot.meta.fuentes] == ["epoch_ai"]
        out = capsys.readouterr().out
        assert "sources=epoch_ai " in out and "families_with_data=1" in out

    def test_both_sources_with_key_and_promotion_drops_aa(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bootstrap = tmp_path / "data" / "benchmarks_bootstrap.json"
        code = _run(
            tmp_path, "--promote-bootstrap", "--bootstrap-path", str(bootstrap),
            red=_Red(), env={VARIABLE_CLAVE: "k"},
        )
        assert code == script.EXIT_OK
        runtime = read_snapshot(tmp_path / "runtime" / "current.json")
        assert [f.nombre for f in runtime.meta.fuentes] == ["epoch_ai", "artificial_analysis"]
        assert "glm-5.3" in runtime.puntuaciones
        promovido = read_snapshot(bootstrap)
        assert [f.nombre for f in promovido.meta.fuentes] == ["epoch_ai"]
        assert "glm-5.3" not in promovido.puntuaciones
        assert "bootstrap=" in capsys.readouterr().out

    def test_aa_failure_is_reported_but_not_fatal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(tmp_path, red=_Red(aa_status=401), env={VARIABLE_CLAVE: "bad"})
        assert code == script.EXIT_OK
        out = capsys.readouterr().out
        assert "skipped_source=artificial_analysis" in out
        assert "API key rejected" in out

    def test_epoch_failure_exits_one_and_keeps_previous(self, tmp_path: Path) -> None:
        assert _run(tmp_path, red=_Red()) == script.EXIT_OK
        destino = tmp_path / "runtime" / "current.json"
        antes = destino.read_bytes()
        assert _run(tmp_path, red=_Red(epoch_status=500)) == script.EXIT_SYNC_FAILED
        assert destino.read_bytes() == antes

    def test_invalid_mapping_exits_two(self, tmp_path: Path) -> None:
        code = script.main(
            ["--output", str(tmp_path / "c.json"), "--mapping", str(tmp_path / "missing.yaml")],
            transport=httpx.MockTransport(_Red()), env={},
        )
        assert code == script.EXIT_POST_STEP_FAILED

    def test_second_run_is_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, red=_Red()) == script.EXIT_OK
        assert _run(tmp_path, red=_Red()) == script.EXIT_OK
        assert "unchanged=True" in capsys.readouterr().out
