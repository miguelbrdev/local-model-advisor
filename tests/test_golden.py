"""Tests for src.evaluation.golden and the committed golden suite.

Unit tests use a tiny in-memory catalogue and the offline heuristic.  The
last class runs the real committed suite against the committed bootstrap:
it is the regression net the suite exists for, so a failure there means
either the pipeline changed behaviour or the suite needs a reviewed update.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.benchmarks.provider import BenchmarksLocales
from src.benchmarks.snapshot import DEFAULT_BENCHMARKS_BOOTSTRAP_PATH
from src.catalog.synced import CatalogoSincronizado
from src.compatibility.heuristica import HeuristicaParametrosProvider
from src.evaluation.golden import (
    DEFAULT_SUITE_PATH,
    CasoGolden,
    SuiteGolden,
    cargar_suite,
    evaluar_caso,
    evaluar_suite,
)
from src.models.schemas import FamiliaModelo
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH


def _fam(fid: str, tareas: list[str], parametros_b: float, licencia: str = "MIT") -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid,
        variantes=[],
        pipeline_tag=tareas[0],
        tareas_soportadas=tareas,
        license_declarada=licencia,
        canirun_id=fid,
        parametros_b=parametros_b,
        parametros_texto=f"{parametros_b:g}B",
    )


class _Catalogo:
    def __init__(self, familias: list[FamiliaModelo], snapshot_id: str = "snap-test") -> None:
        self._familias = familias
        self._snapshot_id = snapshot_id

    @property
    def snapshot_id(self) -> str | None:
        return self._snapshot_id

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def cargar(self) -> list[FamiliaModelo]:
        return list(self._familias)


CATALOGO = _Catalogo(
    [
        _fam("coder-small", ["codigo"], 1.5),
        _fam("coder-big", ["codigo"], 70.0),
        _fam("chat-mid", ["chat", "codigo"], 7.0, licencia="Llama 3.1 Community"),
        _fam("reasoner", ["razonamiento"], 7.0),
    ]
)
PROVIDER = HeuristicaParametrosProvider()


def _caso(**overrides) -> CasoGolden:
    base = {"id": "c", "criterios": {"tarea": "codigo", "vram_gb": 8}}
    base.update(overrides)
    return CasoGolden.model_validate(base)


class TestCasoGoldenValidation:
    def test_included_and_excluded_overlap_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both included and excluded"):
            _caso(debe_incluir=["a"], debe_excluir=["a"])

    def test_primero_cannot_be_excluded(self) -> None:
        with pytest.raises(ValidationError, match="'primero'"):
            _caso(primero="a", debe_excluir=["a"])

    def test_primero_cannot_be_kept_out_of_the_top(self) -> None:
        with pytest.raises(ValidationError, match="'primero'"):
            _caso(primero="a", debe_quedar_fuera_del_top=["a"])

    def test_included_and_kept_out_overlap_rejected(self) -> None:
        with pytest.raises(ValidationError, match="kept out of the top"):
            _caso(debe_incluir=["a"], debe_quedar_fuera_del_top=["a"])

    def test_criteria_must_be_complete(self) -> None:
        with pytest.raises(ValidationError, match="mandatory fields"):
            _caso(criterios={"tarea": "codigo"})

    def test_duplicate_case_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate case ids"):
            SuiteGolden(
                version="1", snapshot_id="snap-x", casos=[_caso(id="dup"), _caso(id="dup")]
            )

    def test_suite_needs_at_least_one_case(self) -> None:
        with pytest.raises(ValidationError):
            SuiteGolden(version="1", snapshot_id="snap-x", casos=[])


class TestEvaluarCaso:
    def test_passing_case(self) -> None:
        caso = _caso(
            debe_incluir=["coder-small", "chat-mid"],
            debe_excluir=["coder-big", "reasoner"],
            primero="coder-small",
        )
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert r.aprobado and r.fallos == []
        assert r.top == ["coder-small", "chat-mid"]
        assert r.candidatos_totales == 2
        assert (r.incluidos_encontrados, r.incluidos_esperados) == (2, 2)
        assert (r.excluidos_respetados, r.excluidos_esperados) == (2, 2)
        assert r.primero_ok is True

    def test_missing_inclusion_fails(self) -> None:
        caso = _caso(debe_incluir=["coder-big"])
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert not r.aprobado
        assert any("missing from top" in f for f in r.fallos)
        assert r.incluidos_encontrados == 0

    def test_exclusion_checked_against_full_ranking_not_top_n(self) -> None:
        """``chat-mid`` is rank 2; with top_n=1 it is outside the window but
        still a candidate, so excluding it must fail."""
        caso = _caso(top_n=1, debe_excluir=["chat-mid"])
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert not r.aprobado
        assert any("should have been filtered out" in f for f in r.fallos)

    def test_kept_out_of_top_fails_when_it_ranks(self) -> None:
        """``coder-small`` ranks first; demanding it stays out of the top 1
        must fail, while the same demand with a model outside the window passes."""
        r = evaluar_caso(
            _caso(top_n=1, debe_quedar_fuera_del_top=["coder-small"]),
            catalogo=CATALOGO, vram_provider=PROVIDER,
        )
        assert not r.aprobado
        assert any("should not rank in top 1" in f for f in r.fallos)

        r = evaluar_caso(
            _caso(top_n=1, debe_quedar_fuera_del_top=["chat-mid"]),
            catalogo=CATALOGO, vram_provider=PROVIDER,
        )
        assert r.aprobado

    def test_wrong_first_fails(self) -> None:
        caso = _caso(primero="chat-mid")
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert r.primero_ok is False
        assert any("expected first" in f for f in r.fallos)

    def test_empty_result_fails_by_default(self) -> None:
        caso = _caso(criterios={"tarea": "chat", "vram_gb": 1})
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert r.candidatos_totales == 0
        assert not r.aprobado
        assert any("at least 1 candidates" in f for f in r.fallos)

    def test_empty_result_can_be_the_expectation(self) -> None:
        caso = _caso(criterios={"tarea": "chat", "vram_gb": 1}, min_candidatos=0,
                     debe_excluir=["chat-mid"])
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert r.aprobado
        assert r.primero_ok is None

    def test_primero_with_empty_ranking_fails_cleanly(self) -> None:
        caso = _caso(criterios={"tarea": "chat", "vram_gb": 1}, primero="chat-mid",
                     min_candidatos=0)
        r = evaluar_caso(caso, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert r.primero_ok is False


class TestEvaluarSuite:
    def _suite(self, *casos: CasoGolden, snapshot_id: str = "snap-test") -> SuiteGolden:
        return SuiteGolden(version="1", snapshot_id=snapshot_id, casos=list(casos))

    def test_aggregates(self) -> None:
        suite = self._suite(
            _caso(id="ok", debe_incluir=["coder-small"], debe_excluir=["coder-big"],
                  primero="coder-small"),
            _caso(id="ko", debe_incluir=["coder-big", "coder-small"], primero="chat-mid"),
        )
        informe = evaluar_suite(suite, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert (informe.total, informe.aprobados) == (2, 1)
        assert informe.tasa_aprobacion == 0.5
        assert informe.hit_rate == pytest.approx(2 / 3)
        assert informe.tasa_exclusion == 1.0
        assert (informe.aciertos_primero, informe.esperados_primero) == (1, 2)
        assert [r.id for r in informe.resultados] == ["ok", "ko"]
        assert informe.snapshot_coincide is True

    def test_ratios_are_none_when_nothing_expected(self) -> None:
        informe = evaluar_suite(self._suite(_caso()), catalogo=CATALOGO, vram_provider=PROVIDER)
        assert informe.hit_rate is None
        assert informe.tasa_exclusion is None
        assert informe.esperados_primero == 0

    def test_snapshot_mismatch_is_reported_not_raised(self) -> None:
        suite = self._suite(_caso(), snapshot_id="snap-other")
        informe = evaluar_suite(suite, catalogo=CATALOGO, vram_provider=PROVIDER)
        assert informe.snapshot_coincide is False
        assert informe.snapshot_id_catalogo == "snap-test"
        assert informe.aprobados == 1


class TestCargarSuite:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "s.yaml"
        f.write_text(
            "version: '1'\nsnapshot_id: snap-x\ncasos:\n"
            "  - id: a\n    criterios: {tarea: chat, vram_gb: 8}\n",
            encoding="utf-8",
        )
        suite = cargar_suite(f)
        assert suite.casos[0].criterios.tarea == "chat"
        assert suite.casos[0].top_n == 3

    def test_invalid_yaml_content_raises_validation_error(self, tmp_path: Path) -> None:
        f = tmp_path / "s.yaml"
        f.write_text("version: '1'\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            cargar_suite(f)

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            cargar_suite(tmp_path / "nope.yaml")


class TestCommittedSuite:
    """The real suite against the real bootstrap: the regression net itself."""

    def test_committed_suite_passes_against_bootstrap(self, tmp_path: Path) -> None:
        suite = cargar_suite(DEFAULT_SUITE_PATH)
        catalogo = CatalogoSincronizado(
            runtime_dir=tmp_path / "absent", bootstrap_path=DEFAULT_BOOTSTRAP_PATH
        )
        benchmarks = BenchmarksLocales(
            runtime_path=tmp_path / "absent.json", bootstrap_path=DEFAULT_BENCHMARKS_BOOTSTRAP_PATH
        )
        informe = evaluar_suite(
            suite, catalogo=catalogo, vram_provider=PROVIDER, benchmarks=benchmarks
        )

        fallos = {r.id: r.fallos for r in informe.resultados if not r.aprobado}
        assert fallos == {}, fallos
        assert informe.snapshot_coincide, (
            "the committed suite must be written for the committed bootstrap; "
            "re-review the cases when the bootstrap is re-promoted"
        )
        assert informe.benchmarks_coinciden, (
            "the committed suite must be written for the committed benchmark bootstrap; "
            "re-review the cases when benchmarks are re-promoted"
        )
        assert informe.hit_rate == 1.0
        assert informe.tasa_exclusion == 1.0

    def test_committed_suite_covers_every_v1_task(self) -> None:
        suite = cargar_suite(DEFAULT_SUITE_PATH)
        assert {c.criterios.tarea for c in suite.casos} == {"chat", "codigo", "razonamiento"}
