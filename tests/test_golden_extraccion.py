"""Tests for src.evaluation.extraccion and scripts.evaluate_extraction (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import evaluate_extraction as script
from src.chains.extraccion_criterios import ExtraccionNoDisponibleError
from src.evaluation.extraccion import (
    DEFAULT_SUITE_PATH,
    CasoExtraccion,
    SuiteExtraccion,
    cargar_suite,
    coincide,
    evaluar_extraccion,
)
from src.models.schemas import CriteriosBusqueda


class _Scripted:
    def __init__(self, respuestas: dict[str, CriteriosBusqueda], fallo: str | None = None):
        self._r = respuestas
        self._fallo = fallo

    def extraer(self, mensaje: str, base=None) -> CriteriosBusqueda:
        if mensaje == self._fallo:
            raise ExtraccionNoDisponibleError("down")
        return self._r.get(mensaje, CriteriosBusqueda())


def _suite(*casos: dict) -> SuiteExtraccion:
    return SuiteExtraccion(version="t", casos=[CasoExtraccion(**c) for c in casos])


class TestCoincide:
    def test_null_means_empty(self) -> None:
        assert coincide("vram_gb", None, None)
        assert not coincide("vram_gb", None, 12.0)
        assert coincide("idiomas_requeridos", None, [])
        assert coincide("requiere_licencia_comercial", None, False)

    def test_languages_as_normalised_sets(self) -> None:
        assert coincide("idiomas_requeridos", ["Español", "en"], ["en", "es"])
        assert not coincide("idiomas_requeridos", ["es"], ["es", "en"])

    def test_vram_tolerance(self) -> None:
        assert coincide("vram_gb", 12, 12.001)
        assert not coincide("vram_gb", 12, 11.5)
        assert not coincide("vram_gb", 12, None)

    def test_runtime_and_licence(self) -> None:
        assert coincide("runtime_preferido", "LM Studio", "lm studio")
        assert coincide("requiere_licencia_comercial", True, True)
        assert not coincide("requiere_licencia_comercial", True, False)
        assert coincide("tarea", "codigo", "codigo")


class TestEvaluar:
    def test_metrics(self) -> None:
        suite = _suite(
            {"id": "a", "mensaje": "m1", "esperado": {"tarea": "codigo", "vram_gb": 12}},
            {"id": "b", "mensaje": "m2", "esperado": {"tarea": "chat", "vram_gb": None}},
            {"id": "c", "mensaje": "m3", "esperado": {"tarea": "razonamiento"}},
        )
        extractor = _Scripted(
            {
                "m1": CriteriosBusqueda(tarea="codigo", vram_gb=12),
                "m2": CriteriosBusqueda(tarea="chat", vram_gb=8),  # invented VRAM → fail
            },
            fallo="m3",
        )
        informe = evaluar_extraccion(suite, extractor)
        assert (informe.total, informe.aprobados) == (3, 1)
        assert informe.errores_backend == 1
        assert informe.exactitud_por_campo == {"tarea": pytest.approx(2 / 3), "vram_gb": 0.5}
        por_id = {r.id: r for r in informe.resultados}
        assert por_id["b"].campos_fallidos == {"vram_gb": {"esperado": None, "obtenido": 8.0}}
        assert por_id["c"].error is not None and por_id["c"].obtenido is None

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown fields"):
            CasoExtraccion(id="x", mensaje="m", esperado={"gpu": "4090"})

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            _suite({"id": "a", "mensaje": "m", "esperado": {"tarea": "chat"}},
                   {"id": "a", "mensaje": "n", "esperado": {"tarea": "chat"}})


class TestCommittedSuite:
    def test_loads_and_covers_the_essentials(self) -> None:
        suite = cargar_suite(DEFAULT_SUITE_PATH)
        assert len(suite.casos) >= 15
        tareas = {c.esperado.get("tarea") for c in suite.casos}
        assert {"chat", "codigo", "razonamiento"} <= tareas
        # the "never invent VRAM" rule must be pinned at least twice
        assert sum(1 for c in suite.casos if c.esperado.get("vram_gb", 1) is None) >= 2
        assert any("idiomas_requeridos" in c.esperado for c in suite.casos)
        assert any("runtime_preferido" in c.esperado for c in suite.casos)


class TestScript:
    def test_exit_codes_with_injected_extractor(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        suite = tmp_path / "s.yaml"
        suite.write_text(
            "version: 't'\ncasos:\n"
            "  - id: a\n    mensaje: m1\n    esperado: {tarea: codigo}\n"
            "  - id: b\n    mensaje: m2\n    esperado: {tarea: chat}\n",
            encoding="utf-8",
        )
        bueno = _Scripted(
            {"m1": CriteriosBusqueda(tarea="codigo"), "m2": CriteriosBusqueda(tarea="chat")}
        )
        assert script.main(["--suite", str(suite)], extractor=bueno) == script.EXIT_OK
        assert "cases passed: 2/2" in capsys.readouterr().out

        regular = _Scripted({"m1": CriteriosBusqueda(tarea="codigo")})
        codigo = script.main(["--suite", str(suite)], extractor=regular)
        assert codigo == script.EXIT_BELOW_THRESHOLD
        assert script.main(
            ["--suite", str(suite), "--min-aprobacion", "0.5"], extractor=regular
        ) == script.EXIT_OK

    def test_all_backend_errors_fail(self, tmp_path: Path) -> None:
        suite = tmp_path / "s.yaml"
        suite.write_text(
            "version: 't'\ncasos:\n  - id: a\n    mensaje: m1\n    esperado: {tarea: codigo}\n",
            encoding="utf-8",
        )
        caido = _Scripted({}, fallo="m1")
        assert script.main(
            ["--suite", str(suite), "--min-aprobacion", "0"], extractor=caido
        ) == script.EXIT_BELOW_THRESHOLD

    def test_invalid_suite(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 't'\n", encoding="utf-8")
        assert script.main(["--suite", str(bad)]) == script.EXIT_SUITE_INVALID

    def test_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        suite = tmp_path / "s.yaml"
        suite.write_text(
            "version: 't'\ncasos:\n  - id: a\n    mensaje: m1\n    esperado: {tarea: codigo}\n",
            encoding="utf-8",
        )
        bueno = _Scripted({"m1": CriteriosBusqueda(tarea="codigo")})
        assert script.main(["--suite", str(suite), "--json"], extractor=bueno) == script.EXIT_OK
        assert json.loads(capsys.readouterr().out)["aprobados"] == 1
