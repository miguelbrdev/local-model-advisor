"""Tests for the offline quantisation-table provider and the snapshot-first chain."""

from __future__ import annotations

import pytest

from src.compatibility.cuantizaciones import (
    CUANTIZACION_REFERENCIA,
    CuantizacionesProvider,
    recomendar_cuantizacion,
)
from src.compatibility.heuristica import HeuristicaParametrosProvider, estimar_vram_gb
from src.compatibility.provider import (
    ProveedorSnapshotPrimero,
    VRAMCompatibilityProvider,
    elegir_provider,
    proveedor_offline,
)
from src.compatibility.vram import FRACCION_VRAM_UTIL
from src.models.schemas import CompatibilityResult, Cuantizacion, FamiliaModelo

TABLA = [
    Cuantizacion(nombre="Q2_K", bits=2, vram_gb=3.1, disco_gb=2.4, calidad="low"),
    Cuantizacion(nombre="Q4_K_M", bits=4, vram_gb=4.6, disco_gb=3.9, calidad="good"),
    Cuantizacion(nombre="Q6_K", bits=6, vram_gb=6.6, disco_gb=5.9, calidad="excellent"),
    Cuantizacion(nombre="Q8_0", bits=8, vram_gb=8.7, disco_gb=7.8, calidad="excellent"),
    Cuantizacion(nombre="F16", bits=16, vram_gb=16.9, disco_gb=15.6, calidad="lossless"),
]


def _fam(cuantizaciones: list[Cuantizacion] | None = TABLA, parametros_b: float | None = 8.0):
    return FamiliaModelo(
        familia_id="qwen3-8b", variantes=[], pipeline_tag="chat", tareas_soportadas=["chat"],
        canirun_id="qwen3-8b", parametros_b=parametros_b, parametros_texto="8B",
        cuantizaciones=list(cuantizaciones or []),
    )


class TestCuantizacionesProvider:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(CuantizacionesProvider(), VRAMCompatibilityProvider)

    def test_comfortable_tight_insufficient(self) -> None:
        p = CuantizacionesProvider()
        comodo = p.check(_fam(), 12.0)
        assert (comodo.compatible, comodo.grado) == (True, "comfortable")
        assert comodo.fuente == "canirun_snapshot"
        assert comodo.vram_required_gb == 4.6
        assert comodo.variante == "8B"
        justo = p.check(_fam(), 5.0)
        assert (justo.compatible, justo.grado) == (True, "tight")
        no = p.check(_fam(), 4.0)
        assert (no.compatible, no.grado) == (False, "insufficient")

    def test_note_names_reference_and_best_comfortable_quantisation(self) -> None:
        r = CuantizacionesProvider().check(_fam(), 12.0)
        assert CUANTIZACION_REFERENCIA in r.nota and "4.6 GB" in r.nota
        assert "Best comfortable fit: Q8_0 (8.7 GB, excellent)" in r.nota

    def test_no_recommendation_line_when_reference_is_the_best(self) -> None:
        # 80 % of 6 GB = 4.8 GB: only Q4_K_M and below fit comfortably
        r = CuantizacionesProvider().check(_fam(), 6.0)
        assert "Best comfortable fit" not in r.nota

    def test_missing_table_is_no_disponible(self) -> None:
        r = CuantizacionesProvider().check(_fam(cuantizaciones=[]), 12.0)
        assert r.compatible is None and r.fuente == "no_disponible"
        assert CUANTIZACION_REFERENCIA in r.nota

    def test_missing_reference_row_is_no_disponible(self) -> None:
        tabla = [c for c in TABLA if c.nombre != CUANTIZACION_REFERENCIA]
        assert CuantizacionesProvider().check(_fam(tabla), 12.0).fuente == "no_disponible"

    def test_zero_vram_is_unknown_but_keeps_requirement(self) -> None:
        r = CuantizacionesProvider().check(_fam(), 0.0)
        assert r.compatible is None and r.fuente == "canirun_snapshot"
        assert r.vram_required_gb == 4.6

    def test_deterministic(self) -> None:
        p = CuantizacionesProvider()
        assert p.check(_fam(), 12.0) == p.check(_fam(), 12.0)


class TestRecomendar:
    def test_highest_bits_within_budget(self) -> None:
        assert recomendar_cuantizacion(TABLA, 12.0).nombre == "Q8_0"  # 80 % of 12 = 9.6
        assert recomendar_cuantizacion(TABLA, 24.0).nombre == "F16"
        assert recomendar_cuantizacion(TABLA, 4.0).nombre == "Q2_K"

    def test_none_when_nothing_fits(self) -> None:
        assert recomendar_cuantizacion(TABLA, 3.0) is None
        assert recomendar_cuantizacion([], 100.0) is None

    def test_threshold_is_the_shared_constant(self) -> None:
        vram = 4.6 / FRACCION_VRAM_UTIL
        assert recomendar_cuantizacion(TABLA, vram).nombre == "Q4_K_M"


class _Respaldo:
    def __init__(self) -> None:
        self.llamadas: list[str] = []

    def check(self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None):
        self.llamadas.append(familia.familia_id)
        return CompatibilityResult(
            modelo_id=familia.familia_id, variante="8B", compatible=True,
            fuente="heuristica_propia", nota="fallback", vram_required_gb=5.8,
        )


class TestSnapshotPrimero:
    def test_snapshot_answer_skips_the_fallback(self) -> None:
        respaldo = _Respaldo()
        chain = ProveedorSnapshotPrimero(CuantizacionesProvider(), respaldo)
        r = chain.check(_fam(), 12.0)
        assert r.fuente == "canirun_snapshot" and respaldo.llamadas == []

    def test_missing_table_delegates(self) -> None:
        respaldo = _Respaldo()
        chain = ProveedorSnapshotPrimero(CuantizacionesProvider(), respaldo)
        r = chain.check(_fam(cuantizaciones=[]), 12.0)
        assert r.fuente == "heuristica_propia" and respaldo.llamadas == ["qwen3-8b"]

    def test_offline_chain_uses_heuristic_without_table(self) -> None:
        r = proveedor_offline().check(_fam(cuantizaciones=[]), 12.0)
        assert r.fuente == "heuristica_propia"
        assert r.vram_required_gb == estimar_vram_gb(8.0)
        assert isinstance(proveedor_offline(), VRAMCompatibilityProvider)

    def test_offline_chain_prefers_table(self) -> None:
        r = proveedor_offline().check(_fam(), 12.0)
        assert r.fuente == "canirun_snapshot" and r.vram_required_gb == 4.6

    def test_production_routing_is_snapshot_first(self) -> None:
        p = elegir_provider("chat")
        assert isinstance(p, ProveedorSnapshotPrimero)
        assert p.check(_fam(), 12.0).fuente == "canirun_snapshot"  # no network needed

    def test_unknown_everywhere_stays_unknown(self) -> None:
        r = proveedor_offline().check(_fam(cuantizaciones=[], parametros_b=None), 12.0)
        assert r.compatible is None

    @pytest.mark.parametrize("vram", [4.0, 5.0, 12.0])
    def test_table_and_heuristic_agree_on_direction_for_small_budgets(self, vram: float) -> None:
        tabla = proveedor_offline().check(_fam(), vram)
        heuristica = HeuristicaParametrosProvider().check(_fam(cuantizaciones=[]), vram)
        # the source table is less conservative than the parameter heuristic
        assert tabla.vram_required_gb < heuristica.vram_required_gb
