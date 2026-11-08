"""Tests for the benchmark mapping, snapshot, synchronisation and provider.

Pure in-memory sources (no HTTP) and files only under ``tmp_path``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.benchmarks.fuentes import DatosFuente, FuenteBenchmarksError, FuenteMeta
from src.benchmarks.mapping import (
    EntradaMapeo,
    MapeoBenchmarks,
    MapeoInvalidoError,
    cargar_mapeo,
)
from src.benchmarks.provider import BenchmarksEnMemoria, BenchmarksLocales, ProveedorBenchmarks
from src.benchmarks.snapshot import (
    BenchmarksSnapshot,
    compute_content_hash,
    construir_snapshot,
    read_snapshot,
    write_snapshot,
)
from src.benchmarks.sync import aplicar_mapeo, sync_benchmarks
from src.models.schemas import EvidenciaBenchmark

AHORA = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


def _meta(
    nombre: str = "epoch_ai", *, redistribuible: bool = True, version: str = "1"
) -> FuenteMeta:
    return FuenteMeta(
        nombre=nombre, version=version, url="https://example.test/data",
        licencia="CC BY 4.0" if redistribuible else "terms",
        atribucion=f"Source: {nombre}", redistribuible=redistribuible,
        registros=1, descargado_en=AHORA,
    )


def _ev(fuente: str, benchmark: str, modelo: str, valor: float = 0.5) -> EvidenciaBenchmark:
    return EvidenciaBenchmark(
        fuente=fuente, benchmark=benchmark, metrica="m", valor=valor,
        valor_normalizado=min(valor, 1.0), modelo_fuente=modelo,
    )


def _epoch() -> DatosFuente:
    return DatosFuente(
        meta=_meta("epoch_ai"),
        puntuaciones={
            "qwen3-8b": [_ev("epoch_ai", "gpqa_diamond", "qwen3-8b", 0.56)],
            "Qwen3-8B": [_ev("epoch_ai", "eci", "Qwen3-8B", 0.53)],
            # same spelling in both namespaces
            "Qwen3-32B": [
                _ev("epoch_ai", "gpqa_diamond", "Qwen3-32B", 0.65),
                _ev("epoch_ai", "eci", "Qwen3-32B", 0.61),
            ],
        },
    )


def _aa() -> DatosFuente:
    return DatosFuente(
        meta=_meta("artificial_analysis", redistribuible=False, version="v5"),
        puntuaciones={
            "glm-5-3": [_ev("artificial_analysis", "intelligence_index", "glm-5-3", 0.3)]
        },
    )


def _mapeo(**familias: EntradaMapeo) -> MapeoBenchmarks:
    return MapeoBenchmarks(version="t", snapshot_id="snap-test", familias=familias)


class _Fuente:
    """In-memory ``FuenteBenchmarks``: returns scripted data or raises."""

    def __init__(self, nombre: str, datos: DatosFuente | None, *, obligatoria: bool) -> None:
        self.nombre = nombre
        self._datos = datos
        self.obligatoria = obligatoria
        self.llamadas = 0

    def descargar(self, *, timeout: float = 60.0, transport=None) -> DatosFuente:
        self.llamadas += 1
        if self._datos is None:
            raise FuenteBenchmarksError(f"{self.nombre}: scripted failure")
        return self._datos


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class TestMapeo:
    def test_consultas_tag_namespaces(self) -> None:
        entrada = EntradaMapeo(
            epoch_model_version="qwen3-8b", epoch_eci_model="Qwen3-8B",
            artificial_analysis_slug="qwen3-8b-slug",
        )
        assert entrada.consultas("epoch_ai") == [("qwen3-8b", False), ("Qwen3-8B", True)]
        assert entrada.consultas("artificial_analysis") == [("qwen3-8b-slug", None)]
        assert entrada.consultas("other") == []
        assert EntradaMapeo().consultas("epoch_ai") == []

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntradaMapeo(epoch_name="x")  # type: ignore[call-arg]

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        ruta = tmp_path / "m.yaml"
        ruta.write_text(
            "version: '1'\nsnapshot_id: snap-x\nfamilias:\n"
            "  a:\n    epoch_model_version: A\n  b:\n    nota: none\n",
            encoding="utf-8",
        )
        mapeo = cargar_mapeo(ruta)
        assert mapeo.familias["a"].epoch_model_version == "A"
        assert mapeo.familias["b"].consultas("epoch_ai") == []

    def test_load_errors_are_wrapped(self, tmp_path: Path) -> None:
        with pytest.raises(MapeoInvalidoError):
            cargar_mapeo(tmp_path / "missing.yaml")
        malo = tmp_path / "bad.yaml"
        malo.write_text("version: '1'\nfamilias: [1, 2]\n", encoding="utf-8")
        with pytest.raises(MapeoInvalidoError):
            cargar_mapeo(malo)

    def test_committed_mapping_is_valid_and_matches_bootstrap(self) -> None:
        import json

        from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH

        mapeo = cargar_mapeo()
        ids = {
            f["canirun_id"]
            for f in json.loads(DEFAULT_BOOTSTRAP_PATH.read_text(encoding="utf-8"))["familias"]
        }
        assert set(mapeo.familias) == ids
        assert mapeo.snapshot_id == "snap-3eb98b50f1e483e0"


class TestAplicarMapeo:
    def test_fields_decide_the_namespace_not_the_spelling(self) -> None:
        mapeo = _mapeo(
            **{
                "qwen3-32b": EntradaMapeo(
                    epoch_model_version="Qwen3-32B", epoch_eci_model="Qwen3-32B"
                )
            }
        )
        puntuaciones, faltan = aplicar_mapeo(mapeo, [_epoch()])
        assert sorted(e.benchmark for e in puntuaciones["qwen3-32b"]) == ["eci", "gpqa_diamond"]
        assert faltan == []

    def test_benchmark_name_never_pulls_the_index(self) -> None:
        mapeo = _mapeo(**{"qwen3-32b": EntradaMapeo(epoch_model_version="Qwen3-32B")})
        puntuaciones, _ = aplicar_mapeo(mapeo, [_epoch()])
        assert [e.benchmark for e in puntuaciones["qwen3-32b"]] == ["gpqa_diamond"]

    def test_missing_names_are_reported_not_guessed(self) -> None:
        mapeo = _mapeo(
            **{"qwen3-8b": EntradaMapeo(epoch_model_version="qwen3-8b", epoch_eci_model="Nope")},
            otra=EntradaMapeo(nota="unmapped"),
        )
        puntuaciones, faltan = aplicar_mapeo(mapeo, [_epoch(), _aa()])
        assert [e.benchmark for e in puntuaciones["qwen3-8b"]] == ["gpqa_diamond"]
        assert puntuaciones["otra"] == []
        assert faltan == ["qwen3-8b: epoch_ai 'Nope' not found"]

    def test_scores_from_every_source_are_combined(self) -> None:
        mapeo = _mapeo(
            **{
                "glm-5.3": EntradaMapeo(
                    artificial_analysis_slug="glm-5-3", epoch_eci_model="Qwen3-8B"
                )
            }
        )
        puntuaciones, _ = aplicar_mapeo(mapeo, [_epoch(), _aa()])
        assert {e.fuente for e in puntuaciones["glm-5.3"]} == {"epoch_ai", "artificial_analysis"}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _snapshot(con_aa: bool = True) -> BenchmarksSnapshot:
    fuentes = [_meta("epoch_ai")]
    if con_aa:
        fuentes.append(_meta("artificial_analysis", redistribuible=False))
    puntuaciones = {
        "qwen3-8b": [_ev("epoch_ai", "gpqa_diamond", "qwen3-8b", 0.56)],
        "glm-5.3": (
            [_ev("artificial_analysis", "intelligence_index", "glm-5-3", 0.3)] if con_aa else []
        ),
        "sin-datos": [],
    }
    return construir_snapshot(
        puntuaciones, fuentes, fecha=AHORA, mapeo_version="t", mapeo_snapshot_id="snap-test",
        familias_mapeadas=3, sin_coincidencia=["x: epoch_ai 'y' not found"],
    )


class TestSnapshot:
    def test_hash_is_deterministic_and_order_independent(self) -> None:
        a = {"m": [_ev("epoch_ai", "eci", "M", 0.5), _ev("epoch_ai", "gpqa_diamond", "m", 0.4)]}
        b = {"m": list(reversed(a["m"]))}
        assert compute_content_hash(a, [_meta()]) == compute_content_hash(b, [_meta()])
        assert compute_content_hash(a, [_meta()]) != compute_content_hash(a, [_meta(version="2")])

    def test_construir_sets_counts_and_id(self) -> None:
        s = _snapshot()
        assert s.meta.familias_con_datos == 2
        assert s.meta.snapshot_id == f"bench-{s.meta.hash_contenido[:16]}"
        assert s.atribuciones == ["Source: epoch_ai", "Source: artificial_analysis"]

    def test_invariants_enforced(self) -> None:
        s = _snapshot()
        datos = s.model_dump(mode="json")
        datos["meta"]["familias_con_datos"] = 5
        with pytest.raises(ValidationError, match="familias_con_datos"):
            BenchmarksSnapshot.model_validate(datos)
        datos = s.model_dump(mode="json")
        datos["puntuaciones"]["qwen3-8b"][0]["valor"] = 0.99
        with pytest.raises(ValidationError, match="hash_contenido"):
            BenchmarksSnapshot.model_validate(datos)
        datos = s.model_dump(mode="json")
        datos["meta"]["fuentes"] = datos["meta"]["fuentes"][:1]
        with pytest.raises(ValidationError, match="undeclared sources"):
            BenchmarksSnapshot.model_validate(datos)

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        ruta = tmp_path / "b" / "current.json"
        write_snapshot(_snapshot(), ruta)
        assert read_snapshot(ruta) == _snapshot()
        assert not ruta.with_name("current.json.tmp").exists()

    def test_read_rejects_corrupt_or_missing(self, tmp_path: Path) -> None:
        assert read_snapshot(tmp_path / "none.json") is None
        ruta = tmp_path / "c.json"
        ruta.write_text("{not json", encoding="utf-8")
        assert read_snapshot(ruta) is None
        ruta.write_text('{"meta": {}, "puntuaciones": {}}', encoding="utf-8")
        assert read_snapshot(ruta) is None

    def test_solo_redistribuible_drops_non_redistributable_sources(self) -> None:
        reducido = _snapshot().solo_redistribuible()
        assert [f.nombre for f in reducido.meta.fuentes] == ["epoch_ai"]
        assert set(reducido.puntuaciones) == {"qwen3-8b"}
        assert reducido.meta.familias_con_datos == 1
        assert reducido.meta.fecha_sincronizacion == AHORA
        assert reducido.meta.sin_coincidencia == ["x: epoch_ai 'y' not found"]
        assert reducido.meta.snapshot_id != _snapshot().meta.snapshot_id
        assert reducido == _snapshot(con_aa=False).solo_redistribuible()


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------


class TestSync:
    def _mapeo(self) -> MapeoBenchmarks:
        return _mapeo(
            **{
                "qwen3-8b": EntradaMapeo(
                    epoch_model_version="qwen3-8b", epoch_eci_model="Qwen3-8B"
                )
            },
            **{"glm-5.3": EntradaMapeo(artificial_analysis_slug="glm-5-3")},
        )

    def test_writes_validated_snapshot(self, tmp_path: Path) -> None:
        destino = tmp_path / "current.json"
        r = sync_benchmarks(
            destino,
            [_Fuente("epoch_ai", _epoch(), obligatoria=True),
             _Fuente("artificial_analysis", _aa(), obligatoria=False)],
            self._mapeo(), ahora=AHORA,
        )
        assert r.ok and not r.unchanged
        assert r.fuentes_ok == ["epoch_ai", "artificial_analysis"]
        assert r.familias_con_datos == 2
        snapshot = read_snapshot(destino)
        assert snapshot is not None and snapshot.meta.snapshot_id == r.snapshot_id
        assert snapshot.meta.mapeo_version == "t"
        assert snapshot.meta.familias_mapeadas == 2

    def test_mandatory_failure_keeps_previous_snapshot(self, tmp_path: Path) -> None:
        destino = tmp_path / "current.json"
        write_snapshot(_snapshot(), destino)
        antes = destino.read_bytes()
        r = sync_benchmarks(
            destino, [_Fuente("epoch_ai", None, obligatoria=True)], self._mapeo(), ahora=AHORA
        )
        assert not r.ok and r.error.startswith("fuente_obligatoria_fallida")
        assert destino.read_bytes() == antes

    def test_optional_failure_continues(self, tmp_path: Path) -> None:
        destino = tmp_path / "current.json"
        r = sync_benchmarks(
            destino,
            [_Fuente("epoch_ai", _epoch(), obligatoria=True),
             _Fuente("artificial_analysis", None, obligatoria=False)],
            self._mapeo(), ahora=AHORA,
        )
        assert r.ok
        assert r.fuentes_ok == ["epoch_ai"]
        assert "artificial_analysis" in r.fuentes_fallidas
        assert r.sin_coincidencia == []  # the AA name was never looked up
        assert [f.nombre for f in read_snapshot(destino).meta.fuentes] == ["epoch_ai"]

    def test_identical_data_is_not_rewritten(self, tmp_path: Path) -> None:
        destino = tmp_path / "current.json"
        fuentes = [_Fuente("epoch_ai", _epoch(), obligatoria=True)]
        primero = sync_benchmarks(destino, fuentes, self._mapeo(), ahora=AHORA)
        mtime = destino.stat().st_mtime_ns
        segundo = sync_benchmarks(
            destino, fuentes, self._mapeo(), ahora=datetime(2027, 1, 1, tzinfo=UTC)
        )
        assert segundo.ok and segundo.unchanged
        assert segundo.snapshot_id == primero.snapshot_id
        assert destino.stat().st_mtime_ns == mtime

    def test_no_data_aborts_without_writing(self, tmp_path: Path) -> None:
        destino = tmp_path / "current.json"
        r = sync_benchmarks(
            destino, [_Fuente("epoch_ai", _epoch(), obligatoria=True)],
            _mapeo(x=EntradaMapeo(epoch_model_version="missing")), ahora=AHORA,
        )
        assert not r.ok and r.error.startswith("sin_datos")
        assert r.sin_coincidencia == ["x: epoch_ai 'missing' not found"]
        assert not destino.exists()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TestProvider:
    def test_runtime_file_wins_over_bootstrap(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime.json"
        bootstrap = tmp_path / "bootstrap.json"
        write_snapshot(_snapshot(), runtime)
        write_snapshot(_snapshot(con_aa=False), bootstrap)
        p = BenchmarksLocales(runtime_path=runtime, bootstrap_path=bootstrap)
        assert isinstance(p, ProveedorBenchmarks)
        assert p.disponible
        assert p.snapshot_id == _snapshot().meta.snapshot_id
        assert p.fuentes == ["epoch_ai", "artificial_analysis"]
        assert p.fecha_sincronizacion == AHORA
        assert [e.benchmark for e in p.puntuaciones_de("glm-5.3")] == ["intelligence_index"]
        assert p.puntuaciones_de("unknown") == []

    def test_falls_back_to_bootstrap(self, tmp_path: Path) -> None:
        bootstrap = tmp_path / "bootstrap.json"
        write_snapshot(_snapshot(con_aa=False), bootstrap)
        p = BenchmarksLocales(runtime_path=tmp_path / "absent.json", bootstrap_path=bootstrap)
        assert p.snapshot_id == _snapshot(con_aa=False).meta.snapshot_id
        assert p.atribuciones == ["Source: epoch_ai"]

    def test_nothing_available_is_neutral_not_fatal(self, tmp_path: Path) -> None:
        corrupto = tmp_path / "corrupt.json"
        corrupto.write_text("{}", encoding="utf-8")
        p = BenchmarksLocales(runtime_path=corrupto, bootstrap_path=tmp_path / "absent.json")
        assert not p.disponible
        assert p.snapshot_id is None
        assert p.fecha_sincronizacion is None
        assert p.atribuciones == []
        assert p.fuentes == []
        assert p.puntuaciones_de("qwen3-8b") == []

    def test_memoized_after_first_access(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtime.json"
        write_snapshot(_snapshot(), runtime)
        p = BenchmarksLocales(runtime_path=runtime, bootstrap_path=tmp_path / "absent.json")
        antes = p.snapshot_id
        runtime.unlink()
        assert p.snapshot_id == antes

    def test_in_memory_double(self) -> None:
        p = BenchmarksEnMemoria({"a": [_ev("epoch_ai", "eci", "A")]}, atribuciones=["x"])
        assert isinstance(p, ProveedorBenchmarks)
        assert p.puntuaciones_de("a")[0].modelo_fuente == "A"
        assert p.puntuaciones_de("b") == []
        assert p.atribuciones == ["x"]

    def test_committed_bootstrap_is_valid_and_epoch_only(self) -> None:
        from src.benchmarks.snapshot import DEFAULT_BENCHMARKS_BOOTSTRAP_PATH

        snapshot = read_snapshot(DEFAULT_BENCHMARKS_BOOTSTRAP_PATH)
        assert snapshot is not None
        assert [f.nombre for f in snapshot.meta.fuentes] == ["epoch_ai"]
        assert all(f.redistribuible for f in snapshot.meta.fuentes)
        assert snapshot.meta.familias_con_datos > 0
