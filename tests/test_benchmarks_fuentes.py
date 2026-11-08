"""Tests for the benchmark source adapters (Epoch AI ZIP, Artificial Analysis API).

Every HTTP call goes through ``httpx.MockTransport``; the Epoch archive is
built in memory.  No network, no real key, no files outside ``tmp_path``.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

import httpx
import pytest

from src.benchmarks import artificial_analysis as aa
from src.benchmarks.artificial_analysis import FuenteArtificialAnalysis
from src.benchmarks.epoch import (
    ECI_ANCLA_MAX,
    ECI_ANCLA_MIN,
    FuenteEpoch,
)
from src.benchmarks.fuentes import FuenteBenchmarks, FuenteBenchmarksError, normalizar

AHORA = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

_COLS = (
    "Model version,mean_score,Best score (across scorers),Release date,Organization,stderr,id\n"
)


def _zip(
    *,
    gpqa: str = _COLS + "qwen3-8b,0.56,0.5676,2025-04-28,Alibaba,0.03,a\n"
    "qwen3-8b_none,0.47,0.475,2025-04-28,Alibaba,0.03,b\n"
    "Llama-3.2-1B-Instruct,0.23,0.2393,2024-09-24,Meta AI,0.03,c\n",
    math: str = _COLS + "phi-4,0.64,0.6494,2024-12-12,Microsoft,0.01,d\n",
    swe: str = _COLS + "glm-5,0.72,0.7208,2026-02-11,Z.ai,0.02,e\n"
    "glm-5,0.70,0.7000,2026-02-11,Z.ai,0.02,e2\n",
    metadata: str = (
        "benchmark,in_eci,source_file,score_column,scale,random_baseline,score_ceiling\n"
        "GPQA diamond,True,gpqa_diamond.csv,Best score (across scorers),1.0,0.25,0.97\n"
        "MATH level 5,True,math_level_5.csv,Best score (across scorers),1.0,0.0,1.0\n"
    ),
    eci: str = "Model,Display name,eci,eci_ci_low,eci_ci_high,date,Accessibility group\n"
    "Qwen3-8B,Qwen3-8B,132.2,130,134,2025-04-28,Open weights\n"
    "qwen3-8b,dup,999,0,0,2025-04-28,Open weights\n",
    omit: tuple[str, ...] = (),
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archivo:
        miembros = {
            "gpqa_diamond.csv": gpqa,
            "math_level_5.csv": math,
            "swe_bench_verified.csv": swe,
            "benchmark_metadata.csv": metadata,
            "epoch_capabilities_index/eci_scores.csv": eci,
            "README.md": "licence",
        }
        for nombre, contenido in miembros.items():
            if nombre not in omit:
                info = zipfile.ZipInfo(nombre, date_time=(2026, 11, 7, 10, 0, 0))
                archivo.writestr(info, contenido)
    return buffer.getvalue()


class TestNormalizar:
    def test_linear_and_clamped(self) -> None:
        assert normalizar(0.61, 0.25, 0.97) == pytest.approx(0.5)
        assert normalizar(0.1, 0.25, 0.97) == 0.0
        assert normalizar(1.5, 0.0, 1.0) == 1.0

    def test_rejects_bad_bounds(self) -> None:
        with pytest.raises(ValueError):
            normalizar(0.5, 1.0, 1.0)


class TestEpochParse:
    def test_scores_keyed_by_exact_source_name(self) -> None:
        datos = FuenteEpoch.parsear(_zip(), descargado_en=AHORA)

        assert datos.meta.nombre == "epoch_ai"
        assert datos.meta.redistribuible is True
        assert datos.meta.version == "2026-11-07"
        assert datos.meta.registros == len(datos.puntuaciones)
        gpqa = [e for e in datos.puntuaciones["qwen3-8b"] if e.benchmark == "gpqa_diamond"]
        assert len(gpqa) == 1
        assert gpqa[0].valor == pytest.approx(0.5676)
        esperado = (0.5676 - 0.25) / (0.97 - 0.25)
        assert gpqa[0].valor_normalizado == pytest.approx(esperado, abs=1e-4)
        assert gpqa[0].fecha_modelo == "2025-04-28"
        # the _none run is a different source name, never merged
        assert datos.puntuaciones["qwen3-8b_none"][0].valor == pytest.approx(0.475)

    def test_below_random_baseline_normalises_to_zero(self) -> None:
        datos = FuenteEpoch.parsear(_zip(), descargado_en=AHORA)
        assert datos.puntuaciones["Llama-3.2-1B-Instruct"][0].valor_normalizado == 0.0

    def test_missing_metadata_row_falls_back_to_unit_bounds(self) -> None:
        datos = FuenteEpoch.parsear(_zip(), descargado_en=AHORA)
        swe = datos.puntuaciones["glm-5"]
        assert [e.benchmark for e in swe] == ["swe_bench_verified"]
        assert swe[0].valor_normalizado == pytest.approx(0.7208)

    def test_duplicate_rows_keep_the_highest_score(self) -> None:
        datos = FuenteEpoch.parsear(_zip(), descargado_en=AHORA)
        assert datos.puntuaciones["glm-5"][0].valor == pytest.approx(0.7208)

    def test_index_rows_are_a_separate_namespace(self) -> None:
        datos = FuenteEpoch.parsear(_zip(), descargado_en=AHORA)
        indice = datos.puntuaciones["Qwen3-8B"]
        assert [e.benchmark for e in indice] == ["eci"]
        assert indice[0].valor == pytest.approx(132.2)
        esperado = (132.2 - ECI_ANCLA_MIN) / (ECI_ANCLA_MAX - ECI_ANCLA_MIN)
        assert indice[0].valor_normalizado == pytest.approx(esperado, abs=1e-4)
        # same spelling in both namespaces: both rows are kept, tagged by benchmark
        assert {e.benchmark for e in datos.puntuaciones["qwen3-8b"]} == {"gpqa_diamond", "eci"}

    def test_unparsable_scores_are_skipped(self) -> None:
        gpqa = _COLS + "broken,,,2025-01-01,X,,z\nok,0.5,0.5,2025-01-01,X,0,y\n"
        datos = FuenteEpoch.parsear(_zip(gpqa=gpqa), descargado_en=AHORA)
        assert "broken" not in datos.puntuaciones
        assert "ok" in datos.puntuaciones

    def test_missing_required_csv_is_an_error(self) -> None:
        with pytest.raises(FuenteBenchmarksError, match="files missing"):
            FuenteEpoch.parsear(_zip(omit=("gpqa_diamond.csv",)), descargado_en=AHORA)

    def test_not_a_zip_is_an_error(self) -> None:
        with pytest.raises(FuenteBenchmarksError, match="not a ZIP"):
            FuenteEpoch.parsear(b"nope", descargado_en=AHORA)

    def test_archive_without_any_score_is_an_error(self) -> None:
        vacio = _zip(gpqa=_COLS, math=_COLS, swe=_COLS, eci="Model,eci\n")
        with pytest.raises(FuenteBenchmarksError, match="no score"):
            FuenteEpoch.parsear(vacio, descargado_en=AHORA)


class TestEpochDownload:
    def test_downloads_through_injected_transport(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=_zip()))
        datos = FuenteEpoch().descargar(transport=transport)
        assert "qwen3-8b" in datos.puntuaciones
        assert datos.meta.descargado_en.tzinfo is not None

    def test_http_error_is_wrapped(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(503))
        with pytest.raises(FuenteBenchmarksError, match="download failed"):
            FuenteEpoch().descargar(transport=transport)

    def test_timeout_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("simulated", request=request)

        with pytest.raises(FuenteBenchmarksError, match="download failed"):
            FuenteEpoch().descargar(transport=httpx.MockTransport(handler))

    def test_satisfies_protocol(self) -> None:
        assert isinstance(FuenteEpoch(), FuenteBenchmarks)
        assert FuenteEpoch.obligatoria is True


def _pagina(modelos: list[dict], *, has_more: bool, version: int | None = 5) -> dict:
    return {
        "tier": "free",
        "intelligence_index_version": version,
        "pagination": {"page": 1, "page_size": 200, "total_pages": 2, "has_more": has_more},
        "data": modelos,
    }


def _modelo(slug: str, indice: float | None, name: str = "") -> dict:
    return {
        "id": slug,
        "name": name or slug,
        "slug": slug,
        "release_date": "2026-02-01",
        "model_creator": {"id": "x", "name": "y"},
        "evaluations": {aa.CAMPO_INDICE: indice, "artificial_analysis_coding_index": None},
        "pricing": {"price_1m_input_tokens": 1.0},
        "performance": {"median_output_tokens_per_second": 50.0},
    }


class _ApiFalsa:
    """Scripted paginated endpoint recording the requests it sees."""

    def __init__(self, paginas: dict[int, dict], status: int = 200) -> None:
        self.paginas = paginas
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        pagina = int(request.url.params.get("page", "1"))
        return httpx.Response(200, json=self.paginas[pagina])


class TestArtificialAnalysis:
    def test_pages_until_has_more_is_false_and_keeps_only_indexed_models(self) -> None:
        api = _ApiFalsa({
            1: _pagina([_modelo("glm-5-3", 31.7, "GLM-5.3 (Max)"), _modelo("sin", None)],
                       has_more=True),
            2: _pagina([_modelo("minimax-m3", 18.9)], has_more=False),
        })
        datos = FuenteArtificialAnalysis("k").descargar(transport=httpx.MockTransport(api))

        assert [r.url.params["page"] for r in api.requests] == ["1", "2"]
        assert all(r.headers["x-api-key"] == "k" for r in api.requests)
        assert set(datos.puntuaciones) == {"glm-5-3", "minimax-m3"}
        e = datos.puntuaciones["glm-5-3"][0]
        assert (e.fuente, e.benchmark) == ("artificial_analysis", "intelligence_index")
        assert e.valor == 31.7
        assert e.valor_normalizado == pytest.approx(0.317)
        assert "GLM-5.3 (Max)" in e.nota
        assert datos.meta.version == "v5"
        assert datos.meta.redistribuible is False
        assert "v5" in datos.meta.atribucion

    def test_rejected_key_is_a_clear_error(self) -> None:
        api = _ApiFalsa({}, status=401)
        with pytest.raises(FuenteBenchmarksError, match="API key rejected"):
            FuenteArtificialAnalysis("bad").descargar(transport=httpx.MockTransport(api))

    def test_unexpected_body_is_an_error(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": "???"}))
        with pytest.raises(FuenteBenchmarksError, match="unexpected response"):
            FuenteArtificialAnalysis("k").descargar(transport=transport)

    def test_no_indexed_model_is_an_error(self) -> None:
        api = _ApiFalsa({1: _pagina([_modelo("a", None)], has_more=False)})
        with pytest.raises(FuenteBenchmarksError, match="no model carries an index"):
            FuenteArtificialAnalysis("k").descargar(transport=httpx.MockTransport(api))

    def test_pagination_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(aa, "MAX_PAGINAS", 2)
        api = _ApiFalsa({n: _pagina([_modelo(f"m{n}", 1.0)], has_more=True) for n in (1, 2, 3)})
        datos = FuenteArtificialAnalysis("k").descargar(transport=httpx.MockTransport(api))
        assert len(api.requests) == 2
        assert set(datos.puntuaciones) == {"m1", "m2"}

    def test_from_environment(self) -> None:
        assert FuenteArtificialAnalysis.desde_entorno({}) is None
        assert FuenteArtificialAnalysis.desde_entorno({aa.VARIABLE_CLAVE: "  "}) is None
        fuente = FuenteArtificialAnalysis.desde_entorno({aa.VARIABLE_CLAVE: "abc"})
        assert isinstance(fuente, FuenteArtificialAnalysis)
        assert isinstance(fuente, FuenteBenchmarks)
        assert fuente.obligatoria is False

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            FuenteArtificialAnalysis("   ")

    def test_version_unknown_when_absent(self) -> None:
        api = _ApiFalsa({1: _pagina([_modelo("a", 2.0)], has_more=False, version=None)})
        datos = FuenteArtificialAnalysis("k").descargar(transport=httpx.MockTransport(api))
        assert datos.meta.version == "unknown"
        json.dumps(datos.meta.model_dump(mode="json"))  # serialisable
