"""Tests for the language/runtime enrichment step and the runtime filter.

One ``httpx.MockTransport`` routes by host and path to scripted canIRun
detail records, Hugging Face cards and model trees.  No network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.models.schemas import CriteriosBusqueda, FamiliaModelo
from src.sync.catalog_policy import CatalogPolicy
from src.sync.catalog_sync import sync_catalog
from src.sync.enrichment import (
    FUENTE_CANIRUN,
    FUENTE_HF_CARD,
    FUENTE_HF_TREE,
    enriquecer_familias,
    repo_de_url,
)
from src.sync.snapshot import read_snapshot
from src.tools.runtimes import RUNTIMES_CANONICOS, normalizar_runtime
from src.tools.structured_filter import filtrar_candidatos
from tests.conftest import MockVRAMProvider

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fam(fid: str, url: str, canirun_id: str | None = "=") -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid, variantes=[], pipeline_tag="chat", tareas_soportadas=["chat"],
        license_url=url, canirun_id=fid if canirun_id == "=" else canirun_id,
    )


class _Red:
    """Scripted responses keyed by path; unknown paths answer 404."""

    def __init__(self) -> None:
        self.detalles: dict[str, Any] = {}
        self.tarjetas: dict[str, Any] = {}
        self.arboles: dict[str, list[Any]] = {}
        self.fallos: set[str] = set()
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if any(f in url for f in self.fallos):
            raise httpx.ConnectTimeout("simulated", request=request)
        host, path = request.url.host, request.url.path
        if host == "canirun.ai" and path.startswith("/api/models/"):
            d = self.detalles.get(path.removeprefix("/api/models/"))
            return httpx.Response(200, json=d) if d else httpx.Response(404)
        if host == "huggingface.co" and path == "/api/models":
            filtro = request.url.params.get("filter", "")
            repo = filtro.removeprefix("base_model:quantized:")
            return httpx.Response(200, json=self.arboles.get(repo, []))
        if host == "huggingface.co" and path.startswith("/api/models/"):
            t = self.tarjetas.get(path.removeprefix("/api/models/"))
            return httpx.Response(200, json=t) if t else httpx.Response(404)
        return httpx.Response(404)


def _detalle(mid: str, **extra: Any) -> dict[str, Any]:
    base = {
        "id": mid, "name": mid, "provider": "P", "family": "F", "params": "8B",
        "paramsBillions": 8, "architecture": "dense", "releaseDate": "2025-04",
        "contextLength": 1, "useCase": ["chat"], "url": "", "license": "MIT",
        "ollamaId": None, "lmStudioId": None, "tools": True, "thinking": False,
        "quants": [{"name": "Q4_K_M", "bits": 4, "vramGB": 4.6, "diskGB": 3.9, "quality": "good"}],
        "futureField": {"ignored": True},
    }
    base.update(extra)
    return base


def _tarjeta(repo: str, language: Any = None, tags: list[str] | None = None, **extra: Any) -> dict:
    card: dict[str, Any] = {"license": "mit"}
    if language is not None:
        card["language"] = language
    card.update(extra)
    return {"id": repo, "cardData": card, "tags": tags or ["safetensors", "transformers"],
            "library_name": "transformers", "gated": False}


def _hoja(mid: str, *tags: str, library: str | None = None) -> dict:
    return {"id": mid, "tags": list(tags), "library_name": library}


@pytest.fixture
def red() -> _Red:
    r = _Red()
    r.detalles["qwen3-8b"] = _detalle("qwen3-8b", ollamaId="qwen3:8b", lmStudioId="qwen3-2504")
    r.detalles["llama3.1-8b"] = _detalle("llama3.1-8b", ollamaId="llama3.1:8b")
    r.tarjetas["Qwen/Qwen3-8B"] = _tarjeta("Qwen/Qwen3-8B")
    r.tarjetas["meta-llama/Llama-3.1-8B-Instruct"] = _tarjeta(
        "meta-llama/Llama-3.1-8B-Instruct", language=["en", "de", "fr", "Code", "multilingual"]
    )
    r.arboles["Qwen/Qwen3-8B"] = [
        _hoja("x/Qwen3-8B-GGUF", "gguf", "llama"),
        _hoja("y/Qwen3-8B-4bit", "mlx", library="mlx"),
        _hoja("z/Qwen3-8B-AWQ", "awq", "compressed-tensors"),
        _hoja("w/Qwen3-8B-Q4-gguf"),  # format only in the name
    ]
    r.arboles["meta-llama/Llama-3.1-8B-Instruct"] = [_hoja("a/L-exl3", "exl3")]
    return r


@pytest.fixture
def policy() -> CatalogPolicy:
    return CatalogPolicy()


def _enriquecer(familias: list[FamiliaModelo], red: _Red, policy: CatalogPolicy):
    with httpx.Client(transport=httpx.MockTransport(red)) as client:
        return enriquecer_familias(familias, policy, client=client, max_workers=2)


class TestPolicy:
    def test_enrichment_tables_loaded(self, policy: CatalogPolicy) -> None:
        assert policy.enrichment_enabled is True
        assert policy.canirun_detail_endpoint == "https://canirun.ai/api/models/{id}"
        assert policy.huggingface_api == "https://huggingface.co/api/models"
        assert policy.runtime_by_canirun_field == {"ollamaId": "ollama", "lmStudioId": "lmstudio"}
        assert policy.runtime_by_format["gguf"] == ["llama.cpp", "lmstudio"]
        assert policy.runtime_by_format["awq"] == ["vllm"]
        todos = {r for rs in policy.runtime_by_format.values() for r in rs}
        todos |= set(policy.runtime_by_canirun_field.values())
        assert todos <= RUNTIMES_CANONICOS

    def test_policy_without_section_disables_enrichment(self, tmp_path: Path) -> None:
        ruta = tmp_path / "p.yaml"
        ruta.write_text("version: '9'\nuse_case_mapping: {chat: [chat]}\n", encoding="utf-8")
        p = CatalogPolicy(ruta)
        assert p.enrichment_enabled is False
        assert p.canirun_detail_endpoint is None and p.huggingface_api is None


class TestRepoDeUrl:
    def test_parses_org_repo(self) -> None:
        assert repo_de_url("https://huggingface.co/Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
        assert repo_de_url("https://www.huggingface.co/a/b/tree/main") == "a/b"

    def test_rejects_other_hosts_and_short_paths(self) -> None:
        assert repo_de_url("https://example.com/a/b") is None
        assert repo_de_url("https://huggingface.co/onlyorg") is None
        assert repo_de_url(None) is None


class TestEnriquecer:
    def test_languages_runtimes_and_formats(self, red: _Red, policy: CatalogPolicy) -> None:
        familias = [
            _fam("qwen3-8b", "https://huggingface.co/Qwen/Qwen3-8B"),
            _fam("llama3.1-8b", "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"),
        ]
        (qwen, llama), informe = _enriquecer(familias, red, policy)

        assert qwen.idiomas_detectados is None  # card declares nothing: unknown, not empty
        assert qwen.ollama_id == "qwen3:8b" and qwen.lmstudio_id == "qwen3-2504"
        assert qwen.formatos_disponibles == [
            "awq", "compressed-tensors", "gguf", "mlx", "safetensors", "transformers"
        ]
        assert qwen.runtimes_disponibles == [
            "llama.cpp", "lmstudio", "mlx", "ollama", "transformers", "vllm"
        ]
        assert qwen.fuentes_enriquecimiento == [FUENTE_CANIRUN, FUENTE_HF_TREE]
        assert [(c.nombre, c.vram_gb, c.calidad) for c in qwen.cuantizaciones] == [
            ("Q4_K_M", 4.6, "good")
        ]
        assert llama.cuantizaciones[0].bits == 4

        assert llama.idiomas_detectados == ["en", "de", "fr"]  # "Code"/"multilingual" dropped
        assert llama.ollama_id == "llama3.1:8b" and llama.lmstudio_id is None
        assert llama.runtimes_disponibles == ["exllama", "ollama", "transformers"]
        assert llama.fuentes_enriquecimiento == [FUENTE_CANIRUN, FUENTE_HF_CARD, FUENTE_HF_TREE]

        assert informe.contadores[FUENTE_CANIRUN] == {"ok": 2, "sin_datos": 0, "error": 0}
        assert informe.contadores[FUENTE_HF_CARD] == {"ok": 1, "sin_datos": 1, "error": 0}
        assert informe.contadores[FUENTE_HF_TREE] == {"ok": 2, "sin_datos": 0, "error": 0}

    def test_input_families_are_not_mutated_and_order_kept(
        self, red: _Red, policy: CatalogPolicy
    ) -> None:
        familias = [
            _fam("llama3.1-8b", "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"),
            _fam("qwen3-8b", "https://huggingface.co/Qwen/Qwen3-8B"),
        ]
        nuevas, _ = _enriquecer(familias, red, policy)
        assert [f.familia_id for f in nuevas] == ["llama3.1-8b", "qwen3-8b"]
        assert familias[0].idiomas_detectados is None and familias[0].runtimes_disponibles == []

    def test_quantised_mirror_follows_base_model(self, red: _Red, policy: CatalogPolicy) -> None:
        red.tarjetas["lmstudio-community/L-GGUF"] = _tarjeta(
            "lmstudio-community/L-GGUF", tags=["gguf"],
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        )
        (fam,), informe = _enriquecer(
            [_fam("llama3.1-8b", "https://huggingface.co/lmstudio-community/L-GGUF")], red, policy
        )
        assert fam.idiomas_detectados == ["en", "de", "fr"]  # from the base card
        assert "gguf" in fam.formatos_disponibles  # the mirror itself
        assert "exl3" in fam.formatos_disponibles  # the base model's tree
        objetivo = "quantized%3Ameta-llama%2FLlama-3.1-8B-Instruct"
        assert any(objetivo in u or objetivo.replace("%3A", ":").replace("%2F", "/") in u
                   for u in red.requests)

    def test_missing_sources_are_counted_not_fatal(self, red: _Red, policy: CatalogPolicy) -> None:
        (fam,), informe = _enriquecer(
            [_fam("nuevo", "https://huggingface.co/org/unknown")],
            red, policy,
        )
        assert fam.idiomas_detectados is None
        assert fam.runtimes_disponibles == [] and fam.formatos_disponibles == []
        assert fam.fuentes_enriquecimiento == []
        assert informe.contadores[FUENTE_CANIRUN]["sin_datos"] == 1
        assert informe.contadores[FUENTE_HF_CARD]["sin_datos"] == 1
        assert informe.contadores[FUENTE_HF_TREE]["sin_datos"] == 1

    def test_transport_errors_are_counted_not_raised(
        self, red: _Red, policy: CatalogPolicy
    ) -> None:
        red.fallos.add("huggingface.co")
        (fam,), informe = _enriquecer(
            [_fam("qwen3-8b", "https://huggingface.co/Qwen/Qwen3-8B")], red, policy
        )
        assert fam.ollama_id == "qwen3:8b"  # canIRun still answered
        assert fam.formatos_disponibles == []
        assert informe.contadores[FUENTE_HF_CARD]["error"] == 1
        assert informe.contadores[FUENTE_HF_TREE]["error"] == 1

    def test_invalid_detail_contract_is_an_error(self, red: _Red, policy: CatalogPolicy) -> None:
        red.detalles["qwen3-8b"] = {"id": "qwen3-8b"}  # missing required list fields
        (fam,), informe = _enriquecer(
            [_fam("qwen3-8b", "https://huggingface.co/Qwen/Qwen3-8B")], red, policy
        )
        assert fam.ollama_id is None
        assert informe.contadores[FUENTE_CANIRUN]["error"] == 1

    def test_family_without_hf_url_or_canirun_id(self, red: _Red, policy: CatalogPolicy) -> None:
        (fam,), informe = _enriquecer(
            [_fam("x", "https://example.com/x", canirun_id=None)], red, policy
        )
        assert red.requests == []
        assert all(informe.contadores[f]["sin_datos"] == 1 for f in informe.contadores)

    def test_empty_input(self, red: _Red, policy: CatalogPolicy) -> None:
        assert _enriquecer([], red, policy)[0] == []


class TestSyncIntegration:
    def _transport(self, red: _Red) -> httpx.MockTransport:
        payload = json.loads(
            (_FIXTURES / "canirun_models_response.json").read_text(encoding="utf-8")
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "canirun.ai" and request.url.path == "/api/models":
                return httpx.Response(200, json=payload)
            return red(request)

        return httpx.MockTransport(handler)

    def test_sync_enriches_and_records_counts(self, tmp_path: Path, red: _Red) -> None:
        result = sync_catalog(tmp_path, transport=self._transport(red))
        assert result.ok
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        qwen = next(f for f in snap.familias if f.familia_id == "qwen3-8b")
        assert qwen.ollama_id == "qwen3:8b" and "llama.cpp" in qwen.runtimes_disponibles
        llama = next(f for f in snap.familias if f.familia_id == "llama3.1-8b")
        assert llama.idiomas_detectados == ["en", "de", "fr"]
        assert snap.meta.enriquecimiento[FUENTE_CANIRUN]["ok"] == 2
        total_card = sum(snap.meta.enriquecimiento[FUENTE_HF_CARD].values())
        assert total_card == snap.meta.modelos_incluidos

    def test_sync_can_disable_enrichment(self, tmp_path: Path, red: _Red) -> None:
        result = sync_catalog(tmp_path, transport=self._transport(red), enriquecer=False)
        assert result.ok
        assert red.requests == []
        snap = read_snapshot(tmp_path, result.snapshot_id or "")
        assert snap.meta.enriquecimiento == {}
        assert all(f.runtimes_disponibles == [] for f in snap.familias)

    def test_enrichment_changes_the_snapshot_identity(self, tmp_path: Path, red: _Red) -> None:
        con = sync_catalog(tmp_path / "a", transport=self._transport(red))
        sin = sync_catalog(tmp_path / "b", transport=self._transport(red), enriquecer=False)
        assert con.snapshot_id != sin.snapshot_id


class TestRuntimeNormalisation:
    @pytest.mark.parametrize(
        ("texto", "esperado"),
        [("Ollama", "ollama"), (" LM Studio ", "lmstudio"), ("llama-cpp", "llama.cpp"),
         ("GGUF", "llama.cpp"), ("exl2", "exllama"), ("vLLM", "vllm"), ("", None), (None, None),
         ("kobold", "kobold")],
    )
    def test_aliases(self, texto: str | None, esperado: str | None) -> None:
        assert normalizar_runtime(texto) == esperado


class TestRuntimeFilter:
    def _familia(self, fid: str, runtimes: list[str]) -> FamiliaModelo:
        return FamiliaModelo(
            familia_id=fid, variantes=["7B"], pipeline_tag="chat", tareas_soportadas=["chat"],
            runtimes_disponibles=runtimes,
        )

    def test_declared_runtimes_are_enforced(self) -> None:
        familias = [self._familia("a", ["ollama", "llama.cpp"]), self._familia("b", ["vllm"])]
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=8, runtime_preferido="Ollama")
        candidatos = filtrar_candidatos(familias, criterios, MockVRAMProvider())
        assert [c.familia.familia_id for c in candidatos] == ["a"]

    def test_unknown_runtimes_never_exclude(self) -> None:
        familias = [self._familia("a", []), self._familia("b", ["vllm"])]
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=8, runtime_preferido="lm studio")
        candidatos = filtrar_candidatos(familias, criterios, MockVRAMProvider())
        assert [c.familia.familia_id for c in candidatos] == ["a"]

    def test_no_preference_passes_all(self) -> None:
        familias = [self._familia("a", ["ollama"]), self._familia("b", [])]
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=8)
        assert len(filtrar_candidatos(familias, criterios, MockVRAMProvider())) == 2
