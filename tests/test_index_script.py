"""Tests for scripts.index_docs — offline: mocked HTTP, fake embedder, in-memory Qdrant."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from scripts import index_docs as script
from src.evidence.docs_store import AlmacenDocumentos
from src.evidence.qdrant_index import IndiceEvidencia
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH
from tests.test_qdrant_index import FakeEmbedder

README = """---
license: apache-2.0
---
# A model

## Introduction

This model is designed for code generation and reasoning over long contexts,
with strong multilingual support across many programming languages.
"""


def _client(ok_substrings: tuple[str, ...]) -> httpx.Client:
    """200 with README for URLs containing any of ``ok_substrings``; 401 otherwise."""

    def handler(request: httpx.Request) -> httpx.Response:
        if any(s in str(request.url) for s in ok_substrings):
            return httpx.Response(200, text=README)
        return httpx.Response(401, text="gated")

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(tmp_path: Path, *extra: str, client=None, embedder=None, qdrant=None) -> int:
    return script.main(
        [
            "--catalog-dir", str(tmp_path / "absent"),
            "--bootstrap-path", str(DEFAULT_BOOTSTRAP_PATH),
            "--docs-dir", str(tmp_path / "docs"),
            *extra,
        ],
        http_client=client,
        embedder=embedder,
        qdrant=qdrant,
    )


class TestParser:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("LMA_QDRANT_URL", "LMA_EMBEDDING_MODEL", "LMA_OLLAMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        args = script.build_parser().parse_args([])
        assert args.docs_dir.parts[-2:] == ("runtime", "docs")
        assert args.qdrant_url == "http://127.0.0.1:6333"
        assert args.embedding_model == "embeddinggemma"
        assert args.ollama_base_url == "http://127.0.0.1:11434"
        assert not (args.skip_fetch or args.skip_index or args.refetch)

    def test_defaults_come_from_the_api_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LMA_QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("LMA_EMBEDDING_MODEL", "other-embed")
        monkeypatch.setenv("LMA_OLLAMA_BASE_URL", "http://ollama:11434")
        args = script.build_parser().parse_args([])
        assert args.qdrant_url == "http://qdrant:6333"
        assert args.embedding_model == "other-embed"
        assert args.ollama_base_url == "http://ollama:11434"
        assert script.build_parser().parse_args(["--qdrant-url", "http://x:1"]).qdrant_url == "http://x:1"


class TestFetchAndIndex:
    def test_full_run_stores_docs_and_indexes_fragments(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        qdrant = QdrantClient(":memory:")
        with _client(("/Qwen/",)) as client:
            code = _run(tmp_path, client=client, embedder=FakeEmbedder(), qdrant=qdrant)

        assert code == script.EXIT_OK
        almacen = AlmacenDocumentos(tmp_path / "docs")
        resumen = almacen.resumen()
        assert resumen["ok"] >= 20          # the Qwen families
        assert resumen["gated"] >= 50       # everything else answered 401
        assert sum(resumen.values()) == 107
        out = capsys.readouterr().out
        assert "fetch: gated=" in out and "ok=" in out
        assert "index: snapshot=snap-3eb98b50f1e483e0 familias=" in out

        indexadas = IndiceEvidencia(qdrant, FakeEmbedder()).familias_indexadas()
        assert len(indexadas) == resumen["ok"]
        assert "qwen2.5-coder-7b" in indexadas

    def test_second_run_uses_the_cache(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        llamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            llamadas.append(str(request.url))
            return httpx.Response(200, text=README)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            _run(tmp_path, "--skip-index", client=client)
            primera = len(llamadas)
            _run(tmp_path, "--skip-index", client=client)
        assert primera == 107
        assert len(llamadas) == 107  # nothing re-downloaded
        assert "cached=107" in capsys.readouterr().out

    def test_refetch_downloads_again(self, tmp_path: Path) -> None:
        llamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            llamadas.append(str(request.url))
            return httpx.Response(200, text=README)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            _run(tmp_path, "--skip-index", client=client)
            _run(tmp_path, "--skip-index", "--refetch", client=client)
        assert len(llamadas) == 214  # 2 x 107 families

    def test_skip_fetch_indexes_stored_docs_only(self, tmp_path: Path) -> None:
        with _client(("/Qwen/",)) as client:
            _run(tmp_path, "--skip-index", client=client)
        qdrant = QdrantClient(":memory:")
        code = _run(tmp_path, "--skip-fetch", embedder=FakeEmbedder(), qdrant=qdrant)
        assert code == script.EXIT_OK
        assert IndiceEvidencia(qdrant, FakeEmbedder()).familias_indexadas()

    def test_transport_errors_are_stored_not_fatal(self, tmp_path: Path) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            code = _run(tmp_path, "--skip-index", client=client)
        assert code == script.EXIT_OK
        assert AlmacenDocumentos(tmp_path / "docs").resumen() == {"error": 107}


class TestFailures:
    def test_no_catalogue_exits_two(self, tmp_path: Path) -> None:
        code = script.main(
            ["--catalog-dir", str(tmp_path / "absent"),
             "--bootstrap-path", str(tmp_path / "absent.json"),
             "--docs-dir", str(tmp_path / "docs")]
        )
        assert code == script.EXIT_NO_CATALOGUE

    def test_embedder_failure_exits_one_and_keeps_docs(self, tmp_path: Path) -> None:
        class _Broken:
            def embed_documents(self, texts):
                raise ConnectionError("ollama down")

            def embed_query(self, text):
                raise ConnectionError("ollama down")

        with _client(("/Qwen/",)) as client:
            code = _run(
                tmp_path, client=client, embedder=_Broken(), qdrant=QdrantClient(":memory:")
            )
        assert code == script.EXIT_INDEX_FAILED
        assert AlmacenDocumentos(tmp_path / "docs").resumen()["ok"] >= 20
