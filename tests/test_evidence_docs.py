"""Tests for src.evidence.hf_readme, docs_store and chunker.

HTTP is always ``httpx.MockTransport``; files only under ``tmp_path``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from src.evidence.chunker import FRAGMENTO_MINIMO, limpiar_markdown, trocear
from src.evidence.docs_store import AlmacenDocumentos, DocumentoModelo
from src.evidence.hf_readme import (
    base_model_de,
    descargar_readme,
    es_espejo_cuantizado,
    separar_front_matter,
    url_readme_raw,
)

README_QWEN = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B
tags:
- code
---
# Qwen2.5-Coder-7B-Instruct

Qwen2.5-Coder is the latest series of code-specific Qwen models.

## Introduction

Significantly improvements in code generation, code reasoning and code fixing.
"""

README_MIRROR = """---
base_model: Qwen/Qwen3-0.6B
tags:
- gguf
---
## Community Model> Qwen3-0.6B by Qwen

GGUF quantization provided by LM Studio team.
"""

README_BASE = """---
license: apache-2.0
---
# Qwen3-0.6B

Qwen3 is the latest generation of large language models in Qwen series,
offering a comprehensive suite of dense and mixture-of-experts models.
"""


def _transport(rutas: dict[str, tuple[int, str]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        status, text = rutas.get(str(request.url), (404, "Not Found"))
        return httpx.Response(status, text=text)

    return httpx.MockTransport(handler)


def _client(rutas: dict[str, tuple[int, str]]) -> httpx.Client:
    return httpx.Client(transport=_transport(rutas))


RAW_QWEN = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/README.md"
RAW_MIRROR = "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF/raw/main/README.md"
RAW_BASE = "https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/README.md"


# ---------------------------------------------------------------------------
# hf_readme helpers
# ---------------------------------------------------------------------------

class TestUrlReadmeRaw:
    @pytest.mark.parametrize(
        ("url", "esperado"),
        [
            ("https://huggingface.co/Qwen/Qwen3-0.6B", RAW_BASE),
            ("https://huggingface.co/Qwen/Qwen3-0.6B/", RAW_BASE),
            ("https://www.huggingface.co/Qwen/Qwen3-0.6B/tree/main", RAW_BASE),
            ("https://github.com/Qwen/Qwen3", None),
            ("https://huggingface.co/Qwen", None),
        ],
    )
    def test_mapping(self, url: str, esperado: str | None) -> None:
        assert url_readme_raw(url) == esperado


class TestFrontMatter:
    def test_split(self) -> None:
        meta, cuerpo = separar_front_matter(README_QWEN)
        assert meta["license"] == "apache-2.0"
        assert cuerpo.lstrip().startswith("# Qwen2.5-Coder-7B-Instruct")

    def test_absent(self) -> None:
        assert separar_front_matter("# Title\n\nBody") == ({}, "# Title\n\nBody")

    def test_invalid_yaml_keeps_body(self) -> None:
        meta, cuerpo = separar_front_matter("---\n: : :\n  - [\n---\nBody")
        assert meta == {}
        assert cuerpo == "Body"

    def test_unterminated_front_matter_is_body(self) -> None:
        meta, cuerpo = separar_front_matter("---\nlicense: mit\nno end")
        assert meta == {}
        assert cuerpo.startswith("---")

    def test_base_model_variants(self) -> None:
        assert base_model_de({"base_model": "Qwen/Qwen3-0.6B"}) == "Qwen/Qwen3-0.6B"
        assert base_model_de({"base_model": ["a/b", "c/d"]}) == "a/b"
        assert base_model_de({"base_model": "not-a-repo"}) is None
        assert base_model_de({}) is None

    def test_mirror_detection(self) -> None:
        assert es_espejo_cuantizado({"tags": ["GGUF"], "base_model": "a/b"}) is True
        assert es_espejo_cuantizado({"tags": ["code"], "base_model": "a/b"}) is False
        assert es_espejo_cuantizado({"tags": ["gguf"]}) is False
        assert es_espejo_cuantizado({"tags": "gguf", "base_model": "a/b"}) is False


# ---------------------------------------------------------------------------
# descargar_readme
# ---------------------------------------------------------------------------

class TestDescargarReadme:
    def test_public_repo_ok(self) -> None:
        with _client({RAW_QWEN: (200, README_QWEN)}) as client:
            r = descargar_readme(
                "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct", client=client
            )
        assert r.estado == "ok"
        assert r.texto.startswith("# Qwen2.5-Coder-7B-Instruct")
        assert r.base_model == "Qwen/Qwen2.5-Coder-7B"
        assert r.redirigido_desde is None  # not a mirror: instruct README kept

    @pytest.mark.parametrize(("status", "estado"), [(401, "gated"), (403, "gated"),
                                                     (404, "not_found"), (500, "error")])
    def test_http_statuses(self, status: int, estado: str) -> None:
        with _client({RAW_QWEN: (status, "x")}) as client:
            r = descargar_readme(
                "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct", client=client
            )
        assert r.estado == estado
        assert r.texto == ""
        assert str(status) in r.detalle

    def test_transport_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            r = descargar_readme("https://huggingface.co/Qwen/Qwen3-0.6B", client=client)
        assert r.estado == "error"
        assert "ConnectTimeout" in r.detalle

    def test_not_huggingface(self) -> None:
        with _client({}) as client:
            r = descargar_readme("https://example.com/model", client=client)
        assert r.estado == "not_hf"
        assert r.url_readme is None

    def test_mirror_follows_base_model_once(self) -> None:
        rutas = {RAW_MIRROR: (200, README_MIRROR), RAW_BASE: (200, README_BASE)}
        with _client(rutas) as client:
            r = descargar_readme(
                "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF", client=client
            )
        assert r.estado == "ok"
        assert "Qwen3 is the latest generation" in r.texto
        assert r.url_readme == RAW_BASE
        assert r.redirigido_desde == "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF"
        assert r.base_model == "Qwen/Qwen3-0.6B"

    def test_mirror_keeps_own_text_when_base_is_gated(self) -> None:
        rutas = {RAW_MIRROR: (200, README_MIRROR), RAW_BASE: (401, "gated")}
        with _client(rutas) as client:
            r = descargar_readme(
                "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF", client=client
            )
        assert r.estado == "ok"
        assert "Community Model" in r.texto
        assert r.redirigido_desde is None

    def test_base_model_hop_is_not_recursive(self) -> None:
        """A base model that is itself a gguf mirror must not trigger a second hop."""
        mirror_of_mirror = README_MIRROR.replace("Qwen/Qwen3-0.6B", "other/Deeper-GGUF")
        rutas = {
            RAW_MIRROR: (200, README_MIRROR),
            RAW_BASE: (200, mirror_of_mirror),
            "https://huggingface.co/other/Deeper-GGUF/raw/main/README.md": (200, README_BASE),
        }
        llamadas: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            llamadas.append(str(request.url))
            status, text = rutas[str(request.url)]
            return httpx.Response(status, text=text)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r = descargar_readme(
                "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF", client=client
            )
        assert llamadas == [RAW_MIRROR, RAW_BASE]
        assert "Community Model" in r.texto


# ---------------------------------------------------------------------------
# AlmacenDocumentos
# ---------------------------------------------------------------------------

class TestAlmacenDocumentos:
    def _doc(
        self, fid: str = "qwen3-0.6b", estado: str = "ok", texto: str = "Qwen3 prose"
    ) -> DocumentoModelo:
        return DocumentoModelo(
            familia_id=fid, url_repo="https://huggingface.co/Qwen/Qwen3-0.6B",
            url_readme=RAW_BASE, estado=estado, texto=texto,
            fecha_descarga=datetime(2026, 11, 7, tzinfo=UTC), hash_texto="x",
        )

    def test_round_trip(self, tmp_path: Path) -> None:
        store = AlmacenDocumentos(tmp_path / "docs")
        ruta = store.guardar(self._doc())
        assert ruta.name == "qwen3-0.6b.json"
        assert not list((tmp_path / "docs").glob("*.tmp"))
        assert store.cargar("qwen3-0.6b") == self._doc()

    def test_missing_and_corrupt_return_none(self, tmp_path: Path) -> None:
        store = AlmacenDocumentos(tmp_path)
        assert store.cargar("nope") is None
        (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
        assert store.cargar("bad") is None
        assert store.listar() == []

    def test_listar_and_resumen(self, tmp_path: Path) -> None:
        store = AlmacenDocumentos(tmp_path)
        store.guardar(self._doc("b", "ok"))
        store.guardar(self._doc("a", "gated", ""))
        store.guardar(self._doc("c", "ok"))
        assert [d.familia_id for d in store.listar()] == ["a", "b", "c"]
        assert store.resumen() == {"ok": 2, "gated": 1}

    def test_desde_descarga_hashes_text(self) -> None:
        from src.evidence.hf_readme import ReadmeDescargado

        descarga = ReadmeDescargado("u", RAW_BASE, "ok", "hello")
        doc = DocumentoModelo.desde_descarga("f", descarga, fecha=datetime(2026, 1, 1, tzinfo=UTC))
        assert doc.hash_texto == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        assert doc.tiene_texto is True
        assert DocumentoModelo.desde_descarga(
            "g", ReadmeDescargado("u", RAW_BASE, "gated"), fecha=doc.fecha_descarga
        ).tiene_texto is False

    def test_default_directory_is_runtime_docs(self) -> None:
        assert AlmacenDocumentos().directorio.parts[-2:] == ("runtime", "docs")


# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------

class TestLimpiarMarkdown:
    def test_drops_code_html_images_and_badges(self) -> None:
        texto = (
            "# Model\n\n<p align=\"center\"><img src=\"x.png\"/></p>\n\n"
            "![badge](https://img.shields.io/x.svg)\n\n"
            "Use it like this:\n\n```python\nprint('hi')\n```\n\n"
            "See [the paper](https://arxiv.org/abs/1) for details.\n\n|---|---|\n"
        )
        limpio = limpiar_markdown(texto)
        assert "print" not in limpio
        assert "<p" not in limpio and "img" not in limpio
        assert "shields.io" not in limpio
        assert "See the paper for details." in limpio
        assert "|---|---|" not in limpio


class TestTrocear:
    def test_deterministic_and_prefixed_with_heading(self) -> None:
        a = trocear(README_QWEN)
        b = trocear(README_QWEN)
        assert a == b
        assert any(f.startswith("Introduction: ") for f in a)
        assert all(len(f) <= 800 for f in a)

    def test_respects_size_limit_with_long_paragraph(self) -> None:
        parrafo = " ".join(f"Sentence number {i} is here." for i in range(200))
        fragmentos = trocear(f"# Long\n\n{parrafo}", tamano=200)
        assert len(fragmentos) > 5
        assert all(len(f) <= 200 for f in fragmentos)
        assert "".join(fragmentos).count("Sentence number 199") == 1

    def test_hard_cuts_a_single_giant_sentence(self) -> None:
        fragmentos = trocear("x" * 1000, tamano=300)
        assert len(fragmentos) == 4
        assert all(len(f) <= 300 for f in fragmentos)

    def test_drops_fragments_below_minimum(self) -> None:
        assert trocear("# T\n\nshort") == []
        assert len("short") < FRAGMENTO_MINIMO

    def test_empty_input(self) -> None:
        assert trocear("") == []

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            trocear("text", tamano=0)
