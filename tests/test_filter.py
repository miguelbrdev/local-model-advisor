"""Tests for src.tools.structured_filter.

Covers: task filtering, VRAM compatibility gate, language subset check,
and commercial-license filter.
"""

from __future__ import annotations

import pytest

from src.models.schemas import (
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.tools.structured_filter import (
    _es_licencia_comercial,
    _tarea_match,
    filtrar_candidatos,
)
from tests.conftest import (
    FAMILIA_CHAT,
    FAMILIA_CODIGO,
    FAMILIA_POCOS_IDIOMAS,
    FAMILIA_SIN_LICENCIA,
    MockVRAMProvider,
)

# A familia that does NOT match any code/chat/reasoning tags
FAMILIA_IMAGE = FamiliaModelo(
    familia_id="image-model",
    variantes=["7B"],
    pipeline_tag="image-generation",
    license_declarada="MIT",
    idiomas_detectados=["en"],
    formatos_disponibles=["gguf"],
    canirun_id="image-model-7b",
)


# ---------------------------------------------------------------------------
# _tarea_match helper
# ---------------------------------------------------------------------------

class TestTareaMatch:
    def test_code_tag_matches_codigo(self) -> None:
        assert _tarea_match("code", "codigo") is True

    def test_text_generation_matches_chat(self) -> None:
        assert _tarea_match("text-generation", "chat") is True

    def test_text_generation_matches_razonamiento(self) -> None:
        assert _tarea_match("text-generation", "razonamiento") is True

    def test_sentence_similarity_matches_embeddings(self) -> None:
        assert _tarea_match("sentence-similarity", "embeddings") is True

    def test_text_classification_matches_rerankers(self) -> None:
        assert _tarea_match("text-classification", "rerankers") is True

    def test_code_does_not_match_chat(self) -> None:
        assert _tarea_match("code", "chat") is False

    def test_unknown_tag_matches_exact_tarea(self) -> None:
        assert _tarea_match("my-custom-tag", "my-custom-tag") is True


# ---------------------------------------------------------------------------
# _es_licencia_comercial helper
# ---------------------------------------------------------------------------

class TestEsLicenciaComercial:
    @pytest.mark.parametrize("lic", ["MIT", "Apache-2.0", "BSD-2-Clause", "ISC", "MPL-2.0"])
    def test_permissive_licenses_return_true(self, lic: str) -> None:
        assert _es_licencia_comercial(lic) is True

    def test_none_returns_false(self) -> None:
        assert _es_licencia_comercial(None) is False

    def test_restrictive_license_returns_false(self) -> None:
        assert _es_licencia_comercial("Llama 3.1 Community") is False

    def test_case_insensitive(self) -> None:
        assert _es_licencia_comercial("mit") is True
        assert _es_licencia_comercial("apache-2.0") is True


# ---------------------------------------------------------------------------
# filtrar_candidatos — task filter
# ---------------------------------------------------------------------------

class TestFiltrarPorTarea:
    def test_codigo_returns_only_code_families(
        self, mock_vram_provider: MockVRAMProvider
    ) -> None:
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_IMAGE], criterios, mock_vram_provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert "qwen2.5-coder" in ids
        assert "image-model" not in ids

    def test_chat_returns_text_generation_families(
        self, mock_vram_provider: MockVRAMProvider
    ) -> None:
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=16.0)
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_CHAT], criterios, mock_vram_provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert "llama3.1-8b" in ids


# ---------------------------------------------------------------------------
# filtrar_candidatos — VRAM filter
# ---------------------------------------------------------------------------

class TestFiltrarPorVRAM:
    def test_incompatible_models_excluded(self) -> None:
        provider = MockVRAMProvider(default_compatible=False)
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=4.0)
        resultado = filtrar_candidatos([FAMILIA_CODIGO], criterios, provider)
        assert len(resultado) == 0

    def test_compatible_models_kept(self) -> None:
        provider = MockVRAMProvider(default_compatible=True)
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        resultado = filtrar_candidatos([FAMILIA_CODIGO], criterios, provider)
        assert len(resultado) == 1

    def test_none_compatible_kept(self) -> None:
        """compatible=None (unknown) should NOT be excluded."""
        provider = MockVRAMProvider()
        provider.results = {
            "qwen2.5-coder": CompatibilityResult(
                modelo_id="qwen2.5-coder",
                variante="7B",
                compatible=None,
                fuente="no_disponible",
            )
        }
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        resultado = filtrar_candidatos([FAMILIA_CODIGO], criterios, provider)
        assert len(resultado) == 1


# ---------------------------------------------------------------------------
# filtrar_candidatos — language filter
# ---------------------------------------------------------------------------

class TestFiltrarPorIdioma:
    def test_missing_language_excludes_family(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, idiomas_requeridos=["en", "es", "ja"]
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_POCOS_IDIOMAS], criterios, provider
        )
        ids = [r.familia.familia_id for r in resultado]
        # zh-only has only ["zh"], so "en","es","ja" not subset → excluded
        assert "zh-only" not in ids
        # qwen2.5-coder has ["en","es","zh","fr","de"], no "ja" → also excluded
        assert "qwen2.5-coder" not in ids
        assert len(resultado) == 0

    def test_family_with_matching_languages_passes(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, idiomas_requeridos=["en", "es"]
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_POCOS_IDIOMAS], criterios, provider
        )
        ids = [r.familia.familia_id for r in resultado]
        # qwen2.5-coder has en+es → passes
        assert "qwen2.5-coder" in ids
        # zh-only has only zh → fails
        assert "zh-only" not in ids

    def test_empty_language_list_passes_all(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0, idiomas_requeridos=[])
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_POCOS_IDIOMAS], criterios, provider
        )
        assert len(resultado) == 2

    def test_family_with_none_languages_passes_language_filter(self) -> None:
        familia_none_lang = FAMILIA_CODIGO.model_copy(
            update={"idiomas_detectados": None}
        )
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, idiomas_requeridos=["en"]
        )
        resultado = filtrar_candidatos([familia_none_lang], criterios, provider)
        assert len(resultado) == 1


# ---------------------------------------------------------------------------
# filtrar_candidatos — license filter
# ---------------------------------------------------------------------------

class TestFiltrarPorLicencia:
    def test_commercial_filter_excludes_restrictive(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, requiere_licencia_comercial=True
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA],
            criterios,
            provider,
        )
        ids = [r.familia.familia_id for r in resultado]
        # Apache 2.0 → permitted; None → not
        assert "qwen2.5-coder" in ids
        assert "mystery-model" not in ids

    def test_commercial_filter_false_passes_all(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, requiere_licencia_comercial=False
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA],
            criterios,
            provider,
        )
        assert len(resultado) == 2


# ---------------------------------------------------------------------------
# filtrar_candidatos — integration of all filters
# ---------------------------------------------------------------------------

class TestFiltrarIntegracion:
    def test_all_filters_combined(self) -> None:
        """Task + language + license + VRAM all must pass."""
        provider = MockVRAMProvider(default_compatible=True)
        criterios = CriteriosBusqueda(
            tarea="codigo",
            vram_gb=16.0,
            idiomas_requeridos=["en", "es"],
            requiere_licencia_comercial=True,
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_CHAT, FAMILIA_SIN_LICENCIA, FAMILIA_POCOS_IDIOMAS],
            criterios,
            provider,
        )
        # Only qwen2.5-coder passes: code tag, en+es, Apache 2.0, compatible
        assert len(resultado) == 1
        assert resultado[0].familia.familia_id == "qwen2.5-coder"

    def test_provider_called_for_each_candidate(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        filtrar_candidatos([FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA], criterios, provider)
        # Provider is called AFTER task/license/language filters
        # FAMILIA_CODIGO passes task, so provider is called
        assert provider.call_count >= 1
