"""Tests for src.tools.structured_filter.

Covers: task filtering on ``tareas_soportadas``, VRAM compatibility gate,
language subset check, and commercial-license filter.
"""

from __future__ import annotations

from src.models.schemas import (
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.tools.structured_filter import filtrar_candidatos, soporta_tarea
from tests.conftest import (
    FAMILIA_CHAT,
    FAMILIA_CODIGO,
    FAMILIA_POCOS_IDIOMAS,
    FAMILIA_SIN_LICENCIA,
    MockVRAMProvider,
)

# A family that declares none of the v1 tasks.
FAMILIA_IMAGE = FamiliaModelo(
    familia_id="image-model",
    variantes=["7B"],
    pipeline_tag="",
    tareas_soportadas=[],
    license_declarada="MIT",
    idiomas_detectados=["en"],
    formatos_disponibles=["gguf"],
    canirun_id="image-model-7b",
)

# Multi-task family: ``pipeline_tag`` only shows the first task.
FAMILIA_MULTITAREA = FamiliaModelo(
    familia_id="generalist",
    variantes=["8B"],
    pipeline_tag="chat",
    tareas_soportadas=["chat", "codigo", "razonamiento"],
    license_declarada="Apache 2.0",
    canirun_id="generalist-8b",
)


# ---------------------------------------------------------------------------
# soporta_tarea helper
# ---------------------------------------------------------------------------

class TestSoportaTarea:
    def test_declared_task_matches(self) -> None:
        assert soporta_tarea(FAMILIA_CODIGO, "codigo") is True

    def test_undeclared_task_does_not_match(self) -> None:
        assert soporta_tarea(FAMILIA_CODIGO, "chat") is False

    def test_reads_every_declared_task_not_only_pipeline_tag(self) -> None:
        assert FAMILIA_MULTITAREA.pipeline_tag == "chat"
        assert soporta_tarea(FAMILIA_MULTITAREA, "codigo") is True
        assert soporta_tarea(FAMILIA_MULTITAREA, "razonamiento") is True

    def test_pipeline_tag_alone_is_not_enough(self) -> None:
        """A family whose ``pipeline_tag`` says ``codigo`` but declares no
        tasks must not match: ``tareas_soportadas`` is the contract."""
        familia = FAMILIA_CODIGO.model_copy(update={"tareas_soportadas": []})
        assert soporta_tarea(familia, "codigo") is False

    def test_empty_tasks_match_nothing(self) -> None:
        for tarea in ("chat", "codigo", "razonamiento"):
            assert soporta_tarea(FAMILIA_IMAGE, tarea) is False


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
        assert ids == ["qwen2.5-coder"]

    def test_chat_returns_chat_families(self, mock_vram_provider: MockVRAMProvider) -> None:
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=16.0)
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_CHAT], criterios, mock_vram_provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["llama3.1-8b"]

    def test_multitask_family_passes_for_every_declared_task(
        self, mock_vram_provider: MockVRAMProvider
    ) -> None:
        for tarea in ("chat", "codigo", "razonamiento"):
            criterios = CriteriosBusqueda(tarea=tarea, vram_gb=16.0)
            resultado = filtrar_candidatos([FAMILIA_MULTITAREA], criterios, mock_vram_provider)
            assert [r.familia.familia_id for r in resultado] == ["generalist"], tarea

    def test_razonamiento_excludes_dedicated_coder(
        self, mock_vram_provider: MockVRAMProvider
    ) -> None:
        criterios = CriteriosBusqueda(tarea="razonamiento", vram_gb=16.0)
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_CHAT, FAMILIA_MULTITAREA], criterios, mock_vram_provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["llama3.1-8b", "generalist"]

    def test_provider_not_consulted_for_filtered_out_families(
        self, mock_vram_provider: MockVRAMProvider
    ) -> None:
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        filtrar_candidatos([FAMILIA_CHAT, FAMILIA_IMAGE], criterios, mock_vram_provider)
        assert mock_vram_provider.call_count == 0


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

    def test_provider_receives_vram_and_runtime(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=12.0, runtime_preferido="ollama")
        filtrar_candidatos([FAMILIA_CODIGO], criterios, provider)
        assert provider.calls == [("qwen2.5-coder", 12.0)]


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
        # zh-only has only ["zh"]; qwen2.5-coder lacks "ja" → both excluded
        assert resultado == []

    def test_family_with_matching_languages_passes(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, idiomas_requeridos=["en", "es"]
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_POCOS_IDIOMAS], criterios, provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["qwen2.5-coder"]

    def test_empty_language_list_passes_all(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0, idiomas_requeridos=[])
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_POCOS_IDIOMAS], criterios, provider
        )
        assert len(resultado) == 2

    def test_family_with_none_languages_passes_language_filter(self) -> None:
        """Synced families declare no languages (``None``); the filter must
        not exclude them, only report the limitation."""
        familia_none_lang = FAMILIA_CODIGO.model_copy(update={"idiomas_detectados": None})
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0, idiomas_requeridos=["en"])
        resultado = filtrar_candidatos([familia_none_lang], criterios, provider)
        assert len(resultado) == 1


# ---------------------------------------------------------------------------
# filtrar_candidatos — license filter
# ---------------------------------------------------------------------------

class TestFiltrarPorLicencia:
    def test_commercial_filter_excludes_restrictive_and_unknown(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, requiere_licencia_comercial=True
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA], criterios, provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["qwen2.5-coder"]  # "Apache 2.0" permitted; None not

    def test_commercial_filter_accepts_source_spelling(self) -> None:
        """The synced source spells it ``"Apache 2.0"`` (space, no dash)."""
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="chat", vram_gb=16.0, requiere_licencia_comercial=True)
        resultado = filtrar_candidatos([FAMILIA_MULTITAREA, FAMILIA_CHAT], criterios, provider)
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["generalist"]  # Llama Community is not permissive

    def test_commercial_filter_false_passes_all(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(
            tarea="codigo", vram_gb=16.0, requiere_licencia_comercial=False
        )
        resultado = filtrar_candidatos(
            [FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA], criterios, provider
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
        # Only qwen2.5-coder passes: declares codigo, en+es, Apache 2.0, compatible
        assert [r.familia.familia_id for r in resultado] == ["qwen2.5-coder"]

    def test_preserves_catalogue_order(self) -> None:
        provider = MockVRAMProvider()
        criterios = CriteriosBusqueda(tarea="codigo", vram_gb=16.0)
        resultado = filtrar_candidatos(
            [FAMILIA_POCOS_IDIOMAS, FAMILIA_CODIGO, FAMILIA_SIN_LICENCIA], criterios, provider
        )
        ids = [r.familia.familia_id for r in resultado]
        assert ids == ["zh-only", "qwen2.5-coder", "mystery-model"]
