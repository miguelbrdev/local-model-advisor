"""Shared fixtures for the test suite.

Provides reusable sample catalogue families, search criteria, and a
deterministic mock VRAM provider that never calls external APIs.
"""

from __future__ import annotations

import pytest

from src.models.schemas import (
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)

# ---------------------------------------------------------------------------
# Sample catalogue families
# ---------------------------------------------------------------------------

FAMILIA_CODIGO = FamiliaModelo(
    familia_id="qwen2.5-coder",
    variantes=["1.5B", "7B", "14B", "32B"],
    pipeline_tag="code",
    license_declarada="Apache 2.0",
    idiomas_detectados=["en", "es", "zh", "fr", "de"],
    formatos_disponibles=["gguf"],
    canirun_id="qwen2.5-coder-7b",
)

FAMILIA_CHAT = FamiliaModelo(
    familia_id="llama3.1-8b",
    variantes=["8B"],
    pipeline_tag="text-generation",
    license_declarada="Llama 3.1 Community",
    idiomas_detectados=["en", "es"],
    formatos_disponibles=["gguf"],
    canirun_id="llama3.1-8b",
)

FAMILIA_SIN_LICENCIA = FamiliaModelo(
    familia_id="mystery-model",
    variantes=["7B"],
    pipeline_tag="code",
    license_declarada=None,
    idiomas_detectados=["en"],
    formatos_disponibles=["gguf"],
    canirun_id=None,
)

FAMILIA_POCOS_IDIOMAS = FamiliaModelo(
    familia_id="zh-only",
    variantes=["7B"],
    pipeline_tag="code",
    license_declarada="MIT",
    idiomas_detectados=["zh"],
    formatos_disponibles=["gguf"],
    canirun_id="zh-only-7b",
)


@pytest.fixture
def familia_codigo() -> FamiliaModelo:
    return FAMILIA_CODIGO


@pytest.fixture
def familia_chat() -> FamiliaModelo:
    return FAMILIA_CHAT


@pytest.fixture
def catalogue() -> list[FamiliaModelo]:
    return [FAMILIA_CODIGO, FAMILIA_CHAT, FAMILIA_SIN_LICENCIA, FAMILIA_POCOS_IDIOMAS]


# ---------------------------------------------------------------------------
# Sample criteria
# ---------------------------------------------------------------------------

@pytest.fixture
def criterios_codigo_11gb() -> CriteriosBusqueda:
    return CriteriosBusqueda(
        tarea="codigo",
        vram_gb=11.0,
        idiomas_requeridos=["en", "es"],
    )


@pytest.fixture
def criterios_chat_8gb() -> CriteriosBusqueda:
    return CriteriosBusqueda(
        tarea="chat",
        vram_gb=8.0,
    )


@pytest.fixture
def criterios_requiere_comercial() -> CriteriosBusqueda:
    return CriteriosBusqueda(
        tarea="codigo",
        vram_gb=16.0,
        requiere_licencia_comercial=True,
    )


# ---------------------------------------------------------------------------
# Mock VRAM provider
# ---------------------------------------------------------------------------

class MockVRAMProvider:
    """Deterministic mock that returns configurable results per familia."""

    def __init__(
        self,
        default_compatible: bool = True,
        results: dict[str, CompatibilityResult] | None = None,
    ):
        self.default_compatible = default_compatible
        self.results = results or {}
        self.call_count = 0
        self.calls: list[tuple[str, float]] = []

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        self.call_count += 1
        self.calls.append((familia.familia_id, vram_gb))

        if familia.familia_id in self.results:
            return self.results[familia.familia_id]

        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=familia.variantes[0] if familia.variantes else "unknown",
            compatible=self.default_compatible,
            grado="comfortable" if self.default_compatible else "insufficient",
            fuente="no_disponible",
            nota="Mock provider result.",
        )


@pytest.fixture
def mock_vram_provider() -> MockVRAMProvider:
    return MockVRAMProvider()


@pytest.fixture
def mock_vram_provider_all_compatible() -> MockVRAMProvider:
    return MockVRAMProvider(default_compatible=True)


@pytest.fixture
def mock_vram_provider_all_incompatible() -> MockVRAMProvider:
    return MockVRAMProvider(default_compatible=False)
