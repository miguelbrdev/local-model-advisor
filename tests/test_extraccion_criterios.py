"""Tests for src.chains.extraccion_criterios.

No Ollama, no network: the language-model extractor is exercised through
an injected fake runnable.  The real ChatOllama chain is only built lazily
on first use, which one test pins so the suite can never reach the
network by accident.
"""

from __future__ import annotations

import pytest

from src.chains.extraccion_criterios import (
    PROMPT_SISTEMA,
    CriteriosExtraidos,
    ExtraccionNoDisponibleError,
    ExtractorCompuesto,
    ExtractorCriterios,
    ExtractorJson,
    ExtractorOllama,
    MensajeNoEstructuradoError,
)
from src.models.schemas import CriteriosBusqueda


class _FakeLLM:
    """Stands in for the structured-output runnable."""

    def __init__(self, salida=None, error: Exception | None = None) -> None:
        self.salida = salida
        self.error = error
        self.llamadas: list = []

    def invoke(self, mensajes):
        self.llamadas.append(mensajes)
        if self.error is not None:
            raise self.error
        return self.salida


# ---------------------------------------------------------------------------
# CriteriosExtraidos.fusionar
# ---------------------------------------------------------------------------

class TestFusionar:
    def test_without_base_yields_the_stated_fields(self) -> None:
        c = CriteriosExtraidos(tarea="codigo", vram_gb=12).fusionar(None)
        assert c == CriteriosBusqueda(tarea="codigo", vram_gb=12)

    def test_stated_fields_override_base(self) -> None:
        base = CriteriosBusqueda(tarea="chat", vram_gb=8, idiomas_requeridos=["en"])
        c = CriteriosExtraidos(vram_gb=24).fusionar(base)
        assert c.tarea == "chat"
        assert c.vram_gb == 24
        assert c.idiomas_requeridos == ["en"]

    def test_unstated_fields_keep_base(self) -> None:
        base = CriteriosBusqueda(tarea="codigo", vram_gb=8, requiere_licencia_comercial=True)
        c = CriteriosExtraidos().fusionar(base)
        assert c == base

    def test_empty_language_list_does_not_erase_base(self) -> None:
        base = CriteriosBusqueda(idiomas_requeridos=["es"])
        fusionado = CriteriosExtraidos(idiomas_requeridos=[]).fusionar(base)
        assert fusionado.idiomas_requeridos == ["es"]

    def test_explicit_false_licence_overrides_true(self) -> None:
        base = CriteriosBusqueda(requiere_licencia_comercial=True)
        c = CriteriosExtraidos(requiere_licencia_comercial=False).fusionar(base)
        assert c.requiere_licencia_comercial is False

    def test_negative_vram_rejected(self) -> None:
        with pytest.raises(ValueError):
            CriteriosExtraidos(vram_gb=-1)


# ---------------------------------------------------------------------------
# Normalisation of languages and runtime
# ---------------------------------------------------------------------------

class TestNormalizacion:
    def test_language_names_become_iso_codes(self) -> None:
        c = CriteriosExtraidos(idiomas_requeridos=["Español", "english", "fr", " Alemán "])
        assert c.idiomas_requeridos == ["es", "en", "fr", "de"]

    def test_duplicates_and_blanks_removed(self) -> None:
        c = CriteriosExtraidos(idiomas_requeridos=["es", "español", "", "ES"])
        assert c.idiomas_requeridos == ["es"]

    def test_unknown_language_kept_lowercase(self) -> None:
        assert CriteriosExtraidos(idiomas_requeridos=["Klingon"]).idiomas_requeridos == ["klingon"]

    def test_runtime_lowercased_and_trimmed(self) -> None:
        c = CriteriosExtraidos(runtime_preferido="  LM  Studio ")
        assert c.runtime_preferido == "lm studio"
        assert CriteriosExtraidos(runtime_preferido="   ").runtime_preferido is None

    def test_json_clients_get_the_same_normalisation(self) -> None:
        c = ExtractorJson().extraer(
            '{"idiomas_requeridos": ["Español"], "runtime_preferido": "Ollama"}'
        )
        assert c.idiomas_requeridos == ["es"]
        assert c.runtime_preferido == "ollama"


# ---------------------------------------------------------------------------
# ExtractorJson
# ---------------------------------------------------------------------------

class TestExtractorJson:
    def test_full_document(self) -> None:
        c = ExtractorJson().extraer(
            '{"tarea": "codigo", "vram_gb": 11, "idiomas_requeridos": ["en"]}'
        )
        assert c == CriteriosBusqueda(tarea="codigo", vram_gb=11, idiomas_requeridos=["en"])

    def test_partial_document_merges_over_base(self) -> None:
        base = CriteriosBusqueda(tarea="chat")
        c = ExtractorJson().extraer('{"vram_gb": 6}', base)
        assert c == CriteriosBusqueda(tarea="chat", vram_gb=6)

    def test_free_text_raises_with_error_items(self) -> None:
        with pytest.raises(MensajeNoEstructuradoError) as excinfo:
            ExtractorJson().extraer("quiero programar con 12 GB")
        assert excinfo.value.errores
        assert {"loc", "msg", "type"} <= set(excinfo.value.errores[0])

    def test_wrong_field_type_names_the_field(self) -> None:
        with pytest.raises(MensajeNoEstructuradoError) as excinfo:
            ExtractorJson().extraer('{"tarea": "codigo", "vram_gb": "many"}')
        assert excinfo.value.errores[0]["loc"] == ["vram_gb"]

    def test_unknown_task_rejected(self) -> None:
        with pytest.raises(MensajeNoEstructuradoError):
            ExtractorJson().extraer('{"tarea": "embeddings", "vram_gb": 8}')


# ---------------------------------------------------------------------------
# ExtractorOllama (fake runnable)
# ---------------------------------------------------------------------------

class TestExtractorOllama:
    def test_model_output_is_validated_and_merged(self) -> None:
        fake = _FakeLLM(salida=CriteriosExtraidos(tarea="codigo", vram_gb=12))
        c = ExtractorOllama(llm=fake).extraer("quiero programar con 12 GB")
        assert c == CriteriosBusqueda(tarea="codigo", vram_gb=12)

    def test_dict_output_is_accepted(self) -> None:
        fake = _FakeLLM(salida={"tarea": "chat", "vram_gb": 8, "idiomas_requeridos": ["es"]})
        c = ExtractorOllama(llm=fake).extraer("un asistente en español, 8 GB")
        assert c == CriteriosBusqueda(tarea="chat", vram_gb=8, idiomas_requeridos=["es"])

    def test_prompt_and_message_reach_the_model(self) -> None:
        fake = _FakeLLM(salida=CriteriosExtraidos())
        ExtractorOllama(llm=fake).extraer("hola")
        (mensajes,) = fake.llamadas
        assert mensajes[0].content == PROMPT_SISTEMA
        assert mensajes[1].content == "hola"

    def test_merges_over_base_for_follow_ups(self) -> None:
        fake = _FakeLLM(salida=CriteriosExtraidos(vram_gb=12))
        base = CriteriosBusqueda(tarea="razonamiento")
        assert ExtractorOllama(llm=fake).extraer("12 GB", base) == CriteriosBusqueda(
            tarea="razonamiento", vram_gb=12
        )

    def test_backend_failure_is_explicit(self) -> None:
        fake = _FakeLLM(error=ConnectionError("connection refused"))
        with pytest.raises(ExtraccionNoDisponibleError, match="ConnectionError"):
            ExtractorOllama(llm=fake).extraer("hola")

    def test_invalid_output_is_not_silently_accepted(self) -> None:
        fake = _FakeLLM(salida={"tarea": "imagen", "vram_gb": 8})
        with pytest.raises(ExtraccionNoDisponibleError, match="invalid structured output"):
            ExtractorOllama(llm=fake).extraer("genera imágenes")

    def test_construction_does_no_io_and_defers_langchain(self) -> None:
        extractor = ExtractorOllama(modelo="nope:0b", base_url="http://127.0.0.1:1")
        assert extractor._llm is None  # chain not built until first use

    def test_calentar_reports_readiness_without_raising(self) -> None:
        assert ExtractorOllama(llm=_FakeLLM(salida=CriteriosExtraidos())).calentar() is True
        caido = ExtractorOllama(llm=_FakeLLM(error=ConnectionError("down")))
        assert caido.calentar() is False

    def test_unreachable_server_yields_no_disponible(self) -> None:
        """Real chain against a closed port: must map to ExtraccionNoDisponibleError."""
        extractor = ExtractorOllama(modelo="nope:0b", base_url="http://127.0.0.1:9")
        with pytest.raises(ExtraccionNoDisponibleError):
            extractor.extraer("hola")


# ---------------------------------------------------------------------------
# ExtractorCompuesto
# ---------------------------------------------------------------------------

class TestExtractorCompuesto:
    def test_json_short_circuits_without_calling_llm(self) -> None:
        fake = _FakeLLM(salida=CriteriosExtraidos(tarea="chat"))
        compuesto = ExtractorCompuesto(ExtractorJson(), ExtractorOllama(llm=fake))
        c = compuesto.extraer('{"tarea": "codigo", "vram_gb": 8}')
        assert c.tarea == "codigo"
        assert fake.llamadas == []

    def test_free_text_goes_to_llm(self) -> None:
        fake = _FakeLLM(salida=CriteriosExtraidos(tarea="chat", vram_gb=6))
        compuesto = ExtractorCompuesto(ExtractorJson(), ExtractorOllama(llm=fake))
        assert compuesto.extraer("un chat con 6 GB").vram_gb == 6
        assert len(fake.llamadas) == 1

    def test_without_llm_free_text_is_a_client_error(self) -> None:
        compuesto = ExtractorCompuesto(ExtractorJson(), None)
        assert compuesto.acepta_texto_libre is False
        with pytest.raises(MensajeNoEstructuradoError):
            compuesto.extraer("un chat con 6 GB")

    def test_llm_failure_propagates(self) -> None:
        fake = _FakeLLM(error=TimeoutError("slow"))
        compuesto = ExtractorCompuesto(ExtractorJson(), ExtractorOllama(llm=fake))
        with pytest.raises(ExtraccionNoDisponibleError):
            compuesto.extraer("un chat con 6 GB")

    def test_all_implementations_satisfy_protocol(self) -> None:
        assert isinstance(ExtractorJson(), ExtractorCriterios)
        assert isinstance(ExtractorOllama(llm=_FakeLLM()), ExtractorCriterios)
        assert isinstance(ExtractorCompuesto(ExtractorJson(), None), ExtractorCriterios)
