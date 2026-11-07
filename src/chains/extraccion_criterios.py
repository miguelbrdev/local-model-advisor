"""Turn an incoming chat message into validated ``CriteriosBusqueda``.

Two ways in, one contract out:

* :class:`ExtractorJson` — the message *is* a JSON document with
  ``CriteriosBusqueda`` fields.  Deterministic, no LLM, always available.
* :class:`ExtractorOllama` — free text ("quiero programar en Python con
  12 GB de VRAM") interpreted by a local Ollama model through LangChain's
  structured-output API.  The model only fills the optional fields of
  :class:`CriteriosExtraidos`; everything it returns is re-validated with
  Pydantic and merged into the criteria already known for the
  conversation.  It never sees the catalogue, never scores and never
  picks candidates (AGENTS.md: LLM only maps text to criteria).

:class:`ExtractorCompuesto` tries JSON first (cheap, exact) and only then
the LLM, so structured clients never pay for an LLM call and the API keeps
working when Ollama is down: a free-text message then yields
:class:`ExtraccionNoDisponibleError`, which the API reports as an explicit 503
instead of inventing criteria.

Privacy: Ollama runs on localhost; the prompt and the message never leave
the machine.  Nothing in this module reads the catalogue.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.models.schemas import CriteriosBusqueda

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
"""Gemma 4 (4B effective, ~6.6 GB at Q4): strong Spanish, reliable JSON-schema
output.  ``gemma4:e2b`` is the lighter choice for small GPUs."""

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

KEEP_ALIVE_OLLAMA = -1
"""Ask Ollama to keep the model loaded indefinitely: the first request after
an unload paid ~100 s of model load on an RTX 2080 Ti."""

PROMPT_SISTEMA = """\
You extract search criteria for choosing a local LLM from the user's message.
Fill ONLY what the user states explicitly; leave everything else null or empty.
Never guess numbers: if the user names a GPU without saying its memory, leave
vram_gb null. Never invent languages or licences.

Fields:
- tarea: "chat" (assistant, conversation, writing), "codigo" (programming,
  code, software), or "razonamiento" (reasoning, maths, logic, step-by-step
  problem solving). Null if unclear.
- vram_gb: GPU memory in gigabytes as a number (e.g. "12 GB" -> 12).
- idiomas_requeridos: ISO 639-1 codes of languages the model must handle
  (e.g. "español e inglés" -> ["es", "en"]). Empty list if not stated.
- runtime_preferido: runtime named by the user (e.g. "ollama", "llama.cpp",
  "lm studio", "vllm"), lowercase. Null if not stated.
- requiere_licencia_comercial: true only if the user says they need
  commercial/business use; false if they say it is personal; null if not stated.

The user may write in Spanish or English. Respond only with the structured object.
"""


ALIAS_IDIOMAS: dict[str, str] = {
    "español": "es", "espanol": "es", "spanish": "es", "castellano": "es",
    "inglés": "en", "ingles": "en", "english": "en",
    "francés": "fr", "frances": "fr", "french": "fr",
    "alemán": "de", "aleman": "de", "german": "de",
    "italiano": "it", "italian": "it",
    "portugués": "pt", "portugues": "pt", "portuguese": "pt",
    "catalán": "ca", "catalan": "ca", "català": "ca",
    "euskera": "eu", "vasco": "eu", "basque": "eu",
    "gallego": "gl", "galician": "gl", "galego": "gl",
    "chino": "zh", "chinese": "zh", "mandarín": "zh", "mandarin": "zh",
    "japonés": "ja", "japones": "ja", "japanese": "ja",
    "coreano": "ko", "korean": "ko",
    "ruso": "ru", "russian": "ru",
    "árabe": "ar", "arabe": "ar", "arabic": "ar",
    "hindi": "hi",
    "neerlandés": "nl", "neerlandes": "nl", "holandés": "nl", "dutch": "nl",
    "polaco": "pl", "polish": "pl",
    "turco": "tr", "turkish": "tr",
}
"""Language names (Spanish and English) → ISO 639-1.  Small models often
answer with the name instead of the code the prompt asks for."""


def normalizar_idiomas(valores: list[str]) -> list[str]:
    """Lower-case, map known names to ISO codes, drop blanks and duplicates."""
    resultado: list[str] = []
    for valor in valores:
        clave = str(valor).strip().lower()
        if not clave:
            continue
        codigo = ALIAS_IDIOMAS.get(clave, clave)
        if codigo not in resultado:
            resultado.append(codigo)
    return resultado


class CriteriosExtraidos(BaseModel):
    """What an extractor may fill in from one message.  Every field optional.

    Mirrors ``CriteriosBusqueda`` but with ``requiere_licencia_comercial``
    nullable so "not stated" can be told apart from "false" when merging.
    Languages are normalised to ISO codes and the runtime to lower case,
    whether they come from the model or from a JSON client.
    """

    tarea: Literal["chat", "codigo", "razonamiento"] | None = None
    vram_gb: float | None = Field(default=None, ge=0)
    idiomas_requeridos: list[str] = Field(default_factory=list)
    runtime_preferido: str | None = None
    requiere_licencia_comercial: bool | None = None

    @field_validator("idiomas_requeridos", mode="after")
    @classmethod
    def _idiomas_iso(cls, valores: list[str]) -> list[str]:
        return normalizar_idiomas(valores)

    @field_validator("runtime_preferido", mode="after")
    @classmethod
    def _runtime_minusculas(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        limpio = " ".join(valor.strip().lower().split())
        return limpio or None

    def fusionar(self, base: CriteriosBusqueda | None) -> CriteriosBusqueda:
        """Overlay these values on ``base``: stated fields win, unstated keep.

        Args:
            base: Criteria already known for the conversation, or ``None``
                for a first message.

        Returns:
            A new, validated ``CriteriosBusqueda``.
        """
        previo = base.model_dump() if base is not None else {}
        nuevo = {k: v for k, v in self.model_dump().items() if v is not None and v != []}
        return CriteriosBusqueda.model_validate({**previo, **nuevo})


class MensajeNoEstructuradoError(Exception):
    """The message is not a ``CriteriosExtraidos`` JSON document.

    Attributes:
        errores: Pydantic error items (``loc``/``msg``/``type``) for the API
            to relay as a 422 when no language-model extractor is available.
    """

    def __init__(self, errores: list[dict[str, Any]]) -> None:
        super().__init__("message is not structured criteria JSON")
        self.errores = errores


class ExtraccionNoDisponibleError(Exception):
    """The language-model extractor could not produce valid criteria.

    Raised for an unreachable Ollama server, a model that is not pulled,
    or output that fails Pydantic validation.  Never swallowed into empty
    criteria: the API must tell the client to send structured JSON instead.
    """


@runtime_checkable
class ExtractorCriterios(Protocol):
    """Contract every extractor satisfies."""

    def extraer(
        self, mensaje: str, base: CriteriosBusqueda | None = None
    ) -> CriteriosBusqueda:
        """Return validated criteria for ``mensaje`` merged over ``base``.

        Raises:
            MensajeNoEstructuradoError: The message cannot be read as criteria
                by this extractor (only meaningful for deterministic ones).
            ExtraccionNoDisponibleError: The extractor's backend failed.
        """
        ...


class ExtractorJson:
    """Deterministic extractor for messages that are criteria JSON."""

    def extraer(
        self, mensaje: str, base: CriteriosBusqueda | None = None
    ) -> CriteriosBusqueda:
        try:
            extraidos = CriteriosExtraidos.model_validate_json(mensaje)
        except ValidationError as exc:
            raise MensajeNoEstructuradoError(
                [
                    {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors(include_url=False)
                ]
            ) from exc
        return extraidos.fusionar(base)


class ExtractorOllama:
    """Free-text extractor backed by a local Ollama model via LangChain.

    Args:
        modelo: Ollama model tag (e.g. ``"gemma4:e4b"``).
        base_url: Ollama server URL; must be local.
        llm: Injected structured-output runnable whose ``invoke(messages)``
            returns a :class:`CriteriosExtraidos` or a dict.  Tests pass a
            fake; production leaves it ``None`` and the ChatOllama chain is
            built lazily on first use, so constructing this class performs
            no I/O and does not import LangChain until needed.
    """

    def __init__(
        self,
        modelo: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        *,
        llm: Any | None = None,
    ) -> None:
        self.modelo = modelo
        self.base_url = base_url
        self._llm = llm

    def extraer(
        self, mensaje: str, base: CriteriosBusqueda | None = None
    ) -> CriteriosBusqueda:
        from langchain_core.messages import HumanMessage, SystemMessage

        mensajes = [SystemMessage(content=PROMPT_SISTEMA), HumanMessage(content=mensaje)]
        try:
            salida = self._runnable().invoke(mensajes)
        except Exception as exc:  # network, missing model, parser failure
            logger.warning(
                "Ollama extraction failed (%s @ %s): %s", self.modelo, self.base_url, exc
            )
            raise ExtraccionNoDisponibleError(f"{type(exc).__name__}: {exc}") from exc

        try:
            extraidos = (
                salida
                if isinstance(salida, CriteriosExtraidos)
                else CriteriosExtraidos.model_validate(salida)
            )
        except ValidationError as exc:
            logger.warning("Ollama returned criteria that failed validation: %s", exc)
            raise ExtraccionNoDisponibleError(f"invalid structured output: {exc}") from exc
        return extraidos.fusionar(base)

    def _runnable(self) -> Any:
        if self._llm is None:
            from langchain_ollama import ChatOllama

            chat = ChatOllama(
                model=self.modelo,
                base_url=self.base_url,
                temperature=0,
                keep_alive=KEEP_ALIVE_OLLAMA,
            )
            self._llm = chat.with_structured_output(CriteriosExtraidos, method="json_schema")
        return self._llm

    def calentar(self) -> bool:
        """Load the model now with a trivial extraction so the first user
        request does not pay the load time.  Never raises."""
        try:
            self.extraer("hola")
        except ExtraccionNoDisponibleError as exc:
            logger.warning("Warm-up of %s failed: %s", self.modelo, exc)
            return False
        return True


class ExtractorCompuesto:
    """JSON first, language model second.

    Args:
        json: Deterministic extractor, always consulted first.
        llm: Free-text extractor, or ``None`` when natural-language input
            is disabled; a non-JSON message then surfaces
            :class:`MensajeNoEstructuradoError` unchanged.
    """

    def __init__(self, json: ExtractorCriterios, llm: ExtractorCriterios | None) -> None:
        self._json = json
        self._llm = llm

    @property
    def acepta_texto_libre(self) -> bool:
        return self._llm is not None

    def calentar(self) -> bool:
        """Warm the language model, if any.  Returns whether it is ready."""
        calentar = getattr(self._llm, "calentar", None)
        return bool(calentar()) if callable(calentar) else False

    def extraer(
        self, mensaje: str, base: CriteriosBusqueda | None = None
    ) -> CriteriosBusqueda:
        try:
            return self._json.extraer(mensaje, base)
        except MensajeNoEstructuradoError:
            if self._llm is None:
                raise
        return self._llm.extraer(mensaje, base)
