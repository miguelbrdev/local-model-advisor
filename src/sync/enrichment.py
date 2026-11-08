"""Enrich normalised families with languages and runtimes from approved sources.

Runs inside the catalogue synchronisation, after normalisation and before
the snapshot is hashed, never during a request.  Three lookups per family,
each independent and each tolerant to failure (a family that cannot be
enriched keeps its fields empty and the snapshot records how many):

* **canIRun detail** ``GET /api/models/{id}`` (contract observed
  2026-11-08): ``ollamaId`` and ``lmStudioId`` name the model in those
  runtimes' libraries; the record also carries ``quants``, ``tools``,
  ``thinking`` and ``activeParams``, which this step does not use yet.
* **Hugging Face model card** ``GET /api/models/{org}/{repo}`` (public,
  also for gated repositories): ``cardData.language`` is the only
  structured language declaration available anywhere for these models,
  so ``idiomas_detectados`` comes from it and from nothing else — never
  from README prose.  Quantised mirrors (``gguf`` tag + ``base_model``)
  are followed one hop to the original repository, like the README
  fetcher does.
* **Hugging Face model tree** ``GET /api/models?filter=base_model:quantized:{repo}``:
  the community's quantised derivatives, whose tags (``gguf``, ``mlx``,
  ``awq``, …) say which formats exist and therefore which runtimes can
  load the model.  The format → runtime table lives in the versioned
  policy (``enrichment.runtime_by_format``).

Nothing here guesses: a language is stored only if the card declares it,
a runtime only if the source names it or publishes a format for it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.chains.extraccion_criterios import normalizar_idiomas
from src.models.schemas import FamiliaModelo
from src.sync.canirun_schemas import CanIRunModelDetail
from src.sync.catalog_policy import CatalogPolicy

logger = logging.getLogger(__name__)

FUENTE_CANIRUN = "canirun_detail"
FUENTE_HF_CARD = "huggingface_card"
FUENTE_HF_TREE = "huggingface_tree"
FUENTES = (FUENTE_CANIRUN, FUENTE_HF_CARD, FUENTE_HF_TREE)

ESTADOS = ("ok", "sin_datos", "error")
"""Per-source outcome for one family: data found, source answered without
usable data (404, empty card, empty tree), or transport/contract failure."""

MAX_WORKERS = 8
HF_HOST = "huggingface.co"
_USER_AGENT = "local-model-advisor (catalogue sync)"


class _HFCardData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    language: list[Any] | str | None = None
    base_model: list[Any] | str | None = None
    tags: list[Any] | None = None


class _HFModel(BaseModel):
    """Subset of ``GET /api/models/{repo}`` read here."""

    model_config = ConfigDict(extra="ignore")

    id: str
    cardData: _HFCardData | None = None  # noqa: N815 — external field name
    tags: list[str] = Field(default_factory=list)
    library_name: str | None = None


class _HFTreeEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tags: list[str] = Field(default_factory=list)
    library_name: str | None = None


@dataclass
class InformeEnriquecimiento:
    """Outcome counts per source, stored in the snapshot header."""

    contadores: dict[str, dict[str, int]] = field(
        default_factory=lambda: {f: dict.fromkeys(ESTADOS, 0) for f in FUENTES}
    )

    def anotar(self, fuente: str, estado: str) -> None:
        self.contadores[fuente][estado] += 1


def repo_de_url(url: str | None) -> str | None:
    """``org/repo`` from a ``huggingface.co`` URL, else ``None``."""
    if not url:
        return None
    partes = urlparse(url.strip())
    if partes.netloc.lower().removeprefix("www.") != HF_HOST:
        return None
    segmentos = [s for s in partes.path.split("/") if s]
    if len(segmentos) < 2:
        return None
    return f"{segmentos[0]}/{segmentos[1]}"


def enriquecer_familias(
    familias: list[FamiliaModelo],
    policy: CatalogPolicy,
    *,
    client: httpx.Client,
    max_workers: int = MAX_WORKERS,
) -> tuple[list[FamiliaModelo], InformeEnriquecimiento]:
    """Enrich every family concurrently; order is preserved.

    Args:
        familias: Normalised families (not mutated).
        policy: Provides endpoints and the format/field → runtime tables.
        client: Shared HTTP client (tests inject a mock transport).
        max_workers: Concurrent lookups; bounded to be polite to both APIs.

    Returns:
        New family instances plus the per-source outcome counts.
    """
    informe = InformeEnriquecimiento()
    enriquecedor = _Enriquecedor(policy, client)
    if not familias:
        return [], informe
    with ThreadPoolExecutor(max_workers=min(max_workers, len(familias))) as pool:
        resultados = list(pool.map(enriquecedor.enriquecer, familias))
    enriquecidas: list[FamiliaModelo] = []
    for familia, estados in resultados:
        enriquecidas.append(familia)
        for fuente, estado in estados.items():
            informe.anotar(fuente, estado)
    logger.info("Enrichment: %s", informe.contadores)
    return enriquecidas, informe


class _Enriquecedor:
    def __init__(self, policy: CatalogPolicy, client: httpx.Client) -> None:
        self._policy = policy
        self._client = client

    def enriquecer(self, familia: FamiliaModelo) -> tuple[FamiliaModelo, dict[str, str]]:
        estados: dict[str, str] = {}
        runtimes: set[str] = set()
        formatos: set[str] = set()
        idiomas: list[str] | None = None
        ollama_id = lmstudio_id = None

        # --- canIRun detail -------------------------------------------------
        detalle, estados[FUENTE_CANIRUN] = self._detalle_canirun(familia.canirun_id)
        if detalle is not None:
            ollama_id = detalle.ollamaId or None
            lmstudio_id = detalle.lmStudioId or None
            for campo, runtime in self._policy.runtime_by_canirun_field.items():
                if getattr(detalle, campo, None):
                    runtimes.add(runtime)

        # --- Hugging Face card ------------------------------------------------
        repo = repo_de_url(familia.license_url)
        canonico = repo
        tarjeta, estados[FUENTE_HF_CARD] = self._tarjeta_hf(repo)
        if tarjeta is not None:
            espejo, base = tarjeta
            formatos |= self._formatos_de(espejo.tags, espejo.library_name, espejo.id)
            fuente_idiomas = base or espejo
            idiomas = _idiomas_de(fuente_idiomas.cardData) or None
            if base is not None:
                canonico = base.id
                formatos |= self._formatos_de(base.tags, base.library_name, base.id)
            if idiomas is None and espejo is not fuente_idiomas:
                idiomas = _idiomas_de(espejo.cardData) or None
            if idiomas is None and estados[FUENTE_HF_CARD] == "ok":
                estados[FUENTE_HF_CARD] = "sin_datos"

        # --- Hugging Face tree ------------------------------------------------
        arbol, estados[FUENTE_HF_TREE] = self._arbol_hf(canonico)
        for entrada in arbol:
            formatos |= self._formatos_de(entrada.tags, entrada.library_name, entrada.id)

        for formato in formatos:
            runtimes.update(self._policy.runtime_by_format.get(formato, []))

        nueva = familia.model_copy(
            update={
                "idiomas_detectados": idiomas,
                "formatos_disponibles": sorted(formatos),
                "runtimes_disponibles": sorted(runtimes),
                "ollama_id": ollama_id,
                "lmstudio_id": lmstudio_id,
                "fuentes_enriquecimiento": [f for f in FUENTES if estados.get(f) == "ok"],
            }
        )
        return nueva, estados

    # ------------------------------------------------------------------
    # Lookups: each returns (data, estado) and never raises
    # ------------------------------------------------------------------

    def _detalle_canirun(self, canirun_id: str | None) -> tuple[CanIRunModelDetail | None, str]:
        plantilla = self._policy.canirun_detail_endpoint
        if not canirun_id or not plantilla:
            return None, "sin_datos"
        respuesta, estado = self._get(plantilla.format(id=canirun_id))
        if respuesta is None:
            return None, estado
        try:
            return CanIRunModelDetail.model_validate_json(respuesta.content), "ok"
        except ValidationError as exc:
            logger.warning("canIRun detail for %s rejected: %s", canirun_id, exc)
            return None, "error"

    def _tarjeta_hf(self, repo: str | None) -> tuple[tuple[_HFModel, _HFModel | None] | None, str]:
        """The repository's card and, for a quantised mirror, its base model's card."""
        if not repo or not self._policy.huggingface_api:
            return None, "sin_datos"
        modelo, estado = self._modelo_hf(repo)
        if modelo is None:
            return None, estado
        base = None
        base_repo = _base_model_de(modelo.cardData)
        if base_repo and _es_espejo(modelo) and base_repo != modelo.id:
            base, _ = self._modelo_hf(base_repo)
        return (modelo, base), "ok"

    def _modelo_hf(self, repo: str) -> tuple[_HFModel | None, str]:
        respuesta, estado = self._get(f"{self._policy.huggingface_api}/{repo}")
        if respuesta is None:
            return None, estado
        try:
            return _HFModel.model_validate_json(respuesta.content), "ok"
        except ValidationError as exc:
            logger.warning("Hugging Face card for %s rejected: %s", repo, exc)
            return None, "error"

    def _arbol_hf(self, repo: str | None) -> tuple[list[_HFTreeEntry], str]:
        if not repo or not self._policy.huggingface_api:
            return [], "sin_datos"
        respuesta, estado = self._get(
            self._policy.huggingface_api,
            params={"filter": f"base_model:quantized:{repo}", "limit": "200"},
        )
        if respuesta is None:
            return [], estado
        try:
            entradas = [_HFTreeEntry.model_validate(e) for e in respuesta.json()]
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Hugging Face tree for %s rejected: %s", repo, exc)
            return [], "error"
        return entradas, ("ok" if entradas else "sin_datos")

    def _get(
        self, url: str, params: dict[str, str] | None = None
    ) -> tuple[httpx.Response | None, str]:
        try:
            respuesta = self._client.get(url, params=params, headers={"User-Agent": _USER_AGENT})
        except httpx.HTTPError as exc:
            logger.warning("Enrichment request failed for %s: %s", url, exc)
            return None, "error"
        if respuesta.status_code == 404:
            return None, "sin_datos"
        if respuesta.status_code != 200:
            logger.warning(
                "Enrichment request for %s returned HTTP %s", url, respuesta.status_code
            )
            return None, "error"
        return respuesta, "ok"

    def _formatos_de(self, tags: list[str], library: str | None, repo_id: str) -> set[str]:
        conocidos = set(self._policy.runtime_by_format)
        encontrados = {t.lower() for t in tags if t.lower() in conocidos}
        if library and library.lower() in conocidos:
            encontrados.add(library.lower())
        if "gguf" in conocidos and "gguf" in repo_id.lower():
            encontrados.add("gguf")
        return encontrados


def _idiomas_de(card: _HFCardData | None) -> list[str]:
    """ISO codes declared in ``cardData.language`` (two/three-letter tokens only)."""
    if card is None or card.language is None:
        return []
    valores = [card.language] if isinstance(card.language, str) else card.language
    codigos = normalizar_idiomas([str(v) for v in valores])
    return [c for c in codigos if 2 <= len(c) <= 3 and c.isalpha()]


def _base_model_de(card: _HFCardData | None) -> str | None:
    if card is None:
        return None
    valor = card.base_model
    if isinstance(valor, list):
        valor = valor[0] if valor else None
    if not isinstance(valor, str) or valor.count("/") != 1:
        return None
    return valor.strip()


def _es_espejo(modelo: _HFModel) -> bool:
    """A repository that only re-packages another model (gguf/mlx tag + base_model)."""
    etiquetas = {t.lower() for t in modelo.tags}
    if modelo.cardData and isinstance(modelo.cardData.tags, list):
        etiquetas |= {str(t).lower() for t in modelo.cardData.tags}
    libreria = (modelo.library_name or "").lower()
    return bool(etiquetas & {"gguf", "mlx"}) or libreria in {"gguf", "mlx"}
