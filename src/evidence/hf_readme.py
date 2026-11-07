"""Download a model's README from Hugging Face.

Data source: the ``url`` field of each canIRun.ai record, which for the
whole catalogue is ``https://huggingface.co/<org>/<repo>``.  The README is
fetched raw from the ``main`` revision.  Observed outcomes, each kept as
an explicit status instead of being collapsed into "no text":

* ``ok`` — public repository, README downloaded.
* ``gated`` — 401/403: licence-gated repositories (Meta, Google) refuse
  anonymous downloads.  No token is used on purpose: the sync must not
  depend on personal credentials.
* ``not_found`` — 404.
* ``error`` — any other HTTP status or transport failure.
* ``not_hf`` — the URL is not a Hugging Face repository.

Quantisation mirrors (e.g. ``lmstudio-community/*-GGUF``) ship a short
README that only credits the original model.  When the front matter tags
the repository as ``gguf`` and names a ``base_model``, that model's README
is fetched instead (one hop, never recursive) and the result records where
it was redirected from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import yaml

HF_HOST = "huggingface.co"
EstadoDescarga = Literal["ok", "gated", "not_found", "error", "not_hf"]


@dataclass(frozen=True)
class ReadmeDescargado:
    """Outcome of one README download.

    Attributes:
        url_repo: Repository URL the download was requested for.
        url_readme: Raw README URL actually fetched, or ``None`` for ``not_hf``.
        estado: One of :data:`EstadoDescarga`.
        texto: README body with the YAML front matter removed; empty unless ``ok``.
        front_matter: Parsed front matter mapping (``{}`` when absent/invalid).
        base_model: ``base_model`` declared in the front matter, if any.
        redirigido_desde: Mirror repository URL when the text comes from the
            base model instead, else ``None``.
        detalle: Short human-readable reason for non-``ok`` states.
    """

    url_repo: str
    url_readme: str | None
    estado: EstadoDescarga
    texto: str = ""
    front_matter: dict[str, Any] = field(default_factory=dict)
    base_model: str | None = None
    redirigido_desde: str | None = None
    detalle: str = ""


def url_readme_raw(url_repo: str) -> str | None:
    """Map a Hugging Face repository URL to its raw ``main`` README URL.

    Returns ``None`` when the URL is not ``huggingface.co/<org>/<repo>``.
    """
    partes = urlparse(url_repo.strip())
    if partes.netloc.lower().removeprefix("www.") != HF_HOST:
        return None
    segmentos = [s for s in partes.path.split("/") if s]
    if len(segmentos) < 2:
        return None
    org, repo = segmentos[0], segmentos[1]
    return f"https://{HF_HOST}/{org}/{repo}/raw/main/README.md"


def separar_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """Split a README into its YAML front matter and the body.

    Returns ``({}, markdown)`` when there is no front matter or it does not
    parse as a mapping; the body is never lost.
    """
    if not markdown.startswith("---"):
        return {}, markdown
    lineas = markdown.split("\n")
    for i in range(1, len(lineas)):
        if lineas[i].strip() == "---":
            bloque = "\n".join(lineas[1:i])
            cuerpo = "\n".join(lineas[i + 1:])
            try:
                meta = yaml.safe_load(bloque)
            except yaml.YAMLError:
                return {}, cuerpo
            return (meta if isinstance(meta, dict) else {}), cuerpo
    return {}, markdown


def base_model_de(front_matter: dict[str, Any]) -> str | None:
    """``base_model`` as ``org/repo``; lists take their first entry."""
    valor = front_matter.get("base_model")
    if isinstance(valor, list):
        valor = valor[0] if valor else None
    if not isinstance(valor, str) or valor.count("/") != 1:
        return None
    return valor.strip()


def es_espejo_cuantizado(front_matter: dict[str, Any]) -> bool:
    """A repository that only re-packages another model (``gguf`` tag + base_model)."""
    tags = front_matter.get("tags") or []
    if not isinstance(tags, list):
        return False
    return any(str(t).lower() == "gguf" for t in tags) and base_model_de(front_matter) is not None


def descargar_readme(
    url_repo: str,
    *,
    client: httpx.Client,
    seguir_base_model: bool = True,
) -> ReadmeDescargado:
    """Download the README for ``url_repo``.

    Args:
        url_repo: Hugging Face repository URL from the catalogue.
        client: HTTP client; tests inject one with ``httpx.MockTransport``.
        seguir_base_model: Follow ``base_model`` once for quantisation
            mirrors.  The recursive call passes ``False``.

    Returns:
        A :class:`ReadmeDescargado`; never raises for HTTP or transport
        problems, which become ``estado="error"``.
    """
    url = url_readme_raw(url_repo)
    if url is None:
        return ReadmeDescargado(
            url_repo, None, "not_hf", detalle="not a huggingface.co repository"
        )

    try:
        respuesta = client.get(url, follow_redirects=True)
    except httpx.HTTPError as exc:
        return ReadmeDescargado(url_repo, url, "error", detalle=f"{type(exc).__name__}: {exc}")

    if respuesta.status_code in (401, 403):
        return ReadmeDescargado(url_repo, url, "gated", detalle=f"HTTP {respuesta.status_code}")
    if respuesta.status_code == 404:
        return ReadmeDescargado(url_repo, url, "not_found", detalle="HTTP 404")
    if respuesta.status_code != 200:
        return ReadmeDescargado(url_repo, url, "error", detalle=f"HTTP {respuesta.status_code}")

    meta, cuerpo = separar_front_matter(respuesta.text)
    base = base_model_de(meta)
    propio = ReadmeDescargado(url_repo, url, "ok", cuerpo.strip(), meta, base)

    if seguir_base_model and es_espejo_cuantizado(meta):
        original = descargar_readme(
            f"https://{HF_HOST}/{base}", client=client, seguir_base_model=False
        )
        if original.estado == "ok":
            return ReadmeDescargado(
                url_repo=url_repo,
                url_readme=original.url_readme,
                estado="ok",
                texto=original.texto,
                front_matter=original.front_matter,
                base_model=base,
                redirigido_desde=url_repo,
            )
    return propio
