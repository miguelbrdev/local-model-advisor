"""Deterministic splitting of a README into fragments for indexing.

Model cards mix prose, tables, badges, HTML and usage code.  Only prose
explains what a model is good for, so the cleaner drops fenced code,
HTML tags, images and badge links, then the splitter cuts the text at
headings and paragraphs into fragments of at most ``tamano`` characters,
each prefixed with its section heading so a fragment read on its own still
says what it is about.  Same input, same fragments: the point ids in the
vector store derive from the fragment index.
"""

from __future__ import annotations

import re

TAMANO_FRAGMENTO = 800
"""Maximum characters per fragment."""

FRAGMENTO_MINIMO = 40
"""Fragments shorter than this (after cleaning) carry no explanation."""

SECCIONES_OMITIDAS = (
    "quickstart", "quick start", "getting started", "usage", "how to use", "how to run",
    "installation", "install", "requirements", "deployment", "deploy", "inference",
    "running", "download", "citation", "cite", "bibtex", "license", "licence",
    "acknowledg", "special thanks", "community model", "contact", "changelog",
    "table of contents", "sponsor", "vllm", "transformers", "llama.cpp", "ollama",
    "lm studio", "serving", "serve", "settings", "sampling", "api", "sdk", "prompt format",
    "chat template", "tokenizer",
)
"""Section headings (case-insensitive substrings) whose content is operational
boilerplate — install commands, API calls, citations — rather than an
explanation of what the model is good for.  Indexing them produced the
generic "Quickstart" fragments that surfaced for every query."""

FRACCION_TABLA_MAXIMA = 0.4
"""Fragments where more than this share of lines are table rows are
benchmark dumps, unreadable as prose evidence."""

_BLOQUE_CODIGO = re.compile(r"```.*?```", re.S)
_HTML = re.compile(r"<[^>]+>")
_IMAGEN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_ENLACE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_ESPACIOS = re.compile(r"[ \t]+")
_LINEAS_VACIAS = re.compile(r"\n{3,}")
_ENCABEZADO = re.compile(r"^(#{1,6})\s+(.*)$")


def limpiar_markdown(texto: str) -> str:
    """Strip code, HTML, images and link targets; keep headings and prose."""
    texto = _BLOQUE_CODIGO.sub("", texto)
    texto = _IMAGEN.sub("", texto)
    texto = _HTML.sub("", texto)
    texto = _ENLACE.sub(r"\1", texto)
    lineas = [_ESPACIOS.sub(" ", linea).strip() for linea in texto.split("\n")]
    lineas = [linea for linea in lineas if not _es_ruido(linea)]
    return _LINEAS_VACIAS.sub("\n\n", "\n".join(lineas)).strip()


def _es_ruido(linea: str) -> bool:
    """Table rules, horizontal rules and badge leftovers add nothing."""
    sin = linea.replace("|", "").replace("-", "").replace(":", "").strip()
    return bool(linea) and not sin


def trocear(texto: str, *, tamano: int = TAMANO_FRAGMENTO) -> list[str]:
    """Split cleaned markdown into fragments of at most ``tamano`` characters.

    Sections start at every heading; the heading text prefixes each
    fragment of its section.  Paragraphs are packed greedily; a single
    paragraph longer than ``tamano`` is cut at sentence boundaries, then
    hard-cut as a last resort.  Fragments below :data:`FRAGMENTO_MINIMO`
    characters are dropped.
    """
    if tamano <= 0:
        raise ValueError(f"tamano must be > 0, got {tamano}")
    fragmentos: list[str] = []
    for encabezado, parrafos in _secciones(limpiar_markdown(texto)):
        if seccion_omitida(encabezado):
            continue
        prefijo = f"{encabezado}: " if encabezado else ""
        for cuerpo in _empaquetar(parrafos, tamano - len(prefijo)):
            fragmento = prefijo + cuerpo
            if len(fragmento.strip()) >= FRAGMENTO_MINIMO and not _es_tabla(cuerpo):
                fragmentos.append(fragmento)
    return fragmentos


def seccion_omitida(encabezado: str) -> bool:
    """Whether a section heading marks operational boilerplate."""
    titulo = encabezado.lower()
    return any(clave in titulo for clave in SECCIONES_OMITIDAS)


def _es_tabla(cuerpo: str) -> bool:
    lineas = [linea for linea in cuerpo.split("\n") if linea.strip()]
    if not lineas:
        return False
    tabla = sum(1 for linea in lineas if linea.count("|") >= 2)
    return tabla / len(lineas) > FRACCION_TABLA_MAXIMA


def _secciones(texto: str) -> list[tuple[str, list[str]]]:
    secciones: list[tuple[str, list[str]]] = []
    encabezado = ""
    actual: list[str] = []
    for bloque in texto.split("\n\n"):
        bloque = bloque.strip()
        if not bloque:
            continue
        lineas = bloque.split("\n")
        m = _ENCABEZADO.match(lineas[0])
        if m:
            if actual:
                secciones.append((encabezado, actual))
            encabezado = m.group(2).strip()
            actual = []
            resto = "\n".join(lineas[1:]).strip()
            if resto:
                actual.append(resto)
        else:
            actual.append(bloque)
    if actual:
        secciones.append((encabezado, actual))
    return secciones


def _empaquetar(parrafos: list[str], limite: int) -> list[str]:
    limite = max(limite, FRAGMENTO_MINIMO)
    piezas: list[str] = []
    for parrafo in parrafos:
        piezas.extend(_partir(parrafo, limite))
    paquetes: list[str] = []
    actual = ""
    for pieza in piezas:
        candidato = f"{actual}\n\n{pieza}" if actual else pieza
        if len(candidato) <= limite:
            actual = candidato
        else:
            if actual:
                paquetes.append(actual)
            actual = pieza
    if actual:
        paquetes.append(actual)
    return paquetes


def _partir(parrafo: str, limite: int) -> list[str]:
    """Cut an over-long paragraph at sentence ends, then hard-cut."""
    if len(parrafo) <= limite:
        return [parrafo]
    frases = re.split(r"(?<=[.!?])\s+", parrafo)
    trozos: list[str] = []
    actual = ""
    for frase in frases:
        while len(frase) > limite:
            trozos.append(frase[:limite])
            frase = frase[limite:]
        candidato = f"{actual} {frase}".strip() if actual else frase
        if len(candidato) <= limite:
            actual = candidato
        else:
            if actual:
                trozos.append(actual)
            actual = frase
    if actual:
        trozos.append(actual)
    return trozos
