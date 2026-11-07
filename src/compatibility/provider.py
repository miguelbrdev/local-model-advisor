"""VRAM compatibility provider interface and implementations.

``VRAMCompatibilityProvider`` is the contract the filter consumes: ``check``
never raises and always returns a ``CompatibilityResult`` whose ``fuente``
says where the verdict came from.

Production routing (:func:`elegir_provider`) is ``ProveedorConFallback``:
canirun.ai first, and the offline parameter-count heuristic when canirun.ai
is unreachable or has no entry for a family.  Every estimated verdict is
labelled ``fuente="heuristica_propia"`` so the response can say which
candidates were measured and which were estimated; nothing is silently
invented.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from src.compatibility.cache import CACHE_CANIRUN, CacheCompatibilidad
from src.compatibility.heuristica import HeuristicaParametrosProvider
from src.models.schemas import CompatibilityResult, FamiliaModelo


@runtime_checkable
class VRAMCompatibilityProvider(Protocol):
    """Protocol for VRAM compatibility checkers."""

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult: ...


CANIRUN_API_BASE = "https://canirun.ai/api"
CANIRUN_TIMEOUT = 10.0


class ProveedorExternoError(Exception):
    """canirun.ai could not be consulted: network, timeout or HTTP failure.

    Raised by :meth:`CanIRunProvider.consultar` only; :meth:`check` keeps
    the never-raise contract and turns it into a ``no_disponible`` result.
    A fallback wrapper uses the exception to tell an *outage* (stop calling
    the service for the rest of the request) from a per-family gap.
    """


def _variante_por_defecto(familia: FamiliaModelo, *, ultima: bool = False) -> str:
    """Label of the variant a provider reports when it cannot choose one.

    Synced families carry no nested ``variantes`` (each canIRun.ai record
    is a single checkpoint), so the parameter-count text (``"7B"``) is the
    only meaningful label left; ``"unknown"`` is the last resort.
    """
    if familia.variantes:
        return familia.variantes[-1] if ultima else familia.variantes[0]
    return familia.parametros_texto or "unknown"


class CanIRunProvider:
    """Checks VRAM compatibility via the canirun.ai public API.

    Used for chat, code, and reasoning tasks.  Maps ``familia.canirun_id``
    to the canirun.ai model identifier.  ``runtime`` is accepted for
    Protocol compatibility but not sent: the compatibility endpoint has no
    runtime parameter in the observed contract.

    Args:
        cache: Measured results are read from and written to it, keyed by
            ``(canirun_id, vram_gb)``; ``None`` disables caching (the
            default for direct construction, so tests see every call).
            Production routing passes the process-wide ``CACHE_CANIRUN``.
    """

    def __init__(self, cache: CacheCompatibilidad | None = None) -> None:
        self._cache = cache

    def consultar(self, familia: FamiliaModelo, vram_gb: float) -> CompatibilityResult:
        """Query canirun.ai for one family.

        Returns:
            The measured verdict (``fuente="canirun_ai"``), or a
            ``no_disponible`` result when the family has no ``canirun_id``
            — that is a gap in the data, not a service failure.

        Raises:
            ProveedorExternoError: On any transport, timeout, HTTP or
                decoding failure, so callers can treat it as an outage.
        """
        canirun_id = familia.canirun_id
        if not canirun_id:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=_variante_por_defecto(familia),
                compatible=None,
                fuente="no_disponible",
                nota="No canirun_id mapping configured for this family.",
            )

        if self._cache is not None:
            cacheado = self._cache.obtener(canirun_id, vram_gb)
            if cacheado is not None:
                return cacheado.model_copy(
                    update={"modelo_id": familia.familia_id}
                )

        try:
            response = httpx.post(
                f"{CANIRUN_API_BASE}/compatibility",
                json={
                    "hardware": {"ramGb": int(vram_gb * 2), "gpu": {"name": "Custom"}},
                    "modelId": canirun_id,
                    "quantization": "Q4_K_M",
                },
                timeout=CANIRUN_TIMEOUT,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise ProveedorExternoError(f"canirun.ai API error: {exc}") from exc

        estimado = data.get("estimated", {})
        resultado = CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=_variante_por_defecto(familia, ultima=True),
            compatible=data.get("compatible"),
            grado=data.get("status"),
            fuente="canirun_ai",
            nota="; ".join(data.get("notes", [])),
            tokens_per_second=estimado.get("tokensPerSecond"),
            vram_required_gb=estimado.get("vramRequiredGb"),
        )
        if self._cache is not None:
            self._cache.guardar(canirun_id, vram_gb, resultado)
        return resultado

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        """Never-raising variant of :meth:`consultar`."""
        try:
            return self.consultar(familia, vram_gb)
        except ProveedorExternoError as exc:
            return CompatibilityResult(
                modelo_id=familia.familia_id,
                variante=_variante_por_defecto(familia, ultima=True),
                compatible=None,
                fuente="no_disponible",
                nota=str(exc),
            )


class ProveedorConFallback:
    """canirun.ai first, offline heuristic when it cannot answer.

    Two situations trigger the fallback, and the result says which:

    * **Outage** — :meth:`CanIRunProvider.consultar` raised.  The service
      is not asked again for the rest of this provider's life (one
      instance per pipeline run), so a down service costs one timeout per
      request instead of one per candidate.
    * **Gap** — canirun.ai answered ``no_disponible`` for this family (no
      ``canirun_id``).  Only this family is estimated; the next one asks
      the service again.

    When the heuristic cannot estimate either (unknown parameter count),
    the honest ``no_disponible`` result is returned unchanged.

    Args:
        primario: Measured source.
        respaldo: Offline estimator; defaults to the parameter heuristic.
    """

    def __init__(
        self,
        primario: CanIRunProvider | None = None,
        respaldo: VRAMCompatibilityProvider | None = None,
    ) -> None:
        self._primario = primario or CanIRunProvider(cache=CACHE_CANIRUN)
        self._respaldo = respaldo or HeuristicaParametrosProvider()
        self._motivo_caida: str | None = None

    @property
    def primario_caido(self) -> bool:
        """Whether canirun.ai failed during this provider's life."""
        return self._motivo_caida is not None

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        medido: CompatibilityResult | None = None
        if not self.primario_caido:
            try:
                medido = self._primario.consultar(familia, vram_gb)
            except ProveedorExternoError as exc:
                self._motivo_caida = str(exc)
        if medido is not None and medido.fuente != "no_disponible":
            return medido

        estimado = self._respaldo.check(familia, vram_gb, runtime)
        if estimado.compatible is None:
            # Neither source can judge: keep the measured source's own
            # explanation when there is one, never upgrade an unknown.
            return medido if medido is not None else estimado

        motivo = self._motivo_caida or (medido.nota if medido else "canirun.ai unavailable")
        return estimado.model_copy(
            update={"nota": f"Estimated locally ({motivo}). {estimado.nota}"}
        )


class TamanoDirectoProvider:
    """Fallback heuristic for small models (embeddings/rerankers).

    These models are small (hundreds of MB to ~2 GB) and rarely VRAM-bound,
    so they are reported as compatible without an external call.  Not
    reachable from the v1 API, whose tasks all route to the canirun.ai
    provider with heuristic fallback.
    """

    def check(
        self, familia: FamiliaModelo, vram_gb: float, runtime: str | None = None
    ) -> CompatibilityResult:
        return CompatibilityResult(
            modelo_id=familia.familia_id,
            variante=_variante_por_defecto(familia),
            compatible=True,
            grado="runs_well",
            fuente="heuristica_propia",
            nota="Size-based heuristic for small models (embeddings/rerankers).",
        )


def elegir_provider(tarea: str) -> VRAMCompatibilityProvider:
    """Route to the right VRAM provider based on task category.

    A fresh ``ProveedorConFallback`` per call, so its outage flag is scoped
    to one pipeline run and a recovered service is used again on the next
    request.
    """
    if tarea in ("chat", "codigo", "razonamiento"):
        return ProveedorConFallback()
    return TamanoDirectoProvider()
