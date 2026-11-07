"""In-process TTL cache for measured canirun.ai compatibility results.

A verdict for ``(model, VRAM)`` does not change between requests, yet every
``/chat`` asked canirun.ai again for each candidate (about one second per
call).  Caching the *measured* results for a few hours removes most of
that latency for repeated budgets without ever serving an estimate as a
measurement: only ``fuente="canirun_ai"`` results are stored.

Thread-safe (the filter checks candidates concurrently).  Bounded by
``max_entradas`` with oldest-first eviction, so a long-running API cannot
grow without limit.  The clock is injectable for deterministic tests.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from src.models.schemas import CompatibilityResult

TTL_POR_DEFECTO_SEGUNDOS = 6 * 3600
MAX_ENTRADAS_POR_DEFECTO = 4096

Clave = tuple[str, float]


class CacheCompatibilidad:
    """TTL cache keyed by ``(canirun_id, vram_gb)``.

    Args:
        ttl_segundos: Lifetime of an entry.  ``0`` disables caching.
        max_entradas: Upper bound; the oldest entry is evicted beyond it.
        reloj: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        ttl_segundos: float = TTL_POR_DEFECTO_SEGUNDOS,
        max_entradas: int = MAX_ENTRADAS_POR_DEFECTO,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_segundos < 0 or max_entradas < 1:
            raise ValueError("ttl_segundos must be >= 0 and max_entradas >= 1")
        self._ttl = ttl_segundos
        self._max = max_entradas
        self._reloj = reloj
        self._entradas: OrderedDict[Clave, tuple[float, CompatibilityResult]] = OrderedDict()
        self._lock = threading.Lock()
        self.aciertos = 0
        self.fallos = 0

    @staticmethod
    def clave(canirun_id: str, vram_gb: float) -> Clave:
        return (canirun_id, round(float(vram_gb), 2))

    def obtener(self, canirun_id: str, vram_gb: float) -> CompatibilityResult | None:
        """Return a fresh cached result or ``None`` (expired entries are dropped)."""
        if self._ttl == 0:
            return None
        clave = self.clave(canirun_id, vram_gb)
        with self._lock:
            entrada = self._entradas.get(clave)
            if entrada is None:
                self.fallos += 1
                return None
            expira, resultado = entrada
            if self._reloj() >= expira:
                del self._entradas[clave]
                self.fallos += 1
                return None
            self.aciertos += 1
            return resultado

    def guardar(self, canirun_id: str, vram_gb: float, resultado: CompatibilityResult) -> None:
        """Store a *measured* result; estimates and unknowns are never cached."""
        if self._ttl == 0 or resultado.fuente != "canirun_ai":
            return
        clave = self.clave(canirun_id, vram_gb)
        with self._lock:
            self._entradas.pop(clave, None)
            self._entradas[clave] = (self._reloj() + self._ttl, resultado)
            while len(self._entradas) > self._max:
                self._entradas.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entradas)

    def vaciar(self) -> None:
        with self._lock:
            self._entradas.clear()


CACHE_CANIRUN = CacheCompatibilidad()
"""Process-wide cache shared by every request's provider."""
