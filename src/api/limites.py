"""Request guards for the API: per-client rate limiting and an optional API key.

Both are deliberately dependency-free and in-process: this service runs on
one machine for one user or a small team, so a fixed-window counter per
client address and a shared secret header are proportionate.  They exist
so that exposing the port beyond ``127.0.0.1`` is not immediately unsafe,
not as a substitute for a real gateway.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable

RATE_LIMIT_POR_DEFECTO = 60
"""Requests per client per minute allowed on ``/chat`` (``0`` disables)."""

VENTANA_SEGUNDOS = 60.0


class LimitadorTasa:
    """Fixed-window request counter per client key.

    Args:
        maximo_por_ventana: Requests allowed per key within a window;
            ``0`` means unlimited.
        ventana_segundos: Window length.
        reloj: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        maximo_por_ventana: int = RATE_LIMIT_POR_DEFECTO,
        ventana_segundos: float = VENTANA_SEGUNDOS,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        if maximo_por_ventana < 0 or ventana_segundos <= 0:
            raise ValueError("maximo_por_ventana must be >= 0 and ventana_segundos > 0")
        self._maximo = maximo_por_ventana
        self._ventana = ventana_segundos
        self._reloj = reloj
        self._ventanas: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    @property
    def activo(self) -> bool:
        return self._maximo > 0

    def admitir(self, clave: str) -> tuple[bool, float]:
        """Register one request for ``clave``.

        Returns:
            ``(admitida, segundos_hasta_reinicio)``.  The second value is
            how long the client should wait when refused (``0`` otherwise).
        """
        if not self.activo:
            return True, 0.0
        ahora = self._reloj()
        with self._lock:
            inicio, cuenta = self._ventanas.get(clave, (ahora, 0))
            if ahora - inicio >= self._ventana:
                inicio, cuenta = ahora, 0
            if cuenta >= self._maximo:
                self._ventanas[clave] = (inicio, cuenta)
                return False, max(0.0, self._ventana - (ahora - inicio))
            self._ventanas[clave] = (inicio, cuenta + 1)
            # Keep memory bounded: drop windows that already expired.
            if len(self._ventanas) > 10_000:
                self._ventanas = {
                    k: v for k, v in self._ventanas.items() if ahora - v[0] < self._ventana
                }
            return True, 0.0


def api_key_valida(configurada: str | None, recibida: str | None) -> bool:
    """Constant-time comparison; ``None`` configured means the API is open."""
    if not configurada:
        return True
    if not recibida:
        return False
    return secrets.compare_digest(configurada, recibida)
