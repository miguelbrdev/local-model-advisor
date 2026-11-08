"""TLS verification context shared by every outbound HTTP client.

httpx verifies servers against the ``certifi`` bundle by default.  That
bundle dropped the GlobalSign Root CA (R1) in late 2026, and canirun.ai
still serves a chain whose last link is GTS Root R4 cross-signed by that
root, so on 2026-11-08 every request to canirun.ai failed with
``CERTIFICATE_VERIFY_FAILED`` while curl and the browser (operating-system
store) accepted it.  :func:`contexto_tls` therefore builds the context
from the operating-system store (Windows certificate store; the
distribution's ``ca-certificates`` on Linux and in the Docker image),
which is what the rest of the machine trusts.

Set ``LMA_TLS_CERTIFI=1`` to go back to the certifi bundle, for example to
reproduce the failure or on a host whose system store is unusable.
"""

from __future__ import annotations

import os
import ssl


def contexto_tls() -> ssl.SSLContext | bool:
    """``verify`` argument for httpx: system trust store, or certifi on request."""
    if os.environ.get("LMA_TLS_CERTIFI", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return ssl.create_default_context()
