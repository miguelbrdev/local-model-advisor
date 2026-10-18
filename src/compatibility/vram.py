"""Shared VRAM budgeting constants.

Both the offline heuristic provider and the ranking scorer need the same
notion of a *comfortable* fit; keeping the number here avoids two copies
drifting apart.
"""

FRACCION_VRAM_UTIL = 0.8
"""Share of the available VRAM a model may occupy and still leave room for
KV cache, context growth and runtime buffers.  At or below it a fit is
``comfortable``; above it (but still fitting) it is ``tight``."""
