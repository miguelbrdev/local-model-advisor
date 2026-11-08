"""Canonical runtime names shared by the catalogue enrichment and the filter.

The user may write a runtime in many ways ("LM Studio", "lmstudio",
"llama-cpp"); the catalogue stores one canonical token per runtime so the
filter can compare them.  Unknown names are kept lower-cased rather than
dropped: a runtime we do not know cannot match any family, and the
response then says the preference was not applied.
"""

from __future__ import annotations

RUNTIMES_CANONICOS: frozenset[str] = frozenset({
    "ollama",
    "lmstudio",
    "llama.cpp",
    "vllm",
    "mlx",
    "transformers",
    "exllama",
    "onnxruntime",
    "openvino",
})

_ALIAS: dict[str, str] = {
    "lm studio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "llamacpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "llama cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "ggml": "llama.cpp",
    "gguf": "llama.cpp",
    "exllamav2": "exllama",
    "exllamav3": "exllama",
    "exl2": "exllama",
    "exl3": "exllama",
    "onnx": "onnxruntime",
    "onnx runtime": "onnxruntime",
    "onnx-runtime": "onnxruntime",
    "hf transformers": "transformers",
    "huggingface transformers": "transformers",
    "apple mlx": "mlx",
    "mlx-lm": "mlx",
}


def normalizar_runtime(texto: str | None) -> str | None:
    """Canonical runtime token for ``texto``, or ``None`` when blank.

    Lower-cases, collapses whitespace and resolves known aliases; names
    outside :data:`RUNTIMES_CANONICOS` are returned as typed (lower-cased)
    so the caller can still report them.
    """
    if texto is None:
        return None
    limpio = " ".join(texto.strip().lower().split())
    if not limpio:
        return None
    return _ALIAS.get(limpio, limpio)
