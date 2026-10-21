from __future__ import annotations

from .base import VLMBackend, Reply

__all__ = ["BACKENDS", "VLMBackend", "Reply", "get_backend"]

BACKENDS = ("transformers", "openrouter")


def get_backend(name: str, model_id: str, **options) -> VLMBackend:
    if name == "transformers":
        from .transformers import TransformersBackend
        return TransformersBackend(model_id, **options)
    if name == "openrouter":
        from .openrouter import OpenRouterBackend
        return OpenRouterBackend(model_id, **options)
    raise ValueError(f"unknown VLM backend {name!r}; one of {BACKENDS}")
