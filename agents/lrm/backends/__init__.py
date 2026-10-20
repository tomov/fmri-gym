"""Language-model backends; :func:`get_backend` builds one by name, importing only its own deps.

========================  ======  ========  ==============================================
backend                   encode  generate  for
========================  ======  ========  ==============================================
``transformers``          yes     yes       any Hugging Face causal LM (Qwen, Llama, gpt-oss,
                                            DeepSeek, Mistral...), one process or torchrun
``vllm``                  no      yes       fast local play with an open-weight model
``openrouter``            no      yes       models only an API serves (GPT, Claude, Gemini...)
``deepseek_v32``          yes     no        DeepSeek V3.2 on its own runtime, under torchrun
``deepseek_v4``           yes     no        DeepSeek V4 on its own runtime, under torchrun
========================  ======  ========  ==============================================

Ported from reason-to-play's ``agents/lrm/backends`` (MIT; the DeepSeek
runtimes keep DeepSeek's MIT notice), reshaped to the two calls in :mod:`.base`.
"""

from __future__ import annotations

from .base import LRMBackend, Reply

__all__ = ["BACKENDS", "LRMBackend", "Reply", "get_backend"]

#: The names :func:`get_backend` takes.
BACKENDS = ("transformers", "vllm", "openrouter", "deepseek_v32", "deepseek_v4")


def get_backend(name: str, model_id: str, **options) -> LRMBackend:
    """Load a model.

    :param name: one of :data:`BACKENDS`.
    :param model_id: the checkpoint, a converted DeepSeek folder, or an API model id.
    :param options: the backend's own constructor arguments.
    :return: the loaded backend.
    :raises ValueError: an unknown name.
    """
    if name == "transformers":
        from .transformers import TransformersBackend
        return TransformersBackend(model_id, **options)
    if name == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend(model_id, **options)
    if name == "openrouter":
        from .openrouter import OpenRouterBackend
        return OpenRouterBackend(model_id, **options)
    if name == "deepseek_v32":
        from .deepseek import DeepSeekV32Backend
        return DeepSeekV32Backend(model_id, **options)
    if name == "deepseek_v4":
        from .deepseek import DeepSeekV4Backend
        return DeepSeekV4Backend(model_id, **options)
    raise ValueError(f"unknown LRM backend {name!r}; one of {BACKENDS}")
