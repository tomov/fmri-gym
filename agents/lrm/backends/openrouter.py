"""Any model OpenRouter serves (GPT, Claude, Gemini, DeepSeek, Llama, Qwen...): replies only.

An API returns text, never activations, so this backend generates and cannot
encode: it is how a closed model plays, not how its internal state is read.
It speaks OpenAI's chat API at OpenRouter's address and needs
``OPENROUTER_API_KEY``; the key, the model id and the model's context length
are all checked when it is built, before a single turn is spent.

Reasoning is always sent explicitly (``reasoning.effort``, ``"none"`` by
default), because a thinking model left to its default thinks. No reasoning
*token cap* is sent: OpenRouter rejects one beside an effort, and providers
were found to ignore it (reason-to-play's probes of Qwen3.5 and DeepSeek
V3.2, April 2026), so the only bound on a turn's cost is what the caller sends.

Transient failures (rate limits, dropped connections, a provider answering
with no choices or no usage) are retried with exponential backoff, up to
``max_wait`` seconds in all; a context-length error is not, since it will
never pass. The billed cost is the response's own ``usage.cost``, which is
what the provider OpenRouter routed to charged.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import LRMBackend, Reply

API_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(LRMBackend):
    """A model on OpenRouter.

    :param model_id: OpenRouter's id (``openai/gpt-5``, ``deepseek/deepseek-v3.2``).
    :param max_new_tokens: the visible answer's cap (reasoning may exceed it).
    :param temperature: sampling temperature.
    :param top_p: nucleus.
    :param reasoning_effort: ``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``.
    :param seed: sent with every call.
    :param max_wait: seconds of backoff a call may spend before its error is raised.
    :raises ValueError: no API key, or a model OpenRouter does not list.
    """

    name = "openrouter"
    can_generate = True

    def __init__(self, model_id: str, max_new_tokens: int = 4096, temperature: float = 0.5,
                 top_p: float = 0.95, reasoning_effort: str = "none", seed: int = 0,
                 max_wait: float = 3600.0) -> None:
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("the openrouter backend needs OPENROUTER_API_KEY "
                             "(https://openrouter.ai/keys)")
        self.model_id = model_id
        self.params = {"max_tokens": max_new_tokens, "temperature": temperature,
                       "top_p": top_p, "seed": seed,
                       "extra_body": {"reasoning": {"effort": reasoning_effort}}}
        self.max_wait = max_wait
        self.client = OpenAI(base_url=API_URL, api_key=key)
        self.info = _model_info(model_id, key)

    def generate(self, chat: list[dict]) -> Reply:
        response = self._call(chat)
        message = response.choices[0].message
        usage = response.usage
        cost = response.model_dump().get("usage", {}).get("cost")
        return Reply(message.content or "", getattr(message, "reasoning", None) or "",
                     usage.prompt_tokens, usage.completion_tokens,
                     None if cost is None else float(cost))

    def record(self) -> dict:
        return {**super().record(), "context_length": self.info["context_length"],
                "reasoning_effort": self.params["extra_body"]["reasoning"]["effort"]}

    def _call(self, chat: list[dict]) -> Any:
        """One completion, retried with backoff on anything transient."""
        from openai import APIConnectionError, APIStatusError, RateLimitError
        waited, delay = 0.0, 2.0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id, messages=chat, **self.params)
                if response.choices and response.usage is not None:
                    return response
                problem: Exception = RuntimeError(f"OpenRouter answered {self.model_id} "
                                                  "with no choices or no usage")
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                if "context length" in str(e).lower() or "context_length" in str(e).lower():
                    raise
                problem = e
            if waited + delay > self.max_wait:
                raise RuntimeError(f"gave up after {waited:.0f}s of retries") from problem
            print(f"[openrouter] {problem}; retrying in {delay:.0f}s")
            time.sleep(delay)
            waited, delay = waited + delay, min(delay * 2, 300.0)


def _model_info(model_id: str, key: str) -> dict:
    """OpenRouter's listing of ``model_id`` (its context length, pricing, parameters).

    :raises ValueError: if OpenRouter does not list it.
    """
    import httpx
    response = httpx.get(f"{API_URL}/models", headers={"Authorization": f"Bearer {key}"},
                         timeout=30.0)
    response.raise_for_status()
    for model in response.json()["data"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"OpenRouter lists no model {model_id!r} (https://openrouter.ai/models)")
