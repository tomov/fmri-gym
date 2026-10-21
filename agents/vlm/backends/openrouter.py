"""Any model OpenRouter serves (GPT, Claude, Gemini, DeepSeek, Llama, Qwen...): replies only.

An API returns text, never activations, so this is how a closed model plays, not how its
internal state is read. Needs OPENROUTER_API_KEY. can_see_images is read off the model's
own listed input modalities, not assumed.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

from .base import VLMBackend, Reply

API_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(VLMBackend):
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
        self.can_see_images = "image" in self.info.get("architecture", {}).get(
            "input_modalities", [])

    def generate(self, chat: list[dict]) -> Reply:
        self._refuse_images_unless_supported(chat)
        chat = [{**m, "content": _to_openai_content(m["content"])} for m in chat]
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


def _to_openai_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if part.get("type") == "image":
            buf = io.BytesIO()
            part["image"].convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        else:
            out.append(part)
    return out


def _model_info(model_id: str, key: str) -> dict:
    import httpx
    response = httpx.get(f"{API_URL}/models", headers={"Authorization": f"Bearer {key}"},
                         timeout=30.0)
    response.raise_for_status()
    for model in response.json()["data"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"OpenRouter lists no model {model_id!r} (https://openrouter.ai/models)")
