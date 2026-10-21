"""Any Hugging Face vision-language checkpoint: hidden states by hooks, replies by generate.

Always loads the full vision-language model (AutoProcessor + AutoModelForImageTextToText),
never just its language model -- native vision weights, not a separately-trained encoder
bolted on after the fact. Decoder hooks expect model.model.layers, falling back to
model.model.language_model.layers; unverified on real hardware, no GPU here to check it.
"""

from __future__ import annotations

import os

import numpy as np

from .base import VLMBackend, Reply
from .chat import assistant_header, ensure_template, header_positions


class TransformersBackend(VLMBackend):
    name = "transformers"
    can_encode = True
    can_generate = True
    can_see_images = True

    def __init__(self, model_id: str, device_map: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 512, temperature: float = 0.5, top_p: float = 0.95,
                 seed: int = 0) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.manual_seed(seed)
        self.model_id = model_id
        self.sampling = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                         "top_p": top_p, "do_sample": temperature > 0}
        self.rank = int(os.environ.get("RANK", "0"))
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.template = ensure_template(self.tokenizer, model_id)
        placement = _placement(model_id, device_map)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation=_attention(), **placement).eval()
        self.placement = next(iter(placement.items()))
        self.devices = sorted({str(p.device) for p in self.model.parameters()})
        self.decoder = _language_decoder(self.model)
        self._device = self.decoder.embed_tokens.weight.device

    @property
    def n_layers(self) -> int:
        return len(self.decoder.layers) + 1

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    def encode(self, chat: list[dict], targets: list[int], layers: list[int]) -> np.ndarray:
        import torch
        _, forward_inputs = self._render(chat)
        header = assistant_header(self.tokenizer, render=self._render_text)
        header_ids = self.tokenizer(header, add_special_tokens=False)["input_ids"]
        positions = header_positions(forward_inputs["input_ids"][0].tolist(), header_ids,
                                     chat, targets)
        caught: dict[int, torch.Tensor] = {}
        handles = [self._module(layer).register_forward_hook(_catch(caught, layer, positions))
                   for layer in layers]
        try:
            with torch.inference_mode():
                self.model(**forward_inputs, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        # Spread over GPUs; each layer's states are on the device that holds it.
        stacked = torch.stack([caught[layer].cpu() for layer in layers], dim=1)
        return stacked.to(torch.float16).numpy()

    def generate(self, chat: list[dict]) -> Reply:
        import torch
        _, inputs = self._render(chat, add_generation_prompt=True)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(**_generate_inputs(self.model, inputs), **self.sampling,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, n_in:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        reasoning, _, answer = text.rpartition("</think>")
        return Reply(answer.strip(), reasoning.replace("<think>", "").strip(), n_in, len(new))

    def _render_text(self, chat: list[dict], **kw) -> str:
        return self.processor.apply_chat_template(chat, tokenize=False, **kw)

    def _render(self, chat: list[dict], **kw) -> tuple[str, dict]:
        text = self._render_text(chat, **kw)
        images = _images_in(chat)
        enc = self.processor(text=[text], images=images or None, return_tensors="pt",
                             add_special_tokens=False)
        return text, {k: v.to(self._device) for k, v in enc.items()}

    def record(self) -> dict:
        where, how = self.placement
        return {**super().record(), "chat_template": self.template,
                "attention": self.model.config._attn_implementation,
                "placement": f"{where}={how}", "devices": ",".join(self.devices),
                "world_size": int(os.environ.get("WORLD_SIZE", "1"))}

    def _module(self, layer: int):
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is not one of {self.model_id}'s "
                             f"0..{self.n_layers - 1}")
        return self.decoder.embed_tokens if layer == 0 else self.decoder.layers[layer - 1]


def _language_decoder(model):
    # Some families keep the decoder at model.model; a VLM wrapper commonly nests it
    # one level deeper, vision tower as a sibling, under language_model (most) or
    # text_model (Idefics3/SmolVLM).
    if hasattr(model.model, "layers"):
        return model.model
    for attr in ("language_model", "text_model"):
        nested = getattr(model.model, attr, None)
        if nested is not None and hasattr(nested, "layers"):
            return nested
    raise AttributeError(
        f"{type(model).__name__} has no model.model.layers, model.model.language_model.layers "
        "or model.model.text_model.layers; tell this backend where the decoder is")


def _placement(model_id: str, device_map: str) -> dict:
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        if device_map == "auto" and not torch.cuda.is_available():
            built = torch.version.cuda or "none"
            raise RuntimeError(f"device_map=auto found no GPU (torch {torch.__version__}, built "
                               f"for CUDA {built}; a build newer than the driver sees none). Use "
                               "a torch the driver runs, or pass device_map=cpu to mean the CPU")
        return {"device_map": device_map}
    import torch.distributed as dist
    from transformers import AutoConfig
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if getattr(AutoConfig.from_pretrained(model_id), "model_type", "") == "qwen3_5_moe":
        from . import qwen35_moe
        qwen35_moe.apply()
    return {"tp_plan": "auto"}


def _attention() -> str | None:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


def _generate_inputs(model, inputs: dict) -> dict:
    # generate() validates kwargs against forward()'s named parameters, but its own
    # multimodal path (_prepare_multimodal_encoder_kwargs_for_generation) actually
    # reads some keys -- e.g. LLaVA's batch_num_images -- off get_image_features
    # instead, so the validator rejects a key the model itself needs. Union both.
    import inspect
    named = set(inspect.signature(model.forward).parameters)
    for method in ("get_image_features", "get_video_features"):
        fn = getattr(model.base_model, method, None)
        if fn is not None:
            named |= set(inspect.signature(fn).parameters)
    return {k: v for k, v in inputs.items() if k in named}


def _images_in(chat: list[dict]) -> list:
    images = []
    for message in chat:
        content = message.get("content")
        if isinstance(content, list):
            images.extend(part["image"] for part in content if part.get("type") == "image")
    return images


def _catch(caught: dict, layer: int, positions: list[int]):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        caught[layer] = hidden[0, positions].detach()
    return hook
