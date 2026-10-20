"""Any Hugging Face causal LM: hidden states by hooks, replies by ``generate``.

One backend for every family ``transformers`` loads (Qwen, Llama, gpt-oss,
DeepSeek V3/V3.2/V4, Mistral...): ``AutoModelForCausalLM`` loads the language
model of a vision-language checkpoint too, which is all a symbolic transcript
needs. Every one of them keeps its decoder at ``model.model``, with
``embed_tokens`` and ``layers``, which is where the hooks go.

``encode`` runs the decoder without the LM head, whose logits over the whole
vocabulary at every position would cost more memory than the rest of the pass.
The hooks cut each layer's output down to the target positions as it is made,
so a long window never holds every layer's full sequence at once
(``output_hidden_states`` would).

Placement: one process spreads the layers over the visible GPUs
(``device_map="auto"``, needs ``accelerate``), and refuses to start if torch
sees none -- ``auto`` would otherwise load a large model into RAM and run it
on the CPU, which looks like a slow GPU run until someone checks; ``cpu`` is
how to ask for the CPU. Under ``torchrun`` every rank
holds a slice of every layer (``tp_plan="auto"``), only rank 0 writes, and
Qwen3.5-MoE gets reason-to-play's patch for its KV heads (:mod:`.qwen35_moe`).
The attention kernel is FlashAttention 2 when ``flash_attn`` is installed, and
otherwise the one ``transformers`` picks for the architecture (SDPA where it
can, eager where it cannot, as for gpt-oss); :meth:`record` says which ran.
"""

from __future__ import annotations

import os

import numpy as np

from .base import LRMBackend, Reply
from .chat import ensure_template, header_positions


class TransformersBackend(LRMBackend):
    """A Hugging Face checkpoint.

    :param model_id: a Hugging Face id or a local folder (``Qwen/Qwen3.5-9B``).
    :param device_map: ``from_pretrained``'s, in one process (ignored under torchrun).
    :param dtype: ``from_pretrained``'s; ``"auto"`` is the checkpoint's own.
    :param max_new_tokens: :meth:`generate`'s cap.
    :param temperature: :meth:`generate`'s sampling temperature; ``0`` is greedy.
    :param top_p: :meth:`generate`'s nucleus.
    :param seed: seeds torch once, so sampling repeats run to run.
    """

    name = "transformers"
    can_encode = True
    can_generate = True

    def __init__(self, model_id: str, device_map: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 512, temperature: float = 0.5, top_p: float = 0.95,
                 seed: int = 0) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.model_id = model_id
        self.sampling = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                         "top_p": top_p, "do_sample": temperature > 0}
        self.rank = int(os.environ.get("RANK", "0"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.template = ensure_template(self.tokenizer, model_id)
        placement = _placement(model_id, device_map)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=_attention(), **placement).eval()
        self.placement = next(iter(placement.items()))
        self.devices = sorted({str(p.device) for p in self.model.parameters()})
        self.decoder = self.model.model
        self._device = self.decoder.embed_tokens.weight.device

    @property
    def n_layers(self) -> int:
        return len(self.decoder.layers) + 1

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    def encode(self, chat: list[dict], targets: list[int], layers: list[int]) -> np.ndarray:
        import torch
        input_ids, positions = header_positions(self.tokenizer, chat, targets)
        caught: dict[int, torch.Tensor] = {}
        handles = [self._module(layer).register_forward_hook(_catch(caught, layer, positions))
                   for layer in layers]
        try:
            with torch.inference_mode():
                self.decoder(input_ids=torch.tensor([input_ids], device=self._device),
                             use_cache=False)
        finally:
            for h in handles:
                h.remove()
        # Spread over GPUs, each layer's states are on the device that holds it.
        stacked = torch.stack([caught[layer].cpu() for layer in layers], dim=1)
        return stacked.to(torch.float16).numpy()

    def generate(self, chat: list[dict]) -> Reply:
        import torch
        inputs = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.sampling,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, n_in:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        # A reasoning model's thinking comes first, closed by </think>.
        reasoning, _, answer = text.rpartition("</think>")
        return Reply(answer.strip(), reasoning.replace("<think>", "").strip(), n_in, len(new))

    def record(self) -> dict:
        where, how = self.placement
        return {**super().record(), "chat_template": self.template,
                "attention": self.model.config._attn_implementation,
                "placement": f"{where}={how}", "devices": ",".join(self.devices),
                "world_size": int(os.environ.get("WORLD_SIZE", "1"))}

    def _module(self, layer: int):
        """Layer 0 is the embedding table, layer ``i`` decoder block ``i - 1``."""
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is not one of {self.model_id}'s "
                             f"0..{self.n_layers - 1}")
        return self.decoder.embed_tokens if layer == 0 else self.decoder.layers[layer - 1]


def _placement(model_id: str, device_map: str) -> dict:
    """``from_pretrained``'s placement: tensor-parallel under torchrun, else ``device_map``."""
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
    """FlashAttention 2 if it is installed, else ``None``: the architecture's default."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


def _catch(caught: dict, layer: int, positions: list[int]):
    """A forward hook that keeps one layer's output at ``positions``."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        caught[layer] = hidden[0, positions].detach()
    return hook
