"""DeepSeek V3.2 and V4 on DeepSeek's own inference code: hidden states, under torchrun.

The full checkpoints (671B and up) run fastest on the inference code DeepSeek
ships with them, vendored here unchanged from reason-to-play
(``deepseek_v32/``, ``deepseek_v4/``: each with its LICENSE and a
``PROVENANCE.json`` of upstream hashes). It shards every layer over the
ranks of a ``torchrun`` launch and loads weights converted for that many
ranks (``python -m agents.lrm.backends.deepseek_v32.convert``, likewise v4).
Every rank runs the same forward pass on the same tokens, so no rank sends
another its input; rank 0 writes.

What a block returns is not the residual stream itself, and each version
needs its own reading:

- V3.2 fuses the residual add into the next norm, so a block returns
  ``(ffn_out, residual)`` and the stream after it is their sum. (The
  reason-to-play adapter keeps ``ffn_out`` alone, which is the MoE layer's
  output, not the hidden state.)
- V4 keeps ``hc_mult`` parallel copies of the stream (Hyper-Connections), a
  block returns ``[batch, seq, hc_mult, dim]``, and there is no single
  stream to read; it is the copies' mean, reason-to-play's convention.

Both models can be loaded by the ``transformers`` backend too, which is
slower on the full checkpoint but needs no conversion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import LRMBackend
from .chat import ensure_template, header_positions

HERE = Path(__file__).parent


class _DeepSeekRuntime(LRMBackend):
    """The part both versions share: torchrun, the converted shards, the hooks.

    :param model_id: the converted checkpoint's folder (shards and tokenizer).
    :param config: the runtime's model config; default, the vendored one.
    :param max_seq_len: the longest window, in tokens, the runtime makes room for.
    """

    can_encode = True
    #: the vendored runtime's package
    runtime: str
    #: its model config, in the runtime's folder
    default_config: str

    def __init__(self, model_id: str, config: str | None = None,
                 max_seq_len: int = 16384) -> None:
        import importlib

        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError(f"the {self.name} backend runs DeepSeek's CUDA kernels and "
                               "needs GPUs; launch it with torchrun on a GPU node, or load "
                               "the model with the transformers backend")
        self.model_id = model_id
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.set_default_dtype(torch.bfloat16)
        self.config = config or str(HERE / self.runtime / self.default_config)
        with open(self.config) as f:
            args = {**json.load(f), "max_seq_len": max_seq_len, "max_batch_size": 1}
        model = importlib.import_module(f".{self.runtime}.model", __package__)
        loader = importlib.import_module(f".{self.runtime}.load_utils", __package__)
        with torch.device("cuda"):
            self.model = model.Transformer(model.ModelArgs(**args))
        loader.load_sharded_model(self.model, model_id, self.rank, self.world_size)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.template = ensure_template(self.tokenizer, model_id)

    @property
    def n_layers(self) -> int:
        return len(self.model.layers) + 1

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    def encode(self, chat: list[dict], targets: list[int], layers: list[int]) -> np.ndarray:
        import torch
        input_ids, positions = header_positions(self.tokenizer, chat, targets)
        caught: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            if not 0 <= layer < self.n_layers:
                raise ValueError(f"layer {layer} is not one of 0..{self.n_layers - 1}")
            module = self.model.embed if layer == 0 else self.model.layers[layer - 1]
            handles.append(module.register_forward_hook(self._catch(caught, layer, positions)))
        try:
            with torch.inference_mode():
                self.model(torch.tensor([input_ids], device="cuda"), start_pos=0)
        finally:
            for h in handles:
                h.remove()
        stacked = torch.stack([caught[layer] for layer in layers], dim=1)
        return stacked.to(torch.float16).cpu().numpy()

    def record(self) -> dict:
        return {**super().record(), "runtime": self.runtime, "config": self.config,
                "chat_template": self.template, "world_size": self.world_size}

    def _catch(self, caught: dict, layer: int, positions: list[int]):
        """A forward hook that keeps one layer's residual stream at ``positions``."""
        def hook(_module, _inputs, output):
            stream = output if layer == 0 else self._stream(output)
            caught[layer] = stream[0, positions].detach()
        return hook

    def _stream(self, block_output):
        """The residual stream after a block, from what the block returns."""
        raise NotImplementedError


class DeepSeekV32Backend(_DeepSeekRuntime):
    name = "deepseek_v32"
    runtime = "deepseek_v32"
    default_config = "config_671B_v3.2.json"

    def _stream(self, block_output):
        ffn_out, residual = block_output
        return ffn_out + residual


class DeepSeekV4Backend(_DeepSeekRuntime):
    name = "deepseek_v4"
    runtime = "deepseek_v4"
    default_config = "config_V4_Flash.json"

    def __init__(self, model_id: str, config: str | None = None,
                 max_seq_len: int = 16384) -> None:
        super().__init__(model_id, config, max_seq_len)
        import torch
        # V4's attention helpers make index tensors with torch.arange and no
        # device; without this default they land on the CPU and the pass fails.
        torch.set_default_device("cuda")

    def _stream(self, block_output):
        return block_output.mean(dim=2)
