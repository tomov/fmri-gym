"""A local model served by vLLM: fast replies, no hidden states.

vLLM batches and pages its KV cache, which makes it the fast way to let an
open-weight model play; it does not expose activations, so the same model's
internal state is read with the ``transformers`` backend. It shards the model
over every visible GPU (tensor parallelism) unless told how many to use.

The prompt goes in as token ids from the model's own chat template, which is
what Mistral's tokenizers require; a tokenizer with no template gets the Llama
3.1 one, and :meth:`record` says so.
"""

from __future__ import annotations

from .base import LRMBackend, Reply
from .chat import ensure_template


class VLLMBackend(LRMBackend):
    """A checkpoint loaded into vLLM.

    :param model_id: a Hugging Face id or a local folder.
    :param max_model_len: the context vLLM reserves KV cache for, in tokens.
    :param n_gpus: tensor-parallel width; ``None`` is every visible GPU.
    :param max_new_tokens: the reply's cap.
    :param temperature: sampling temperature.
    :param top_p: nucleus.
    :param seed: vLLM's sampling seed.
    """

    name = "vllm"
    can_generate = True

    def __init__(self, model_id: str, max_model_len: int = 32768, n_gpus: int | None = None,
                 max_new_tokens: int = 512, temperature: float = 0.5, top_p: float = 0.95,
                 seed: int = 0) -> None:
        import torch
        from vllm import LLM, SamplingParams
        self.model_id = model_id
        self.n_gpus = n_gpus or torch.cuda.device_count()
        if not self.n_gpus:
            raise RuntimeError("the vllm backend needs a CUDA GPU")
        self.llm = LLM(model=model_id, tensor_parallel_size=self.n_gpus,
                       max_model_len=max_model_len, gpu_memory_utilization=0.95, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.template = ensure_template(self.tokenizer, model_id)
        self.sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                                       top_p=top_p, seed=seed)

    def generate(self, chat: list[dict]) -> Reply:
        from vllm.inputs import TokensPrompt
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
        out = self.llm.generate(TokensPrompt(prompt_token_ids=ids), self.sampling,
                                use_tqdm=False)[0].outputs[0]
        reasoning, _, answer = out.text.rpartition("</think>")
        return Reply(answer.strip(), reasoning.replace("<think>", "").strip(), len(ids),
                     len(out.token_ids))

    def record(self) -> dict:
        return {**super().record(), "chat_template": self.template, "n_gpus": self.n_gpus}
