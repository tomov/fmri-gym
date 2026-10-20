"""Where, in a model's own rendering of a chat, each assistant turn's header ends.

A turn's internal state is read at the last token of its assistant header
(:mod:`.base`), and every model family writes that header differently:
``<|im_start|>assistant\\n`` (Qwen), ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
(Llama 3), ``<|start|>assistant`` (gpt-oss), ``<｜Assistant｜>`` (DeepSeek). So the
header is read off the tokenizer's own chat template rather than written down
per family: it is the longest start of the template's generation prompt that
appears before every assistant message of a probe chat
(:func:`assistant_header`). The generation prompt can carry more than the
header on the last turn only (Qwen's opened ``<think>``), which is why it is
cut back to what every turn shares.

A tokenizer that ships no template gets the Llama 3.1 one, or DeepSeek's for a
DeepSeek checkpoint (both from reason-to-play, ``agents/lrm/backends``); the
backend records which, so a fallback is never silent.
"""

from __future__ import annotations

from typing import Any

#: Llama 3.1's chat format, for a tokenizer with no template.
LLAMA_31_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

#: DeepSeek V3's chat format without tool calls, for a DeepSeek tokenizer with no template.
DEEPSEEK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}"
    "{% set add_generation_prompt = false %}{% endif %}"
    "{% set ns = namespace(system_prompt='', is_first_sp=true, is_last_user=false) %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% if ns.is_first_sp %}{% set ns.system_prompt = message['content'] %}"
    "{% set ns.is_first_sp = false %}"
    "{% else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{{ bos_token }}{{ ns.system_prompt }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{% set ns.is_last_user = true %}"
    "{{ '<｜User｜>' + message['content'] }}{% endif %}"
    "{% if message['role'] == 'assistant' %}{% if ns.is_last_user %}"
    "{{ '<｜Assistant｜></think>' }}{% endif %}{% set ns.is_last_user = false %}"
    "{{ message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and ns.is_last_user %}{{ '<｜Assistant｜></think>' }}{% endif %}"
)


def ensure_template(tokenizer: Any, model_id: str) -> str:
    """Give ``tokenizer`` a chat template if it has none, and say where it came from.

    :param tokenizer: a Hugging Face tokenizer.
    :param model_id: the checkpoint, to recognise a DeepSeek one.
    :return: ``"model"``, ``"fallback: deepseek"`` or ``"fallback: llama-3.1"``.
    """
    if tokenizer.chat_template:
        return "model"
    deepseek = "deepseek" in model_id.lower()
    tokenizer.chat_template = DEEPSEEK_CHAT_TEMPLATE if deepseek else LLAMA_31_CHAT_TEMPLATE
    source = "fallback: deepseek" if deepseek else "fallback: llama-3.1"
    print(f"{model_id} has no chat template; using the {source.split(': ')[1]} one")
    return source


def header_positions(tokenizer: Any, chat: list[dict], targets: list[int]) -> tuple[list, list]:
    """Tokenize ``chat`` and find the token that ends each target's assistant header.

    :param tokenizer: a fast Hugging Face tokenizer with a chat template.
    :param chat: ``[{"role", "content"}, ...]``.
    :param targets: indices into ``chat`` of assistant messages.
    :return: ``(input_ids, positions)``: the chat's token ids, and one position per target.
    :raises ValueError: if the header cannot be found, or ends inside a token.
    """
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    starts = _content_starts(text, chat)
    header = assistant_header(tokenizer)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_by_end = {end: i for i, (_, end) in enumerate(enc["offset_mapping"])}
    positions = []
    for t in targets:
        if chat[t]["role"] != "assistant":
            raise ValueError(f"message {t} is a {chat[t]['role']} message, not an assistant one")
        previous_end = starts[t - 1] + len(chat[t - 1]["content"])
        at = text.rfind(header, previous_end, starts[t])
        if at < 0:
            raise ValueError(f"no assistant header {header!r} before message {t}")
        if at + len(header) not in token_by_end:
            raise ValueError(f"the assistant header {header!r} of message {t} ends inside a "
                             "token; this tokenizer merges across it")
        positions.append(token_by_end[at + len(header)])
    return enc["input_ids"], positions


#: A chat with an assistant turn before the last one and the last one itself, whose
#: contents appear nowhere in any template's markup.
PROBE = [{"role": "user", "content": "probe-user-1"},
         {"role": "assistant", "content": "probe-reply-1"},
         {"role": "user", "content": "probe-user-2"},
         {"role": "assistant", "content": "probe-reply-2"}]


def assistant_header(tokenizer: Any) -> str:
    """The text a tokenizer's chat template writes before every assistant message.

    It is the longest start of the template's generation prompt that is found
    before both assistant turns of :data:`PROBE`. A probe and not the chat at
    hand, because a chat whose only assistant turn is its last one sees just the
    last turn's form (Qwen's opened ``<think>`` with it), not the one every turn shares.

    :param tokenizer: a Hugging Face tokenizer with a chat template.
    :return: the header text.
    :raises ValueError: if the generation prompt is not an addition to the chat, or
        shares nothing with what precedes the assistant turns.
    """
    plain = tokenizer.apply_chat_template(PROBE[:1], tokenize=False)
    prompted = tokenizer.apply_chat_template(PROBE[:1], tokenize=False, add_generation_prompt=True)
    if not prompted.startswith(plain):
        raise ValueError("the chat template's generation prompt is not an addition to the chat")
    prompt = prompted[len(plain):]
    text = tokenizer.apply_chat_template(PROBE, tokenize=False)
    starts = _content_starts(text, PROBE)
    gaps = [text[starts[i - 1] + len(PROBE[i - 1]["content"]):starts[i]] for i in (1, 3)]
    for n in range(len(prompt), 0, -1):
        if all(prompt[:n] in gap for gap in gaps):
            return prompt[:n]
    raise ValueError(f"the generation prompt {prompt!r} is not before the assistant messages")


def _content_starts(text: str, chat: list[dict]) -> list[int]:
    """Where each message's content begins in the rendered chat, found in order."""
    starts, cursor = [], 0
    for i, message in enumerate(chat):
        at = text.find(message["content"], cursor)
        if at < 0:
            raise ValueError(f"message {i}'s content is not in the rendered chat verbatim; "
                             "the chat template rewrites it")
        starts.append(at)
        cursor = at + len(message["content"])
    return starts
