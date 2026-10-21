"""Where, in a model's own rendering of a chat, each assistant turn's header ends.

Every family writes that header differently, so it's read off the tokenizer's own chat
template rather than hardcoded per family. A tokenizer with no template of its own gets
the Llama 3.1 one, or DeepSeek's for a DeepSeek checkpoint.
"""

from __future__ import annotations

from typing import Any

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
    if tokenizer.chat_template:
        return "model"
    deepseek = "deepseek" in model_id.lower()
    tokenizer.chat_template = DEEPSEEK_CHAT_TEMPLATE if deepseek else LLAMA_31_CHAT_TEMPLATE
    source = "fallback: deepseek" if deepseek else "fallback: llama-3.1"
    print(f"{model_id} has no chat template; using the {source.split(': ')[1]} one")
    return source


def header_positions(input_ids: list[int], header_ids: list[int], chat: list[dict],
                     targets: list[int]) -> list[int]:
    """The token ending each target assistant message's header, in the ids the model reads.

    Searched in the processor's ids, not the chat's text: an image renders as one
    placeholder there but as one token per patch here, which shifts every later position.

    :param input_ids: the forward pass's token ids, images expanded.
    :param header_ids: :func:`assistant_header` tokenized.
    :param chat: the chat they encode; turn N's header is the Nth occurrence.
    :param targets: indices into ``chat`` of assistant messages.
    :raises ValueError: a target that is not an assistant message, or too few headers.
    """
    n = len(header_ids)
    ends = [i + n - 1 for i in range(len(input_ids) - n + 1) if input_ids[i:i + n] == header_ids]
    assistants = [i for i, m in enumerate(chat) if m["role"] == "assistant"]
    if len(ends) < len(assistants):
        raise ValueError(f"found {len(ends)} assistant header(s) in the tokens but "
                         f"{len(assistants)} assistant message(s); does the header tokenize "
                         "the same alone as in the chat?")
    for t in targets:
        if chat[t]["role"] != "assistant":
            raise ValueError(f"message {t} is a {chat[t]['role']} message, not an assistant one")
    return [ends[assistants.index(t)] for t in targets]


#: Structured, not plain-string, content: a VLM processor's own chat template commonly
#: expects the multimodal content list and silently drops plain-string content instead
#: of rendering it, which real turns never hit -- :func:`.observation.frame_content`
#: always builds a user message's content as parts.
PROBE = [{"role": "user", "content": [{"type": "text", "text": "probe-user-1"}]},
         {"role": "assistant", "content": [{"type": "text", "text": "probe-reply-1"}]},
         {"role": "user", "content": [{"type": "text", "text": "probe-user-2"}]},
         {"role": "assistant", "content": [{"type": "text", "text": "probe-reply-2"}]}]


def assistant_header(tokenizer: Any, render: Any = None) -> str:
    render = render or (lambda c, **kw: tokenizer.apply_chat_template(c, tokenize=False, **kw))
    plain = render(PROBE[:1])
    prompted = render(PROBE[:1], add_generation_prompt=True)
    if not prompted.startswith(plain):
        raise ValueError("the chat template's generation prompt is not an addition to the chat")
    prompt = prompted[len(plain):]
    text = render(PROBE)
    starts = _content_starts(text, PROBE)
    gaps = [text[starts[i - 1] + len(_text_of(PROBE[i - 1]["content"])):starts[i]] for i in (1, 3)]
    for n in range(len(prompt), 0, -1):
        if all(prompt[:n] in gap for gap in gaps):
            return prompt[:n]
    raise ValueError(f"the generation prompt {prompt!r} is not before the assistant messages")


def _text_of(content: Any) -> str:
    """A message's content as plain text: itself if a string, else its one text part."""
    if isinstance(content, str):
        return content
    return next(part["text"] for part in content if part.get("type") == "text")


def _content_starts(text: str, chat: list[dict]) -> list[int]:
    # Only ever called on PROBE, whose content renders to plain text.
    starts, cursor = [], 0
    for i, message in enumerate(chat):
        content = _text_of(message["content"])
        at = text.find(content, cursor)
        if at < 0:
            raise ValueError(f"message {i}'s content is not in the rendered chat verbatim; "
                             "the chat template rewrites it")
        starts.append(at)
        cursor = at + len(content)
    return starts
