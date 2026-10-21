# VLM harness: vision-language models that play the games and are read while they do

Two uses, one turn format. A turn is the frame a player saw, the HUD lines beside it, and the game's state as text (`symbolic/`), all as they stood before that frame's action.

- **Play live:** `BackendVLMPolicy` (`agents/policies.py`) plays a curriculum through any backend that sees images and generates; its blocks are logged like a subject's.
- **Read a logged block:** `extract_features.py` turns every recorded frame of a block (a subject's or a model's) into a turn, with the logged action as the reply, and writes the model's hidden state at each turn to `<block>/vlm_states_<model>.h5`, beside the turn's `frame`, `episode_id`, `ep_frame` and `run_time` (the scan's alignment keys). Most readers need only the log; one whose fields cost real per-frame engine work (vizdoom, coom) replays the block through a fresh adapter instead (`symbolic.base.SymbolicReader.wants_engine_state`).

```bash
python agents/agent_play.py --curriculum configs/curricula/crafter/levels/L4.json \
    --policy vlm-backend --backend openrouter --model openai/gpt-5 --max-frames 60
python agents/vlm/extract_features.py --block data/<run>/block-01_crafter_L4 \
    --backend transformers --model Qwen/Qwen3.5-9B --layers 0 16 -1
```

## Symbolic state

`symbolic/` holds one reader per game, which sees each logged frame's `variables` in order and says what the player knows: status and inventory, the things in view, and what just happened. Most backends log their whole state every frame (`tests/test_capture.py`), so their reader needs nothing else; vizdoom and coom cost real per-frame engine work nothing on the live path needs, so `capture()` does not log it, and `extract_features.py` reads it from a replayed adapter instead.

| backend | reads |
|---|---|
| vizdoom, coom | `player_variables` (status bar, position, kill/hit counters) and `labels` (actors on screen, by bearing and distance), read from a replayed adapter; `notifications` from the log |
| crafter | `inventory`, `daylight`, `facing`, `objects` within the 9x9 view |
| minihack, nethack | `blstats`, `inv_letters`/`inv_strs`, `message` |
| rushhour | `cars` |
| baba | `ruleset` |
| any other | the logged scalars |

## Backends

| `--backend` | hidden states | replies | for |
|---|---|---|---|
| `transformers` | yes | yes | a Hugging Face vision-language checkpoint (`AutoModelForImageTextToText`), one process or `torchrun` |
| `openrouter` | no | yes | API models (GPT, Claude, Gemini, Qwen-VL...); needs `OPENROUTER_API_KEY` |

Only vision-language models: a text-only checkpoint with a separately trained image encoder attached would not be one model's own perception. A turn's state is the hidden state at the last token of the assistant header (`backends/chat.py`); layer `0` is the embeddings, layer `i` the output of decoder block `i`. A new runtime is a `VLMBackend` in `backends/<name>.py`, registered in `backends/__init__.py`.
