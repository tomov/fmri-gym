# LRM harness: a subject's game as text, and a language model's internal state reading it

The fMRI side records what the subject did; this turns that record into something a language model can read, and saves what the model's internal state is at every moment of it, aligned to the block's frames and so to the scan. The model does not play: it reads the subject's own game, turn by turn, with the subject's actions as its answers.

Two stages, each a module you run, joined by a file:

```text
block-NN_<backend>_<game>/events.jsonl        the subject's block, as fmri_play logged it
        |  python -m agents.lrm.transcript      replays it (needs the game installed)
        v
<block>/lrm/transcript_stride<N>.jsonl         one turn per sampled frame: symbolic observation + action
        |  python -m agents.lrm.encode          reads it through the model (needs the GPU, not the game)
        v
<block>/lrm/states_<transcript>_<model>.h5     the model's hidden state at every turn, every chosen layer
```

## Stage 1: the transcript

```bash
python -m agents.lrm.transcript --block data/<run>/block-01_vizdoom_VizdoomDefendCenter-v1 --stride 9
```

The block is replayed from its seeds and actions through its own adapter (`fmri_gym.replay`), and at every `stride`-th frame of an episode the backend's reader (`symbolic/`) writes down the state the subject acted on and the action they took. Every replayed step's reward and `terminated` must equal the logged ones, or the run stops: a transcript of a game that diverged would describe frames nobody saw.

A reader writes down what the subject could know and nothing else: what was drawn (the session's strip above the frame, from the adapter's own `hud()`; what the game drew in its picture), and what a player knows without it being written anywhere (that a shot landed, that they were hit), as events since the last turn. An engine knows far more -- every actor in the level, the map, the player's coordinates -- and the reader leaves that out, so the model reads the subject's game and not a better-informed one. A ViZDoom turn is shown to the model as:

```text
[USER  t=172]
Observation:
HUD: 287 s, Score: 0
Inventory: [Ammo: 18, Health: 72%, Arms: 2, Armor: 0%, Weapon: Pistol, Bullets: 18, Shells: 0, Rockets: 0, Cells: 0]
Objects:
- MarineChainsawVzd: 3 deg right, 499 away
- BulletPuff: ahead, 788 away
Events: You were hit 1 time(s)
[ASSISTANT t=172]
Action: NOOP
```

The inventory is Doom's status bar, the objects are the actors with pixels on screen (ViZDoom's labels buffer) by bearing from the crosshair and distance, and the events come from the engine's kill, hit and hits-taken counters and Doom's pickup messages; `symbolic/vizdoom.py` lists each source and what is left out.

`t` is the frame's index in the block (`frame` in `events.jsonl`), which is what lines a turn up with the frames, the triggers and the scan. `stride` is the temporal sampling: at 35 fps, `--stride 9` is a turn every ~257 ms, `--stride 1` is every frame. The actions between two turns are played in the replay but not shown.

The system prompt is `prompts/system.txt` with the game's rules: `prompts/game_rules/<backend>/<game>.txt` when there is one, else the instructions screen the subject read. The header of the transcript says which (`rules_source`), and which reader wrote the observations (`reader`).

## Stage 2: the internal state

```bash
python -m agents.lrm.encode --transcript <block>/lrm/transcript_stride9.jsonl --backend transformers --model Qwen/Qwen3.5-9B --layers all --window 64
torchrun --nproc-per-node 8 -m agents.lrm.encode --transcript <...>.jsonl --backend deepseek_v32 --model /ckpt/DeepSeek-V3.2-mp8 --option max_seq_len=32768
```

Each episode is one conversation: the system prompt, then a user message per observation and an assistant message per action. An episode longer than `--window` turns is read in windows that overlap by half, and each window reports only the turns in its back half, so every turn has at least `window / 2` turns of history before it (all of them near the start of the episode). One forward pass per window gives every reported turn its state. Under `torchrun` every rank reads every window and rank 0 writes.

A turn's state is the hidden state at the last token of its assistant header: the model has read the rules, the history and this observation, and nothing of the action. The header is read off the model's own chat template (`backends/chat.py`), so it is `<|im_start|>assistant\n` for Qwen, `<|start_header_id|>assistant<|end_header_id|>\n\n` for Llama 3, `<|start|>assistant` for gpt-oss and `<｜Assistant｜>` for DeepSeek without anything written per family. `states.h5` holds, per turn, `hidden (n_layers, hidden_size)` float16 and the turn's `frame`, `episode_id`, `ep_frame`, `run_time` and `context_start`; its attributes hold the model, how it was run (`backend`: placement, attention kernel, chat template, world size), the layers, the window and the full system prompt, so the exact chat any row was read from can be rebuilt (`internal_state.py`). Layer `0` is the token embeddings, layer `i` the residual stream after decoder block `i` (before the final norm). On GPU in bfloat16 a turn's state depends slightly on how long the window it was read in is: the hooks return exactly the model's hidden states, but the same turn read with a different context length drifts in the deeper layers (Qwen3.5-35B-A3B on L40S: cosine 1.0 through layer 4, about 0.96–0.99 by layers 16–39), because the attention and MoE kernels round differently at different lengths. So a states file reproduces under the same `--window`, and states read with different window plans are not interchangeable.

## Backends

| `--backend` | hidden states | replies | for |
|---|---|---|---|
| `transformers` | yes | yes | any Hugging Face causal LM: Qwen, Llama, gpt-oss, DeepSeek V3/V3.2/V4, Mistral; one process over the visible GPUs, or tensor-parallel under `torchrun` |
| `vllm` | no | yes | fast local play with an open-weight model |
| `openrouter` | no | yes | models only an API serves (GPT, Claude, Gemini...); needs `OPENROUTER_API_KEY` |
| `deepseek_v32` | yes | no | DeepSeek V3.2 on DeepSeek's own runtime, under `torchrun`, from converted shards |
| `deepseek_v4` | yes | no | DeepSeek V4 likewise |

An API returns text, never activations, so a closed model can play but its internal state cannot be read; `encode` refuses such a backend before loading anything. A backend's own settings go in as `--option key=value` (`device_map=cpu`, `max_seq_len=32768`, `temperature=0`). The backends, the DeepSeek runtimes and the Qwen3.5-MoE tensor-parallel patch are ported from [reason-to-play](https://github.com/botcs/reason-to-play-src/tree/d94b686666c34743356fef2b13b4b9c57f3ab994/agents/lrm/backends) (MIT); `deepseek_v32/` and `deepseek_v4/` are vendored unchanged, with DeepSeek's LICENSE and reason-to-play's `PROVENANCE.json` of upstream hashes. Each backend's dependencies are in `backends/requirements.txt`; none of them is something the game side installs.

DeepSeek's blocks do not return the residual stream itself: V3.2 returns `(ffn_out, residual)`, whose sum is the stream (reason-to-play's adapter keeps `ffn_out` alone, the MoE output), and V4 keeps four parallel copies of the stream, which are averaged as reason-to-play does (`backends/deepseek.py`).

## Layout

```text
transcript.py        stage 1: replay a block into Turns; read/write transcript.jsonl
symbolic/base.py     Observation, and the generic reader (logged scalar variables, key names)
symbolic/vizdoom.py  ViZDoom: status bar, actors in view, events, button names
prompting.py         turns -> chat messages; the window plan; the [USER]/[ASSISTANT] view
prompts/             system.txt, and game_rules/<backend>/<game>.txt
backends/base.py     LRMBackend: encode(chat, targets, layers) -> states; generate(chat) -> Reply
backends/chat.py     where each assistant header ends, from the model's own chat template
backends/*.py        transformers, vllm, openrouter, deepseek (v3.2, v4), qwen35_moe (TP patch)
backends/deepseek_*  DeepSeek's inference runtimes, vendored
internal_state.py    states.h5: StateConfig, StateWriter
encode.py            stage 2
```

## Adding a game

A backend with no reader of its own gets the generic one, whose inventory is the adapter's logged scalar `variables` and whose actions are the phase's key names; for a game whose state is a grid or a picture, that says almost nothing (the header records `reader: generic`, so it is never mistaken for a real one). A reader is a `SymbolicReader` subclass in `symbolic/<backend>.py`, registered in `symbolic/__init__.py`: `observe()` reads the live env into an `Observation` of what the subject could know, `reset()` and `after_step()` let it report what changed between turns, `action_name()` names a logged action in the game's terms, and `replay_spec()` may switch on engine outputs the subject's block did not need (labels, messages), never anything the game plays by (the replay's reward check enforces it). Add `prompts/game_rules/<backend>/<game>.txt` to tell the model the rules in the reader's vocabulary.

## Adding a model

A model `transformers` loads needs nothing new: `--backend transformers --model <id>`. Another runtime is an `LRMBackend` in `backends/<name>.py`, registered in `backends/__init__.py`, that sets `can_encode` / `can_generate` and implements those calls (with `n_layers` for `encode`); `backends/chat.py`'s `header_positions` gives it the target tokens from any chat template.
