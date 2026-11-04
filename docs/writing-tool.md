# The writing tool

The part of this repository that is in use: a 4B voice model that continues the owner's scenes in Traditional
Chinese, and the `write-continue` skill that puts it in the writing flow.

This file is the single source of truth for the artifact. Numbers elsewhere are history.

## Current artifact

| | |
|---|---|
| model | `voice-zh-v10-Q4_K_M.gguf` (2.6 GB) |
| location | `~/models/gguf/`, served on `http://127.0.0.1:8771` |
| started by | `scripts/write_continue_note.py` on demand — it probes, launches, and waits for health |
| base | `Qwen/Qwen3.5-4B`, LoRA r32, merged and quantised |

Three artifacts were trained in sequence. v1 and v9 are superseded and deleted; the frozen held-out set that
compared them is not.

## Corpus

`dataset/story_rl/voice.json` — 141 tasks from the six posted stories: **108 train / 33 held out**. The
held-out tasks are byte-identical to the ones every earlier round was scored on, so a result measured today
is comparable with one measured yesterday.

`dataset/story_rl/archive/voice-v1.json` is the superseded 81-task corpus, kept because it is the provenance of
the shipped comparison.

The 33 held-out openings are the measuring instrument. Do not add tasks from a held-out story to the training
side; the builder will not stop you, and the split then measures nothing.

## Training

```
shape       prose (src/story_rl/prose.py)
recipe      GRPO, 68 steps = 5 epochs at batch 8 over 108 tasks
reward      judge craft score x length factor, gated by a lint veto and a character veto
judge       deepseek-chat via api.deepseek.com — never through OpenRouter (owner ruling, enforced in code)
```

Regenerate and run:

```
uv run python3 scripts/build_traditional_map.py --corpus <posted stories>
uv run python3 scripts/story_rl.py estimate --shape prose --steps 68
STORY_RL_APPROVED_USD=4.74 uv run python3 scripts/story_rl.py train --shape prose --steps 68
```

A training run takes about an hour and must be detached (`nohup … & disown`): the harness reaps jobs that
outlive their turn, which is how five earlier runs died at 6–15 steps with no traceback.

## Characters

The author writes Traditional Chinese. The model used to write simplified characters in **52 %** of its
generations, and the reward could not see it.

`src/story_rl/traditional.py` owns the rule. `traditional_map.json` is generated from her own prose
(`scripts/build_traditional_map.py`), so her usage is the authority and no character she writes can be
rewritten. One-to-many characters are word rules or untouched.

The negative control is the contract: `audit()` returns zero findings on her 59,563 characters and `rewrite()`
leaves that text byte-identical. **A repair that "corrects" her prose is worse than no repair**, because a
wrong-but-Traditional character reads as correct.

It is enforced twice — the reward vetoes a rollout carrying one, and the skill rewrites the draft and names
every correction in the callout.

## Measured, and what the numbers mean

| | |
|---|---|
| simplified characters in a generation | v1 52 % → v9 25 % → **v10 2 %** (0 of 12 on device) |
| v10 vs v9, frozen 33, one pass | **32 better / 0 worse / 1 unchanged**, p ≈ 0 |
| v9 vs v1, frozen 33, one pass | 19 better / 12 worse, p = 0.28 |

**Absolute craft scores are a property of the judge route, not of the prose.** The same 99 generations scored
0.697 through OpenRouter and 0.283 through `api.deepseek.com` with an identical request body. Only paired
comparisons inside one pass are portable; no score in this repository may be compared with one from another
route.

## Scripts

| script | what it is for |
|---|---|
| `write_continue_note.py` | the writing surface: continue a note, append a callout, keep or delete |
| `write_scene.py` | the same model on a bare opening, for a flow that supplies its own context |
| `build_traditional_map.py` | regenerate the character table from the corpus |
| `build_blind_read.py` | build the unlabelled sheet and its key |
| `score_blind_read.py` | turn keep/drop marks into a two-sided exact test |
| `story_rl.py` | estimate, train, sample, evaluate |

## Open

- **No human has read the drafts.** `📤-OUTPUT/blind-read-v10-2026-11-04.md` holds 20 unlabelled continuations
  from v10 and the base model; the key is `bindings/blind-read/key-v10.json` in the 2026-11-04 capsule. Until
  those are marked, every craft claim here is a judge's.
- **The context axis is unmeasured.** The skill sends 600 characters; the model trained on openings averaging
  42. A 10-task probe found −0.014 and could not resolve an effect at that power. Not "no effect".
