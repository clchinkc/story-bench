# Shapes

A **shape** is one task family: a corpus, a reward and the environment that turns a prompt into a rollout.
Three exist. Only `prose` is in daily use; the other two are kept because they still run, and
`tests/test_shapes.py` fails if either stops.

| shape | task | corpus | reward | status |
|---|---|---|---|---|
| `prose` | continue a scene opening in the author's voice | `voice.json` — 141 tasks, 108 train / 33 held out | judge craft × length, gated by a lint veto and a character veto | **the product**, see [writing-tool](writing-tool.md) |
| `repair` | repair one broken continuation into a passing one | `tasks_repair.json` — 12 train / 8 held out | deterministic lint and parser checks, no judge | experiment |
| `shape-a` | answer a temporal question about a passage | `shape_a.json` — 32 train / 16 held out | exact-match against the task's own truth | experiment |

Every shape is selected the same way and only the corpus and reward differ:

```
uv run python3 scripts/bench.py estimate --shape prose --steps 68
uv run python3 scripts/bench.py train    --shape prose --steps 68
uv run python3 scripts/bench.py sample   --shape repair --half held_out --out base.jsonl
uv run python3 scripts/bench.py evaluate --shape shape-a --generations trained.jsonl --against base.jsonl
```

## What keeps a shape alive

Each shape module pins its own corpus path, defines its own split, and refuses a split that overlaps:

- `prose.py` → `voice.json`, judged by the frozen assessor
- `corpus.py` → `tasks_repair.json`, judged by the lint layer
- `shape_a.py` → `shape_a.json`, judged by exact match

`tests/test_shapes.py` asserts three things per shape: both halves exist, they do not overlap, and the builder
produces a batch. That is the minimum a shape cannot survive without. It deliberately does not assert the
reward, the difficulty or convergence — a test that proves its own fixture is worse than no test.

## Adding a shape

1. A module that holds the task type, `load_tasks()`, `load_split()` and a `load_split().by_id()`.
2. A dataset builder that takes `batch_size`, `group_size`, `n_batches`.
3. An entry in `SHAPE_BUILDERS` and `RECIPES` in `train.py`, and in `SHAPES` in `sample.py`/`evaluate.py`.
4. A row in the table above, and a case in `tests/test_shapes.py`.

If a shape cannot hold both halves apart, it is not a shape — it is a demo.
