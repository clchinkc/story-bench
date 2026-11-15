# Function ledger (structural metric)

`src/function_ledger.py` reports **order** findings for a scene-level annotation of
Propp's 31 functions. It is not a coverage score.

## Why not coverage

Propp's functions may be omitted but do not reorder. A story that fills every slot
by padding is worse than one that omits half of them, so a coverage number would
reward padding. The only outputs are:

- `missing-predecessor` — a function appears whose predecessor never does;
- `order` — a function appears before a predecessor that comes later;
- `no-branch` — neither opening branch (8a villainy, 8b lack) is present, which
  means the ledger does not apply to this story.

Coverage is emitted for completeness and must not be ranked. It divides by the 31
slots one sequence can occupy, not by the 32 ids in `FUNCTION_SEQUENCE`: a story
takes 8a or 8b and never both, so dividing by 32 could never reach 1.

## Annotation format

```json
{"annotations": [{"function": "8a", "scene": 3, "evidence": "..."}]}
```

One entry per occurrence. The first scene wins for ordering. `function` is a
string id from `FUNCTION_SEQUENCE` (`"1"`..`"31"`, plus `"8a"` and `"8b"`).

## Running

```bash
uv run --no-sync --offline python -m src.function_ledger --annotations x.json
```

Exit 0 when clean, 1 when any finding is present, so a pipeline step can gate on it.

## Status

The metric is implemented and tested; it is **not yet scored in the evaluation
pipeline** because the pipeline has no scene-level function annotations to consume.
The first dependency is an annotation pass over a small corpus with two independent
annotators; until that exists, wiring the number into scoring would produce a
metric over zero rows.
