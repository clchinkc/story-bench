# Story-bench

Story-bench contains a historical script-based narrative benchmark and the planned evaluation environment for author-directed story development with Narrative Craft.

The legacy scores and leaderboards are historical outputs, not validated literary quality or current comparable model rankings. Frozen-code review found coverage-biased averaging, missing judge-field credit, invalid numeric acceptance, truncated context, oracle truth errors and incomplete cost reporting. The old 50/50 composite and ensemble median do not establish objectivity or remove judge bias.

## Current authority

- [Experiment contract](docs/experiment-contract.md): tasks, native environment, four skill/tool arms, dataset/annotation, critics, reward, statistics and spending.
- [Repair register](docs/repair-register.md): observed defects, unresolved candidates, planned regressions and historical disposition.
- [Strategic plan](https://linear.app/firstory/issue/DAILY-105).
- [DAILY-105](https://linear.app/firstory/issue/DAILY-105): owner-approved implementation in dependency-ordered waves. W0 local installed-consumer qualification and W1 enabled offline measurement checks passed; native environment and spend/provider work follow.

Narrative Craft is the domain/runtime authority; this repository owns experiments and adapters. The initial study is revision/continuity across English, Traditional Chinese and Simplified Chinese. Generation, long form, training compatibility and actual author use follow gated phases. No benefit has been established.

## Historical implementation

The existing `run.py`, `src/`, `dataset/`, `config/` and `results/` contain the prior benchmark. Results and original protocols remain preserved. Published counts must be derived from actual manifests, not this README. Historic reports are subject to the [repair register](docs/repair-register.md); valid raw records may be reaggregated, invalid judge inputs require rejudging, and invalid generation/oracle episodes require regeneration under a new version.

The original frozen repo used broad `requirements.txt` bounds. The W1 candidate now captures Python3.14.5 with `pyproject.toml`/`uv.lock` and114 passing regression tests. Prepare with `uv sync --locked`; run `uv run --no-sync pytest -q`. The enabled offline report requires explicit assignment and record manifests; invalid or incomplete data cannot acquire default success credit. Whole legacy live provider/evaluator paths, semantic oracle and legacy rankings remain disabled and unqualified. W3 must qualify a new provider adapter before any live call.

W1 evidence is in capsule `2026-09-21-story-evaluation-plan-consolidation/bindings/execution/w1/acceptance.json`:15 baseline failures and six repair regressions were reproduced independently;114 tests pass with exact IDs. The historical disposition accounts for all869 generation and2,586 evaluation rows while preserving all3,468 raw files. Rejudge/regenerate labels describe future required work, not completed model calls. Unknown costs remain unknown; diagnostics are not literary quality.

## Evidence and approval boundary

The selected source snapshots are Story-bench `2722769c79c7aa79a8bb7a90fa9da724974ee5e8` and Narrative Craft `38428703b976bf0a23c750b14a356d4303230ff6`. W0 qualified the unchanged installed Narrative Craft artifact against the vault consumer candidate `57fe227b4bcb5b5aafa6feb1f0c1ff39dac5f279`: independent runtime 74, Firstory 81, React 40 and Ghost Reader 61 tests passed. Capsule `2026-09-21-story-evaluation-plan-consolidation/bindings/execution/w0/acceptance.json` records exact qualification. Firstory qualification covers local export and the actual receiving schema, not live storage or embedding. Environment isolation, paid dispatch, human calibration and empirical benefit remain later gates. Source tests and synthetic records do not prove literary calibration.

See [LICENSE](LICENSE) for repository terms; that license does not automatically cover external story texts. Publication or distribution of datasets/results requires the recorded rights and privacy decisions.
