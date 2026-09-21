# Story-bench

Story-bench contains a historical script-based narrative benchmark and the planned evaluation environment for author-directed story development with Narrative Craft.

The legacy scores and leaderboards are historical outputs, not validated literary quality or current comparable model rankings. Frozen-code review found coverage-biased averaging, missing judge-field credit, invalid numeric acceptance, truncated context, oracle truth errors and incomplete cost reporting. The old 50/50 composite and ensemble median do not establish objectivity or remove judge bias.

## Current authority

- [Experiment contract](docs/experiment-contract.md): tasks, native environment, four skill/tool arms, dataset/annotation, critics, reward, statistics and spending.
- [Repair register](docs/repair-register.md): observed defects, unresolved candidates, planned regressions and historical disposition.
- [Strategic plan](https://linear.app/firstory/issue/DAILY-105).
- [DAILY-105](https://linear.app/firstory/issue/DAILY-105): implementation Backlog, awaiting approval.

Narrative Craft is the domain/runtime authority; this repository owns experiments and adapters. The initial study is revision/continuity across English, Traditional Chinese and Simplified Chinese. Generation, long form, training compatibility and actual author use follow gated phases. No benefit has been established.

## Historical implementation

The existing `run.py`, `src/`, `dataset/`, `config/` and `results/` contain the prior benchmark. Results and original protocols remain preserved. Published counts must be derived from actual manifests, not this README. Historic reports are subject to the [repair register](docs/repair-register.md); valid raw records may be reaggregated, invalid judge inputs require rejudging, and invalid generation/oracle episodes require regeneration under a new version.

The frozen repo uses `requirements.txt` with broad lower bounds and has no package/lockfile or Narrative Craft consumer binding. Query 2 must capture a reproducible environment before measurements. Do not assume legacy CLI operations are offline: generation, grading, retries and force/recompute options can call paid providers. Query 1 ran no such calls or improvements.

## Evidence and approval boundary

The selected source snapshots are Story-bench `2722769c79c7aa79a8bb7a90fa9da724974ee5e8` and Narrative Craft `38428703b976bf0a23c750b14a356d4303230ff6`. The installed Narrative Craft artifact exists, but native consumer parity remains blocked. [Baseline evidence](https://linear.app/firstory/issue/DAILY-105) records exact qualification. Source tests and synthetic records do not prove literary calibration.

See [LICENSE](LICENSE) for repository terms; that license does not automatically cover external story texts. Publication or distribution of datasets/results requires the recorded rights and privacy decisions.
