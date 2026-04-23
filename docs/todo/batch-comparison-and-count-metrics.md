# Element Counting & A/B Batch Comparison

**Goal**: Enhance `story-bench` to support deep structural evaluation using count-correct metrics and enable direct A/B batch comparisons of different models/prompts.

## Context
Currently, `story-bench` relies on LLM-as-a-judge for qualitative analysis. Inspired by `np-tp-benchmark`, we need to introduce a quantitative "Count-Correctness" formula (`1 - |gt - pd| / (gt + pd)`) specifically for structural elements, and a dedicated module to compare two benchmark runs side-by-side.

## Tasks

- [ ] **1. Implement `element_count_score` Metric**
  - In `src/scoring.py`, add a new programmatic scoring function for tasks that have strict structural counts (e.g., `constrained_continuation` where the prompt demands exactly N constraints be met).
  - Formula: `score = 1 - abs(gt_count - pd_count) / (gt_count + pd_count)` (handling 0/0 edge cases).
  - Integrate this score into the `calculate_programmatic_scores` pipeline (with appropriate weighting).

- [ ] **2. Create `comparison.py` Module**
  - Create `src/comparison.py`.
  - Define a `StoryBenchmarkComparison` class that takes two `benchmark_results.json` files (e.g., `left_result` and `right_result`).

- [ ] **3. Implement Comparison Analytics**
  - Add methods to calculate the delta (absolute and percentage improvement) across different task types.
  - **Task Type Breakdown**: Which specific narrative tasks did Model B improve on compared to Model A?
  - **Component Breakdown**: Did Model B improve Programmatic scores (less slop) or LLM Judge scores (better narrative)?

- [ ] **4. Add CLI Command for Comparison**
  - Update `run.py` to accept a new command: `python run.py --compare results_A.json results_B.json`.
  - Output a formatted Markdown table summarizing the head-to-head performance.