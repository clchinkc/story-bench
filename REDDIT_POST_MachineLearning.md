# Reddit Post: r/MachineLearning

**Subreddit**: https://www.reddit.com/r/MachineLearning/
**Flair**: [R] Research

---

## Title

`[R] Story Theory Benchmark: Using narrative frameworks to evaluate LLM creative capabilities (34 tasks, 21 models, open-source)`

---

## Post Content

**The problem**: Most LLM benchmarks work because they have verifiable answers. Code runs. Math checks. Facts are right or wrong. Creative writing has no such ground truth — no automatic verifier, no clear reward signal for training or evaluation.

**The approach**: Classical story theory frameworks (Hero's Journey, Save the Cat, Story Circle, etc.) provide surprisingly objective criteria. The Hero's Journey has 12 defined beats. Save the Cat has 15. Either the model executes a beat correctly, or it doesn't. This isn't perfect — structure isn't everything — but it's objective enough to enable rigorous comparison.

**Story Theory Benchmark** tests whether LLMs understand narrative structure using these frameworks.

---

## Key Results

**Model Rankings** (21 models evaluated):

| Model | Score | Cost/Gen | Notes |
|-------|-------|----------|-------|
| DeepSeek v3.2 | 91.9% | $0.20 | Best value |
| Claude Opus 4.5 | 90.8% | $2.85 | Most consistent |
| Claude Sonnet 4.5 | 90.1% | $1.74 | |
| o3 | 89.3% | $0.96 | |
| Gemini 3 Flash | 88.3% | $0.59 | |

DeepSeek v3.2 matches frontier quality at 1/14th the cost — unexpected for narrative tasks.

**Agentic vs Standard Tasks**:
- Standard (single-shot) tasks: ~31% average spread between best/worst models
- Agentic (multi-turn) tasks: ~57% average spread — nearly 2x

Multi-turn tasks (iterative revision, constraint discovery, planning execution) expose capability gaps that single-shot benchmarks miss.

**Hardest Task**: Constraint Discovery (asking strategic YES/NO questions to uncover hidden story rules)
- Average: 59%
- Best (GPT-5.2): 81%
- Worst: 26%

---

## Methodology

- **34 tasks** across 9 types (5 standard + 4 agentic)
- **5 story frameworks**: Hero's Journey, Save the Cat, Story Circle, Freytag's Pyramid, Three-Act Structure
- **3 evaluators** per generation (Claude Haiku 4.5, Gemini 2.5-Flash, GPT-5-Mini) with median aggregation
- **Scoring**: 50% programmatic (word count, repetition, coherence) + 50% LLM judge (task-specific criteria)
- **714 generations, 2,145 evaluations**

All tasks are YAML-defined for reproducibility.

---

## Why This Matters

The verification gap in creative AI isn't just an evaluation problem — it's a training bottleneck. Code models improve rapidly because you can generate training data with verifiable solutions. Creative writing can't do this directly.

Story theory frameworks suggest a path forward: objective-enough structure to enable both evaluation and potential training signals. This won't solve the full problem (prose quality, emotional resonance, originality), but it's a foundation.

---

## Links

- **GitHub**: https://github.com/clchinkc/story-bench
- **Full Leaderboard**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md
- **Task Analysis**: https://github.com/clchinkc/story-bench/blob/main/results/TASK_ANALYSIS.md
- **Medium**: https://medium.com/@clchinkc (full analysis post)
- **Twitter**: https://twitter.com/firstoryapp

---

## CTA Flow

```
Reddit → GitHub (run it yourself, see full results)
      → Medium (full analysis post)
      → Twitter (follow for updates)
```

---

## Posting Notes

- Use [R] flair for research
- Be prepared for methodology questions
- Visualizations help: benchmark_cost_vs_score.png, benchmark_heatmap.png
