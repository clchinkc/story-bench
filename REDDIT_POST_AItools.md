# Reddit Post: r/AItools

**Subreddit**: https://www.reddit.com/r/AItools/

---

## Title

`Story Theory Benchmark: Open-source evaluation framework for comparing narrative AI (21 models, 34 tasks)`

---

## Post Content

**Story Theory Benchmark** — an open-source framework for evaluating AI models on narrative generation using classical story theory criteria.

### Model Comparison

| Model | Score | Cost/Gen | Value Score |
|-------|-------|----------|-------------|
| DeepSeek v3.2 | 91.9% | $0.20 | 426.9 |
| Claude Opus 4.5 | 90.8% | $2.85 | 29.0 |
| Claude Sonnet 4.5 | 90.1% | $1.74 | 46.6 |
| o3 | 89.3% | $0.96 | 83.2 |
| Gemini 3 Flash | 88.3% | $0.59 | 132.3 |
| DeepSeek R1 | 86.9% | $0.42 | 181.5 |

*Value = Score²/Cost (rewards quality quadratically)*

DeepSeek v3.2 achieves near-frontier quality at 1/14th the cost of Claude Opus.

### Features

- **34 tasks** across 9 types (standard + agentic multi-turn)
- **5 story frameworks**: Hero's Journey, Save the Cat, Story Circle, Freytag's Pyramid, Three-Act Structure
- **3 evaluators** per generation with median aggregation
- **YAML-defined tasks** — fork and modify for your use case
- **Reproducible** — run on your own machine

### Best for

- Comparing narrative generation quality across models
- Understanding cost-effectiveness trade-offs
- Evaluating whether models improve through feedback
- Testing constraint-handling and planning ability

### Links

**GitHub**: https://github.com/clchinkc/story-bench
**Leaderboard**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md
**Medium**: https://medium.com/@clchinkc (full analysis post)
**Twitter**: [@firstoryapp](https://twitter.com/firstoryapp)

---

## CTA Flow

```
Reddit → GitHub (run it yourself, see full results)
      → Medium (full analysis post)
      → Twitter (follow for updates)
```

---

## Posting Notes

- Share the comparison table prominently
- Emphasize open-source + reproducibility
- Include benchmark_cost_vs_score.png visualization
- Be ready for questions about setup and usage
