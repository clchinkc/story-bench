# Reddit Post: r/ArtificialIntelligence

**Subreddit**: https://www.reddit.com/r/ArtificialIntelligence/

---

## Title

`Story Theory Benchmark: Multi-turn agentic tasks reveal ~2x larger capability gaps than single-shot benchmarks`

---

## Post Content

Released an open-source benchmark testing LLM narrative generation using classical story theory frameworks. The most interesting finding isn't about which model wins — it's about **what kind of tasks reveal capability differences**.

### The finding

- **Standard (single-shot) tasks**: ~31% average spread between best and worst models
- **Agentic (multi-turn) tasks**: ~57% average spread — nearly 2x

Multi-turn tasks (iterative revision, constraint discovery, planning-then-execution) expose gaps that single-shot benchmarks don't reveal.

### Why this matters

Most benchmarks test single-shot generation. But real-world use often involves iteration — revising based on feedback, discovering constraints, planning before execution.

Models that score similarly on simple generation tasks show **wide variance** when required to iterate, plan, and respond to feedback.

### Example: Iterative Revision task

| Model | Score |
|-------|-------|
| Claude Sonnet 4 | 90.8% |
| o3 | 93.9% |
| DeepSeek v3.2 | 89.5% |
| Llama 4 Maverick | 39.6% |

**51-point spread** on a single task type. This isn't about "bad at narrative" — it reveals differences in multi-turn reasoning capability.

### Model rankings (overall)

| Model | Score | Cost/Gen |
|-------|-------|----------|
| DeepSeek v3.2 | 91.9% | $0.20 |
| Claude Opus 4.5 | 90.8% | $2.85 |
| Claude Sonnet 4.5 | 90.1% | $1.74 |
| o3 | 89.3% | $0.96 |

DeepSeek leads on value. Claude leads on consistency.

### Hardest task: Constraint Discovery

Asking strategic YES/NO questions to uncover hidden story rules.
- Average: 59%
- Best (GPT-5.2): 81%
- Worst: 26%

This tests strategic questioning, not just generation.

### Open questions

- Do models that improve through feedback develop coherent internal representations?
- Is narrative understanding distinct from general reasoning ability?
- Do these results predict downstream task performance?

### Links

**GitHub**: https://github.com/clchinkc/story-bench
**Full analysis**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md
**Task breakdown**: https://github.com/clchinkc/story-bench/blob/main/results/TASK_ANALYSIS.md
**Medium**: https://medium.com/@clchinkc (full analysis post)
**Twitter**: [@firstoryapp](https://twitter.com/firstoryapp)

---

## CTA Flow

```
Reddit → GitHub (detailed methodology, run yourself)
      → Medium (full analysis post)
      → Twitter (follow for updates)
```

---

## Posting Notes

- Lead with the agentic vs standard finding
- Use Iterative Revision as concrete example
- Pose open questions to encourage discussion
- Share visualizations showing score spreads
