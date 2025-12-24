# Reddit Post: r/WritingWithAI

**Subreddit**: https://www.reddit.com/r/WritingWithAI/

---

## Title

`Story Theory Benchmark: Which AI models actually understand narrative structure? (34 tasks, 25 models compared)`

---

## Post Content

If you're using AI to help with fiction writing, you've probably noticed some models handle story structure better than others. But how do you actually compare them?

I built **Story Theory Benchmark** — an open-source framework that tests AI models against classical story frameworks (Hero's Journey, Save the Cat, Story Circle, etc.). These frameworks have defined beats. Either the model executes them correctly, or it doesn't.

### What it tests

- Can your model execute story beats correctly?
- Can it manage multiple constraints simultaneously?
- Does it actually improve when given feedback?
- Can it convert between different story frameworks?

### Results snapshot

| Model | Score | Cost/Gen | Best for |
|-------|-------|----------|----------|
| DeepSeek v3.2 | 92.2% | $0.20 | Best value |
| Claude Opus 4.5 | 90.9% | $2.85 | Most consistent |
| Claude Sonnet 4.5 | 90.1% | $1.74 | Balance |
| o3 | 89.5% | $0.96 | Long-range planning |
| glm-4.7 | 88.8% | $0.61 | Strong value |
| kimi-k2-thinking | 88.7% | $0.58 | Reasoning strength |
| minimax-m2.1 | 86.9% | $0.38 | Highest LLM judge (97%) |
| mistral-small-creative | 84.3% | $0.21 | Budget option |

DeepSeek matches frontier quality at a fraction of the cost. Mistral-small-creative offers solid budget alternative for straightforward narrative generation.

### Why multi-turn matters for writers

Multi-turn tasks (iterative revision, feedback loops) showed nearly **2x larger capability gaps** between models than single-shot generation.

Some models improve substantially through feedback. Others plateau quickly. If you're doing iterative drafting with AI, this matters more than single-shot benchmarks suggest.

### Try it yourself

The benchmark is open source. You can test your preferred model or explore the full leaderboard.

**GitHub**: https://github.com/clchinkc/story-bench
**Full leaderboard**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md
**Medium**: https://medium.com/@clchinkc (full analysis post)
**Twitter**: [@firstoryapp](https://twitter.com/firstoryapp)

---

## CTA Flow

```
Reddit → GitHub (explore results, test your model)
      → Medium (full analysis post)
      → Twitter (follow for updates)
```

---

## Posting Notes

- Emphasize practical utility for writers
- Lead with the model comparison angle
- Share benchmark_heatmap.png for visual
- Engage with comments about specific models or workflows
