# Reddit Post: r/LLM

**Subreddit**: https://www.reddit.com/r/LLM/

---

## Title

`DeepSeek v3.2 achieves 91.9% on Story Theory Benchmark at $0.20 — Claude Opus scores 90.8% at $2.85. Which is worth it?`

---

## Post Content

I built a benchmark specifically for narrative generation using story theory frameworks (Hero's Journey, Save the Cat, etc.). Tested 23 models. Here's what I found.

### Leaderboard

| Rank | Model | Score | Cost/Gen | Notes |
|------|-------|-------|----------|-------|
| 1 | DeepSeek v3.2 | 91.9% | $0.20 | Best value |
| 2 | Claude Opus 4.5 | 90.8% | $2.85 | Most consistent |
| 3 | Claude Sonnet 4.5 | 90.1% | $1.74 | Balance |
| 4 | Claude Sonnet 4 | 89.6% | $1.59 | |
| 5 | o3 | 89.3% | $0.96 | |
| 6 | kimi-k2-thinking | 88.8% | $0.58 | Solid performer |
| 7 | Gemini 3-flash-preview | 88.3% | $0.59 | Fast & cheap |
| ... | ... | ... | ... | ... |
| 14 | mistral-small-creative | 84.3% | $0.21 | Budget gem |

### Analysis

**DeepSeek v3.2** (Best Value)
- Highest absolute score (91.9%)
- 14× cheaper than Claude Opus
- Strong across most tasks
- Some variance (drops to 72% on hardest tasks)

**Claude Opus** (Premium Consistency)
- Second-highest score (90.8%)
- Most consistent across ALL task types (88-93% range)
- Better on constraint discovery tasks
- 14× more expensive for 1.1% lower score

**The middle ground: Claude Sonnet 4.5**
- 90.1% (only 1.8% below DeepSeek)
- $1.74 (39% of Opus cost)
- Best for cost-conscious production use

### Use case recommendations

- **Unlimited budget**: Claude Opus (consistency across edge cases)
- **Budget-conscious production**: Claude Sonnet 4.5 (90%+ at 39% the cost)
- **High volume / research**: DeepSeek v3.2 (save money for more runs)

### Interesting finding

Multi-turn agentic tasks showed **~2x larger capability spreads** than single-shot tasks:
- Standard tasks: ~31% spread between best/worst
- Agentic tasks: ~57% spread

Models that handle iterative feedback well are qualitatively different from those that don't.

### Links

**GitHub**: https://github.com/clchinkc/story-bench
**Full leaderboard**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md
**Task analysis**: https://github.com/clchinkc/story-bench/blob/main/results/TASK_ANALYSIS.md
**Medium**: https://medium.com/@clchinkc (full analysis post)
**Twitter**: [@firstoryapp](https://twitter.com/firstoryapp)

---

## CTA Flow

```
Reddit → GitHub (detailed results, run yourself)
      → Medium (full analysis post)
      → Twitter (follow for updates)
```

---

## Posting Notes

- Frame as practical decision-making tool
- Lead with cost vs quality trade-off
- Include benchmark_cost_vs_score.png
- Be prepared for model debates
