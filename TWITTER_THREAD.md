# Twitter Thread: Story Theory Benchmark Launch

**Format**: Post as continuous thread (Post 1 is main tweet, Posts 2-8 are replies to build narrative)
**Timing**: Space 30-60 minutes apart between posts
**Include images**: benchmark_cost_vs_score.png, benchmark_heatmap.png, benchmark_rankings_stacked.png

## CTA Flow

```
Twitter Thread → GitHub (detailed results)
```

---

## Post 1: Main Launch (Thread Starter)

**Character count: 247/280**

```
Story Theory Benchmark launched

34 tasks | 23 models | 5 narrative theories

Tests whether LLMs understand story structure — using frameworks screenwriters have used for decades.

Open source. Reproducible.

Full results below
github.com/clchinkc/story-bench
```

---

## Post 2: The Problem

**Character count: 271/280**

```
Most LLM benchmarks have verifiable answers.

Code runs. Math checks. Facts are right or wrong.

Creative writing? No ground truth. No automatic verifier.

Story theory provides objective-enough criteria: Hero's Journey has 12 defined beats. Either the model executes them correctly, or it doesn't.
```

---

## Post 3: Key Findings

**Character count: 258/280**

```
Results:

DeepSeek v3.2: 91.9% @ $0.20 (best value)
Claude Opus 4.5: 90.8% (most consistent)
Claude Sonnet 4.5: 90.1%
o3: 89.3%
Gemini 3 Flash: 88.3%

DeepSeek matches frontier quality at 1/14th the cost.

Full leaderboard: github.com/clchinkc/story-bench
```

**[Include: benchmark_rankings_stacked.png]**

---

## Post 4: Agentic Tasks Insight

**Character count: 274/280**

```
Agentic (multi-turn) tasks reveal larger capability gaps:

Standard tasks: ~31% spread between best/worst
Agentic tasks: ~57% spread — nearly 2x

Hardest: Constraint Discovery (59% avg)
Best: GPT-5.2 (81%)
Worst: 26%

Multi-turn exposes gaps that single-shot benchmarks miss.
```

---

## Post 5: Cost-Effectiveness

**Character count: 256/280**

```
Best value (Score²/Cost):

DeepSeek v3.2: 427 (91.9% @ $0.20)
GPT-4o-mini: 365 (81.7% @ $0.18)
Qwen 3: 321 (83.6% @ $0.22)

Premium tier:
Claude Opus: Best consistency
o3: Strong all-rounder

Your use case determines your tier.
```

**[Include: benchmark_cost_vs_score.png]**

---

## Post 6: Why This Matters

**Character count: 268/280**

```
Creative AI has a verification problem.

Code models improve fast because answers are checkable.
Creative writing can't do that directly.

Story theory frameworks provide objective-enough structure — for evaluation now, potentially for training signals later.

It's a foundation, not a complete solution.
```

---

## Post 7: Task Examples

**Character count: 264/280**

```
Example: Multi-Beat Synthesis

Write 3 connected beats where:
- Each has specific requirements
- Beat 3 can't contradict Beat 1
- Character voice stays consistent

Average: 79.8%
Best (o3): 92.9%
Worst: 43.7%

47-point spread = high discrimination between models.
```

**[Include: benchmark_heatmap.png]**

---

## Post 8: Call to Action

**Character count: 259/280**

```
Story Theory Benchmark is open source

→ Run it on your model
→ Add new tasks
→ Check the full leaderboard

If you're building AI writing tools, the methodology might be useful.

Follow for more on LLMs and creative writing!

#LLM #AIWriting #Benchmark
```

---

## Posting Checklist

- [ ] Post 1 (main announcement)
- [ ] Post 2 (the problem — verification gap)
- [ ] Post 3 (key findings + image)
- [ ] Post 4 (agentic tasks insight)
- [ ] Post 5 (cost analysis + image)
- [ ] Post 6 (why this matters)
- [ ] Post 7 (task example + image)
- [ ] Post 8 (call to action)
- [ ] Share Medium post link as bonus tweet after thread completes
- [ ] Respond to comments/retweets

**Each post should reply to the previous one to form a continuous thread.**
