# Medium Post: Story Theory Benchmark

**Publication Target**: Medium (personal or Towards Data Science)
**URL**: https://medium.com/@clchinkc
**Word Count**: ~1,100

---

## Title

**Why Most LLM Benchmarks Miss What Matters for Creative Writing**

*Subtitle*: And how story theory provides an unexpected solution

---

## Post Content

Most LLM benchmarks work because they have answers you can check.

Code benchmarks run the program. Math benchmarks verify the calculation. Trivia benchmarks look up the fact. There's a right answer — a ground truth that enables evaluation and, increasingly, training through reinforcement learning.

Creative writing doesn't work this way.

Ask an LLM to write a scene. Is it good? That depends on who you ask. There's no objective test, no automatic verification, no reward signal you can use at scale. This is why progress on creative AI benchmarks has lagged far behind coding and math.

I spent several months exploring this problem while building tools for fiction writers. The solution I found came from an unexpected source: **story theory frameworks that have been taught in film schools for decades**.

---

### The Verification Gap

Modern LLM improvements lean heavily on verifiable rewards. Models like GPT-5, Gemini 3, and Claude Opus 4.5 use sophisticated reasoning chains, but the training signal that guides them still relies on being able to check answers.

For creative writing, what would that check look like?

- "Is this paragraph engaging?" — *Depends on the reader*
- "Is this character believable?" — *Subjective judgment*
- "Does this plot work?" — *Genre and audience dependent*

Without objective criteria, you can't build reliable evaluation frameworks. Without reliable evaluation, you can't systematically improve models for creative tasks.

---

### Story Theory as Objective Criteria

Here's what I realized: **classical story frameworks provide surprisingly objective criteria for creative work**.

The Hero's Journey has 12 defined beats. Save the Cat has 15. These aren't vague suggestions — they're structural frameworks used by professional screenwriters and novelists. Each beat has specific requirements. Either the model executes "Crossing the Threshold" correctly, or it doesn't.

| Framework | Beats | Origin |
|-----------|-------|--------|
| Hero's Journey | 12 | Joseph Campbell |
| Save the Cat | 15 | Blake Snyder |
| Story Circle | 8 | Dan Harmon |
| Three-Act Structure | 3 | Aristotle |
| Freytag's Pyramid | 5 | Gustav Freytag |

This isn't a perfect solution — story theory captures structure, not prose quality or emotional resonance. But it provides **objective-enough criteria** to enable rigorous comparison between models.

---

### The Benchmark

I built **Story Theory Benchmark** to test whether LLMs can reason about narrative structure.

**34 tasks** across 9 types. **25 frontier models** evaluated. **5 story theory frameworks** as evaluation criteria.

The tasks range from straightforward (generate a missing story beat) to complex (write three interconnected beats while maintaining cross-dependencies and character consistency).

#### Task Types

**Standard tasks** test single-shot generation:
- Beat Interpolation — generate the missing beat between two given beats
- Beat Revision — identify and correct an incorrectly executed beat
- Multi-Beat Synthesis — coordinate three beats with cross-constraints

**Agentic tasks** test multi-turn interaction:
- Constraint Discovery — ask strategic questions to uncover hidden story rules
- Planning Execution — create a plan, then execute it
- Iterative Revision — improve through structured feedback loops

---

### Key Findings

#### DeepSeek v3.2 Leads — At 1/14th the Cost

This was unexpected.

| Model | Score | Cost/Generation | Value |
|-------|-------|-----------------|-------|
| **DeepSeek v3.2** | **91.9%** | **$0.20** | **426.9** |
| Claude Opus 4.5 | 90.8% | $2.85 | 29.0 |
| Claude Sonnet 4.5 | 90.1% | $1.74 | 46.6 |
| o3 | 89.3% | $0.96 | 83.2 |
| Gemini 3 Flash | 88.3% | $0.59 | 132.3 |

DeepSeek v3.2 achieves near-frontier quality on narrative tasks at a fraction of the cost. For high-volume creative applications, this changes the calculus significantly.

*(Value = Score² / Cost — quadratically rewards quality while penalizing cost)*

#### Agentic Tasks Show Larger Capability Gaps

Standard single-shot tasks showed **~31% average spread** between best and worst models.

Agentic multi-turn tasks showed **~57% average spread** — nearly double.

| Task Type | Average Spread | Most Revealing |
|-----------|---------------|----------------|
| Iterative Revision | 57.6% | 90.2% spread on hardest variant |
| Constraint Discovery | 57.3% | Tests strategic questioning |
| Beat Interpolation | 31.7% | Near-ceiling for top models |

The hardest task was **Constraint Discovery** — asking YES/NO questions to uncover hidden story rules. Average score: 59%. Best model (GPT-5.2): 81%. Worst: 26%.

Multi-turn tasks expose gaps that single-shot benchmarks don't reveal. Models that score similarly on simple generation tasks show wide variance when required to iterate, plan, and respond to feedback.

#### Claude Models Show Highest Consistency

While DeepSeek wins on value, **Claude models scored between 88-93% on every task type** — no weak spots.

Other models showed more variance. GPT-5 excels at Beat Revision (97.2%) but struggles with Constraint Discovery. Grok 4 leads Planning Execution (96.4%) but lags elsewhere.

For production systems requiring reliable performance across diverse narrative tasks, consistency matters.

---

### What This Means for Choosing a Model

If you're using AI to assist your writing:

**For cost-sensitive, high-volume work**: DeepSeek v3.2 offers frontier-quality narrative understanding at budget pricing.

**For maximum reliability**: Claude models provide the most consistent performance across different narrative tasks.

**For complex multi-step work** (revision, iteration, feedback): Check agentic task scores, not just single-shot generation benchmarks.

The full leaderboard with task-by-task breakdowns is in the GitHub repository.

---

### Update: Four Strong Additions

Since the initial benchmark run, four new models have been evaluated:

**kimi-k2-thinking** (Rank #7, 88.7%) — Moonshotai's reasoning model shows surprising strength on narrative tasks. Scored 95.9% on LLM judge evaluation and handles both standard and agentic tasks well. At $0.58/M, it's a solid mid-range option.

**glm-4.7** (Rank #6, 88.8%) — Z-AI's latest model edges out kimi-k2-thinking for 6th place. Strong across both standard and agentic tasks. At $0.61/gen, it offers competitive value for creative applications.

**minimax-m2.1** (Rank #9, 86.9%) — MiniMax's updated model scores the highest LLM Judge rating (97.0%) of any model tested, though programmatic scores bring the overall down. Major improvement over minimax-m2 (77.1% → 86.9%). At $0.38/gen, excellent value.

**mistral-small-creative** (Rank #18, 84.3%) — The budget alternative that actually competes. Outperforms similarly-priced models like gpt-4o-mini on narrative structure. Strong on single-shot tasks (beat generation, theory conversion) but weaker on multi-turn work. At $0.21/M, it's the best value in the budget tier.

All four show that the narrative task landscape is competitive beyond just the frontier models.

---

### Beyond Benchmarks

The verification gap in creative AI isn't just an evaluation problem — it's a training bottleneck.

Code models improve rapidly because you can generate training data with verifiable solutions. Creative writing can't do this directly. Story theory frameworks suggest a path forward: **objective-enough structure to enable both evaluation and potential training signals**.

This won't solve the full problem. Structure isn't everything — prose quality, emotional resonance, and originality matter too. But it's a foundation for more rigorous creative AI development.

---

### Try It

The benchmark is open source.

34 tasks. 25 models evaluated. Full results and methodology documented.

If you're working on AI writing tools or interested in narrative evaluation, I'd love to hear what you find.

**GitHub**: [github.com/clchinkc/story-bench](https://github.com/clchinkc/story-bench)

**Twitter**: [@firstoryapp](https://twitter.com/firstoryapp)

---

## Publishing Checklist

- [ ] Copy content to Medium editor
- [ ] Add header image (benchmark_cost_vs_score.png or custom)
- [ ] Add inline images for tables if needed
- [ ] Set tags: `LLM`, `Benchmark`, `AI Writing`, `Creative Writing`, `Machine Learning`
- [ ] Preview and publish
- [ ] Share on Twitter with link
- [ ] Cross-post link to Reddit threads

## CTA Flow

```
Medium Post
    ↓
GitHub (detailed results, run yourself)
    ↓
Twitter (follow for updates)
```
