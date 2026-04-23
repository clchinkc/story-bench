# Story Theory Benchmark

**A rigorous, reproducible benchmark for evaluating LLM narrative generation capabilities using objective story theory frameworks.**

> "Hard to solve, easy to verify." — 34 tasks × 42 models × 3 evaluators = definitive narrative AI evaluation

## What is Story-Bench?

Story-bench is a production benchmark that evaluates how well LLMs understand and generate narrative structure. Unlike subjective benchmarks that rely on human ratings, story-bench uses **objective, falsifiable story theory frameworks** (Hero's Journey, Save the Cat, etc.) as ground truth — making evaluation rigorous and reproducible.

**Why story theory?** Frameworks like Hero's Journey and Save the Cat are:
- **Objective**: Defined beats with specific narrative functions
- **Falsifiable**: A beat is either correctly executed or it isn't
- **Widely understood**: Proven across thousands of films and novels

This transforms "creative writing assessment" from a subjective judgment into a structured evaluation task.

---

## Architecture

### Two-Component 50/50 Scoring

Every generation is scored using a dual-evaluation system:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Programmatic** | 50% | Word count accuracy, repetition penalties, "slop" detection (overused LLM phrases) |
| **LLM Judge** | 50% | Narrative criteria specific to each task type (beat execution, character consistency, etc.) |

The LLM judge uses a **3-model ensemble** to prevent evaluator bias:
- Claude Haiku 4.5
- Gemini 2.5 Flash
- GPT-5 Mini

Final scores are **median aggregated** across all three evaluators.

### Why This Matters

1. **Programmatic component** catches obvious failures: wrong word counts, excessive repetition, "tapestry of" GPT-isms
2. **LLM judge component** evaluates narrative quality: did the model correctly execute the beat? Does the character voice stay consistent?
3. **3-model ensemble** reduces single-evaluator bias — if one model is too harsh or too lenient, the median corrects

### Scoring Details

#### Programmatic Metrics (50%)

| Metric | Weight | Description |
|--------|--------|-------------|
| Word Count | 40% | Gaussian penalty for deviation from target range |
| Repetition | 35% | Penalizes overused words/phrases (1.0 = no repetition) |
| Slop | 25% | Detects "GPT-isms" like "tapestry", "delve", "realm" |

#### LLM Judge Criteria (50%)

Task-specific criteria with weighted scoring:

| Task Type | Key Criteria |
|-----------|--------------|
| **Beat Interpolation** | Elements (25%), Beat Execution (25%), Must-Not (15%), Character (10%), Bridge (15%), Continuity (10%) |
| **Beat Revision** | Diagnosis (20%), Flaw Fix (20%), Beat Satisfaction (20%), Preservation (10%), Minimal Change (10%) |
| **Multi-Beat Synthesis** | Beat Requirements (40%), Cross-Beat Coherence (35%), Context (15%), Coherence (10%) |
| **Theory Conversion** | Beats (35%), Preservation (30%), Structural (20%), Tone (15%) |

---

## Benchmark Design

### 34 Tasks Across 9 Task Types

#### Standard Tasks (Single-shot generation)

| Task Type | Count | Challenge | Example |
|-----------|-------|-----------|---------|
| Beat Interpolation | 5 | Generate missing story beat A→B→C | "Write the mentor encounter between refusal and threshold crossing" |
| Beat Revision | 5 | Identify and fix incorrect beat | "Find what's wrong with this 'Call to Adventure' segment" |
| Constrained Continuation | 4 | Write with 8-10 simultaneous constraints | "Continue the story with: 3 dialogues, specific ending, no 'tapestry'" |
| Theory Conversion | 4 | Rewrite Story A's beats into Story B framework | "Convert this Hero's Journey story to Save the Cat" |
| Multi-Beat Synthesis | 3 | Write 3 beats with cross-beat dependencies | "Write all three phases of the Threshold Crossing sequence" |

#### Agentic Tasks (Multi-turn, iterative)

| Task Type | Count | Challenge | Key Metric |
|-----------|-------|-----------|-----------|
| Constraint Discovery | 3 | Ask YES/NO questions to find hidden rules | Questions per discovery |
| Planning Execution | 3 | Plan first, then execute | Plan adherence score |
| Iterative Revision | 3 | Improve through rule-based feedback | Improvement trajectory |
| Critique Improvement | 4 | Improve through LLM critic feedback | Score progression |

### Story Theories Covered

- **Hero's Journey** (12 beats) — Joseph Campbell
- **Save the Cat** (15 beats) — Blake Snyder
- **Story Circle** (8 beats) — Dan Harmon
- **Freytag's Pyramid** (5 stages) — Gustav Freytag
- **Three-Act Structure** (3 acts) — Aristotle/Syd Field

---

## Current Leaderboard

*Last updated: 2026-03-06 | 42 models evaluated*

### Top 10 by Score

| Rank | Model | Company | Score | Gen Cost | Value Score |
|------|-------|---------|-------|----------|-------------|
| 1 | glm-5 | Z-Ai | 99.6% | $0.0033 | 30,193.8 |
| 2 | gpt-5.4 | OpenAI | 99.6% | $0.0128 | 7,781.0 |
| 3 | mercury-2 | Inception | 99.1% | $0.0006 | 176,698.6 |
| 4 | qwen3.5-27b | Alibaba | 98.9% | $0.0115 | 8,517.5 |
| 5 | qwen3.5-flash-02-23 | Alibaba | 98.6% | $0.0015 | 66,735.0 |
| 6 | gemini-3.1-flash-lite-preview | Google | 98.4% | $0.0050 | 19,549.9 |
| 7 | claude-opus-4.6 | Anthropic | 98.1% | $0.0226 | 4,248.1 |
| 8 | gemini-3.1-flash-image-preview | Google | 97.4% | $0.0057 | 16,549.6 |
| 9 | deepseek-v3.2 | DeepSeek | 92.2% | $0.1978 | 430.2 |
| 10 | claude-opus-4.5 | Anthropic | 90.9% | $2.8457 | 29.0 |

### Top 10 by Value (Score²/Cost)

| Rank | Model | Score | Gen Cost | Value |
|------|-------|-------|----------|-------|
| 1 | step-3.5-flash | 90.1% | $0.0004 | 184,280.7 |
| 2 | mercury-2 | 99.1% | $0.0006 | 176,698.6 |
| 3 | qwen3.5-flash-02-23 | 98.6% | $0.0015 | 66,735.0 |
| 4 | minimax-m2.5 | 85.2% | $0.0021 | 34,862.9 |
| 5 | glm-5 | 99.6% | $0.0033 | 30,193.8 |
| 6 | gemini-3.1-flash-lite-preview | 98.4% | $0.0050 | 19,549.9 |
| 7 | gemini-3.1-flash-image-preview | 97.4% | $0.0057 | 16,549.6 |
| 8 | qwen3.5-27b | 98.9% | $0.0115 | 8,517.5 |
| 9 | gpt-5.4 | 99.6% | $0.0128 | 7,781.0 |
| 10 | claude-opus-4.6 | 98.1% | $0.0226 | 4,248.1 |

### Performance by Task Type

| Task Type | Avg Score | Spread | Best Model | Worst Model |
|-----------|-----------|--------|------------|-------------|
| Beat Interpolation | 92.9% | 31.7% | o3-mini (99.2%) | minimax-m2 (69.2%) |
| Beat Revision | 90.1% | 31.0% | gemini-2.5-flash (95.8%) | llama-4-maverick (72.6%) |
| Multi-Beat Synthesis | 80.3% | 35.8% | o3 (92.9%) | llama-4-maverick (57.2%) |
| Theory Conversion | 86.6% | 32.9% | deepseek-r1 (96.4%) | kimi-k2-thinking (56.0%) |
| Constrained Continuation | 86.1% | 31.2% | deepseek-v3.2 (98.3%) | ministral-14b-2512 (65.0%) |
| Agentic Constraint Discovery | 60.2% | 57.9% | gpt-5.2 (81.4%) | minimax-m2 (26.0%) |
| Agentic Planning Execution | 88.9% | 30.5% | grok-4 (96.4%) | o3-mini (79.4%) |
| Agentic Iterative Revision | 86.7% | 57.6% | claude-sonnet-4 (97.7%) | llama-4-maverick (39.6%) |
| Critique Improvement | 84.4% | 36.8% | deepseek-r1 (89.8%) | llama-4-maverick (11.5%) |

---

## Key Findings

### 1. Cost-Effectiveness Leaders

**Best-in-class value**: Models like `mercury-2`, `step-3.5-flash`, and `qwen3.5-flash-02-23` achieve 98%+ scores at under $0.002 per generation. The value score (Score²/Cost) reveals that these budget models outperform premium models by 2-4 orders of magnitude on cost-efficiency.

**DeepSeek v3.2** is the sweet spot: 92.2% score at $0.20 — high quality at reasonable cost.

### 2. Quality Leaders

Anthropic models (Claude Opus 4.5/4.6, Claude Sonnet 4/4.5) demonstrate **consistent quality across all task types**. No major weak points, making them reliable for narrative generation.

### 3. Agentic Tasks Reveal Capability Gaps

**Constraint Discovery** is the hardest task type (avg 60.2%, 57.9% spread). This tests strategic questioning — models must ask optimal YES/NO questions to discover hidden rules. Top performer gpt-5.2 (81.4%) vs worst minimax-m2 (26.0%) shows 3× discrimination.

**Iterative Revision** also shows huge spread (57.6%). o3 and Claude Sonnet 4 excel; llama-4-maverick struggles (39.6%).

### 4. Multi-Beat Synthesis Tests Long-Range Planning

This task requires coordinating 3 beats with cross-beat constraints. Average score (80.3%) shows clear capability gaps compared to single-beat tasks. Best discriminator for planning abilities.

### 5. Beat Interpolation Near-Ceiling for Top Models

Top models (o3-mini, gpt-5, Claude Opus 4.5) achieve 99%+ on beat interpolation — this task is largely solved at the frontier.

---

## How to Run

### Installation

```bash
git clone https://github.com/clchinkc/story-bench.git
cd story-bench
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### Quick Commands

```bash
# Check benchmark status
python run.py --status

# Run benchmark for a model
python run.py --gen-model "anthropic/claude-opus-4.6"

# Run specific task type
python run.py --gen-model "anthropic/claude-opus-4.6" --task-type "multi_beat_synthesis"

# Generate leaderboard
python run.py --leaderboard

# List missing evaluations for a model
python run.py --list-missing "anthropic/claude-opus-4.6" -v

# Compare two benchmark runs
python run.py --compare results_A.json results_B.json
```

### Python API

```python
import sys
sys.path.insert(0, 'src')

from results_db import ResultsDatabase

# Get leaderboard
db = ResultsDatabase()
leaderboard = db.generate_leaderboard_md()
print(leaderboard)

# Get task analysis
analysis = db.generate_task_analysis_md()
print(analysis)
```

---

## Repository Structure

```
story-bench/
├── run.py                    # CLI entry point
├── visualize.py               # Visualization dashboard
├── src/
│   ├── generator.py           # Story generation
│   ├── evaluator.py           # LLM-as-judge evaluation
│   ├── scoring.py             # Two-component scoring
│   ├── results_db.py          # JSON database
│   ├── agentic_generator.py   # Multi-turn generation
│   ├── agentic_evaluator.py   # Agentic task evaluation
│   └── comparison.py          # A/B model comparison
├── dataset/
│   └── tasks/                 # 34 task YAML files
├── config/
│   └── models.yaml           # Model configurations
├── results/
│   ├── LEADERBOARD.md        # Full rankings
│   ├── TASK_ANALYSIS.md      # Task-level breakdown
│   ├── benchmark_results.json # Full database
│   └── evaluations/           # Individual evaluation YAMLs
└── docs/
    └── ai-evaluation-criteria.md  # Evaluation rubric
```

---

## The Engineering Behind Story-Bench

### Systematic Evaluation Design

This wasn't a quick hack — it's a **systematic evaluation platform** built with production principles:

1. **Three-source verification**: Programmatic metrics (word count, repetition, slop) catch objective failures; LLM judge evaluates narrative quality; ensemble aggregation reduces bias
2. **Reproducible YAML format**: Every generation and evaluation is saved as versioned YAML with full metadata (tokens, cost, timestamp)
3. **A/B comparison tooling**: `comparison.py` enables head-to-head model comparison with statistical breakdowns

### Evidence of Production Thinking

- **Incremental execution**: `run.py` supports `--dry-run`, `--force`, and resumable runs — handles API failures gracefully
- **Cost tracking**: Every generation logs prompt/completion/reasoning tokens and cost; total benchmark cost: $43.18
- **Multi-turn agentic tasks**: Full implementation of constraint discovery, planning-execution, iterative revision, and critique improvement
- **Slop detection**: Custom dictionary of 60+ "GPT-isms" with weighted penalties — not just word matching but phrase-level detection

### Design Patterns

- **Median aggregation**: Reduces outlier influence from any single evaluator
- **Gaussian word count scoring**: Smooth penalty curve vs. harsh binary pass/fail
- **Value metric (Score²/Cost)**: Quadratically rewards quality, penalizes cost — practical for real-world model selection
- **Task type routing**: Different scoring criteria per task type reflects actual narrative requirements

---

## Citation

If you use Story Theory Benchmark in your research:

```bibtex
@software{story_theory_benchmark_2025,
  title = {Story Theory Benchmark: Narrative Generation Evaluation Framework},
  author = {Kevin Chin},
  year = {2025},
  url = {https://github.com/clchinkc/story-bench},
  note = {34 tasks across 9 task types, 42 models evaluated, LLM-as-judge with 3-model ensemble}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Last updated: 2026-04-23*