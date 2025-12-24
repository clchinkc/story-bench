# Story Theory Benchmark

A rigorous benchmark for evaluating LLM narrative generation capabilities across mainstream story theory frameworks. **Hard to solve, easy to verify.**

## Overview

This benchmark evaluates how well language models understand and apply narrative structure theory through **9 task types** (5 standard + 4 agentic):

### Standard Tasks (Single-shot)

| Task Type | Count | Challenge | Verification |
|-----------|-------|-----------|--------------|
| **Beat Interpolation** | 5 | Generate missing beat between two given beats | Logical bridge + beat elements |
| **Beat Revision** | 5 | Diagnose and fix incorrectly executed beat (1 has no flaw) | Correct diagnosis + flaw corrected |
| **Constrained Continuation** | 4 | Continue story with 8-10 constraints | Partial credit on each constraint |
| **Theory Conversion** | 4 | Rewrite from Theory A to Theory B | Structure + core preservation |
| **Multi-Beat Synthesis** | 3 | Write 3 beats with cross-beat constraints | Beat-specific + setup/payoff |

### Agentic Tasks (Multi-turn)

| Task Type | Count | Challenge | Key Metric |
|-----------|-------|-----------|------------|
| **Constraint Discovery** | 3 | Ask YES/NO questions to discover hidden constraints | Discovery efficiency |
| **Planning Execution** | 3 | Create plan/outline, then execute it | Plan quality + execution |
| **Iterative Revision** | 3 | Improve through rule-based feedback rounds | Improvement trajectory |
| **Critique Improvement** | 4 | Improve through LLM critic feedback rounds | Score trajectory + diminishing returns |

**Total**: 34 tasks across 9 task types

### Story Theories Covered

- **Hero's Journey** (12 beats) - Joseph Campbell
- **Save the Cat** (15 beats) - Blake Snyder
- **Story Circle** (8 beats) - Dan Harmon
- **Freytag's Pyramid** (5 stages) - Gustav Freytag
- **Three-Act Structure** (3 acts) - Aristotle/Syd Field

## Benchmark Design

**Scale**: 34 tasks x N models x 1 sample = flexible generations

**Model Tiers**:
- **Strong** ($8-15/1M output tokens): Performance ceiling
- **Cheap** ($0.20-0.80/1M output tokens): Cost-effective baseline

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/story-theory-benchmark.git
cd story-theory-benchmark

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenRouter API key
```

## Quick Start

### Using the CLI (Recommended)

```bash
# Check current status
python run.py --status

# Run benchmark for a model (all task types including agentic)
python run.py --gen-model "anthropic/claude-sonnet-4"

# Run specific task type only
python run.py --gen-model "anthropic/claude-sonnet-4" --task-type "beat_interpolation"
python run.py --gen-model "anthropic/claude-sonnet-4" --task-type "critique_improvement"
python run.py --gen-model "anthropic/claude-sonnet-4" --task-type "agentic_constraint_discovery"

# Run with specific evaluator models
python run.py --gen-model "anthropic/claude-sonnet-4" --eval-model "anthropic/claude-haiku-4.5" "openai/gpt-5-mini"

# Run multiple generation models
python run.py --gen-model "anthropic/claude-sonnet-4" "openai/gpt-4o"

# Preview what would be run (dry run)
python run.py --gen-model "anthropic/claude-sonnet-4" --dry-run

# Show what's missing for a model
python run.py --list-missing "anthropic/claude-sonnet-4" -v

# Generate leaderboard
python run.py --leaderboard

# Clean up failed generations/evaluations for retry
python run.py --clean-failed
python run.py --clean-failed "anthropic/claude-sonnet-4"  # Filter by model
```

### Multi-Evaluator Support

The benchmark supports **multiple evaluator models** for more robust evaluation:

- Each generation is evaluated by all specified evaluator models
- Pass/fail is determined by **median** across evaluators
- Results from each evaluator are stored separately for transparency

Default evaluators: `anthropic/claude-haiku-4.5`, `openai/gpt-5-mini`, `google/gemini-2.5-flash`

### Incremental Evaluation

The benchmark supports **incremental updates**:

- **Add new model**: Only runs that model on all tasks
- **Add new evaluator**: Only runs new evaluator on existing generations
- **Add new task**: Only runs that task on all existing models
- **Re-run failed**: Use `python run.py --clean-failed` to remove failed records, then re-run

### Python API

```python
import sys
sys.path.insert(0, 'src')

from results_db import ResultsDatabase
from utils import load_all_tasks

# Check status
db = ResultsDatabase()
db.print_status()

# Get missing work for a model
task_ids = [t["task_id"] for t in load_all_tasks()]
missing = db.get_missing_generations(task_ids, ["anthropic/claude-sonnet-4"], samples=3)
print(f"Missing generations: {len(missing)}")

# Generate leaderboard
print(db.generate_leaderboard_md())
```

## Project Structure

```
story-theory-benchmark/
├── run.py                 # CLI entry point
├── visualize.py           # Visualization dashboard generator
├── dataset/
│   ├── tasks/
│   │   ├── beat_interpolation/          # 5 tasks
│   │   ├── beat_revision/               # 5 tasks (4 flawed + 1 correctly executed)
│   │   ├── constrained_continuation/    # 4 tasks
│   │   ├── theory_conversion/           # 4 tasks
│   │   ├── multi_beat_synthesis/        # 3 tasks
│   │   ├── agentic_constraint_discovery/  # 3 tasks
│   │   ├── agentic_planning_execution/    # 3 tasks
│   │   ├── agentic_iterative_revision/    # 3 tasks
│   │   └── critique_improvement/          # 4 tasks
│   └── schemas/
│       ├── task_schema.json
│       └── evaluation_schema.json
├── src/
│   ├── __init__.py
│   ├── generator.py         # Story generation via OpenRouter
│   ├── evaluator.py         # LLM-as-judge evaluation
│   ├── agentic_generator.py # Multi-turn agentic generation
│   ├── agentic_evaluator.py # Agentic task evaluation
│   ├── scoring.py           # Score computation (weighted partial credit)
│   ├── analyzer.py          # Statistical analysis
│   ├── results_db.py        # Persistent results database
│   └── utils.py             # Shared utilities
├── results/
│   ├── generations/            # Individual generation YAML files
│   ├── evaluations/            # Individual evaluation YAML files
│   ├── agentic/                # Agentic task results
│   ├── visualizations/         # PNG charts and dashboards
│   ├── benchmark_results.json  # Aggregated results database
│   └── LEADERBOARD.md          # Auto-generated leaderboard
├── config/
│   ├── models.yaml     # Model configuration
│   └── settings.yaml   # Benchmark settings
├── requirements.txt
└── CLAUDE.md           # This file
```

## Task Types Explained

### Standard Tasks (Single-shot)

#### 1. Beat Interpolation

Given two consecutive story beats (A and C), generate the missing beat (B) that logically connects them. Tasks include **beat definitions** that models must execute correctly, and **must_not_include** constraints that prevent easy shortcuts.

**Tests**: Narrative theory understanding, causal reasoning, beat execution
**Verification**: Beat elements + beat execution + must-not avoidance + character voice + logical bridge + continuity

#### 2. Beat Revision

Given a story segment with an incorrectly executed beat, identify the flaw and rewrite it to properly satisfy the beat definition. **The flaw is NOT disclosed**—models must identify the problem themselves.

**Tests**: Error identification, narrative theory understanding, correction ability
**Verification**: Correct diagnosis + flaw corrected + beat definition satisfied + elements preserved

#### 3. Constrained Continuation

Continue a story opening through specified beats while satisfying 8-10 simultaneous constraints.

**Tests**: Multi-constraint planning, attention to detail
**Verification**: Binary check on each constraint, >=80% required to pass

#### 4. Theory Conversion

Rewrite a story from one structural framework to another while preserving core elements.

**Tests**: Understanding multiple frameworks, mapping between theories
**Verification**: All target beats present + core preserved + structural accuracy >=70%

#### 5. Multi-Beat Synthesis

Write 3 connected beats with individual requirements AND cross-beat constraints (e.g., Chekhov's Gun).

**Tests**: Long-range planning, setup/payoff execution
**Verification**: All cross-beat constraints + >=70% beat-specific requirements

### Agentic Tasks (Multi-turn)

#### 6. Constraint Discovery (`agentic_constraint_discovery`)

Model asks YES/NO questions to discover hidden story constraints before writing. Tests strategic information gathering.

**Process**: Question loop -> Discover constraints -> Generate story satisfying discovered constraints
**Metrics**: Discovery efficiency (constraints found / questions asked), question quality, constraint satisfaction in output
**Key insight**: Tests whether models can strategically gather requirements before execution

#### 7. Planning Execution (`agentic_planning_execution`)

Model creates a detailed plan/outline before writing, then executes the plan.

**Process**: Planning phase -> Execution phase
**Metrics**: Plan quality, plan adherence, execution quality
**Key insight**: Tests whether explicit planning improves generation quality

#### 8. Iterative Revision (`agentic_iterative_revision`)

Model writes initial draft, receives structured rule-based feedback, and revises through multiple rounds.

**Process**: Initial generation -> Feedback -> Revision (xN rounds)
**Metrics**: Improvement trajectory, feedback responsiveness, preservation of good elements
**Feedback source**: Rule-based feedback generator (checks specific criteria per round)

#### 9. Critique Improvement (`critique_improvement`)

Model writes initial draft, receives **LLM critic feedback**, and revises. Tests whether critique/revision actually improves quality.

**Process**: Initial generation -> LLM critique -> Revision (xN rounds)
**Key difference from Iterative Revision**: Uses LLM critic (not rule-based), explicitly measures score trajectory
**Metrics**: Initial score, score at each revision, improvement delta, diminishing returns analysis
**Key insight**: Answers "Do revision rounds actually help? By how much? When do returns diminish?"

## Scoring

All tasks use a **two-component scoring system** (0.0-1.0):

### Final Score Components (50/50 Split)

| Component | Weight | Description |
|-----------|--------|-------------|
| **Programmatic** | 50% | Word count (40%) + Repetition (35%) + Slop detection (25%) |
| **LLM Judge** | 50% | Task-specific criteria evaluation |

### LLM Judge Criteria by Task Type

| Task Type | Scoring Criteria (within LLM Judge component) |
|-----------|----------------------------------------------|
| Beat Interpolation | Elements (25%) + Beat Execution (25%) + Must-Not (15%) + Character (10%) + Bridge (15%) + Continuity (10%) |
| Beat Revision | Diagnosis (20%) + Flaw Fix (20%) + Beat Satisfaction (20%) + Preservation (10%) + Required Preserved (10%) + Minimal Change (10%) + Quality (10%) |
| Constrained Continuation | Beats (20%) + Must-Include (30%) + Must-Not (25%) + Tone (15%) + Ending (10%) |
| Theory Conversion | Beats (35%) + Preservation (30%) + Structural (20%) + Tone (15%) |
| Multi-Beat Synthesis | Beat Reqs (40%) + Cross-Beat (35%) + Context (15%) + Coherence (10%) |

### Agentic Task Scoring

| Task Type | Process Scores | Output Scores |
|-----------|---------------|---------------|
| Constraint Discovery | Discovery efficiency, Question quality, Question coverage | Constraint satisfaction, Beat execution, Narrative quality |
| Planning Execution | Plan completeness, Plan coherence | Execution quality, Constraint satisfaction |
| Iterative Revision | Improvement trajectory, Feedback responsiveness, Preservation | Constraint satisfaction, Beat execution, Narrative quality |
| Critique Improvement | Improvement trajectory, Feedback responsiveness, Preservation | Constraint satisfaction, Beat execution, Narrative quality |

## Configuration

### Model Selection (config/models.yaml)

```yaml
model_tiers:
  strong:
    price_range: [8, 15]  # $/1M output tokens
    min_context: 100000
  cheap:
    price_range: [0.20, 0.80]
    min_context: 32000

generation:
  temperature: 0.7
  max_tokens: 2000
  samples_per_task: 3
```

## Why This Design Works

**Hard to Solve**:
- Requires deep theory understanding (not pattern matching)
- Unique constraint combinations (can't memorize)
- Tests: understanding, revision, planning, conversion, synthesis
- Agentic tasks test meta-cognitive abilities (questioning, planning, self-improvement)

**Easy to Verify**:
- All constraints explicit and binary
- Evidence-based (quote from text)
- LLM-as-judge checks objective criteria only

**Diverse**:
- 9 different generation challenges (5 standard + 4 agentic)
- All 5 major theories covered
- Multiple narrative competencies tested
- Both single-shot and multi-turn capabilities

## Task Taxonomy: One Skill Per Task

Each task type tests exactly one distinct narrative capability with no overlap:

| Task Type | Primary Skill | Why No Overlap |
|-----------|--------------|----------------|
| **beat_interpolation** | Beat execution | Given explicit A→C context, execute beat B correctly |
| **beat_revision** | Error identification | Find and fix flaws (not just generate) |
| **constrained_continuation** | Multi-constraint juggling | Handle 8-10 simultaneous requirements |
| **theory_conversion** | Cross-framework mapping | Know BOTH theories and transform between them |
| **multi_beat_synthesis** | Long-range planning | Cross-beat dependencies require planning ahead |
| **constraint_discovery** | Strategic questioning | Efficient info gathering before generation |
| **planning_execution** | Plan → Execute | Explicit planning phase before writing |
| **iterative_revision** | Rule-based feedback | Respond to structured, deterministic feedback |
| **critique_improvement** | LLM-critic feedback | Respond to natural language critique |

**Key principle**: Adding new tasks should test a capability NOT already covered by existing tasks.

## Overlap Reduction Strategy

Two pairs of tasks have potential overlap. Here's how they're differentiated:

### beat_interpolation vs beat_revision

| Aspect | beat_interpolation | beat_revision |
|--------|-------------------|---------------|
| **Core skill** | Generation | Diagnosis + Correction |
| **Context given** | Beat A and Beat C (explicit) | Single flawed segment |
| **Task** | Write beat B to bridge A→C | Find what's wrong, then fix |
| **Prompt hint** | Told exactly what beat to write | NOT told there's definitely a flaw |
| **Discriminating** | Can model execute a beat well? | Can model identify beat violations? |

**Key difference**: Interpolation is pure generation (execute beat B). Revision requires diagnosis first (what's wrong?) before generation. A model good at generation might fail at diagnosis.

### iterative_revision vs critique_improvement

| Aspect | iterative_revision | critique_improvement |
|--------|-------------------|---------------------|
| **Feedback source** | Rule-based (deterministic) | LLM critic (natural language) |
| **Feedback format** | Structured checklist | Prose critique |
| **Predictability** | Same rules every time | Critique varies by run |
| **Tests** | Can model follow explicit rules? | Can model interpret critique? |
| **Metric focus** | Feedback responsiveness | Score trajectory over rounds |

**Key difference**: Rule-based feedback tests instruction-following (like a linter). LLM critic tests natural language understanding of critique. A model might excel at one and struggle with the other.

## Task Discriminative Power

Not all task types are equally effective at distinguishing between models:

| Task Type | Discriminative Power | Why |
|-----------|---------------------|-----|
| **multi_beat_synthesis** | Excellent | Tests long-range planning. Cross-beat constraints require planning ahead. |
| **critique_improvement** | High (expected) | Tests self-improvement. Do revision rounds actually help? |
| **theory_conversion** | Good | Tests deep structural understanding. Models must know BOTH theories. |
| **constrained_continuation** | Good | Tests multi-constraint handling. 8-10 simultaneous requirements. |
| **constraint_discovery** | Good | Tests strategic questioning. Efficient discovery = better understanding. |
| **beat_interpolation** | Moderate-Good | Tests beat execution and theory understanding. |
| **beat_revision** | Moderate | Models must identify flaws themselves from beat definition. |

## Technical Notes

### Task-Specific Token Budgets

The benchmark uses task-aware token allocation. `max_tokens` must be strictly higher than `max_reasoning` to ensure tokens are available for the final response after thinking.

**Note**: These budgets are set generously based on P95 observed usage (reasoning budget ≈ P95 × 2) to handle outlier models like gemini-3-pro and gpt-5 that use extensive reasoning.

| Task Type | Max Tokens | Max Reasoning | Output Buffer |
|-----------|------------|---------------|---------------|
| beat_interpolation | 6000 | 3000 | 3000 |
| beat_revision | 6000 | 3000 | 3000 |
| constrained_continuation | 8000 | 4500 | 3500 |
| theory_conversion | 7000 | 3500 | 3500 |
| multi_beat_synthesis | 10000 | 5500 | 4500 |
| agentic_constraint_discovery | 6000 | 3000 | 3000 |
| agentic_planning_execution | 7000 | 3500 | 3500 |
| agentic_iterative_revision | 6000 | 3000 | 3000 |
| critique_improvement | 7000 | 3500 | 3500 |

### Reasoning Token Handling

All models use `reasoning.max_tokens` - OpenRouter handles mapping to effort levels automatically for models that require it. See [OpenRouter docs](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

- Minimum 1024 reasoning tokens (OpenRouter requirement)
- Maximum 32,000 reasoning tokens (capped by OpenRouter)

Token tracking via `completion_tokens_details.reasoning_tokens` when available.

### Finish Reason Handling

The benchmark tracks `finish_reason` from API responses to detect generation issues:

| Finish Reason | Meaning | Action |
|---------------|---------|--------|
| `stop` | Normal completion | ✓ Generation marked successful |
| `length` | Token limit reached | ✗ Generation marked failed |
| `content_filter` | Content filtered by API | ✗ Generation marked failed |
| Other | Unknown issue | ✗ Generation marked failed |

**Error Logging**: When a generation fails due to non-stop finish reason, an error is logged to stderr:
```
ERROR - [task_id] [model] Generation failed: finish_reason=length (completion=X, reasoning=Y, output=Z)
```

**Stored in Results**: The `finish_reason` is stored in generation metadata:
```yaml
metadata:
  finish_reason: stop  # or "length", "content_filter", etc.
  success: true  # false if finish_reason != "stop"
  error: null  # or error message if failed
```

## Benchmark Status

**Current dataset** (as of 2025-12-25T00:31:56Z):
- **25 models evaluated** across 34 tasks (9 task types: 5 standard + 4 agentic)
- **850 generations** with 100% completion
- **2,553 evaluations** with 100% completion (3 evaluators per generation: claude-haiku-4.5, gemini-2.5-flash, gpt-5-mini)
- **Median aggregation** for robust pass/fail determination
- **Total cost**: $41.73

**Evaluation format**: All evaluations use a unified score breakdown structure with programmatic (50%) and LLM judge (50%) components. Scores are computed consistently across all task types.

**Data quality**: Results are fully validated with 100% completion. The benchmark includes validation that fails hard on empty outputs or malformed responses. Some models exhibit intermittent behavior on specific agentic tasks (e.g., qwen3-235b-a22b on planning execution), which are marked as failed and can be retried.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new tasks following the schema in `dataset/schemas/task_schema.json`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{story_theory_benchmark,
  title = {Story Theory Benchmark: LLM Narrative Generation Evaluation},
  year = {2024},
  url = {https://github.com/your-username/story-theory-benchmark}
}
```

## Acknowledgments

- Story theory frameworks from Joseph Campbell, Blake Snyder, Dan Harmon, Gustav Freytag
- Built with OpenRouter API for model access
- Inspired by evaluation benchmarks like HELM and BIG-bench
