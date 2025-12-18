# Story Theory Benchmark Leaderboard

*Last updated: 2025-12-19T00:45:15.651196*

## Overview

- **Models evaluated**: 21
- **Tasks**: 34
- **Evaluator models**: claude-haiku-4.5, gemini-2.5-flash, gpt-5-mini
- **Aggregation**: Median across evaluators
- **Scoring**: Programmatic (50%) + LLM Judge (50%)
- **Total generations**: 714
- **Total evaluations**: 2145
- **Total cost**: $36.2910

## Model Rankings

| Rank | Model | Company | Score | Gen Cost | Value | LLM Judge |
|------|-------|---------|-------|----------|---------|-----------|
| 1 | deepseek-v3.2 | DeepSeek | 91.9% | $0.1978 | 426.9 | 94.9% |
| 2 | claude-opus-4.5 | Anthropic | 90.8% | $2.8457 | 29.0 | 93.8% |
| 3 | claude-sonnet-4.5 | Anthropic | 90.1% | $1.7390 | 46.6 | 93.6% |
| 4 | claude-sonnet-4 | Anthropic | 89.6% | $1.5932 | 50.4 | 93.3% |
| 5 | o3 | OpenAI | 89.3% | $0.9582 | 83.2 | 92.9% |
| 6 | gemini-3-flash-preview | Google | 88.3% | $0.5896 | 132.3 | 91.2% |
| 7 | claude-haiku-4.5 | Anthropic | 86.9% | $0.6956 | 108.5 | 90.0% |
| 8 | deepseek-r1 | DeepSeek | 86.9% | $0.4157 | 181.5 | 93.4% |
| 9 | o3-mini | OpenAI | 86.7% | $0.5620 | 133.7 | 86.8% |
| 10 | gemini-2.5-flash | Google | 85.7% | $0.4177 | 175.8 | 89.9% |
| 11 | gpt-4o | OpenAI | 85.6% | $0.8069 | 90.9 | 88.4% |
| 12 | grok-4 | xAI | 85.3% | $1.8858 | 38.6 | 92.9% |
| 13 | gemini-3-pro-preview | Google | 83.9% | $2.1086 | 33.4 | 93.0% |
| 14 | qwen3-235b-a22b | Alibaba | 83.6% | $0.2177 | 321.4 | 94.0% |
| 15 | gpt-4o-mini | OpenAI | 81.7% | $0.1829 | 364.7 | 82.1% |
| 16 | gpt-5 | OpenAI | 81.1% | $1.6176 | 40.6 | 91.8% |
| 17 | gpt-5.2 | OpenAI | 80.3% | $1.7564 | 36.7 | 92.4% |
| 18 | gpt-5.1 | OpenAI | 77.9% | $1.5230 | 39.9 | 94.6% |
| 19 | minimax-m2 | MiniMax | 76.7% | $0.3116 | 188.7 | 93.4% |
| 20 | ministral-14b-2512 | Mistral | 76.6% | $0.1906 | 308.1 | 86.5% |
| 21 | llama-4-maverick | Meta | 72.2% | $0.1938 | 268.8 | 71.0% |

## Best Value (Score²/Cost)

*Higher = better value. Formula: Score² / Cost (rewards quality quadratically)*

| Rank | Model | Company | Score | Gen Cost | Value |
|------|-------|---------|-------|----------|-------|
| 1 | deepseek-v3.2 | DeepSeek | 91.9% | $0.1978 | 426.9 |
| 2 | gpt-4o-mini | OpenAI | 81.7% | $0.1829 | 364.7 |
| 3 | qwen3-235b-a22b | Alibaba | 83.6% | $0.2177 | 321.4 |
| 4 | ministral-14b-2512 | Mistral | 76.6% | $0.1906 | 308.1 |
| 5 | llama-4-maverick | Meta | 72.2% | $0.1938 | 268.8 |
| 6 | minimax-m2 | MiniMax | 76.7% | $0.3116 | 188.7 |
| 7 | deepseek-r1 | DeepSeek | 86.9% | $0.4157 | 181.5 |
| 8 | gemini-2.5-flash | Google | 85.7% | $0.4177 | 175.8 |
| 9 | o3-mini | OpenAI | 86.7% | $0.5620 | 133.7 |
| 10 | gemini-3-flash-preview | Google | 88.3% | $0.5896 | 132.3 |
| 11 | claude-haiku-4.5 | Anthropic | 86.9% | $0.6956 | 108.5 |
| 12 | gpt-4o | OpenAI | 85.6% | $0.8069 | 90.9 |
| 13 | o3 | OpenAI | 89.3% | $0.9582 | 83.2 |
| 14 | claude-sonnet-4 | Anthropic | 89.6% | $1.5932 | 50.4 |
| 15 | claude-sonnet-4.5 | Anthropic | 90.1% | $1.7390 | 46.6 |
| 16 | gpt-5 | OpenAI | 81.1% | $1.6176 | 40.6 |
| 17 | gpt-5.1 | OpenAI | 77.9% | $1.5230 | 39.9 |
| 18 | grok-4 | xAI | 85.3% | $1.8858 | 38.6 |
| 19 | gpt-5.2 | OpenAI | 80.3% | $1.7564 | 36.7 |
| 20 | gemini-3-pro-preview | Google | 83.9% | $2.1086 | 33.4 |
| 21 | claude-opus-4.5 | Anthropic | 90.8% | $2.8457 | 29.0 |

## Scores by Task Type

| Model | agentic_constraint_discovery | agentic_iterative_revision | agentic_planning_execution | beat_interpolation | beat_revision | constrained_continuation | critique_improvement | multi_beat_synthesis | theory_conversion |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| deepseek-v3.2 | 72.8% | 89.5% | 96.1% | 98.1% | 96.0% | 95.3% | 87.9% | 90.9% | 95.8% |
| claude-opus-4.5 | 71.6% | 88.1% | 88.9% | 99.1% | 94.7% | 94.7% | 86.7% | 90.7% | 93.8% |
| claude-sonnet-4.5 | 69.0% | 89.7% | 86.9% | 98.1% | 94.9% | 95.2% | 88.0% | 84.7% | 93.4% |
| claude-sonnet-4 | 69.6% | 90.8% | 87.8% | 98.2% | 94.0% | 93.4% | 87.0% | 83.1% | 92.3% |
| o3 | 59.3% | 93.9% | 91.6% | 99.0% | 96.7% | 87.9% | 88.1% | 92.9% | 89.2% |
| gemini-3-flash-preview | 62.2% | 90.4% | 94.4% | 97.4% | 86.9% | 91.5% | 86.3% | 89.8% | 90.1% |
| claude-haiku-4.5 | 62.0% | 85.9% | 90.8% | 98.0% | 86.7% | 94.2% | 86.3% | 74.0% | 92.6% |
| deepseek-r1 | 65.3% | 92.2% | 92.5% | 93.1% | 95.0% | 79.2% | 88.2% | 73.0% | 93.5% |
| o3-mini | 64.6% | 85.5% | 79.4% | 99.2% | 81.8% | 95.8% | 83.1% | 88.8% | 91.7% |
| gemini-2.5-flash | 62.1% | 85.2% | 71.2% | 98.3% | 95.8% | 84.9% | 80.4% | 88.3% | 90.4% |
| gpt-4o | 54.0% | 82.8% | 80.9% | 96.7% | 92.5% | 93.8% | 75.4% | 88.8% | 92.4% |
| grok-4 | 64.4% | 88.0% | 96.4% | 82.6% | 95.0% | 82.8% | 85.5% | 84.4% | 87.7% |
| gemini-3-pro-preview | 43.0% | 93.1% | 94.4% | 97.9% | 85.1% | 82.1% | 90.7% | 72.2% | 88.6% |
| qwen3-235b-a22b | 38.7% | 90.9% | 90.1% | 96.2% | 94.7% | 77.9% | 88.0% | 74.6% | 89.7% |
| gpt-4o-mini | 52.9% | 71.0% | 81.2% | 94.8% | 80.6% | 93.2% | 71.2% | 86.7% | 91.7% |
| gpt-5 | 54.2% | 92.8% | 93.9% | 99.0% | 97.2% | 67.5% | 87.3% | 66.7% | 66.1% |
| gpt-5.2 | 81.4% | 92.2% | 91.5% | 82.7% | 86.4% | 67.4% | 88.8% | 66.1% | 66.4% |
| gpt-5.1 | 65.0% | 90.9% | 92.7% | 74.4% | 91.5% | 67.3% | 88.3% | 67.0% | 66.2% |
| minimax-m2 | 26.0% | 90.2% | 93.0% | 72.8% | 93.5% | 84.2% | 86.4% | 60.0% | 75.6% |
| ministral-14b-2512 | 63.0% | 84.2% | 81.6% | 79.2% | 78.6% | 65.9% | 87.2% | 67.5% | 78.9% |
| llama-4-maverick | 46.7% | 39.6% | 81.3% | 93.2% | 72.6% | 84.9% | 60.0% | 74.5% | 79.9% |

## Component Breakdown by Task Type


### deepseek-v3.2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 72.8% | - | 76.7% |
| agentic_iterative_revision | 89.5% | - | 89.7% |
| agentic_planning_execution | 96.1% | - | 96.7% |
| beat_interpolation | 98.1% | 92.7% | 100.0% |
| beat_revision | 96.0% | 97.9% | 96.8% |
| constrained_continuation | 95.3% | 88.5% | 100.0% |
| critique_improvement | 87.9% | - | 92.8% |
| multi_beat_synthesis | 90.9% | 69.9% | 99.1% |
| theory_conversion | 95.8% | 86.1% | 98.7% |

### claude-opus-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 71.6% | - | 80.0% |
| agentic_iterative_revision | 88.1% | - | 90.0% |
| agentic_planning_execution | 88.9% | - | 88.9% |
| beat_interpolation | 99.1% | 96.5% | 100.0% |
| beat_revision | 94.7% | 94.0% | 94.8% |
| constrained_continuation | 94.7% | 78.7% | 100.0% |
| critique_improvement | 86.7% | - | 89.2% |
| multi_beat_synthesis | 90.7% | 73.2% | 95.5% |
| theory_conversion | 93.8% | 77.3% | 98.9% |

### claude-sonnet-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 69.0% | - | 73.9% |
| agentic_iterative_revision | 89.7% | - | 90.0% |
| agentic_planning_execution | 86.9% | - | 86.9% |
| beat_interpolation | 98.1% | 92.3% | 100.0% |
| beat_revision | 94.9% | 93.5% | 95.9% |
| constrained_continuation | 95.2% | 80.6% | 100.0% |
| critique_improvement | 88.0% | - | 89.1% |
| multi_beat_synthesis | 84.7% | 69.9% | 99.1% |
| theory_conversion | 93.4% | 75.4% | 99.2% |

### claude-sonnet-4

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 69.6% | - | 72.2% |
| agentic_iterative_revision | 90.8% | - | 91.1% |
| agentic_planning_execution | 87.8% | - | 90.0% |
| beat_interpolation | 98.2% | 93.2% | 99.9% |
| beat_revision | 94.0% | 93.9% | 94.8% |
| constrained_continuation | 93.4% | 77.4% | 98.0% |
| critique_improvement | 87.0% | - | 90.9% |
| multi_beat_synthesis | 83.1% | 69.6% | 96.1% |
| theory_conversion | 92.3% | 72.8% | 98.4% |

### o3

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 59.3% | - | 62.2% |
| agentic_iterative_revision | 93.9% | - | 94.6% |
| agentic_planning_execution | 91.6% | - | 90.9% |
| beat_interpolation | 99.0% | 96.0% | 100.0% |
| beat_revision | 96.7% | 97.3% | 94.8% |
| constrained_continuation | 87.9% | 79.5% | 100.0% |
| critique_improvement | 88.1% | - | 91.2% |
| multi_beat_synthesis | 92.9% | 72.0% | 99.8% |
| theory_conversion | 89.2% | 80.4% | 98.2% |

### gemini-3-flash-preview

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 62.2% | - | 71.1% |
| agentic_iterative_revision | 90.4% | - | 90.3% |
| agentic_planning_execution | 94.4% | - | 95.6% |
| beat_interpolation | 97.4% | 95.5% | 99.2% |
| beat_revision | 86.9% | 95.5% | 78.2% |
| constrained_continuation | 91.5% | 83.1% | 99.9% |
| critique_improvement | 86.3% | - | 88.6% |
| multi_beat_synthesis | 89.8% | 82.0% | 97.6% |
| theory_conversion | 90.1% | 81.5% | 98.7% |

### claude-haiku-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 62.0% | - | 71.1% |
| agentic_iterative_revision | 85.9% | - | 87.0% |
| agentic_planning_execution | 90.8% | - | 90.3% |
| beat_interpolation | 98.0% | 91.8% | 100.0% |
| beat_revision | 86.7% | 91.8% | 77.5% |
| constrained_continuation | 94.2% | 77.8% | 99.6% |
| critique_improvement | 86.3% | - | 89.4% |
| multi_beat_synthesis | 74.0% | 73.0% | 91.8% |
| theory_conversion | 92.6% | 74.0% | 98.6% |

### deepseek-r1

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 65.3% | - | 69.4% |
| agentic_iterative_revision | 92.2% | - | 92.2% |
| agentic_planning_execution | 92.5% | - | 92.2% |
| beat_interpolation | 93.1% | 94.1% | 100.0% |
| beat_revision | 95.0% | 96.9% | 95.8% |
| constrained_continuation | 79.2% | 89.6% | 99.2% |
| critique_improvement | 88.2% | - | 90.2% |
| multi_beat_synthesis | 73.0% | 69.5% | 93.4% |
| theory_conversion | 93.5% | 75.8% | 99.1% |

### o3-mini

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 64.6% | - | 66.1% |
| agentic_iterative_revision | 85.5% | - | 84.2% |
| agentic_planning_execution | 79.4% | - | 82.8% |
| beat_interpolation | 99.2% | 97.2% | 99.8% |
| beat_revision | 81.8% | 94.8% | 66.2% |
| constrained_continuation | 95.8% | 84.1% | 99.6% |
| critique_improvement | 83.1% | - | 84.2% |
| multi_beat_synthesis | 88.8% | 72.7% | 92.1% |
| theory_conversion | 91.7% | 72.5% | 97.2% |

### gemini-2.5-flash

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 62.1% | - | 65.6% |
| agentic_iterative_revision | 85.2% | - | 86.1% |
| agentic_planning_execution | 71.2% | - | 68.6% |
| beat_interpolation | 98.3% | 93.0% | 100.0% |
| beat_revision | 95.8% | 97.0% | 95.0% |
| constrained_continuation | 84.9% | 77.6% | 99.6% |
| critique_improvement | 80.4% | - | 85.6% |
| multi_beat_synthesis | 88.3% | 69.3% | 97.6% |
| theory_conversion | 90.4% | 77.0% | 96.9% |

### gpt-4o

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 54.0% | - | 56.7% |
| agentic_iterative_revision | 82.8% | - | 81.7% |
| agentic_planning_execution | 80.9% | - | 83.3% |
| beat_interpolation | 96.7% | 87.2% | 99.8% |
| beat_revision | 92.5% | 95.2% | 90.6% |
| constrained_continuation | 93.8% | 75.1% | 100.0% |
| critique_improvement | 75.4% | - | 77.8% |
| multi_beat_synthesis | 88.8% | 68.0% | 93.6% |
| theory_conversion | 92.4% | 71.7% | 98.9% |

### grok-4

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 64.4% | - | 66.7% |
| agentic_iterative_revision | 88.0% | - | 87.4% |
| agentic_planning_execution | 96.4% | - | 95.3% |
| beat_interpolation | 82.6% | 91.0% | 99.7% |
| beat_revision | 95.0% | 93.5% | 93.3% |
| constrained_continuation | 82.8% | 78.7% | 99.6% |
| critique_improvement | 85.5% | - | 89.0% |
| multi_beat_synthesis | 84.4% | 69.7% | 98.9% |
| theory_conversion | 87.7% | 71.2% | 98.2% |

### gemini-3-pro-preview

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 43.0% | - | - |
| agentic_iterative_revision | 93.1% | - | 93.3% |
| agentic_planning_execution | 94.4% | - | 95.2% |
| beat_interpolation | 97.9% | 94.6% | 100.0% |
| beat_revision | 85.1% | 91.0% | 74.8% |
| constrained_continuation | 82.1% | 75.0% | 99.8% |
| critique_improvement | 90.7% | - | 92.6% |
| multi_beat_synthesis | 72.2% | 69.7% | 96.7% |
| theory_conversion | 88.6% | 71.1% | 97.7% |

### qwen3-235b-a22b

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 38.7% | - | - |
| agentic_iterative_revision | 90.9% | - | 91.1% |
| agentic_planning_execution | 90.1% | - | 90.1% |
| beat_interpolation | 96.2% | 92.7% | 99.7% |
| beat_revision | 94.7% | 95.0% | 92.5% |
| constrained_continuation | 77.9% | 72.1% | 99.8% |
| critique_improvement | 88.0% | - | 89.8% |
| multi_beat_synthesis | 74.6% | 74.4% | 88.9% |
| theory_conversion | 89.7% | 85.1% | 96.8% |

### gpt-4o-mini

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 52.9% | - | 57.2% |
| agentic_iterative_revision | 71.0% | - | 70.0% |
| agentic_planning_execution | 81.2% | - | 81.7% |
| beat_interpolation | 94.8% | 91.8% | 99.5% |
| beat_revision | 80.6% | 92.6% | 64.9% |
| constrained_continuation | 93.2% | 75.4% | 98.7% |
| critique_improvement | 71.2% | - | 74.2% |
| multi_beat_synthesis | 86.7% | 69.6% | 88.5% |
| theory_conversion | 91.7% | 74.1% | 96.4% |

### gpt-5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 54.2% | - | 62.8% |
| agentic_iterative_revision | 92.8% | - | 93.9% |
| agentic_planning_execution | 93.9% | - | 96.7% |
| beat_interpolation | 99.0% | 99.4% | 100.0% |
| beat_revision | 97.2% | 96.0% | 96.3% |
| constrained_continuation | 67.5% | 69.8% | 100.0% |
| critique_improvement | 87.3% | - | 90.2% |
| multi_beat_synthesis | 66.7% | 70.0% | 98.4% |
| theory_conversion | 66.1% | 70.0% | 97.1% |

### gpt-5.2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 81.4% | - | 86.7% |
| agentic_iterative_revision | 92.2% | - | 92.2% |
| agentic_planning_execution | 91.5% | - | 90.7% |
| beat_interpolation | 82.7% | 85.0% | 100.0% |
| beat_revision | 86.4% | 90.6% | 77.9% |
| constrained_continuation | 67.4% | 69.9% | 99.8% |
| critique_improvement | 88.8% | - | 90.4% |
| multi_beat_synthesis | 66.1% | 70.0% | 97.2% |
| theory_conversion | 66.4% | 70.0% | 97.9% |

### gpt-5.1

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 65.0% | - | 78.9% |
| agentic_iterative_revision | 90.9% | - | 90.6% |
| agentic_planning_execution | 92.7% | - | 93.9% |
| beat_interpolation | 74.4% | 81.3% | 100.0% |
| beat_revision | 91.5% | 91.5% | 95.9% |
| constrained_continuation | 67.3% | 70.0% | 99.6% |
| critique_improvement | 88.3% | - | 90.9% |
| multi_beat_synthesis | 67.0% | 70.0% | 99.1% |
| theory_conversion | 66.2% | 69.9% | 97.4% |

### minimax-m2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 26.0% | - | - |
| agentic_iterative_revision | 90.2% | - | 90.8% |
| agentic_planning_execution | 93.0% | - | 94.4% |
| beat_interpolation | 72.8% | 78.0% | 97.7% |
| beat_revision | 93.5% | 94.9% | 89.5% |
| constrained_continuation | 84.2% | 70.8% | 98.8% |
| critique_improvement | 86.4% | - | 89.2% |
| multi_beat_synthesis | 60.0% | 70.0% | 84.9% |
| theory_conversion | 75.6% | 71.0% | 98.0% |

### ministral-14b-2512

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 63.0% | - | 65.6% |
| agentic_iterative_revision | 84.2% | - | 82.4% |
| agentic_planning_execution | 81.6% | - | 79.5% |
| beat_interpolation | 79.2% | 78.4% | 99.2% |
| beat_revision | 78.6% | 83.2% | 73.7% |
| constrained_continuation | 65.9% | 69.9% | 96.9% |
| critique_improvement | 87.2% | - | 87.7% |
| multi_beat_synthesis | 67.5% | 69.8% | 88.2% |
| theory_conversion | 78.9% | 70.0% | 97.6% |

### llama-4-maverick

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| agentic_constraint_discovery | 46.7% | - | 48.9% |
| agentic_iterative_revision | 39.6% | - | 36.1% |
| agentic_planning_execution | 81.3% | - | 80.2% |
| beat_interpolation | 93.2% | 88.1% | 92.2% |
| beat_revision | 72.6% | 84.6% | 61.1% |
| constrained_continuation | 84.9% | 70.6% | 84.6% |
| critique_improvement | 60.0% | - | 60.9% |
| multi_beat_synthesis | 74.5% | 73.1% | 78.8% |
| theory_conversion | 79.9% | 73.8% | 83.4% |

## Cost Efficiency

*Note: Reasoning tokens (for CoT models) are billed but don't produce output, affecting cost efficiency.*

| Model | Gen Cost | Output Tokens | Reasoning % | $/1K Output |
|-------|----------|---------------|-------------|-------------|
| deepseek-v3.2 | $0.1978 | 50,503 | 46.6% | $0.0039 |
| claude-opus-4.5 | $2.8457 | 66,128 | 34.5% | $0.0430 |
| claude-sonnet-4.5 | $1.7390 | 67,672 | 32.4% | $0.0257 |
| claude-sonnet-4 | $1.5932 | 59,896 | 35.1% | $0.0266 |
| o3 | $0.9582 | 65,172 | 34.5% | $0.0147 |
| gemini-3-flash-preview | $0.5896 | 74,378 | 49.6% | $0.0079 |
| claude-haiku-4.5 | $0.6956 | 65,063 | 38.6% | $0.0107 |
| deepseek-r1 | $0.4157 | 52,697 | 51.4% | $0.0079 |
| o3-mini | $0.5620 | 54,722 | 41.7% | $0.0103 |
| gemini-2.5-flash | $0.4177 | 60,183 | 50.4% | $0.0069 |
| gpt-4o | $0.8069 | 54,647 | 17.8% | $0.0148 |
| grok-4 | $1.8858 | 63,896 | 43.1% | $0.0295 |
| gemini-3-pro-preview | $2.1086 | 76,538 | 53.1% | $0.0275 |
| qwen3-235b-a22b | $0.2177 | 51,903 | 52.8% | $0.0042 |
| gpt-4o-mini | $0.1829 | 60,234 | 17.4% | $0.0030 |
| gpt-5 | $1.6176 | 81,045 | 49.1% | $0.0200 |
| gpt-5.2 | $1.7564 | 99,368 | 19.7% | $0.0177 |
| gpt-5.1 | $1.5230 | 104,962 | 28.3% | $0.0145 |
| minimax-m2 | $0.3116 | 78,335 | 41.5% | $0.0040 |
| ministral-14b-2512 | $0.1906 | 86,964 | 12.1% | $0.0022 |
| llama-4-maverick | $0.1938 | 55,273 | 17.8% | $0.0035 |