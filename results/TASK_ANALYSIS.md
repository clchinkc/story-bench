# Story Theory Benchmark - Task Analysis

*Last updated: 2025-12-18T20:36:36.173966*

## Overview

This document provides detailed analysis of each benchmark task, including:
- **Score Spread**: Difference between best and worst model (higher = more discriminative)
- **Average Score**: Mean score across all models
- **Best/Worst Models**: Which models excel or struggle on each task
- **Cost Analysis**: Generation costs per task

## Task Type Summary

| Task Type | Tasks | Avg Score | Avg Spread | Best Spread | Discriminative Power |
|-----------|-------|-----------|------------|-------------|---------------------|
| agentic_iterative_revision | 3 | 86.1% | 57.6% | 90.2% | Excellent |
| agentic_constraint_discovery | 3 | 63.2% | 45.3% | 52.0% | Excellent |
| critique_improvement | 4 | 84.5% | 36.8% | 78.3% | Excellent |
| multi_beat_synthesis | 3 | 79.3% | 35.8% | 48.6% | Excellent |
| beat_interpolation | 5 | 92.1% | 31.7% | 33.9% | Excellent |
| constrained_continuation | 4 | 85.1% | 31.2% | 33.3% | Excellent |
| beat_revision | 5 | 89.2% | 31.0% | 61.0% | Excellent |
| theory_conversion | 4 | 85.7% | 30.8% | 32.7% | Excellent |
| agentic_planning_execution | 3 | 88.4% | 30.4% | 36.6% | Excellent |

## Most Discriminative Tasks

*Tasks with highest score spread - best for distinguishing model capabilities*

| Rank | Task ID | Type | Spread | Avg Score | Best Model | Worst Model |
|------|---------|------|--------|-----------|------------|-------------|
| 1 | agentic_iterative_revision_003 | agentic_iterative_revision | 90.2% | 84.2% | claude-sonnet-4 | llama-4-maverick |
| 2 | critique_improvement_003 | critique_improvement | 78.3% | 79.9% | deepseek-r1 | llama-4-maverick |
| 3 | agentic_iterative_revision_002 | agentic_iterative_revision | 62.0% | 86.9% | gpt-5.2 | llama-4-maverick |
| 4 | beat_revision_002 | beat_revision | 61.0% | 91.8% | gemini-2.5-flash | llama-4-maverick |
| 5 | beat_revision_005 | beat_revision | 55.9% | 72.5% | gemini-2.5-flash | ministral-14b-2512 |
| 6 | agentic_constraint_discovery_003 | agentic_constraint_discovery | 52.0% | 67.6% | gemini-3-pro-preview | qwen3-235b-a22b |
| 7 | multi_beat_synthesis_003 | multi_beat_synthesis | 48.6% | 73.1% | o3 | llama-4-maverick |
| 8 | agentic_constraint_discovery_001 | agentic_constraint_discovery | 47.0% | 57.3% | claude-opus-4.5 | minimax-m2 |
| 9 | agentic_constraint_discovery_002 | agentic_constraint_discovery | 37.0% | 64.8% | gpt-5.2 | gpt-4o |
| 10 | agentic_planning_execution_001 | agentic_planning_execution | 36.6% | 85.4% | deepseek-v3.2 | gemini-2.5-flash |

## Hardest Tasks

*Tasks with lowest average scores across models*

| Rank | Task ID | Type | Avg Score | Spread | Best Score | Best Model |
|------|---------|------|-----------|--------|------------|------------|
| 1 | agentic_constraint_discovery_001 | agentic_constraint_discovery | 57.3% | 47.0% | 76.3% | claude-opus-4.5 |
| 2 | agentic_constraint_discovery_002 | agentic_constraint_discovery | 64.8% | 37.0% | 81.0% | gpt-5.2 |
| 3 | agentic_constraint_discovery_003 | agentic_constraint_discovery | 67.6% | 52.0% | 95.3% | gemini-3-pro-preview |
| 4 | beat_revision_005 | beat_revision | 72.5% | 55.9% | 94.5% | gemini-2.5-flash |
| 5 | multi_beat_synthesis_003 | multi_beat_synthesis | 73.1% | 48.6% | 92.2% | o3 |
| 6 | multi_beat_synthesis_001 | multi_beat_synthesis | 78.8% | 29.6% | 92.5% | o3 |
| 7 | critique_improvement_003 | critique_improvement | 79.9% | 78.3% | 89.8% | deepseek-r1 |
| 8 | theory_conversion_002 | theory_conversion | 82.7% | 26.6% | 92.1% | gpt-4o |
| 9 | critique_improvement_002 | critique_improvement | 82.9% | 29.7% | 91.4% | gpt-5.2 |
| 10 | constrained_continuation_003 | constrained_continuation | 83.0% | 29.1% | 94.6% | claude-sonnet-4.5 |

## Easiest Tasks

*Tasks with highest average scores across models (potential ceiling effects)*

| Rank | Task ID | Type | Avg Score | Spread | Worst Score | Worst Model |
|------|---------|------|-----------|--------|-------------|-------------|
| 1 | beat_revision_003 | beat_revision | 96.8% | 7.3% | 92.0% | gpt-4o-mini |
| 2 | beat_interpolation_004 | beat_interpolation | 94.1% | 26.9% | 71.9% | gpt-5.2 |
| 3 | beat_interpolation_003 | beat_interpolation | 93.5% | 33.9% | 66.1% | minimax-m2 |
| 4 | beat_revision_004 | beat_revision | 93.0% | 11.3% | 86.0% | ministral-14b-2512 |
| 5 | beat_interpolation_002 | beat_interpolation | 91.8% | 32.5% | 67.5% | minimax-m2 |
| 6 | beat_revision_001 | beat_revision | 91.8% | 19.7% | 77.4% | ministral-14b-2512 |
| 7 | beat_revision_002 | beat_revision | 91.8% | 61.0% | 37.3% | llama-4-maverick |
| 8 | agentic_planning_execution_003 | agentic_planning_execution | 90.8% | 27.8% | 70.0% | gemini-2.5-flash |
| 9 | beat_interpolation_001 | beat_interpolation | 90.7% | 31.2% | 68.8% | ministral-14b-2512 |
| 10 | beat_interpolation_005 | beat_interpolation | 90.2% | 33.9% | 65.8% | ministral-14b-2512 |

## Detailed Task Breakdown


### agentic_constraint_discovery

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| agentic_constraint_discovery_003 | Freytag's Pyramid | 67.6% | 43.3% | 95.3% | 52.0% | 0.13 | gemini-3-pro-preview | qwen3-235b-a22b |
| agentic_constraint_discovery_001 | Hero's Journey | 57.3% | 29.3% | 76.3% | 47.0% | 0.14 | claude-opus-4.5 | minimax-m2 |
| agentic_constraint_discovery_002 | Save the Cat | 64.8% | 44.0% | 81.0% | 37.0% | 0.11 | gpt-5.2 | gpt-4o |

### agentic_iterative_revision

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| agentic_iterative_revision_003 | Freytag's Pyramid | 84.2% | 7.5% | 97.7% | 90.2% | 0.20 | claude-sonnet-4 | llama-4-maverick |
| agentic_iterative_revision_002 | Save the Cat | 86.9% | 33.7% | 95.7% | 62.0% | 0.13 | gpt-5.2 | llama-4-maverick |
| agentic_iterative_revision_001 | Story Circle | 87.2% | 77.7% | 98.3% | 20.7% | 0.05 | gpt-5.1 | llama-4-maverick |

### agentic_planning_execution

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| agentic_planning_execution_001 | Save the Cat | 85.4% | 60.1% | 96.7% | 36.6% | 0.09 | deepseek-v3.2 | gemini-2.5-flash |
| agentic_planning_execution_003 | Three-Act Structure | 90.8% | 70.0% | 97.8% | 27.8% | 0.08 | gpt-5.2 | gemini-2.5-flash |
| agentic_planning_execution_002 | Hero's Journey | 89.0% | 70.5% | 97.3% | 26.8% | 0.07 | deepseek-r1 | o3-mini |

### beat_interpolation

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| beat_interpolation_005 | Three-Act Structure | 90.2% | 65.8% | 99.7% | 33.9% | 0.13 | claude-sonnet-4 | ministral-14b-2512 |
| beat_interpolation_003 | Story Circle | 93.5% | 66.1% | 100.0% | 33.9% | 0.11 | o3-mini | minimax-m2 |
| beat_interpolation_002 | Save the Cat | 91.8% | 67.5% | 100.0% | 32.5% | 0.10 | claude-opus-4.5 | minimax-m2 |
| beat_interpolation_001 | Hero's Journey | 90.7% | 68.8% | 100.0% | 31.2% | 0.13 | gpt-5 | ministral-14b-2512 |
| beat_interpolation_004 | Freytag's Pyramid | 94.1% | 71.9% | 98.8% | 26.9% | 0.07 | deepseek-r1 | gpt-5.2 |

### beat_revision

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| beat_revision_002 | Save the Cat | 91.8% | 37.3% | 98.2% | 61.0% | 0.14 | gemini-2.5-flash | llama-4-maverick |
| beat_revision_005 | Save the Cat | 72.5% | 38.5% | 94.5% | 55.9% | 0.22 | gemini-2.5-flash | ministral-14b-2512 |
| beat_revision_001 | Hero's Journey | 91.8% | 77.4% | 97.2% | 19.7% | 0.05 | gpt-5 | ministral-14b-2512 |
| beat_revision_004 | Freytag's Pyramid | 93.0% | 86.0% | 97.3% | 11.3% | 0.03 | deepseek-v3.2 | ministral-14b-2512 |
| beat_revision_003 | Story Circle | 96.8% | 92.0% | 99.2% | 7.3% | 0.02 | grok-4 | gpt-4o-mini |

### constrained_continuation

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| constrained_continuation_001 | Hero's Journey | 83.6% | 65.0% | 98.3% | 33.3% | 0.13 | deepseek-v3.2 | ministral-14b-2512 |
| constrained_continuation_002 | Save the Cat | 88.7% | 65.0% | 96.6% | 31.6% | 0.12 | gemini-2.5-flash | ministral-14b-2512 |
| constrained_continuation_004 | Three-Act Structure | 85.1% | 67.1% | 97.9% | 30.8% | 0.12 | grok-4 | ministral-14b-2512 |
| constrained_continuation_003 | Story Circle | 83.0% | 65.5% | 94.6% | 29.1% | 0.11 | claude-sonnet-4.5 | minimax-m2 |

### critique_improvement

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| critique_improvement_003 | Hero's Journey | 79.9% | 11.5% | 89.8% | 78.3% | 0.17 | deepseek-r1 | llama-4-maverick |
| critique_improvement_002 | Save the Cat | 82.9% | 61.7% | 91.4% | 29.7% | 0.08 | gpt-5.2 | gpt-4o |
| critique_improvement_004 | Story Circle | 85.4% | 70.0% | 92.2% | 22.2% | 0.06 | gemini-3-pro-preview | gpt-4o-mini |
| critique_improvement_001 | Three-Act Structure | 89.7% | 78.4% | 95.5% | 17.1% | 0.05 | gpt-5.2 | llama-4-maverick |

### multi_beat_synthesis

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| multi_beat_synthesis_003 | Save the Cat | 73.1% | 43.7% | 92.2% | 48.6% | 0.15 | o3 | llama-4-maverick |
| multi_beat_synthesis_001 | Hero's Journey | 78.8% | 62.9% | 92.5% | 29.6% | 0.11 | o3 | deepseek-r1 |
| multi_beat_synthesis_002 | Hero's Journey | 86.0% | 64.8% | 94.0% | 29.1% | 0.11 | o3 | ministral-14b-2512 |

### theory_conversion

| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |
|---------|--------|-----|-----|-----|--------|---------|------------|-------------|
| theory_conversion_004 | Hero's Journey | 88.2% | 65.5% | 98.2% | 32.7% | 0.10 | deepseek-v3.2 | gpt-5 |
| theory_conversion_003 | Save the Cat | 83.8% | 64.4% | 96.4% | 32.1% | 0.12 | deepseek-r1 | llama-4-maverick |
| theory_conversion_001 | Three-Act Structure | 88.1% | 66.1% | 97.9% | 31.8% | 0.13 | qwen3-235b-a22b | gpt-5 |
| theory_conversion_002 | Story Circle | 82.7% | 65.5% | 92.1% | 26.6% | 0.11 | gpt-4o | gpt-5.1 |

## Cost Analysis

*Generation cost per task (evaluation costs not included)*

| Task Type | Tasks | Total Cost | Avg Cost/Task |
|-----------|-------|------------|---------------|
| agentic_constraint_discovery | 3 | $1.7224 | $0.0273 |
| agentic_iterative_revision | 3 | $3.1958 | $0.0507 |
| agentic_planning_execution | 3 | $2.1968 | $0.0349 |
| beat_interpolation | 5 | $0.9518 | $0.0091 |
| beat_revision | 5 | $0.8942 | $0.0085 |
| constrained_continuation | 4 | $1.0905 | $0.0130 |
| critique_improvement | 4 | $8.8245 | $0.1051 |
| multi_beat_synthesis | 3 | $1.2338 | $0.0196 |
| theory_conversion | 4 | $1.0982 | $0.0131 |

## Per-Task Model Scores

*Detailed scores for each model on each task (median across evaluators)*


### agentic_constraint_discovery

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| agentic_constraint_discovery_001 | 56.0% | 76.3% | 59.3% | 68.0% | 70.3% | 61.3% | 50.0% | 76.3% | 69.5% | 34.7% | 29.3% | 51.0% | 51.3% | 32.0% | 36.7% | 53.3% | 70.7% | 58.0% | 61.3% | 68.8% | 68.7% |
| agentic_constraint_discovery_002 | 70.7% | 76.0% | 74.7% | 64.0% | 67.7% | 72.0% | 63.3% | 61.0% | 70.7% | 48.0% | 45.3% | 69.3% | 44.0% | 68.0% | 69.0% | 77.3% | 81.0% | 64.0% | 68.0% | 47.3% | 58.7% |
| agentic_constraint_discovery_003 | 59.3% | 62.3% | 74.7% | 75.0% | 58.0% | 85.0% | 68.3% | 89.0% | 95.3% | 57.3% | 57.3% | 68.7% | 66.7% | 58.7% | 57.0% | 64.3% | 92.7% | 56.0% | 64.3% | 43.3% | 66.0% |

### agentic_iterative_revision

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| agentic_iterative_revision_001 | 84.4% | 84.3% | 88.7% | 80.2% | 88.6% | 88.6% | 77.8% | 89.3% | 90.0% | 77.7% | 89.3% | 87.7% | 91.0% | 82.0% | 89.3% | 98.3% | 91.0% | 95.4% | 86.3% | 90.0% | 82.0% |
| agentic_iterative_revision_002 | 80.0% | 90.0% | 86.0% | 93.3% | 92.7% | 86.7% | 89.3% | 94.9% | 95.7% | 33.7% | 91.3% | 83.0% | 82.0% | 79.7% | 93.5% | 92.3% | 95.7% | 92.6% | 87.2% | 92.8% | 93.3% |
| agentic_iterative_revision_003 | 93.3% | 90.0% | 97.7% | 95.7% | 95.3% | 93.3% | 88.6% | 91.2% | 93.7% | 7.5% | 90.0% | 82.0% | 75.5% | 51.3% | 95.7% | 82.0% | 90.0% | 93.8% | 83.0% | 90.0% | 88.6% |

### agentic_planning_execution

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| agentic_planning_execution_001 | 82.4% | 82.4% | 87.4% | 76.7% | 87.0% | 96.7% | 60.1% | 94.3% | 96.1% | 86.8% | 91.0% | 72.4% | 80.4% | 81.7% | 90.7% | 91.0% | 84.0% | 87.2% | 85.9% | 83.9% | 95.6% |
| agentic_planning_execution_002 | 96.7% | 90.9% | 83.9% | 90.8% | 97.3% | 96.7% | 83.9% | 94.3% | 91.5% | 79.3% | 96.1% | 82.2% | 76.1% | 84.1% | 95.5% | 90.0% | 92.8% | 91.9% | 70.5% | 89.6% | 95.7% |
| agentic_planning_execution_003 | 93.3% | 93.3% | 92.0% | 93.3% | 93.3% | 94.9% | 70.0% | 94.3% | 95.7% | 77.8% | 91.9% | 90.2% | 86.3% | 77.8% | 95.5% | 97.2% | 97.8% | 95.7% | 81.7% | 96.7% | 97.8% |

### beat_interpolation

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| beat_interpolation_001 | 98.8% | 99.7% | 97.9% | 99.1% | 73.6% | 98.0% | 98.6% | - | 99.4% | 94.9% | 69.2% | 68.8% | 97.6% | 99.4% | 100.0% | 69.3% | 88.8% | - | 99.4% | 99.1% | 71.8% |
| beat_interpolation_002 | 98.2% | 100.0% | 97.0% | 98.2% | 97.9% | 97.8% | 97.3% | - | 96.9% | 90.5% | 67.5% | 97.2% | 96.5% | 83.0% | - | 72.8% | 85.1% | 98.8% | 99.7% | 97.9% | 72.4% |
| beat_interpolation_003 | 99.1% | 99.7% | 99.7% | 99.7% | 100.0% | - | 100.0% | - | 97.4% | 99.2% | 66.1% | 69.7% | 98.3% | 97.4% | 98.0% | 73.2% | 97.5% | 100.0% | 100.0% | 99.3% | 82.6% |
| beat_interpolation_004 | 96.0% | 96.9% | 96.9% | 94.6% | 98.8% | 98.5% | 96.6% | - | - | 84.4% | 93.8% | 94.5% | 94.9% | 96.7% | - | 89.1% | 71.9% | 97.9% | 98.7% | 97.2% | 96.4% |
| beat_interpolation_005 | 97.8% | 99.4% | 99.7% | 98.8% | 95.2% | 98.0% | 98.8% | - | - | 96.7% | 67.5% | 65.8% | 96.1% | 97.6% | - | 67.5% | 70.0% | 99.2% | 98.2% | 87.5% | 89.7% |

### beat_revision

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| beat_revision_001 | 92.0% | 93.6% | 93.8% | 94.8% | 94.4% | 96.0% | 94.6% | - | 93.2% | 88.9% | 95.0% | 77.4% | 90.5% | 87.0% | 97.2% | 82.4% | 91.9% | 95.6% | - | 94.5% | 91.3% |
| beat_revision_002 | 96.8% | 95.2% | 95.7% | 96.3% | 97.7% | 98.0% | 98.2% | - | 97.1% | 37.3% | 90.9% | 96.0% | 95.2% | 79.3% | 96.7% | - | 96.1% | 96.7% | 87.5% | 96.6% | 96.2% |
| beat_revision_003 | 97.7% | 97.1% | 95.4% | 95.4% | 98.7% | 98.1% | 98.2% | - | 96.5% | 98.4% | 97.9% | 95.1% | 93.2% | 92.0% | 97.7% | 97.7% | 97.7% | 98.5% | 95.2% | - | 99.2% |
| beat_revision_004 | 94.2% | 93.8% | 94.4% | 95.1% | 94.8% | 97.3% | 93.5% | - | - | 87.3% | 90.1% | 86.0% | 91.6% | 92.0% | - | 95.0% | 94.3% | 96.0% | 91.7% | 93.0% | 93.1% |
| beat_revision_005 | 52.8% | 93.9% | 90.4% | 92.8% | 89.2% | 90.7% | 94.5% | - | 53.8% | 51.0% | - | 38.5% | 92.2% | 52.6% | - | 90.9% | 52.1% | - | 52.8% | - | - |

### constrained_continuation

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| constrained_continuation_001 | 94.2% | 96.0% | 97.9% | 95.5% | 82.5% | 98.3% | 67.7% | - | 67.5% | 80.2% | 85.4% | 65.0% | 96.1% | 94.1% | - | 67.5% | 67.4% | 96.3% | 96.8% | 73.0% | 67.2% |
| constrained_continuation_002 | 94.9% | 96.5% | 94.8% | 96.1% | 72.2% | - | 96.6% | - | 95.3% | 92.2% | 93.4% | 65.0% | 94.4% | 96.1% | - | 67.5% | 67.2% | 94.0% | 95.7% | - | 96.1% |
| constrained_continuation_003 | 92.5% | 93.1% | 88.4% | 94.6% | 84.7% | 90.9% | 79.1% | - | - | 75.3% | 65.5% | 66.6% | 92.1% | 91.3% | - | 66.6% | 67.5% | 90.1% | 94.1% | 92.4% | 70.1% |
| constrained_continuation_004 | 95.3% | 93.1% | 92.3% | 94.5% | 77.3% | 96.8% | 96.4% | - | 83.4% | 91.8% | 92.5% | 67.1% | 92.4% | 91.4% | 67.5% | 67.5% | 67.5% | 71.4% | 96.8% | 68.3% | 97.9% |

### critique_improvement

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| critique_improvement_001 | 92.0% | 85.0% | 93.2% | 93.8% | 90.3% | 89.7% | 82.5% | 94.3% | 92.0% | 78.4% | 93.8% | 93.8% | 86.6% | 81.2% | 93.5% | 93.2% | 95.5% | 91.6% | 86.7% | 85.5% | 91.0% |
| critique_improvement_002 | 80.9% | 87.2% | 87.2% | 85.2% | 86.3% | 89.8% | 77.8% | 88.5% | 91.3% | 73.4% | 83.7% | 82.8% | 61.7% | 71.2% | 79.5% | 88.2% | 91.4% | 90.0% | 75.9% | 84.8% | 84.2% |
| critique_improvement_003 | 83.7% | 84.8% | 80.8% | 81.2% | 89.8% | 89.4% | 79.8% | 87.2% | 87.2% | 11.5% | 88.2% | 87.7% | 77.8% | 62.7% | 84.7% | 84.8% | 81.2% | 82.5% | 84.5% | 89.5% | 79.2% |
| critique_improvement_004 | 88.8% | 90.0% | 87.0% | 91.7% | 86.5% | 82.5% | 81.6% | 86.3% | 92.2% | 76.8% | 80.0% | 84.3% | 75.3% | 70.0% | 91.7% | 87.2% | 87.2% | 88.5% | 85.5% | 92.2% | 87.6% |

### multi_beat_synthesis

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| multi_beat_synthesis_001 | 67.3% | 92.5% | 92.5% | 70.5% | 62.9% | 89.0% | 81.3% | - | 70.7% | 89.7% | 65.5% | 82.6% | 89.3% | 85.9% | 67.5% | 67.5% | 66.8% | 92.5% | 89.8% | 65.2% | 87.0% |
| multi_beat_synthesis_002 | 90.3% | 90.5% | 92.5% | 92.5% | 90.5% | 92.4% | 92.4% | - | - | 90.1% | 67.3% | 64.8% | 90.0% | 90.1% | - | 67.5% | 67.5% | 94.0% | 89.4% | 93.6% | 92.2% |
| multi_beat_synthesis_003 | 64.4% | 89.1% | 64.4% | 91.1% | 65.5% | 91.1% | 91.0% | - | 73.7% | 43.7% | 47.1% | 55.1% | 87.0% | 84.0% | 65.9% | 66.1% | 64.1% | 92.2% | 87.1% | 65.1% | 74.0% |

### theory_conversion

| Task ID | claude-haiku-4.5 | claude-opus-4.5 | claude-sonnet-4 | claude-sonnet-4.5 | deepseek-r1 | deepseek-v3.2 | gemini-2.5-flash | gemini-3-flash-preview | gemini-3-pro-preview | llama-4-maverick | minimax-m2 | ministral-14b-2512 | gpt-4o | gpt-4o-mini | gpt-5 | gpt-5.1 | gpt-5.2 | o3 | o3-mini | qwen3-235b-a22b | grok-4 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| theory_conversion_001 | 96.8% | 96.7% | 95.3% | 95.3% | 94.0% | 97.5% | 95.5% | - | 93.0% | 93.5% | 70.0% | 66.5% | 95.7% | 95.1% | 66.1% | 66.6% | 66.5% | 95.8% | 92.5% | 97.9% | 92.7% |
| theory_conversion_002 | 91.9% | 92.0% | 91.1% | 91.7% | 91.3% | 91.6% | 79.6% | - | 78.1% | 70.4% | 69.8% | 91.5% | 92.1% | 88.6% | 66.6% | 65.5% | 66.5% | - | 90.1% | 72.1% | 91.0% |
| theory_conversion_003 | 91.5% | 94.2% | 90.7% | 91.6% | 96.4% | 95.8% | 93.4% | - | 91.0% | 64.4% | 69.7% | 65.5% | 90.9% | 91.7% | 66.1% | 66.1% | 66.1% | 76.7% | 90.8% | 92.9% | 91.0% |
| theory_conversion_004 | 90.5% | 92.2% | 92.2% | 95.0% | 92.3% | 98.2% | 93.0% | - | 92.4% | 91.2% | 92.7% | 92.0% | 90.9% | 91.6% | 65.5% | 66.6% | 66.6% | 95.0% | 93.6% | 96.1% | 76.1% |