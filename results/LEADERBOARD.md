# Story Theory Benchmark Leaderboard

*Last updated: 2025-12-17T00:52:23.185811*

## Overview

- **Models evaluated**: 20
- **Tasks**: 21
- **Evaluator models**: claude-haiku-4.5, gemini-2.5-flash, gpt-5-mini
- **Aggregation**: Median across evaluators
- **Scoring**: Programmatic (50%) + LLM Judge (50%)
- **Total generations**: 420
- **Total evaluations**: 1260
- **Total cost**: $16.9564

## Model Rankings

| Rank | Model | Company | Score | Gen Cost | Value | LLM Judge |
|------|-------|---------|-------|----------|---------|-----------|
| 1 | deepseek-v3.2 | DeepSeek | 95.5% | $0.0156 | 5845.8 | 98.7% |
| 2 | claude-opus-4.5 | Anthropic | 95.0% | $0.6352 | 142.1 | 97.9% |
| 3 | claude-sonnet-4.5 | Anthropic | 94.0% | $0.4016 | 219.9 | 98.7% |
| 4 | o3 | OpenAI | 93.4% | $0.2601 | 335.2 | 98.5% |
| 5 | gpt-4o | OpenAI | 93.2% | $0.1758 | 494.1 | 96.6% |
| 6 | claude-sonnet-4 | Anthropic | 93.0% | $0.3791 | 228.1 | 97.5% |
| 7 | gemini-2.5-flash | Google | 92.2% | $0.0898 | 947.1 | 97.8% |
| 8 | o3-mini | OpenAI | 92.0% | $0.1217 | 695.7 | 91.4% |
| 9 | claude-haiku-4.5 | Anthropic | 90.1% | $0.1488 | 546.0 | 93.1% |
| 10 | gpt-4o-mini | OpenAI | 89.4% | $0.0113 | 7087.3 | 88.9% |
| 11 | deepseek-r1 | DeepSeek | 88.1% | $0.0687 | 1130.3 | 97.7% |
| 12 | qwen3-235b-a22b | Alibaba | 87.9% | $0.0277 | 2783.3 | 96.1% |
| 13 | grok-4 | xAI | 86.4% | $0.4826 | 154.7 | 98.0% |
| 14 | gemini-3-pro-preview | Google | 86.2% | $0.5838 | 127.3 | 92.6% |
| 15 | llama-4-maverick | Meta | 81.5% | $0.0144 | 4612.5 | 79.8% |
| 16 | gpt-5 | OpenAI | 79.6% | $0.5945 | 106.5 | 97.9% |
| 17 | minimax-m2 | MiniMax | 77.8% | $0.0429 | 1412.3 | 94.4% |
| 18 | gpt-5.2 | OpenAI | 75.2% | $0.5032 | 112.4 | 93.9% |
| 19 | ministral-14b-2512 | Mistral | 74.8% | $0.0097 | 5787.8 | 90.8% |
| 20 | gpt-5.1 | OpenAI | 73.6% | $0.5298 | 102.4 | 98.4% |

## Best Value (Score²/Cost)

*Higher = better value. Formula: Score² / Cost (rewards quality quadratically)*

| Rank | Model | Company | Score | Gen Cost | Value |
|------|-------|---------|-------|----------|-------|
| 1 | gpt-4o-mini | OpenAI | 89.4% | $0.0113 | 7087.3 |
| 2 | deepseek-v3.2 | DeepSeek | 95.5% | $0.0156 | 5845.8 |
| 3 | ministral-14b-2512 | Mistral | 74.8% | $0.0097 | 5787.8 |
| 4 | llama-4-maverick | Meta | 81.5% | $0.0144 | 4612.5 |
| 5 | qwen3-235b-a22b | Alibaba | 87.9% | $0.0277 | 2783.3 |
| 6 | minimax-m2 | MiniMax | 77.8% | $0.0429 | 1412.3 |
| 7 | deepseek-r1 | DeepSeek | 88.1% | $0.0687 | 1130.3 |
| 8 | gemini-2.5-flash | Google | 92.2% | $0.0898 | 947.1 |
| 9 | o3-mini | OpenAI | 92.0% | $0.1217 | 695.7 |
| 10 | claude-haiku-4.5 | Anthropic | 90.1% | $0.1488 | 546.0 |
| 11 | gpt-4o | OpenAI | 93.2% | $0.1758 | 494.1 |
| 12 | o3 | OpenAI | 93.4% | $0.2601 | 335.2 |
| 13 | claude-sonnet-4 | Anthropic | 93.0% | $0.3791 | 228.1 |
| 14 | claude-sonnet-4.5 | Anthropic | 94.0% | $0.4016 | 219.9 |
| 15 | grok-4 | xAI | 86.4% | $0.4826 | 154.7 |
| 16 | claude-opus-4.5 | Anthropic | 95.0% | $0.6352 | 142.1 |
| 17 | gemini-3-pro-preview | Google | 86.2% | $0.5838 | 127.3 |
| 18 | gpt-5.2 | OpenAI | 75.2% | $0.5032 | 112.4 |
| 19 | gpt-5 | OpenAI | 79.6% | $0.5945 | 106.5 |
| 20 | gpt-5.1 | OpenAI | 73.6% | $0.5298 | 102.4 |

## Scores by Task Type

| Model | beat_interpolation | beat_revision | constrained_continuation | multi_beat_synthesis | theory_conversion |
|-------|-------|-------|-------|-------|-------|
| deepseek-v3.2 | 98.1% | 96.0% | 95.3% | 90.9% | 95.8% |
| claude-opus-4.5 | 99.1% | 94.7% | 94.7% | 90.7% | 93.8% |
| claude-sonnet-4.5 | 98.1% | 94.9% | 95.2% | 84.7% | 93.4% |
| o3 | 99.0% | 96.7% | 87.9% | 92.9% | 89.2% |
| gpt-4o | 96.7% | 92.5% | 93.8% | 88.8% | 92.4% |
| claude-sonnet-4 | 98.2% | 94.0% | 93.4% | 83.1% | 92.3% |
| gemini-2.5-flash | 98.3% | 95.8% | 84.9% | 88.3% | 90.4% |
| o3-mini | 99.2% | 81.8% | 95.8% | 88.8% | 91.7% |
| claude-haiku-4.5 | 98.0% | 86.7% | 94.2% | 74.0% | 92.6% |
| gpt-4o-mini | 94.8% | 80.6% | 93.2% | 86.7% | 91.7% |
| deepseek-r1 | 93.1% | 95.0% | 79.2% | 73.0% | 93.5% |
| qwen3-235b-a22b | 96.2% | 94.7% | 77.9% | 74.6% | 89.7% |
| grok-4 | 82.6% | 95.0% | 82.8% | 84.4% | 87.7% |
| gemini-3-pro-preview | 97.9% | 85.1% | 82.1% | 72.2% | 88.6% |
| llama-4-maverick | 93.2% | 72.6% | 84.9% | 74.5% | 79.9% |
| gpt-5 | 99.0% | 97.2% | 67.5% | 66.7% | 66.1% |
| minimax-m2 | 72.8% | 93.5% | 84.2% | 60.0% | 75.6% |
| gpt-5.2 | 82.7% | 86.4% | 67.4% | 66.1% | 66.4% |
| ministral-14b-2512 | 79.2% | 78.6% | 65.9% | 67.5% | 78.9% |
| gpt-5.1 | 74.4% | 91.5% | 67.3% | 67.0% | 66.2% |

## Component Breakdown by Task Type


### deepseek-v3.2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 98.1% | 92.7% | 100.0% |
| beat_revision | 96.0% | 97.9% | 96.8% |
| constrained_continuation | 95.3% | 88.5% | 100.0% |
| multi_beat_synthesis | 90.9% | 69.9% | 99.1% |
| theory_conversion | 95.8% | 86.1% | 98.7% |

### claude-opus-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 99.1% | 96.5% | 100.0% |
| beat_revision | 94.7% | 94.0% | 94.8% |
| constrained_continuation | 94.7% | 78.7% | 100.0% |
| multi_beat_synthesis | 90.7% | 73.2% | 95.5% |
| theory_conversion | 93.8% | 77.3% | 98.9% |

### claude-sonnet-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 98.1% | 92.3% | 100.0% |
| beat_revision | 94.9% | 93.5% | 95.9% |
| constrained_continuation | 95.2% | 80.6% | 100.0% |
| multi_beat_synthesis | 84.7% | 69.9% | 99.1% |
| theory_conversion | 93.4% | 75.4% | 99.2% |

### o3

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 99.0% | 96.0% | 100.0% |
| beat_revision | 96.7% | 97.3% | 94.8% |
| constrained_continuation | 87.9% | 79.5% | 100.0% |
| multi_beat_synthesis | 92.9% | 72.0% | 99.8% |
| theory_conversion | 89.2% | 80.4% | 98.2% |

### gpt-4o

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 96.7% | 87.2% | 99.8% |
| beat_revision | 92.5% | 95.2% | 90.6% |
| constrained_continuation | 93.8% | 75.1% | 100.0% |
| multi_beat_synthesis | 88.8% | 68.0% | 93.6% |
| theory_conversion | 92.4% | 71.7% | 98.9% |

### claude-sonnet-4

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 98.2% | 93.2% | 99.9% |
| beat_revision | 94.0% | 93.9% | 94.8% |
| constrained_continuation | 93.4% | 77.4% | 98.0% |
| multi_beat_synthesis | 83.1% | 69.6% | 96.1% |
| theory_conversion | 92.3% | 72.8% | 98.4% |

### gemini-2.5-flash

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 98.3% | 93.0% | 100.0% |
| beat_revision | 95.8% | 97.0% | 95.0% |
| constrained_continuation | 84.9% | 77.6% | 99.6% |
| multi_beat_synthesis | 88.3% | 69.3% | 97.6% |
| theory_conversion | 90.4% | 77.0% | 96.9% |

### o3-mini

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 99.2% | 97.2% | 99.8% |
| beat_revision | 81.8% | 94.8% | 66.2% |
| constrained_continuation | 95.8% | 84.1% | 99.6% |
| multi_beat_synthesis | 88.8% | 72.7% | 92.1% |
| theory_conversion | 91.7% | 72.5% | 97.2% |

### claude-haiku-4.5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 98.0% | 91.8% | 100.0% |
| beat_revision | 86.7% | 91.8% | 77.5% |
| constrained_continuation | 94.2% | 77.8% | 99.6% |
| multi_beat_synthesis | 74.0% | 73.0% | 91.8% |
| theory_conversion | 92.6% | 74.0% | 98.6% |

### gpt-4o-mini

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 94.8% | 91.8% | 99.5% |
| beat_revision | 80.6% | 92.6% | 64.9% |
| constrained_continuation | 93.2% | 75.4% | 98.7% |
| multi_beat_synthesis | 86.7% | 69.6% | 88.5% |
| theory_conversion | 91.7% | 74.1% | 96.4% |

### deepseek-r1

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 93.1% | 94.1% | 100.0% |
| beat_revision | 95.0% | 96.9% | 95.8% |
| constrained_continuation | 79.2% | 89.6% | 99.2% |
| multi_beat_synthesis | 73.0% | 69.5% | 93.4% |
| theory_conversion | 93.5% | 75.8% | 99.1% |

### qwen3-235b-a22b

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 96.2% | 92.7% | 99.7% |
| beat_revision | 94.7% | 95.0% | 92.5% |
| constrained_continuation | 77.9% | 72.1% | 99.8% |
| multi_beat_synthesis | 74.6% | 74.4% | 88.9% |
| theory_conversion | 89.7% | 85.1% | 96.8% |

### grok-4

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 82.6% | 91.0% | 99.7% |
| beat_revision | 95.0% | 93.5% | 93.3% |
| constrained_continuation | 82.8% | 78.7% | 99.6% |
| multi_beat_synthesis | 84.4% | 69.7% | 98.9% |
| theory_conversion | 87.7% | 71.2% | 98.2% |

### gemini-3-pro-preview

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 97.9% | 94.6% | 100.0% |
| beat_revision | 85.1% | 91.0% | 74.8% |
| constrained_continuation | 82.1% | 75.0% | 99.8% |
| multi_beat_synthesis | 72.2% | 69.7% | 96.7% |
| theory_conversion | 88.6% | 71.1% | 97.7% |

### llama-4-maverick

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 93.2% | 88.1% | 92.2% |
| beat_revision | 72.6% | 84.6% | 61.1% |
| constrained_continuation | 84.9% | 70.6% | 84.6% |
| multi_beat_synthesis | 74.5% | 73.1% | 78.8% |
| theory_conversion | 79.9% | 73.8% | 83.4% |

### gpt-5

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 99.0% | 99.4% | 100.0% |
| beat_revision | 97.2% | 96.0% | 96.3% |
| constrained_continuation | 67.5% | 69.8% | 100.0% |
| multi_beat_synthesis | 66.7% | 70.0% | 98.4% |
| theory_conversion | 66.1% | 70.0% | 97.1% |

### minimax-m2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 72.8% | 78.0% | 97.7% |
| beat_revision | 93.5% | 94.9% | 89.5% |
| constrained_continuation | 84.2% | 70.8% | 98.8% |
| multi_beat_synthesis | 60.0% | 70.0% | 84.9% |
| theory_conversion | 75.6% | 71.0% | 98.0% |

### gpt-5.2

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 82.7% | 85.0% | 100.0% |
| beat_revision | 86.4% | 90.6% | 77.9% |
| constrained_continuation | 67.4% | 69.9% | 99.8% |
| multi_beat_synthesis | 66.1% | 70.0% | 97.2% |
| theory_conversion | 66.4% | 70.0% | 97.9% |

### ministral-14b-2512

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 79.2% | 78.4% | 99.2% |
| beat_revision | 78.6% | 83.2% | 73.7% |
| constrained_continuation | 65.9% | 69.9% | 96.9% |
| multi_beat_synthesis | 67.5% | 69.8% | 88.2% |
| theory_conversion | 78.9% | 70.0% | 97.6% |

### gpt-5.1

| Task Type | Score | Programmatic | LLM Judge |
|-----------|-------|--------------|-----------|
| beat_interpolation | 74.4% | 81.3% | 100.0% |
| beat_revision | 91.5% | 91.5% | 95.9% |
| constrained_continuation | 67.3% | 70.0% | 99.6% |
| multi_beat_synthesis | 67.0% | 70.0% | 99.1% |
| theory_conversion | 66.2% | 69.9% | 97.4% |

## Cost Efficiency

*Note: Reasoning tokens (for CoT models) are billed but don't produce output, affecting cost efficiency.*

| Model | Gen Cost | Output Tokens | Reasoning % | $/1K Output |
|-------|----------|---------------|-------------|-------------|
| deepseek-v3.2 | $0.0156 | 12,937 | 41.3% | $0.0012 |
| claude-opus-4.5 | $0.6352 | 14,556 | 31.0% | $0.0436 |
| claude-sonnet-4.5 | $0.4016 | 15,888 | 29.3% | $0.0253 |
| o3 | $0.2601 | 15,794 | 42.4% | $0.0165 |
| gpt-4o | $0.1758 | 12,872 | 0.0% | $0.0137 |
| claude-sonnet-4 | $0.3791 | 13,715 | 34.6% | $0.0276 |
| gemini-2.5-flash | $0.0898 | 15,061 | 55.1% | $0.0060 |
| o3-mini | $0.1217 | 13,039 | 43.2% | $0.0093 |
| claude-haiku-4.5 | $0.1488 | 16,038 | 37.0% | $0.0093 |
| gpt-4o-mini | $0.0113 | 14,096 | 0.0% | $0.0008 |
| deepseek-r1 | $0.0687 | 12,060 | 50.9% | $0.0057 |
| qwen3-235b-a22b | $0.0277 | 11,359 | 62.3% | $0.0024 |
| grok-4 | $0.4826 | 17,058 | 38.8% | $0.0283 |
| gemini-3-pro-preview | $0.5838 | 16,931 | 62.7% | $0.0345 |
| llama-4-maverick | $0.0144 | 12,660 | 0.0% | $0.0011 |
| gpt-5 | $0.5945 | 22,466 | 60.6% | $0.0265 |
| minimax-m2 | $0.0429 | 18,695 | 46.8% | $0.0023 |
| gpt-5.2 | $0.5032 | 26,073 | 22.2% | $0.0193 |
| ministral-14b-2512 | $0.0097 | 27,086 | 0.0% | $0.0004 |
| gpt-5.1 | $0.5298 | 30,241 | 40.2% | $0.0175 |