# Infrastructure Sharing Proposal: Unifying Standard and Agentic Tasks

## Current State

### Separate Infrastructure
- **Standard tasks**: `generator.py` + `evaluator.py`
- **Agentic tasks**: `agentic_generator.py` + `agentic_evaluator.py`

### Already Shared ✅
1. **LLM Client** (`llm_client.py`) - Both use same API interface
2. **Results Database** (`results_db.py`) - Both store results in same DB
3. **Utilities** (`utils.py`) - Shared helper functions
4. **Scoring** (`scoring.py`) - Shared scoring logic

### Currently Duplicated ❌
1. **LLM calling patterns** - Similar `_call_model` logic
2. **Result structures** - `GenerationResult` vs `AgenticResult`
3. **Token tracking** - Both accumulate tokens/costs
4. **Prompt building** - Both have prompt builders
5. **Error handling** - Similar retry/error patterns

## Key Insight: Standard Tasks Are Single-Turn Agentic Tasks

A standard generation is just:
- **1 system message** + **1 user prompt** → **1 assistant response**

An agentic task has:
- **Multiple turns** with conversation state

## Proposed Unified Architecture

### 1. Unified Turn-Based Model

```python
@dataclass
class Turn:
    """A single conversational turn."""
    role: str  # "system", "user", "assistant"
    content: str
    turn_type: str  # "system", "prompt", "generation", "feedback", etc.
    metadata: dict = field(default_factory=dict)

@dataclass
class GenerationResult:
    """Unified result for all generation types."""
    generation_id: str
    task_id: str
    task_type: str
    theory: str
    model: str
    sample_index: int

    # Turn history (standard tasks have 3 turns, agentic have N)
    turns: list[Turn]

    # Final output (extracted from last assistant turn)
    final_output: str

    # Token/cost tracking
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_cost: float

    # Task-specific metrics
    metrics: dict[str, Any]  # {questions_asked, revision_count, etc.}

    timestamp: str
    success: bool
    error: str | None = None
```

**Benefits:**
- Single result format for all tasks
- Standard tasks: `turns = [system, user, assistant]`
- Agentic tasks: `turns = [system, user, assistant, user, assistant, ...]`
- Full conversation history preserved for both

### 2. Unified Generator Class

```python
class BaseGenerator:
    """Base class for all generation tasks."""

    def __init__(self, config: GeneratorConfig, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    def _call_model(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None
    ) -> tuple[str, int, int, int, float, str]:
        """Shared LLM calling logic with token tracking."""
        # Already implemented in agentic_generator.py
        # Move to base class

    def _get_last_non_empty_generation(self, turns: list[Turn]) -> str:
        """Extract last valid generation."""
        # Already implemented - move to base class

    def generate(self, task: dict, model: str, sample_index: int) -> GenerationResult:
        """Main entry point - delegates to task-specific logic."""
        task_type = task["task_type"]

        if task_type in ["beat_interpolation", "beat_revision", ...]:
            return self._generate_standard(task, model, sample_index)
        elif task_type == "agentic_iterative_revision":
            return self._generate_iterative_revision(task, model, sample_index)
        # ... etc

    def _generate_standard(self, task, model, sample_index) -> GenerationResult:
        """Single-turn generation."""
        turns = []
        system_prompt = self._build_system_prompt(task)
        user_prompt = self._build_user_prompt(task)

        turns.append(Turn(role="system", content=system_prompt, turn_type="system"))
        turns.append(Turn(role="user", content=user_prompt, turn_type="prompt"))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        output, pt, ct, rt, cost, finish_reason = self._call_model(model, messages)

        turns.append(Turn(
            role="assistant",
            content=output,
            turn_type="generation",
            metadata={"finish_reason": finish_reason}
        ))

        return GenerationResult(
            ...,
            turns=turns,
            final_output=output,
            total_prompt_tokens=pt,
            total_completion_tokens=ct,
            total_reasoning_tokens=rt,
            total_cost=cost,
            metrics={}
        )

    def _generate_iterative_revision(self, task, model, sample_index) -> GenerationResult:
        """Multi-turn iterative revision."""
        # Existing agentic logic, but using unified Turn and GenerationResult
```

**Benefits:**
- Single generator class handles all task types
- Shared LLM calling logic
- Shared token tracking
- Shared error handling
- Task-specific logic in separate methods

### 3. Unified Evaluator

```python
class BaseEvaluator:
    """Base class for all evaluation."""

    def evaluate(self, generation: GenerationResult, task: dict) -> EvaluationResult:
        """Main entry point."""
        task_type = task["task_type"]

        # All evaluations have:
        # 1. Output evaluation (constraint checking, quality)
        # 2. Process evaluation (optional, for agentic tasks)

        output_scores = self._evaluate_output(generation.final_output, task)

        if task_type.startswith("agentic_"):
            process_scores = self._evaluate_process(generation.turns, task)
            return self._compute_agentic_score(output_scores, process_scores)
        else:
            return self._compute_standard_score(output_scores)
```

**Benefits:**
- Single evaluator for all tasks
- Shared constraint-checking logic
- Standard tasks: output evaluation only
- Agentic tasks: output + process evaluation

## Migration Plan

### Phase 1: Unify Result Structures (Low Risk)
1. Create unified `Turn` and `GenerationResult` in `generator.py`
2. Update `agentic_generator.py` to use new structures (backward compatible)
3. Update `results_db.py` to handle both old and new formats

### Phase 2: Extract Shared Methods (Medium Risk)
1. Create `BaseGenerator` class with shared methods:
   - `_call_model`
   - `_get_last_non_empty_generation`
   - Token tracking logic
2. Update both generators to inherit from base

### Phase 3: Merge Generators (Higher Risk)
1. Move all generation logic into single `Generator` class
2. Keep task-specific methods separate
3. Deprecate `agentic_generator.py` (but keep for reference)

### Phase 4: Merge Evaluators (Higher Risk)
1. Create unified `Evaluator` class
2. Merge constraint-checking logic
3. Keep process/output evaluation separate

## Benefits Summary

### Code Quality
- **-40% duplication**: Eliminate repeated LLM calling, token tracking
- **Easier maintenance**: Fix bugs in one place
- **Consistent error handling**: Same patterns everywhere

### Extensibility
- **New task types**: Just add a new method, not a new file
- **Shared improvements**: Bug fixes benefit all tasks
- **Clear hierarchy**: Standard tasks are special case of agentic

### Performance
- **Same**: No performance impact (same underlying logic)
- **Better diagnostics**: Unified turn history for debugging

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Incremental migration, keep both formats temporarily |
| Regression bugs | Comprehensive testing before/after each phase |
| Complexity increase | Clear separation between shared and task-specific logic |

## Recommendation

**Start with Phase 1** (unify result structures) because:
- Low risk (additive change)
- High value (enables all future work)
- Easy to test (just data structures)
- Backward compatible

Then evaluate whether Phases 2-4 are worth the migration effort.

## Example: How Standard Task Becomes Agentic-Style

**Before (generator.py):**
```python
def generate_story(task, model):
    prompt = build_prompt(task)
    response = llm_client.call(model, prompt)
    return GenerationResult(output=response.content, ...)
```

**After (unified):**
```python
def generate_story(task, model):
    turns = [
        Turn(role="system", content=system_prompt),
        Turn(role="user", content=user_prompt),
    ]
    output, *tokens = self._call_model(model, turns_to_messages(turns))
    turns.append(Turn(role="assistant", content=output))
    return GenerationResult(turns=turns, final_output=output, ...)
```

The logic is nearly identical, but now uses the unified turn-based model that scales to multi-turn tasks.
