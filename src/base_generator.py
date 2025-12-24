"""
Base generator infrastructure for the Story Theory Benchmark.

Provides unified structures and base classes that both standard and agentic
generators inherit from. This eliminates code duplication and provides a
consistent architecture.

Key insight: Standard tasks are just single-turn agentic tasks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """
    A single conversational turn (universal for all task types).

    Standard tasks have 3 turns: [system, user, assistant]
    Agentic tasks have N turns: [system, user, assistant, user, assistant, ...]
    """

    role: str  # "system", "user", "assistant"
    content: str
    turn_type: str  # "system", "prompt", "generation", "feedback", "critique", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """
    Unified result structure for all generation types.

    This replaces both the old GenerationResult (standard) and AgenticResult.
    """

    generation_id: str
    task_id: str
    task_type: str
    theory: str
    model: str
    sample_index: int

    # Universal: All tasks have conversation history
    turns: list[Turn]

    # Universal: Final output extracted from last valid generation
    final_output: str

    # Universal: Token and cost tracking
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_cost: float

    # Task-specific metrics (extensible via dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Examples: {"questions_asked": 5, "revision_count": 3, "plan_quality": 0.8}

    # Status
    timestamp: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation_id": self.generation_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "theory": self.theory,
            "model": self.model,
            "sample_index": self.sample_index,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "turn_type": t.turn_type,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "final_output": self.final_output,
            "metrics": self.metrics,
            "token_usage": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "output_tokens": self.total_completion_tokens
                - self.total_reasoning_tokens,
                "total_cost": self.total_cost,
            },
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


class BaseGenerator:
    """
    Base class for all generators (standard and agentic).

    Provides shared infrastructure:
    - LLM calling with token tracking
    - Turn management
    - Error handling
    - Final output validation

    Subclasses implement task-specific generation logic.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize base generator.

        Args:
            llm_client: Shared LLM client (created if not provided)
        """
        self.llm_client = llm_client or LLMClient()

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        max_reasoning_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, int, int, int, float, str]:
        """
        Shared LLM calling logic with token tracking and logging.

        Args:
            model: Model identifier
            messages: Conversation history
            max_tokens: Max total tokens (completion + reasoning)
            max_reasoning_tokens: Max reasoning/thinking tokens
            temperature: Sampling temperature

        Returns:
            (output, prompt_tokens, completion_tokens, reasoning_tokens, cost, finish_reason)

        Raises:
            RuntimeError: If API call fails
        """
        response = self.llm_client.call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_reasoning_tokens=max_reasoning_tokens,
            retry_attempts=3,
            retry_delay=5,
        )

        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")

        # Log warnings for empty or non-stop responses
        finish_reason = response.finish_reason or "unknown"
        if not response.content.strip():
            logger.warning(
                f"Empty response from {model} (finish_reason={finish_reason}, "
                f"completion_tokens={response.completion_tokens}, "
                f"reasoning_tokens={response.reasoning_tokens})"
            )
        elif finish_reason != "stop":
            logger.warning(
                f"Non-stop finish_reason from {model}: {finish_reason} "
                f"(completion_tokens={response.completion_tokens})"
            )

        return (
            response.content,
            response.prompt_tokens,
            response.completion_tokens,
            response.reasoning_tokens,
            response.cost,
            finish_reason,
        )

    def _validate_final_output(self, turns: list[Turn]) -> str:
        """
        Extract and validate final output from conversation turns.

        Args:
            turns: List of conversation turns

        Returns:
            Final generation content (stripped)

        Raises:
            RuntimeError: If no valid generation found
        """
        # Find the last assistant generation
        final_output = ""
        for turn in reversed(turns):
            if turn.role == "assistant" and turn.turn_type == "generation":
                final_output = turn.content
                break

        # Fail if empty
        if not final_output.strip():
            logger.error("Generation failed: final output is empty")
            raise RuntimeError("Empty generation - no valid output produced")

        return final_output.strip()

    def _turns_to_messages(self, turns: list[Turn]) -> list[dict[str, str]]:
        """Convert Turn objects to LLM API message format."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in turns
            if turn.role in ["system", "user", "assistant"]
        ]
