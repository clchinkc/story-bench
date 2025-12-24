"""
Shared LLM client infrastructure for the Story Theory Benchmark.

This module provides a unified interface for making LLM API calls,
handling token tracking, cost extraction, and retry logic.
Used by both standard and agentic generators.
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Task-specific token budgets (same for all models for fairness)
# - max_tokens: total completion budget (MUST be > reasoning + expected output)
# - max_reasoning_tokens: reasoning budget (min 1024 per OpenRouter)
#
# IMPORTANT: These budgets are set generously based on P95 observed usage:
# - Reasoning budget = P95 * 2 (to handle outlier models like gemini-3-pro)
# - Output buffer = 3000 tokens minimum (for ~1000+ word outputs)
# - Some reasoning models (gemini, gpt-5) may exceed reasoning limits
#
# OpenRouter handles mapping to effort levels automatically for all models.
# See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
TASK_TOKEN_CONFIG = {
    # Standard tasks: generous reasoning + 3000 output buffer
    "beat_interpolation": {"max_tokens": 6000, "max_reasoning_tokens": 3000},
    "beat_revision": {"max_tokens": 6000, "max_reasoning_tokens": 3000},
    "constrained_continuation": {"max_tokens": 8000, "max_reasoning_tokens": 4500},
    "theory_conversion": {"max_tokens": 7000, "max_reasoning_tokens": 3500},
    "multi_beat_synthesis": {"max_tokens": 10000, "max_reasoning_tokens": 5500},
    # Agentic tasks (per-turn limits): generous reasoning + 3000 output buffer
    "agentic_constraint_discovery": {"max_tokens": 6000, "max_reasoning_tokens": 3000},
    "agentic_planning_execution": {"max_tokens": 7000, "max_reasoning_tokens": 3500},
    "agentic_iterative_revision": {"max_tokens": 6000, "max_reasoning_tokens": 3000},
    "critique_improvement": {"max_tokens": 7000, "max_reasoning_tokens": 3500},
}

# Models that produce reasoning tokens (include_reasoning=True enables token tracking)
# OpenRouter handles mapping reasoning.max_tokens to effort levels automatically for all models.
REASONING_MODELS = [
    # Anthropic
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4.5",
    # DeepSeek
    "deepseek/deepseek-r1",
    "deepseek-r1",
    "deepseek/deepseek-v3.2",
    # Google
    "google/gemini-2.5-flash",
    "google/gemini-3-pro-preview",
    # MiniMax
    "minimax/minimax-m2",
    # OpenAI
    "openai/gpt-4o-mini",
    "openai/gpt-5",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/o3",
    "openai/o3-mini",
    # Qwen
    "qwen/qwen3-235b-a22b",
    # xAI
    "x-ai/grok-4",
]


@dataclass
class LLMResponse:
    """Standardized response from LLM API call."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: float
    success: bool
    error: str | None = None
    finish_reason: str | None = None  # "stop", "length", "content_filter", etc.


class LLMClient:
    """Shared LLM client for making API calls.

    This client provides:
    - Unified API calling interface
    - Token tracking (prompt, completion, reasoning)
    - Cost extraction
    - Retry logic with exponential backoff
    - Support for reasoning models
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )

    def call(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_reasoning_tokens: int | None = None,
        retry_attempts: int = 3,
        retry_delay: int = 5,
    ) -> LLMResponse:
        """Make an LLM API call with retry logic.

        Args:
            model: Model identifier (e.g., "anthropic/claude-haiku-4.5")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens for completion
            max_reasoning_tokens: Optional limit for reasoning tokens
            retry_attempts: Number of retry attempts on failure
            retry_delay: Base delay between retries (exponential backoff)

        Returns:
            LLMResponse with content, tokens, cost, and success status
        """
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        extra_body = {}

        # Configure reasoning tokens (OpenRouter handles mapping to effort for all models)
        # Minimum 1024 per OpenRouter, max_tokens must be > reasoning to leave room for output
        if max_reasoning_tokens:
            effective_reasoning = max(max_reasoning_tokens, 1024)
            extra_body["reasoning"] = {"max_tokens": effective_reasoning}

        # Enable reasoning token tracking for supported models
        if any(rm in model.lower() for rm in REASONING_MODELS):
            extra_body["include_reasoning"] = True

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        last_error = None
        for attempt in range(retry_attempts):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff

        return LLMResponse(
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
            cost=0.0,
            success=False,
            error=str(last_error),
            finish_reason=None,
        )

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI-style response into standardized LLMResponse."""
        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        # Extract finish_reason from the response
        finish_reason = None
        if response.choices and len(response.choices) > 0:
            finish_reason = getattr(response.choices[0], "finish_reason", None)

        # Extract usage dict for cost and reasoning tokens
        usage_dict = {}
        if response.usage:
            usage_dict = (
                response.usage.model_dump()
                if hasattr(response.usage, "model_dump")
                else vars(response.usage)
            )

        cost = float(usage_dict.get("cost", 0) or 0)
        reasoning_tokens = (usage_dict.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ) or 0

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=cost,
            success=True,
            finish_reason=finish_reason,
        )

    @staticmethod
    def get_token_config(task_type: str) -> dict[str, int]:
        """Get token configuration for a task type."""
        return TASK_TOKEN_CONFIG.get(
            task_type, {"max_tokens": 4000, "max_reasoning_tokens": 1024}
        )


# Singleton instance for convenience
_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the default LLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
