"""
Agentic generation module for the Story Theory Benchmark.

This module handles multi-turn agentic tasks where models must:
- Ask questions to discover constraints (Constraint Discovery)
- Plan before executing (Planning-then-Execution)
- Revise based on feedback (Iterative Revision)

These tasks test higher-order capabilities beyond single-shot generation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from llm_client import LLMClient
from utils import (
    generate_id,
    get_timestamp,
)


@dataclass
class AgenticTurn:
    """A single turn in an agentic conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    turn_type: str  # "question", "answer", "plan", "feedback", "generation", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticResult:
    """Result of an agentic multi-turn generation."""

    generation_id: str
    task_id: str
    task_type: str
    agentic_type: (
        str  # "constraint_discovery", "planning_execution", "iterative_revision"
    )
    theory: str
    model: str
    sample_index: int

    # Conversation history
    turns: list[AgenticTurn]

    # Final output
    output: str

    # Metrics
    total_turns: int
    questions_asked: int  # For constraint discovery
    constraints_discovered: int  # For constraint discovery
    plan_quality: float | None  # For planning-execution
    revision_count: int  # For iterative revision

    # Token/cost tracking
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    total_cost: float

    timestamp: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation_id": self.generation_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "agentic_type": self.agentic_type,
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
            "output": self.output,
            "metrics": {
                "total_turns": self.total_turns,
                "questions_asked": self.questions_asked,
                "constraints_discovered": self.constraints_discovered,
                "plan_quality": self.plan_quality,
                "revision_count": self.revision_count,
            },
            "token_usage": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "total_cost": self.total_cost,
            },
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class AgenticConfig:
    """Configuration for agentic generation.

    Token budgets must satisfy: max_tokens > max_reasoning_tokens
    OpenRouter enforces minimum 1024 for reasoning tokens.
    """

    temperature: float = 0.7
    max_tokens_per_turn: int = 4000  # Default per-turn budget
    max_reasoning_tokens: int = (
        1500  # Reasoning budget (min 1024 enforced by OpenRouter)
    )

    # Constraint discovery settings
    max_questions: int = 5

    # Iterative revision settings
    max_revisions: int = 3

    # Planning-execution settings (must be > max_reasoning_tokens)
    plan_max_tokens: int = 3500  # 1500 reasoning + 2000 plan output
    execution_max_tokens: int = 5000  # 1500 reasoning + 3500 story output

    retry_attempts: int = 3
    retry_delay: int = 5


class AgenticPromptBuilder:
    """Builds prompts for agentic task types."""

    SYSTEM_PROMPT_CONSTRAINT_DISCOVERY = """You are a professional storyteller who gathers requirements before writing.

Your task has TWO PHASES:
1. DISCOVERY PHASE: Ask YES/NO questions to discover hidden story constraints
2. GENERATION PHASE: Write the story satisfying all discovered constraints

IMPORTANT:
- In the discovery phase, output ONLY your question (one at a time)
- Questions must be answerable with YES or NO
- You have a limited number of questions - be strategic
- After discovery, you'll be told to generate the story

Think about what constraints typically matter for this type of story:
- Character traits, motivations, relationships
- Setting details, time period, atmosphere
- Plot requirements, must-include events
- Tone, style, thematic elements
- Things to avoid"""

    SYSTEM_PROMPT_PLANNING_EXECUTION = """You are a professional storyteller who plans before writing.

Your task has TWO PHASES:
1. PLANNING PHASE: Create a detailed story plan/outline
2. EXECUTION PHASE: Write the actual story following your plan

In the PLANNING phase, output a structured plan including:
- Beat-by-beat outline with key events
- Character arcs and development points
- Key dialogue moments or revelations
- Pacing and tension curve
- How constraints will be satisfied

In the EXECUTION phase, follow your plan closely while writing engaging prose."""

    SYSTEM_PROMPT_ITERATIVE_REVISION = """You are a professional storyteller who revises based on feedback.

Your task has MULTIPLE ROUNDS:
1. INITIAL GENERATION: Write your best attempt
2. FEEDBACK ROUNDS: Receive specific feedback and revise

When revising:
- Address ALL feedback points
- Preserve what was working well
- Make targeted changes, not complete rewrites
- Improve overall quality with each iteration"""

    SYSTEM_PROMPT_CRITIQUE_IMPROVEMENT = """You are a professional storyteller who improves through critique.

Your task involves iterative improvement:
1. Write your best initial attempt
2. Receive detailed critique from a professional editor
3. Revise to address the critique while preserving strengths
4. Repeat until the story meets all requirements

When revising:
- Address EVERY critique point specifically
- Don't introduce new issues while fixing old ones
- Preserve elements that received positive feedback
- Aim for measurable improvement each round"""

    SYSTEM_PROMPT_CRITIC = """You are a professional story editor providing constructive critique.

Evaluate the story against these criteria:
{criteria}

For each criterion:
1. Score from 0.0-1.0
2. Provide SPECIFIC feedback with quotes from the text
3. Suggest concrete improvements

Format your response as:
CRITERION: [name]
SCORE: [0.0-1.0]
EVIDENCE: [quote from text]
FEEDBACK: [specific improvement needed]

End with OVERALL_ASSESSMENT: [2-3 sentence summary]"""

    @classmethod
    def build_constraint_discovery_system(cls, task: dict[str, Any]) -> str:
        """Build system prompt for constraint discovery."""
        return cls.SYSTEM_PROMPT_CONSTRAINT_DISCOVERY

    @classmethod
    def build_constraint_discovery_initial(cls, task: dict[str, Any]) -> str:
        """Build initial user prompt for constraint discovery."""
        context = task["story_context"]
        max_questions = task.get("max_questions", 5)
        hidden_constraints = task["hidden_constraints"]
        num_hidden = len(hidden_constraints)

        prompt = f"""You will write a story following the {task.get("theory", "narrative")} framework.

STORY CONTEXT:
- Genre: {context.get("genre", "Not specified")}
- Protagonist: {context.get("protagonist", "Not specified")}
- Setting: {context.get("setting", "Not specified")}
- Required beats: {", ".join(task.get("required_beats", []))}

HIDDEN CONSTRAINTS: There are {num_hidden} constraints you don't know yet.

You have {max_questions} YES/NO questions to discover them.

Ask your first question:"""
        return prompt

    @classmethod
    def build_constraint_discovery_generate(
        cls, task: dict[str, Any], discovered_constraints: list[str]
    ) -> str:
        """Build generation prompt after constraint discovery."""
        constraints_str = "\n".join(f"- {c}" for c in discovered_constraints)
        word_range = task.get("word_count", [400, 600])

        prompt = f"""DISCOVERY COMPLETE. Now write the story.

DISCOVERED CONSTRAINTS:
{constraints_str}

REQUIREMENTS:
- Word count: {word_range[0]}-{word_range[1]} words
- Follow the {task.get("theory", "narrative")} framework
- Satisfy ALL discovered constraints

Output ONLY the story (no explanations)."""
        return prompt

    @classmethod
    def build_oracle_prompt(
        cls,
        question: str,
        story_context: dict[str, Any],
        constraint_info: list[dict[str, Any]],
    ) -> str:
        """Build prompt for constraint discovery oracle.

        The oracle uses LLM to semantically match questions to hidden constraints.

        Args:
            question: The YES/NO question asked by the model
            story_context: Story context (genre, protagonist, setting)
            constraint_info: List of constraint dicts with id, description, answer

        Returns:
            Prompt for LLM to determine if question matches any constraint
        """
        constraints_desc = "\n".join(
            f"{i + 1}. ({c['id']}): {c['description']}"
            for i, c in enumerate(constraint_info)
        )

        return f"""You are an oracle for a story constraint discovery game.

STORY CONTEXT:
- Genre: {story_context.get("genre", "fantasy")}
- Protagonist: {story_context.get("protagonist", "Not specified")}
- Setting: {story_context.get("setting", "Not specified")}

HIDDEN CONSTRAINTS (the storyteller must discover these through questions):
{constraints_desc}

A storyteller is asking YES/NO questions to discover these constraints.

QUESTION: "{question}"

TASK: Determine if this question is semantically asking about ANY of the hidden constraints.
- Match based on MEANING, not just keywords
- "Does the protagonist have special abilities?" matches a constraint about "cannot use magic"
- "Will a mentor guide the protagonist?" matches a constraint about "mentor death"
- "Is there a time pressure?" matches a constraint about "deadline" or "time limit"
- "Is there family history involved?" matches a constraint about "father failed hero"
- "Is a special item required?" matches a constraint about "reforging a sword"
- Questions about protagonist's motivation, duty, or revenge do NOT match unless a constraint specifically mentions them

If the question matches a constraint, respond with that constraint's number (1-{len(constraint_info)}).
If the question does NOT match any constraint, respond with "NONE".

Respond with ONLY the number (1-{len(constraint_info)}) or "NONE":"""

    @classmethod
    def build_planning_execution_plan(cls, task: dict[str, Any]) -> str:
        """Build planning phase prompt."""
        context = task.get("story_context", {})
        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)

        prompt = f"""Create a detailed story plan for the following:

CONTEXT:
- Theory: {task.get("theory", "narrative")}
- Protagonist: {context.get("protagonist", "Not specified")}
- Setting: {context.get("setting", "Not specified")}
- Conflict: {context.get("central_conflict", "Not specified")}
- Required beats: {", ".join(task.get("required_beats", []))}

CONSTRAINTS:
{constraints_str}

Create a beat-by-beat plan that:
1. Maps out each required beat with specific events
2. Shows how constraints will be satisfied
3. Plans character development moments
4. Identifies key dialogue or revelations

Output your plan (will be used for execution):"""
        return prompt

    @classmethod
    def build_planning_execution_execute(cls, task: dict[str, Any], plan: str) -> str:
        """Build execution phase prompt."""
        word_range = task.get("word_count", [600, 900])

        prompt = f"""Execute your plan and write the story.

YOUR PLAN:
{plan}

REQUIREMENTS:
- Word count: {word_range[0]}-{word_range[1]} words
- Follow your plan closely
- Write engaging narrative prose
- Satisfy all constraints mentioned in your plan

Output ONLY the story (no beat labels or explanations)."""
        return prompt

    @classmethod
    def build_iterative_revision_initial(cls, task: dict[str, Any]) -> str:
        """Build initial generation prompt for iterative revision."""
        context = task.get("story_context", {})
        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        word_range = task.get("word_count", [400, 600])

        prompt = f"""Write a story with the following requirements:

CONTEXT:
- Theory: {task.get("theory", "narrative")}
- Protagonist: {context.get("protagonist", "Not specified")}
- Setting: {context.get("setting", "Not specified")}
- Required beats: {", ".join(task.get("required_beats", []))}

CONSTRAINTS:
{constraints_str}

REQUIREMENTS:
- Word count: {word_range[0]}-{word_range[1]} words

Output ONLY the story:"""
        return prompt

    @classmethod
    def build_iterative_revision_feedback(
        cls, previous_output: str, feedback: list[str]
    ) -> str:
        """Build revision prompt with feedback."""
        feedback_str = "\n".join(f"- {f}" for f in feedback)

        prompt = f"""Your previous version:
{previous_output[:2000]}{"..." if len(previous_output) > 2000 else ""}

FEEDBACK TO ADDRESS:
{feedback_str}

Revise your story to address ALL feedback points while preserving what worked well.
Output ONLY the revised story:"""
        return prompt

    @classmethod
    def build_critique_improvement_initial(cls, task: dict[str, Any]) -> str:
        """Build initial generation prompt for critique improvement."""
        context = task.get("story_context", {})
        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        word_range = task.get("word_count", [400, 600])
        criteria = task.get("evaluation_criteria", [])
        criteria_str = "\n".join(
            f"- {c['criterion']}: {c.get('description', '')}" for c in criteria
        )

        prompt = f"""Write a story with the following requirements:

STORY THEORY: {task.get("theory", "narrative")}

STORY CONTEXT:
- Protagonist: {context.get("protagonist", "Not specified")}
- Setting: {context.get("setting", "Not specified")}
- Central conflict: {context.get("central_conflict", "Not specified")}
- Tone: {context.get("tone", "Not specified")}

REQUIRED BEATS: {", ".join(task.get("required_beats", []))}

CONSTRAINTS:
{constraints_str}

EVALUATION CRITERIA (what you'll be judged on):
{criteria_str}

REQUIREMENTS:
- Word count: {word_range[0]}-{word_range[1]} words

Output ONLY the story:"""
        return prompt

    @classmethod
    def build_critique_prompt(cls, task: dict[str, Any], story: str) -> str:
        """Build critique prompt for the critic model."""
        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        criteria = task.get("evaluation_criteria", [])
        criteria_str = "\n".join(
            f"- {c['criterion']} (weight: {c.get('weight', 0.2)}): {c.get('description', '')}"
            for c in criteria
        )

        prompt = f"""Critique this story against the requirements:

STORY:
{story}

REQUIRED BEATS: {", ".join(task.get("required_beats", []))}

CONSTRAINTS:
{constraints_str}

EVALUATION CRITERIA:
{criteria_str}

Provide specific, actionable feedback for improvement. Quote the text when pointing out issues."""
        return prompt

    @classmethod
    def build_critique_revision_prompt(cls, previous_output: str, critique: str) -> str:
        """Build revision prompt with critique feedback."""
        prompt = f"""Your previous version:
{previous_output}

EDITOR'S CRITIQUE:
{critique}

Revise your story to address ALL critique points while preserving what worked well.
Focus on the specific issues identified.
Output ONLY the revised story:"""
        return prompt


class AgenticGenerator:
    """Generator for multi-turn agentic tasks.

    Uses the shared LLMClient for API calls, ensuring consistent
    infrastructure with standard generators.
    """

    def __init__(
        self,
        config: AgenticConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config or AgenticConfig()
        self.llm_client = llm_client or LLMClient()

    def _validate_output(self, turns: list[AgenticTurn], agentic_type: str) -> str:
        """
        Validate final output is non-empty.

        Args:
            turns: Conversation turns
            agentic_type: Type of agentic task

        Returns:
            Final output content

        Raises:
            RuntimeError: If final output is empty
        """
        import logging

        logger = logging.getLogger(__name__)

        # Find last assistant generation
        output = ""
        for turn in reversed(turns):
            if turn.role == "assistant" and turn.turn_type == "generation":
                output = turn.content
                break

        # FAIL if empty
        if not output.strip():
            logger.error(f"[{agentic_type}] Generation failed: final output is empty")
            raise RuntimeError("Empty generation - no valid output produced")

        return output.strip()

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[str, int, int, int, float]:
        """Make a single model call using shared LLMClient.

        Returns: (output, prompt_tokens, completion_tokens, reasoning_tokens, cost)
        """
        max_tokens = max_tokens or self.config.max_tokens_per_turn

        response = self.llm_client.call(
            model=model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            max_reasoning_tokens=self.config.max_reasoning_tokens,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay,
        )

        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")

        return (
            response.content,
            response.prompt_tokens,
            response.completion_tokens,
            response.reasoning_tokens,
            response.cost,
        )

    def run_constraint_discovery(
        self,
        task: dict[str, Any],
        model: str,
        sample_index: int,
        answer_oracle: Callable[[str], str],
    ) -> AgenticResult:
        """Run constraint discovery agentic task.

        Args:
            task: Task definition with hidden_constraints
            model: Model to use
            sample_index: Sample index
            answer_oracle: Function that takes a question and returns "YES" or "NO"
        """
        task_id = task["task_id"]
        task_type = task["task_type"]
        theory = task.get("theory", "Unknown")

        generation_id = generate_id()
        timestamp = get_timestamp()

        turns: list[AgenticTurn] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0

        discovered_constraints: list[str] = []
        questions_asked = 0
        max_questions = task.get("max_questions", self.config.max_questions)

        # Build conversation
        system_prompt = AgenticPromptBuilder.build_constraint_discovery_system(task)
        initial_prompt = AgenticPromptBuilder.build_constraint_discovery_initial(task)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]

        turns.append(
            AgenticTurn(
                role="system",
                content=system_prompt,
                turn_type="system",
            )
        )
        turns.append(
            AgenticTurn(
                role="user",
                content=initial_prompt,
                turn_type="context",
            )
        )

        try:
            # Discovery phase
            for q_num in range(max_questions):
                # Get model's question
                output, pt, ct, rt, cost = self._call_model(model, messages)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_reasoning_tokens += rt
                total_cost += cost

                question = output.strip()
                questions_asked += 1

                turns.append(
                    AgenticTurn(
                        role="assistant",
                        content=question,
                        turn_type="question",
                        metadata={"question_number": q_num + 1},
                    )
                )
                messages.append({"role": "assistant", "content": question})

                # Get answer from oracle
                answer = answer_oracle(question)

                # Track discovered constraint if YES
                if answer.upper() == "YES":
                    discovered_constraints.append(f"Q{q_num + 1}: {question} -> YES")

                turns.append(
                    AgenticTurn(
                        role="user",
                        content=answer,
                        turn_type="answer",
                        metadata={"answer": answer},
                    )
                )
                messages.append({"role": "user", "content": answer})

            # Generation phase
            gen_prompt = AgenticPromptBuilder.build_constraint_discovery_generate(
                task, discovered_constraints
            )
            messages.append({"role": "user", "content": gen_prompt})

            turns.append(
                AgenticTurn(
                    role="user",
                    content=gen_prompt,
                    turn_type="generation_prompt",
                )
            )

            output, pt, ct, rt, cost = self._call_model(
                model, messages, max_tokens=self.config.execution_max_tokens
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_reasoning_tokens += rt
            total_cost += cost

            turns.append(
                AgenticTurn(
                    role="assistant",
                    content=output,
                    turn_type="generation",
                )
            )

            # Validate final output is non-empty
            self._validate_output(turns, "constraint_discovery")

            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="constraint_discovery",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output=output,
                total_turns=len(turns),
                questions_asked=questions_asked,
                constraints_discovered=len(discovered_constraints),
                plan_quality=None,
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=True,
            )

        except Exception as e:
            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="constraint_discovery",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output="",
                total_turns=len(turns),
                questions_asked=questions_asked,
                constraints_discovered=len(discovered_constraints),
                plan_quality=None,
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=False,
                error=str(e),
            )

    def run_planning_execution(
        self,
        task: dict[str, Any],
        model: str,
        sample_index: int,
    ) -> AgenticResult:
        """Run planning-then-execution agentic task."""
        task_id = task["task_id"]
        task_type = task["task_type"]
        theory = task.get("theory", "Unknown")

        generation_id = generate_id()
        timestamp = get_timestamp()

        turns: list[AgenticTurn] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0

        system_prompt = AgenticPromptBuilder.SYSTEM_PROMPT_PLANNING_EXECUTION

        try:
            # Planning phase
            plan_prompt = AgenticPromptBuilder.build_planning_execution_plan(task)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": plan_prompt},
            ]

            turns.append(
                AgenticTurn(role="system", content=system_prompt, turn_type="system")
            )
            turns.append(
                AgenticTurn(role="user", content=plan_prompt, turn_type="plan_prompt")
            )

            plan_output, pt, ct, rt, cost = self._call_model(
                model, messages, max_tokens=self.config.plan_max_tokens
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_reasoning_tokens += rt
            total_cost += cost

            turns.append(
                AgenticTurn(
                    role="assistant",
                    content=plan_output,
                    turn_type="plan",
                )
            )
            messages.append({"role": "assistant", "content": plan_output})

            # Execution phase
            exec_prompt = AgenticPromptBuilder.build_planning_execution_execute(
                task, plan_output
            )
            messages.append({"role": "user", "content": exec_prompt})

            turns.append(
                AgenticTurn(
                    role="user", content=exec_prompt, turn_type="execution_prompt"
                )
            )

            output, pt, ct, rt, cost = self._call_model(
                model, messages, max_tokens=self.config.execution_max_tokens
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_reasoning_tokens += rt
            total_cost += cost

            turns.append(
                AgenticTurn(
                    role="assistant",
                    content=output,
                    turn_type="generation",
                )
            )

            # Validate final output is non-empty
            self._validate_output(turns, "planning_execution")

            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="planning_execution",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output=output,
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,  # Evaluated separately
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=True,
            )

        except Exception as e:
            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="planning_execution",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output="",
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=False,
                error=str(e),
            )

    def run_iterative_revision(
        self,
        task: dict[str, Any],
        model: str,
        sample_index: int,
        feedback_generator: Callable[[str, dict[str, Any]], list[str]],
    ) -> AgenticResult:
        """Run iterative revision agentic task.

        Args:
            task: Task definition
            model: Model to use
            sample_index: Sample index
            feedback_generator: Function that takes (output, task) and returns list of feedback items
        """
        task_id = task["task_id"]
        task_type = task["task_type"]
        theory = task.get("theory", "Unknown")

        generation_id = generate_id()
        timestamp = get_timestamp()

        turns: list[AgenticTurn] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0

        system_prompt = AgenticPromptBuilder.SYSTEM_PROMPT_ITERATIVE_REVISION

        try:
            # Initial generation
            initial_prompt = AgenticPromptBuilder.build_iterative_revision_initial(task)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt},
            ]

            turns.append(
                AgenticTurn(role="system", content=system_prompt, turn_type="system")
            )
            turns.append(
                AgenticTurn(
                    role="user", content=initial_prompt, turn_type="initial_prompt"
                )
            )

            current_output, pt, ct, rt, cost = self._call_model(model, messages)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_reasoning_tokens += rt
            total_cost += cost

            turns.append(
                AgenticTurn(
                    role="assistant",
                    content=current_output,
                    turn_type="generation",
                    metadata={"revision": 0},
                )
            )
            messages.append({"role": "assistant", "content": current_output})

            # Revision rounds
            revision_count = 0
            max_revisions = task.get("max_revisions", self.config.max_revisions)

            for rev_num in range(max_revisions):
                # Get feedback
                feedback = feedback_generator(current_output, task)

                if not feedback:
                    # No more feedback - we're done
                    break

                revision_count += 1

                # Build revision prompt
                rev_prompt = AgenticPromptBuilder.build_iterative_revision_feedback(
                    current_output, feedback
                )
                messages.append({"role": "user", "content": rev_prompt})

                turns.append(
                    AgenticTurn(
                        role="user",
                        content=rev_prompt,
                        turn_type="feedback",
                        metadata={"feedback": feedback, "revision": rev_num + 1},
                    )
                )

                # Get revision
                current_output, pt, ct, rt, cost = self._call_model(model, messages)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_reasoning_tokens += rt
                total_cost += cost

                turns.append(
                    AgenticTurn(
                        role="assistant",
                        content=current_output,
                        turn_type="generation",
                        metadata={"revision": rev_num + 1},
                    )
                )
                messages.append({"role": "assistant", "content": current_output})

            # Validate final output is non-empty
            self._validate_output(turns, "iterative_revision")

            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="iterative_revision",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output=current_output,
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,
                revision_count=revision_count,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=True,
            )

        except Exception as e:
            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="iterative_revision",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output="",
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=False,
                error=str(e),
            )

    def run_critique_improvement(
        self,
        task: dict[str, Any],
        model: str,
        sample_index: int,
        critic_model: str | None = None,
    ) -> AgenticResult:
        """Run critique improvement task with LLM critic.

        Args:
            task: Task definition with constraints and evaluation criteria
            model: Model to generate story
            sample_index: Sample index
            critic_model: Model to use as critic (defaults to task's critic_model or claude-haiku)
        """
        task_id = task["task_id"]
        task_type = task["task_type"]
        theory = task.get("theory", "Unknown")

        generation_id = generate_id()
        timestamp = get_timestamp()

        turns: list[AgenticTurn] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0

        # Use task-specified critic model or default
        critic = critic_model or task.get("critic_model", "anthropic/claude-haiku-4.5")

        system_prompt = AgenticPromptBuilder.SYSTEM_PROMPT_CRITIQUE_IMPROVEMENT

        try:
            # Initial generation
            initial_prompt = AgenticPromptBuilder.build_critique_improvement_initial(
                task
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt},
            ]

            turns.append(
                AgenticTurn(role="system", content=system_prompt, turn_type="system")
            )
            turns.append(
                AgenticTurn(
                    role="user", content=initial_prompt, turn_type="initial_prompt"
                )
            )

            current_output, pt, ct, rt, cost = self._call_model(model, messages)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_reasoning_tokens += rt
            total_cost += cost

            turns.append(
                AgenticTurn(
                    role="assistant",
                    content=current_output,
                    turn_type="generation",
                    metadata={"revision": 0, "version": "initial"},
                )
            )
            messages.append({"role": "assistant", "content": current_output})

            # Critique-revision rounds
            revision_count = 0
            critique_rounds = task.get("critique_rounds", 3)

            for round_num in range(critique_rounds):
                # Get critique from critic model
                critique_prompt = AgenticPromptBuilder.build_critique_prompt(
                    task, current_output
                )

                critic_messages = [
                    {
                        "role": "system",
                        "content": AgenticPromptBuilder.SYSTEM_PROMPT_CRITIC.format(
                            criteria="\n".join(
                                f"- {c['criterion']}: {c.get('description', '')}"
                                for c in task.get("evaluation_criteria", [])
                            )
                        ),
                    },
                    {"role": "user", "content": critique_prompt},
                ]

                critique, cpt, cct, crt, ccost = self._call_model(
                    critic, critic_messages
                )
                total_prompt_tokens += cpt
                total_completion_tokens += cct
                total_reasoning_tokens += crt
                total_cost += ccost

                turns.append(
                    AgenticTurn(
                        role="user",
                        content=critique,
                        turn_type="critique",
                        metadata={"round": round_num + 1, "critic_model": critic},
                    )
                )

                # Build revision prompt
                rev_prompt = AgenticPromptBuilder.build_critique_revision_prompt(
                    current_output, critique
                )
                messages.append({"role": "user", "content": rev_prompt})

                turns.append(
                    AgenticTurn(
                        role="user",
                        content=rev_prompt,
                        turn_type="revision_prompt",
                        metadata={"round": round_num + 1},
                    )
                )

                # Get revised version
                current_output, pt, ct, rt, cost = self._call_model(model, messages)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_reasoning_tokens += rt
                total_cost += cost

                revision_count += 1

                turns.append(
                    AgenticTurn(
                        role="assistant",
                        content=current_output,
                        turn_type="generation",
                        metadata={
                            "revision": round_num + 1,
                            "version": f"revision_{round_num + 1}",
                        },
                    )
                )
                messages.append({"role": "assistant", "content": current_output})

            # Validate final output is non-empty
            self._validate_output(turns, "critique_improvement")

            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="critique_improvement",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output=current_output,
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,
                revision_count=revision_count,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=True,
            )

        except Exception as e:
            return AgenticResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                agentic_type="critique_improvement",
                theory=theory,
                model=model,
                sample_index=sample_index,
                turns=turns,
                output="",
                total_turns=len(turns),
                questions_asked=0,
                constraints_discovered=0,
                plan_quality=None,
                revision_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_reasoning_tokens=total_reasoning_tokens,
                total_cost=total_cost,
                timestamp=timestamp,
                success=False,
                error=str(e),
            )


if __name__ == "__main__":
    print("Agentic generator module loaded successfully.")
    print("\nSupported agentic task types:")
    print(
        "  - constraint_discovery: Model asks questions to discover hidden constraints"
    )
    print("  - planning_execution: Model plans before executing")
    print("  - iterative_revision: Model revises based on feedback")
    print("  - critique_improvement: Model improves through LLM critique rounds")
