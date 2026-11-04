"""Prose: the writer task, its reward, and why the reward is shaped the way it is.

The task is a scene opening from a published novel; the model writes the next two sentences. There is no
reference answer to copy — the only thing that can say whether the continuation is worth keeping is the
frozen assessor, which is why this shape exists at all.

The reward is deliberately **not** the assessor's number alone:

    veto     the author's deterministic lint layer rejects the rollout outright if it carries an
             error-severity finding or falls below the pristine ceiling. It pays nothing — D001 measured that
             a hollow continuation loses one warning (5 points), so partial credit from lint would be noise.
    signal   the assessor's 0-100 craft score, normalised. This is the only term with range.

A rollout that lints badly scores 0 no matter how the judge liked it, so "beautiful but broken" cannot win;
a rollout that lints cleanly but reads hollow scores whatever the judge says, which is the discrimination the
whole capsule rests on.

The human continuation is carried per task and never shown to the policy: it is the ceiling the judge is
calibrated against in measurement ("how far is the model from publishable prose"), not a training target.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import chz
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from holdout_guard import HoldoutSet, assert_disjoint, prompt_digest
from story_rl import traditional
from story_rl.assessor import Assessment, assess
from story_rl.env import DEFAULT_MODEL, DEFAULT_POLICY, DEFAULT_RENDERER, TaskDataset, permuted

DATA = Path(__file__).resolve().parent.parent.parent / "dataset" / "story_rl"

# The corpus this shape trains on, pinned here the way `shape_a` and `corpus` pin theirs. It used to be two
# environment variables so the same builder could reach a second corpus; with one corpus in use that was a
# switch nobody needed, and twice a run was launched without it and silently trained on the wrong text.
TASKS = DATA / "voice.json"
SPLIT = DATA / "voice_split.json"

NARRATIVE_CRAFT_SRC = Path("/Users/clchinkc/Documents/GitHub/narrative-craft/src")

# The lint veto threshold: the pristine ceiling. The engine is quantised in 15/5/1 steps, so anything below
# 100 has already lost a finding; `error` severity is fatal regardless of the total.
LINT_CEILING = 100
FATAL_SEVERITY = "error"

# Training collapsed continuations to ~33 characters against the base model's ~70, and the judge paid the
# shorter ones less — not because brevity is wrong, but because a 30-character line has no room to show
# craft. A hard veto at 45 rejected 45 % of rollouts and halved the learning, so length is now graded: a
# floor below which nothing is payable, a ramp, and a ceiling.
MIN_CONTINUATION_CHARS = 20
LENGTH_TARGET_CHARS = 60
MAX_CONTINUATION_CHARS = 400

# Both arms leaked foreign tokens at 3.3 % (base included), so this is not a training defect — but a
# Chinese draft with `wet` or a Greek word in it is unusable, and the check costs nothing.
FOREIGN_SCRIPT = re.compile(r"[A-Za-z\u0370-\u03ff\u0400-\u04ff]")


class ProseTaskError(ValueError):
    """A prose row is malformed, or its halves do not reassemble the source passage."""


@dataclass(frozen=True)
class ProseTask:
    task_id: str
    book: str
    opening: str
    human_continuation: str
    prompt_text: str
    words: int
    lang: str = "en"

    def __post_init__(self) -> None:
        for name in ("task_id", "book", "opening", "human_continuation", "prompt_text"):
            raw = getattr(self, name)
            if not isinstance(raw, str) or not raw.strip():
                raise ProseTaskError(f"ProseTask.{name} must be a non-blank str, got {raw!r}")
        if self.opening not in self.prompt_text:
            raise ProseTaskError(f"{self.task_id}: the prompt does not carry the opening verbatim")
        if self.human_continuation.strip() in self.prompt_text:
            raise ProseTaskError(
                f"{self.task_id}: the human continuation is printed in the prompt; the ceiling would be a "
                "copy target rather than a measurement"
            )

    def prompt(self) -> str:
        """The exact text the model sees."""
        return self.prompt_text

    def content_digest(self) -> str:
        return prompt_digest(self.opening)

    @property
    def true_value(self) -> str:
        """There is no reference answer. Kept for the env's logging shape, never shown to the policy."""
        return ""


@dataclass(frozen=True)
class ProseScore:
    """The reward: a lint veto and a judged craft term. Shape mirrors `RolloutScore` for `TaskEnv`."""

    task_id: str
    parsed: bool
    how: str
    strict: float
    correction: float
    lint_score: int
    lint_veto: bool
    craft: int
    band: str
    note: str
    text: str = ""


def lint(text: str, lang: str = "en") -> tuple[int, list[tuple[str, str]]]:
    """The author's deterministic engine. Imported from the package, never reimplemented."""
    if str(NARRATIVE_CRAFT_SRC) not in sys.path:
        sys.path.insert(0, str(NARRATIVE_CRAFT_SRC))
    from narrative_craft.lint.core import lint_text

    result = lint_text(text, lang=lang)
    return result.score, [(f.severity, f.pattern_id) for f in result.findings]


def length_factor(text: str) -> float:
    """0 below the floor, ramping to 1 at the target. Graded because a binary floor starved the run."""
    span = max(1, LENGTH_TARGET_CHARS - MIN_CONTINUATION_CHARS)
    return max(0.0, min(1.0, (len(text) - MIN_CONTINUATION_CHARS) / span))


def guard(text: str, lang: str = "en") -> list[str]:
    """Deterministic rejections the judge should never have to see. Empty means admissible."""
    reasons = []
    if len(text) < MIN_CONTINUATION_CHARS:
        reasons.append(f"too-short:{len(text)}")
    if len(text) > MAX_CONTINUATION_CHARS:
        reasons.append(f"too-long:{len(text)}")
    if lang == "zh" and FOREIGN_SCRIPT.search(text):
        reasons.append("foreign-script")
    if lang == "zh":
        # The author writes Traditional; a simplified character is not a style choice, it is a character she
        # does not use. Measured before this check existed: 54 % of the shipped model's generations carried
        # one, and the reward paid them exactly as much as correct text. Only unambiguous findings veto —
        # `traditional.audit()` only returns characters whose correct form is certain.
        found = traditional.audit(text)
        if found:
            reasons.append("simplified:" + "".join(f.matched for f in found[:4]))
    return reasons


def score_prose(raw: str, task: ProseTask, policy: str = DEFAULT_POLICY) -> ProseScore:
    """One rollout: parse, apply the lint veto, then ask the frozen judge. Never invents a score."""
    from continuation_parser import extract

    extraction = extract(raw, policy)
    text = extraction.text or ""
    base = ProseScore(
        task_id=task.task_id, parsed=not extraction.unparsed, how=extraction.how,
        strict=0.0, correction=0.0, lint_score=0, lint_veto=True, craft=0, band="", note="", text=text,
    )
    if extraction.unparsed or not text.strip():
        return base

    lint_score, findings = lint(text, task.lang)
    reasons = guard(text, task.lang)
    # Only an error-severity finding vetoes. Requiring a pristine 100 meant a single warning — a repeated
    # sentence opening, an adverb — zeroed the rollout, and that vetoed ~45 % of every batch, halving the
    # signal the run could learn from.
    # Only an error-severity finding vetoes. Requiring a pristine score meant a single warning — a repeated
    # sentence opening, an adverb — zeroed the rollout, and that rejected ~45 % of every batch. Lint vetoes
    # breakage; pricing hollowness is the judge's job, which is what D001 already said it could not do.
    vetoed = any(severity == FATAL_SEVERITY for severity, _ in findings) or bool(reasons)

    # A judge that cannot answer is an instrument failure, and it must not look like bad prose. Returning 0
    # here once let a 402 (out of credit) score every rollout zero for nine steps while the run kept billing;
    # the caller must hear about it, not average it away.
    judged: Assessment = assess(task.opening, text, task.lang)

    reward = 0.0 if vetoed else judged.normalised() * length_factor(text)
    return ProseScore(
        task_id=task.task_id, parsed=True, how=extraction.how, strict=reward, correction=judged.normalised(),
        lint_score=lint_score, lint_veto=vetoed, craft=judged.score, band=judged.band,
        note=judged.note, text=text,
    )


def from_row(row: dict) -> ProseTask:
    return ProseTask(
        task_id=row["task_id"], book=row.get("book", row["task_id"].split("-")[0]),
        opening=row["opening"],
        human_continuation=row["human_continuation"], prompt_text=row["prompt"],
        words=int(row.get("words", len(row["opening"].split()) + len(row["human_continuation"].split()))),
        lang=row.get("lang", "en"),
    )


def load_tasks(path: Path = TASKS) -> tuple[ProseTask, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    tasks = tuple(from_row(row) for row in rows)
    ids = [t.task_id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ProseTaskError(f"duplicate task ids: {duplicates}")
    return tasks


def rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def split_manifest(tasks: tuple[ProseTask, ...], salt: str, holdout: int) -> dict:
    """Per-book proportional hold-out, largest remainder, exact count. A pure function of the ids."""
    books = sorted({t.book for t in tasks})
    counts = {book: sum(1 for t in tasks if t.book == book) for book in books}
    quotas = {book: counts[book] * holdout / len(tasks) for book in books}
    take = {book: int(q) for book, q in quotas.items()}
    for book in sorted(quotas, key=lambda b: (-(quotas[b] - take[b]), b))[: holdout - sum(take.values())]:
        take[book] += 1
    held = [
        t.task_id
        for book in books
        for t in sorted((x for x in tasks if x.book == book), key=lambda x: rank(x.task_id, salt))[: take[book]]
    ]
    chosen = set(held)
    return {"salt": salt, "holdout_count": holdout,
            "train": [t.task_id for t in tasks if t.task_id not in chosen], "held_out": sorted(held)}


@dataclass(frozen=True)
class ProseTaskSet:
    train: tuple[ProseTask, ...]
    held_out: tuple[ProseTask, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.held_out:
            raise ProseTaskError("both halves must be non-empty")
        shared = {t.task_id for t in self.train} & {t.task_id for t in self.held_out}
        if shared:
            raise ProseTaskError(f"task ids in both halves: {sorted(shared)}")

    def by_id(self) -> dict[str, ProseTask]:
        return {t.task_id: t for t in self.train + self.held_out}


def load_split(tasks: tuple[ProseTask, ...] | None = None, path: Path = SPLIT) -> ProseTaskSet:
    corpus = tasks if tasks is not None else load_tasks()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    known = {t.task_id for t in corpus}
    missing = [i for i in manifest["train"] + manifest["held_out"] if i not in known]
    if missing:
        raise ProseTaskError(f"{path.name} names task ids the corpus does not carry: {missing}")
    train_ids, held_ids = set(manifest["train"]), set(manifest["held_out"])
    train = tuple(t for t in corpus if t.task_id in train_ids)
    held = tuple(t for t in corpus if t.task_id in held_ids)
    assert_disjoint(
        [t.task_id for t in train], [t.opening for t in train],
        HoldoutSet(task_ids=frozenset(t.task_id for t in held),
                   digests=frozenset(t.content_digest() for t in held)),
    )
    return ProseTaskSet(train=train, held_out=held)


class ProseEnv(ProblemEnv):
    """One prose task, with the judge call off the event loop.

    `ProblemEnv.step` calls `check_answer` synchronously, and the judge is network I/O: measured at 2.8 s per
    call, which serialised 128 rollouts into a six-minute step. Running the judgement in a worker thread lets
    the group's rollouts overlap, which is the difference between a three-hour run and a fifteen-minute one.
    The reward itself is unchanged — it is still `score_prose`, lint veto included.
    """

    def __init__(self, task: ProseTask, renderer: renderers.Renderer, policy: str = DEFAULT_POLICY):
        super().__init__(renderer, convo_prefix=None, format_coef=0.0,
                         require_stop_sequence_for_format=False)
        self.task = task
        self.policy = policy
        self._scored: tuple[str, ProseScore] | None = None

    def get_question(self) -> str:
        return self.task.prompt()

    def score(self, sample_str: str) -> ProseScore:
        if self._scored is not None and self._scored[0] == sample_str:
            return self._scored[1]
        result = score_prose(sample_str, self.task, self.policy)
        self._scored = (sample_str, result)
        return result

    def check_answer(self, sample_str: str) -> bool:
        return bool(self.score(sample_str).strict)

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return ""

    async def step(self, action, *, extra=None):  # noqa: ANN001, ANN201 - matches the cookbook signature
        message, _termination = self.renderer.parse_response(action)
        text = renderers.get_text_content(message)
        scored: ProseScore = await asyncio.to_thread(score_prose, text, self.task, self.policy)
        return replace_step(reward=float(scored.strict), scored=scored)


def replace_step(*, reward: float, scored: ProseScore):
    """Build the StepResult without importing the cookbook's dataclass fields at module scope."""
    from tinker_cookbook.rl.types import StepResult

    return StepResult(
        reward=reward,
        episode_done=True,
        next_observation=tinker.ModelInput.empty(),
        next_stop_condition=None,
        metrics={
            "strict": float(scored.strict),
            "craft": float(scored.craft),
            "lint": float(scored.lint_score),
            "veto": float(scored.lint_veto),
            "parsed": float(scored.parsed),
        },
    )


class ProseDataset(TaskDataset):
    """Same batching as any shape; only the env class differs, so the judge call can be threaded."""

    def get_batch(self, index: int):
        from functools import partial

        from tinker_cookbook.rl.problem_env import ProblemGroupBuilder

        epoch, offset = divmod(index * self.batch_size, len(self.tasks))
        order = permuted(self.tasks, epoch, self.salt)
        chosen = [order[(offset + step) % len(order)] for step in range(self.batch_size)]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(ProseEnv, task=task, renderer=self.renderer, policy=self.policy),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for task in chosen
        ]


@chz.chz
class ProseDatasetBuilder(RLDatasetBuilder):
    """What `tinker_cookbook.rl.train.Config` takes for prose. The training half only."""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    n_batches: int = 200
    policy: str = DEFAULT_POLICY
    dataset_name: str = "prose"

    async def __call__(self) -> tuple[TaskDataset, None]:
        corpus = load_split()
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer: renderers.Renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            ProseDataset(
                tasks=corpus.train, renderer=renderer, batch_size=self.batch_size,
                group_size=self.group_size, n_batches=self.n_batches, policy=self.policy,
                dataset_name=self.dataset_name,
            ),
            None,
        )
