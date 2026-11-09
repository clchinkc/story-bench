"""The critique/revision episode: which request comes next, given what has been said so far.

WHY THIS IS A SEPARATE, PURE MODULE. `critique_improvement` is not one call. The harness's own
`AgenticGenerator.run_critique_improvement` makes SEVEN: an initial draft, then three rounds of
"critic writes a critique, subject rewrites". The pilot's single-call path cannot express it, so
those 72 rows were deferred rather than quietly measured as if a one-shot answer were an episode.

`next_turn` is deliberately pure: it reads a transcript of completed turns and returns the next
request, or None when the episode is complete. Nothing here spends, holds a ledger slot, or touches
the filesystem. The sending loop is therefore mechanical, and the part with the actual risk -- the
prompt that a rewrite is asked to satisfy -- is testable without money.

THE CRITIC IS A SECOND MODEL, NEVER THE SUBJECT. Asking a model to critique its own draft measures
self-agreement, not responsiveness to criticism. The critic is named by the caller (the study
declares it) and must itself be a qualified route, because the ledger will only admit a qualified
one.
"""
from __future__ import annotations

from dataclasses import dataclass

SUBJECT = "subject"
CRITIC = "critic"


@dataclass(frozen=True)
class Turn:
    """One request in the episode.

    `kind` is `initial`, `critique` or `revision`; `role` is which model makes it. The attempt id is
    built by the caller from `(unit, round, kind)` so every turn gets its own immutable ledger slot
    and its own evidence directory.
    """
    index: int
    kind: str
    role: str
    messages: tuple[tuple[str, str], ...]


def criteria_text(task: dict) -> str:
    """The rubric the critic is given, in the form the harness's own critic prompt expects."""
    items = task.get("evaluation_criteria") or []
    return "\n".join(f"- {c['criterion']}: {c.get('description', '')}" for c in items)


def _initial_messages(task: dict, builder) -> tuple[tuple[str, str], ...]:
    return (("system", builder.SYSTEM_PROMPT_CRITIQUE_IMPROVEMENT),
            ("user", builder.build_critique_improvement_initial(task)))


def completed_rounds(transcript) -> int:
    """How many critique/revision rounds are finished, counted from the revisions."""
    return sum(1 for turn in transcript if turn[0] == "revision")


def next_turn(task: dict, transcript, *, rounds: int, builder):
    """The next request, or None when every round is done.

    `transcript` is a tuple of `(kind, text)` for completed turns, in order. A critique is not
    produced without a draft to critique, and a revision does not happen without a critique -- so an
    out-of-order transcript is a programming error, not an empty answer, and it raises.
    """
    if not isinstance(transcript, tuple):
        raise TypeError("transcript must be a tuple of (kind, text)")
    if not isinstance(rounds, int) or rounds < 0:
        raise ValueError("rounds must be a non-negative integer")

    draft = _latest(transcript, "initial", "revision")
    index = len(transcript)

    if draft is None:
        if transcript:
            raise ValueError("a critique or revision exists without a draft to act on")
        return Turn(index, "initial", SUBJECT, _initial_messages(task, builder))

    pending = _latest(transcript, "critique")
    revisions = completed_rounds(transcript)
    if pending is not None and transcript[-1][0] == "critique":
        # A critique has been written and not yet answered: the subject revises.
        return Turn(index, "revision", SUBJECT,
                    _history(task, transcript, builder) + (("user", builder.build_critique_revision_prompt(
                        draft, pending)),))
    if revisions >= rounds:
        return None
    # A fresh round: the critic reads the current draft.
    return Turn(index, "critique", CRITIC,
                (("system", builder.SYSTEM_PROMPT_CRITIC.format(criteria=criteria_text(task))),
                 ("user", builder.build_critique_prompt(task, draft))))


def _latest(transcript, *kinds):
    for kind, text in reversed(transcript):
        if kind in kinds:
            return text
    return None


def _history(task: dict, transcript, builder) -> tuple[tuple[str, str], ...]:
    """The conversation the subject has seen: its own draft turns, and the revision prompts.

    The critic's text is NOT replayed as an assistant turn -- it reaches the subject inside the
    revision prompt, exactly as `run_critique_improvement` does. Replaying it as assistant output
    would claim the subject wrote its critic's words.
    """
    messages = list(_initial_messages(task, builder))
    for kind, text in transcript:
        if kind == "initial":
            messages.append(("assistant", text))
        elif kind == "revision":
            messages.append(("assistant", text))
    return tuple(messages)


def run_episode(task, send, *, rounds, builder, subject_model, critic_model):
    """Drive the episode to completion, returning the transcript.

    `send(turn, model) -> text` is INJECTED rather than imported, and that is the whole point: the
    orchestration can be exercised to the last turn without a network, a ledger slot or a dollar.
    The real sender is the pilot runner's attempt path; a test's sender is a lambda.

    A failed turn raises out of `send` and aborts the episode. There is no partial-episode result:
    a transcript that stops mid-round is not a shorter episode, it is a different measurement, and
    silently returning it would let a 3-call accident be filed as a 7-call episode.
    """
    if not callable(send):
        raise TypeError("send must be callable")
    transcript = ()
    while True:
        turn = next_turn(task, transcript, rounds=rounds, builder=builder)
        if turn is None:
            return transcript
        model = critic_model if turn.role == CRITIC else subject_model
        text = send(turn, model)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"turn {turn.index} ({turn.kind}) returned no text; an empty turn is a "
                             f"failed call, not an episode")
        transcript = transcript + ((turn.kind, text),)
