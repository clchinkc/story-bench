"""The episode shape is the thing under test: 1 initial + 3x(critique, revision) = 7 turns."""
import importlib.util
import pathlib
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "critique_episode", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "critique_episode.py")
ce = importlib.util.module_from_spec(_spec)
# A dataclass resolves its own module through sys.modules; a module loaded by path is not there yet.
sys.modules["critique_episode"] = ce
_spec.loader.exec_module(ce)


class Builder:
    """A stand-in for `AgenticPromptBuilder` that makes each prompt identifiable."""
    SYSTEM_PROMPT_CRITIQUE_IMPROVEMENT = "SUBJECT-SYSTEM"
    SYSTEM_PROMPT_CRITIC = "CRITIC-SYSTEM[{criteria}]"

    def build_critique_improvement_initial(self, task):
        return f"INITIAL for {task['task_id']}"

    def build_critique_prompt(self, task, story):
        return f"CRITIQUE THIS: {story}"

    def build_critique_revision_prompt(self, previous_output, critique):
        return f"REVISE: {previous_output} || BECAUSE: {critique}"


def task():
    return dict(task_id="critique_improvement_001",
                evaluation_criteria=[dict(criterion="constraint_satisfaction",
                                          description="Were all constraints satisfied?")])


def run_episode(rounds=3):
    """Drive the state machine to completion, returning the turns it produced."""
    turns, transcript = [], ()
    while True:
        turn = ce.next_turn(task(), transcript, rounds=rounds, builder=Builder())
        if turn is None:
            return turns
        turns.append(turn)
        transcript = transcript + ((turn.kind, f"{turn.kind}-{len(turns)}"),)


def test_an_episode_is_seven_turns_for_three_rounds():
    turns = run_episode(rounds=3)
    assert [t.kind for t in turns] == ["initial", "critique", "revision",
                                       "critique", "revision", "critique", "revision"]
    assert [t.role for t in turns] == [ce.SUBJECT, ce.CRITIC, ce.SUBJECT,
                                       ce.CRITIC, ce.SUBJECT, ce.CRITIC, ce.SUBJECT]
    assert [t.index for t in turns] == list(range(7))


def test_the_critic_is_never_the_subject():
    """Self-critique measures self-agreement, not responsiveness to criticism."""
    for turn in run_episode():
        if turn.kind == "critique":
            assert turn.role == ce.CRITIC
            assert turn.messages[0][1].startswith("CRITIC-SYSTEM[")
        else:
            assert turn.role == ce.SUBJECT


def test_zero_rounds_is_a_single_initial_call():
    turns = run_episode(rounds=0)
    assert [t.kind for t in turns] == ["initial"]
    assert turns[0].messages[1][1] == "INITIAL for critique_improvement_001"


def test_the_revision_carries_the_critique_verbatim():
    """A revision that cannot see the critique is not a revision."""
    transcript = (("initial", "DRAFT-1"), ("critique", "THE CRITICISM"))
    turn = ce.next_turn(task(), transcript, rounds=3, builder=Builder())
    assert turn.kind == "revision"
    assert turn.messages[-1] == ("user", "REVISE: DRAFT-1 || BECAUSE: THE CRITICISM")
    # The draft is replayed as the subject's own turn; the critique is not.
    assert ("assistant", "DRAFT-1") in turn.messages
    assert not any(text == "THE CRITICISM" for role, text in turn.messages if role == "assistant")


def test_the_second_round_critiques_the_revised_draft_not_the_first():
    transcript = (("initial", "DRAFT-1"), ("critique", "C1"), ("revision", "DRAFT-2"))
    turn = ce.next_turn(task(), transcript, rounds=3, builder=Builder())
    assert turn.kind == "critique"
    assert turn.messages[-1][1] == "CRITIQUE THIS: DRAFT-2"


def test_the_rubric_reaches_the_critic():
    turn = ce.next_turn(task(), (("initial", "DRAFT-1"),), rounds=3, builder=Builder())
    assert turn.kind == "critique"
    assert "constraint_satisfaction" in turn.messages[0][1]
    assert "Were all constraints satisfied?" in turn.messages[0][1]


def test_an_out_of_order_transcript_raises_rather_than_answering():
    with pytest.raises(ValueError, match="without a draft"):
        ce.next_turn(task(), (("critique", "C1"),), rounds=3, builder=Builder())
    with pytest.raises(TypeError):
        ce.next_turn(task(), [("initial", "D")], rounds=3, builder=Builder())
    with pytest.raises(ValueError):
        ce.next_turn(task(), (), rounds=-1, builder=Builder())


def test_run_episode_drives_all_seven_turns_without_a_network():
    calls = []

    def send(turn, model):
        calls.append((turn.kind, model))
        return f"{turn.kind}-text-{len(calls)}"

    transcript = ce.run_episode(task(), send, rounds=3, builder=Builder(),
                                subject_model="subject/model", critic_model="critic/model")
    assert len(transcript) == 7
    assert [k for k, _ in transcript] == ["initial", "critique", "revision",
                                          "critique", "revision", "critique", "revision"]
    # The critic is routed to the critic model and the subject to the subject model, every time.
    assert calls == [(k, "critic/model" if k == "critique" else "subject/model") for k, _ in transcript]


def test_run_episode_refuses_an_empty_turn_instead_of_recording_it():
    """An empty turn is a failed call; returning a 6-turn 'episode' would hide it."""
    state = {"n": 0}

    def send(turn, model):
        state["n"] += 1
        return "" if state["n"] == 4 else f"text-{state['n']}"

    with pytest.raises(ValueError, match="returned no text"):
        ce.run_episode(task(), send, rounds=3, builder=Builder(),
                       subject_model="s/m", critic_model="c/m")


def test_run_episode_propagates_a_failed_call():
    def send(turn, model):
        if turn.kind == "revision":
            raise RuntimeError("provider said no")
        return "ok"

    with pytest.raises(RuntimeError, match="provider said no"):
        ce.run_episode(task(), send, rounds=3, builder=Builder(),
                       subject_model="s/m", critic_model="c/m")


def test_run_episode_rejects_a_non_callable_sender():
    with pytest.raises(TypeError):
        ce.run_episode(task(), "send", rounds=3, builder=Builder(),
                       subject_model="s/m", critic_model="c/m")
