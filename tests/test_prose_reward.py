"""The prose reward, the judge, the stop rule and the drafting prompt — the parts that had no tests.

These exist because the reward path failed *open* once: an out-of-credit judge returned an error, `score_prose`
turned it into a reward of zero, and a training run spent nine steps averaging zeros before anyone noticed. A
test that asserts the failure is loud would have caught it in a second.

No network. The judge is stubbed; the lint engine is the real one, because that is the part whose behaviour is
being asserted.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from story_rl.assessor import Assessment, JudgeError, band_of, build_prompt, parse  # noqa: E402
from story_rl.prose import ProseTask, score_prose  # noqa: E402
from scripts.write_scene import prompt_for  # noqa: E402


def a_task(lang: str = "zh") -> ProseTask:
    return ProseTask(
        task_id="t-001", book="t", opening="他推開門，雨聲忽然變大。",
        human_continuation="她沒有回頭，只是把傘收了起來。",
        prompt_text="續寫這個場景。\n\n他推開門，雨聲忽然變大。", words=10, lang=lang,
    )


# ── the judge's own contract ────────────────────────────────────────────────────────────────────────────

def test_bands_are_exhaustive_and_language_specific():
    assert band_of(95) == "publishable"
    assert band_of(95, "zh") == "出版級"
    assert band_of(0, "zh") == "崩壞"
    for score in range(101):
        assert band_of(score)
        assert band_of(score, "zh")


def test_the_chinese_prompt_is_chinese_and_cites_the_opening():
    prompt = build_prompt("他推開門。", "雨聲變大。", "zh")
    assert "續寫" in prompt and "他推開門。" in prompt
    assert "Continue this scene" not in prompt


def test_parse_refuses_a_missing_or_out_of_range_score():
    with pytest.raises(JudgeError, match="no JSON"):
        parse("I think it is quite good.")
    with pytest.raises(JudgeError, match="no content"):
        parse(None)
    with pytest.raises(JudgeError, match="not an int"):
        parse('{"score": "eighty"}')


def test_parse_refuses_a_band_that_contradicts_its_own_score():
    with pytest.raises(JudgeError, match="disagrees"):
        parse('{"score": 95, "band": "broken"}')


def test_parse_accepts_a_matching_band_in_either_language():
    assert parse('{"score": 95, "band": "publishable"}').band == "publishable"
    assert parse('{"score": 95, "band": "出版級"}', "zh").band == "出版級"


# ── the reward: veto, and a loud failure ────────────────────────────────────────────────────────────────

def test_a_missing_judge_raises_instead_of_scoring_zero(monkeypatch):
    """The defect that cost a run: an unreachable judge must not look like bad prose."""

    def dead(*_args, **_kwargs):
        raise JudgeError("HTTP Error 402: Payment Required")

    monkeypatch.setattr("story_rl.prose.assess", dead)
    with pytest.raises(JudgeError, match="402"):
        score_prose("她把傘收起來，水滴沿著手背往下滑。", a_task())


def test_a_clean_continuation_is_paid_the_judge_score(monkeypatch):
    monkeypatch.setattr("story_rl.prose.assess", lambda *a, **k: Assessment(score=80, band="合格", note="ok"))
    scored = score_prose("她把傘收起來，水滴沿著手背往下滑，沒有回頭，也沒有等他把話說完就往前走，傘尖在石階上拖出一道很長的水痕。", a_task())
    assert scored.strict == pytest.approx(0.8)
    assert scored.lint_veto is False


def test_a_hollow_continuation_is_vetoed_whatever_the_judge_says(monkeypatch):
    """Lint is a veto, not a discount: a high score cannot buy past a broken sentence."""
    monkeypatch.setattr("story_rl.prose.assess", lambda *a, **k: Assessment(score=99, band="出版級", note="ok"))
    hollow = "她走了。他走了。天黑了。燈亮了。" * 3
    scored = score_prose(hollow, a_task())
    assert scored.lint_veto is True
    assert scored.strict == 0.0


# ── the stop rule, as arithmetic ────────────────────────────────────────────────────────────────────────

def test_the_stop_rule_counts_consecutive_skips_and_resets():
    mod = importlib.import_module("scripts.stop_on_skip")
    assert mod.skip_of({"env/all/total_episodes": 128}, 8, 16) == pytest.approx(0.0)
    assert mod.skip_of({"env/all/total_episodes": 16}, 8, 16) == pytest.approx(0.875)
    streak = 0
    for _ in range(3):
        streak = mod.advance(0.7, streak, 0.6)
    assert streak == 3
    assert mod.advance(0.1, streak, 0.6) == 0


def test_the_stop_rule_reads_a_run_directory(tmp_path):
    mod = importlib.import_module("scripts.stop_on_skip")
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        '{"step": 0, "env/all/total_episodes": 128}\n{"step": 1, "env/all/total_episodes": 16}\n', encoding="utf-8")
    rows = mod.steps_seen(run / "metrics.jsonl")
    assert [r["step"] for r in rows] == [0, 1]
    assert mod.skip_of(rows[-1], 8, 16) == pytest.approx(0.875)


def test_a_half_written_metrics_line_is_ignored(tmp_path):
    mod = importlib.import_module("scripts.stop_on_skip")
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0}\n{"step": 1, "env/all/tot', encoding="utf-8")
    assert len(mod.steps_seen(path)) == 1


# ── the drafting prompt ────────────────────────────────────────────────────────────────────────────────

def test_the_drafting_prompt_matches_the_corpus_language():
    assert "續寫" in prompt_for("zh") and "Continue this scene" not in prompt_for("zh")
    assert "Continue this scene" in prompt_for("en") and "續寫" not in prompt_for("en")


# ── transient vs fatal judge failures ───────────────────────────────────────────────────────────────────

def test_a_transient_status_is_worth_retrying_and_a_fatal_one_is_not():
    from story_rl.assessor import retryable

    for status in (429, 500, 502, 503, 504):
        assert retryable(status) is True
    for status in (400, 401, 402, 403, 404):
        assert retryable(status) is False


# ── deterministic guards (length, language) ────────────────────────────────────────────────────────────

def test_a_too_short_continuation_is_rejected_before_the_judge_sees_it():
    from story_rl.prose import guard

    assert guard("他走了。", "zh")[0].startswith("too-short")


def test_a_foreign_script_in_chinese_prose_is_rejected(monkeypatch):
    from story_rl.prose import guard

    leaked = "他把手插進口袋，腳步聲在走廊裡拖出很長的回響，雨聲卻忽然變成wet的雜訊，整條街都濕了。"
    assert "foreign-script" in guard(leaked, "zh")
    assert guard(leaked, "en") == [] or "foreign-script" not in guard(leaked, "en")


def test_a_clean_chinese_continuation_passes_the_guards():
    from story_rl.prose import guard

    clean = "他把手插在口袋裡，指節微微發白，腳步聲在空蕩蕩的走廊裡拖出兩段很長的回響，久久不散，也沒有人應他。"
    assert guard(clean, "zh") == []
