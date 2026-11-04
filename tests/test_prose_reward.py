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


def test_a_clean_continuation_is_paid_the_judge_score_scaled_by_length(monkeypatch):
    """Length is graded, not binary: a 54-character line earns most of the score, not all or none."""
    from story_rl.prose import length_factor

    monkeypatch.setattr("story_rl.prose.assess", lambda *a, **k: Assessment(score=80, band="合格", note="ok"))
    text = "她把傘收起來，水滴沿著手背往下滑，沒有回頭，也沒有等他把話說完就往前走，傘尖在石階上拖出一道很長的水痕。"
    scored = score_prose(text, a_task())
    assert scored.lint_veto is False
    assert scored.strict == pytest.approx(0.8 * length_factor(text))


def test_an_error_severity_finding_vetoes_whatever_the_judge_says(monkeypatch):
    """Lint is a veto on breakage, not a discount: a high score cannot buy past it."""
    monkeypatch.setattr("story_rl.prose.assess", lambda *a, **k: Assessment(score=99, band="出版級", note="ok"))
    monkeypatch.setattr("story_rl.prose.lint", lambda *a, **k: (40, [("error", "broken-sentence")]))
    scored = score_prose("他把手插在口袋裡，指節微微發白，腳步聲在空蕩蕩的走廊裡拖出兩段很長的回響。", a_task())
    assert scored.lint_veto is True
    assert scored.strict == 0.0


def test_a_single_warning_does_not_zero_a_good_rollout(monkeypatch):
    """The starvation bug: requiring a pristine 100 vetoed ~45 % of every batch and halved the signal."""
    monkeypatch.setattr("story_rl.prose.assess", lambda *a, **k: Assessment(score=80, band="合格", note="ok"))
    monkeypatch.setattr("story_rl.prose.lint", lambda *a, **k: (95, [("warning", "sentence-monotone")]))
    text = "他把手插在口袋裡，指節微微發白，腳步聲在空蕩蕩的走廊裡拖出兩段很長的回響，久久不散，也沒有人應他。"
    scored = score_prose(text, a_task())
    assert scored.lint_veto is False
    assert scored.strict > 0


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


# ── the judge route: DeepSeek is always api.deepseek.com ──────────────────────────────────────────────

def test_deepseek_model_id_cannot_be_sent_through_openrouter(monkeypatch):
    """Owner ruling 2026-11-04. The same weights scored 0.596 direct and 0.697 through OpenRouter, so the
    route is part of the measurement. An OpenRouter id for DeepSeek must raise, not silently route."""
    from story_rl import assessor

    monkeypatch.setattr(assessor, "OPENROUTER_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setattr(assessor, "JUDGE_PROVIDER", "")
    try:
        assessor.check_route()
    except assessor.JudgeError as exc:
        assert "api.deepseek.com" in str(exc)
    else:
        raise AssertionError("an OpenRouter DeepSeek id must not pass the route check")

    monkeypatch.setattr(assessor, "JUDGE_PROVIDER", "deepseek")
    assessor.check_route()  # the direct provider is the sanctioned route


def test_naming_a_provider_is_enough_to_reach_the_remote_judge(monkeypatch):
    """`PROSE_JUDGE_PROVIDER=deepseek` with no backend set must not fall through to the local server."""
    from story_rl import assessor

    monkeypatch.setattr(assessor, "BACKEND", "local")
    monkeypatch.setattr(assessor, "JUDGE_PROVIDER", "deepseek")
    assert assessor.remote() is True

    monkeypatch.setattr(assessor, "JUDGE_PROVIDER", "")
    assert assessor.remote() is False

    monkeypatch.setattr(assessor, "BACKEND", "openrouter")
    assert assessor.remote() is True


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


def test_a_previous_draft_block_is_replaced_not_stacked():
    """Run the command twice and the note must not fill with suggestions; the author's prose survives."""
    from scripts.write_continue_note import render, strip_previous_drafts

    prose = "他推開門，雨聲忽然變大。\n\n走廊盡頭的燈一閃一閃，像有人在數著什麼。\n"
    first = prose + "\n" + render([(0, "第一次")], model="m", prompt_hash="h", url="u")
    second = prose + "\n" + render([(0, "第二次")], model="m", prompt_hash="h", url="u")

    cleaned = strip_previous_drafts(first)
    assert "第一次" not in cleaned
    assert "走廊盡頭的燈一閃一閃" in cleaned
    assert cleaned.count("[!draft]") == 0
    assert "第二次" in second and "第一次" not in second


# ── the judge gate: a concurrency ceiling, measured rather than asserted ───────────────────────────────

def test_the_judge_gate_caps_requests_in_flight(monkeypatch):
    """A step fires 128 judge calls at once and the provider answers 429. The gate is the fix, so this
    measures the peak concurrency while 24 calls are in flight instead of trusting the semaphore exists."""
    import threading
    import time

    from story_rl import assessor

    live = 0
    peak = 0
    lock = threading.Lock()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"choices": []}'

    def slow_urlopen(request, timeout=None):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return FakeResponse()

    monkeypatch.setattr(assessor.urllib.request, "urlopen", slow_urlopen)

    threads = [threading.Thread(target=assessor._call_once, args=(None,)) for _ in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak <= assessor.JUDGE_CONCURRENCY, f"{peak} concurrent calls exceeded the gate"
    assert peak > 1, "the gate serialised every call, which would make a 68-step run untenable"


def test_a_rate_limited_reply_is_retried_and_a_refused_one_is_not(monkeypatch):
    """429 is a burst; 402 is a bill. The first gets another attempt, the second must stop the run."""
    import urllib.error

    from story_rl import assessor

    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "0"}, None)
        return {"ok": True}

    monkeypatch.setattr(assessor, "_call_once", flaky)
    assert assessor._with_retries(None) == {"ok": True}
    assert calls["n"] == 2

    def refused(request):
        raise urllib.error.HTTPError("u", 402, "no credit", {}, None)

    monkeypatch.setattr(assessor, "_call_once", refused)
    try:
        assessor._with_retries(None)
    except assessor.JudgeError as exc:
        assert "402" in str(exc)
    else:
        raise AssertionError("a 402 must not be retried")


def test_a_draft_that_quotes_the_note_back_loses_the_quote():
    """The first smoke test after the v9 swap came back with the note's last sentence repeated verbatim as
    the draft's first sentence. In the callout that reads as a duplicate line, so the echo is stripped."""
    from scripts.write_continue_note import strip_echo

    context = "他把手插進口袋，指節微微發白，走廊盡頭的燈一閃一閃。"
    echoed = context + "風在門外低語，像誰在遠處輕聲低唸。他的呼吸在空氣中凝結成霧。"

    cleaned, stripped = strip_echo(echoed, context)
    assert stripped is True
    assert cleaned.startswith("風在門外低語")
    assert "指節微微發白" not in cleaned


def test_a_short_overlap_is_not_treated_as_an_echo():
    """Two or three shared characters are a coincidence; stripping them would eat the draft's opening."""
    from scripts.write_continue_note import strip_echo

    cleaned, stripped = strip_echo("一閃，像有人在數著什麼。", "走廊盡頭的燈一閃一閃。")
    assert stripped is False
    assert cleaned == "一閃，像有人在數著什麼。"
