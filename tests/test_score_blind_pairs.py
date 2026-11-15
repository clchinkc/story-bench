"""The paired blind read: it pairs, it stays blind, and its scorer never guesses.

The 2026-11-04 flat sheet had ten openings continued twice and asked for twenty independent marks. This
pair of scripts halves the work and strengthens the test, and these tests pin the three things that would
quietly destroy either property:

1. **The pairing.** An opening with three continuations, or a pair whose two texts came from one model,
   must be refused rather than paired arbitrarily — a mis-paired sheet still produces a number.
2. **The blind.** The sheet must not contain the words that name either model. A leak does not look like
   an error; it looks like a slightly better result.
3. **The score.** A blank is unrated, a tie is not a vote, and an unrecognised word is reported rather
   than read as a preference. Guessing here decides the answer.

The sign test is checked against hand-computed values, including the two edges: 5/5 gives 1.0, and
10/0 gives 2/1024.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Loaded by path, NOT by putting scripts/ on sys.path: a module named like the package would shadow
# `src/story_rl/` for every test collected afterwards.
def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_pairs = _load("build_blind_pairs")
score_pairs = _load("score_blind_pairs")

SHEET = """## #1

*opening:* 開頭甲

續甲一

verdict:

## #2

*opening:* 開頭甲

續甲二

verdict:

## #3

*opening:* 開頭乙

續乙一

verdict:

## #4

*opening:* 開頭乙

續乙二

verdict:
"""

KEY = {
    "#1": {"arm": "base"},
    "#2": {"arm": "trained"},
    "#3": {"arm": "base"},
    "#4": {"arm": "trained"},
}


def items():
    return build_pairs.parse_sheet(SHEET)


def test_parse_sheet_keeps_the_opening_apart_from_the_continuation():
    parsed = items()
    assert parsed[1] == ("開頭甲", "續甲一")
    assert parsed[4] == ("開頭乙", "續乙二")


def test_pairing_groups_by_the_opening_that_repeats():
    pairs = build_pairs.group_pairs(items(), KEY)
    assert [numbers for _, numbers in pairs] == [[1, 2], [3, 4]]


def test_pairing_refuses_an_opening_with_three_continuations():
    sheet = SHEET + "\n## #5\n\n*opening:* 開頭甲\n\n續甲三\n\nverdict:\n"
    key = {**KEY, "#5": {"arm": "trained"}}
    with pytest.raises(build_pairs.SheetError, match="3 continuations"):
        build_pairs.group_pairs(build_pairs.parse_sheet(sheet), key)


def test_pairing_refuses_a_pair_that_came_from_one_model():
    key = {**KEY, "#2": {"arm": "base"}}
    with pytest.raises(build_pairs.SheetError, match="nothing to compare"):
        build_pairs.group_pairs(items(), key)


def test_pairing_refuses_identical_continuations():
    sheet = SHEET.replace("續甲二", "續甲一")
    with pytest.raises(build_pairs.SheetError, match="identical"):
        build_pairs.group_pairs(build_pairs.parse_sheet(sheet), KEY)


def test_the_sheet_never_names_which_model_wrote_either_side():
    document, _ = build_pairs.build(items(), KEY, seed="test")
    for word in ("trained", "base", "v10", "arm"):
        assert word not in document.lower(), f"the sheet leaks {word!r}"


def test_the_key_holds_both_sides_and_both_judgements_are_readable():
    document, record = build_pairs.build(items(), KEY, seed="test")
    assert set(record["pairs"]) == {"P1", "P2"}
    for entry in record["pairs"].values():
        assert {entry["A_arm"], entry["B_arm"]} == {"base", "trained"}
    # The sheet must not carry the flat sheet's item numbers: they would let the reader look the
    # verdicts up in the old sheet and un-blind themselves.
    for number in KEY:
        assert number not in document


def test_the_key_does_not_put_the_same_side_first_every_time():
    """A seed that always answers A would make the sheet a coin with one face."""
    sides = set()
    for seed in range(20):
        _, record = build_pairs.build(items(), KEY, seed=str(seed))
        sides.add(record["pairs"]["P1"]["A_arm"])
    assert sides == {"base", "trained"}


def test_an_unmarked_pair_is_unrated_and_never_a_vote():
    choices = {"P1": "", "P2": "a"}
    _, record = build_pairs.build(items(), KEY, seed="test")
    result = score_pairs.score(choices, record)
    assert result["unrated"] == 1
    assert result["decided"] == 1


def test_tie_and_neither_are_reported_and_excluded_from_the_test():
    _, record = build_pairs.build(items(), KEY, seed="test")
    result = score_pairs.score({"P1": "tie", "P2": "neither"}, record)
    assert result["tie"] == 1 and result["neither"] == 1
    assert result["decided"] == 0
    assert result["p"] == 1.0


def test_an_unrecognised_mark_is_invalid_rather_than_a_preference():
    _, record = build_pairs.build(items(), KEY, seed="test")
    result = score_pairs.score({"P1": "maybe", "P2": "b"}, record)
    assert result["invalid"] == 1
    assert result["decided"] == 1


def test_a_sheet_the_key_does_not_match_is_refused():
    _, record = build_pairs.build(items(), KEY, seed="test")
    with pytest.raises(score_pairs.SheetError, match="key does not"):
        score_pairs.score({"P1": "a", "P2": "a", "P3": "a"}, record)
    with pytest.raises(score_pairs.SheetError, match="does not have it"):
        score_pairs.score({"P1": "a"}, record)


@pytest.mark.parametrize(("trained", "base", "expected"), [
    (5, 5, 1.0),
    (10, 0, 2 / 1024),
    (9, 1, 22 / 1024),
    (8, 2, 112 / 1024),
    (0, 0, 1.0),
])
def test_the_sign_test_matches_the_hand_computation(trained, base, expected):
    assert score_pairs.sign_test(trained, base) == pytest.approx(expected)


def test_the_rendered_result_names_the_winner_only_when_it_is_significant():
    _, record = build_pairs.build(items(), KEY, seed="test")
    flat = score_pairs.render(score_pairs.score({"P1": "a", "P2": "b"}, record))
    assert "no detectable preference" in flat
    assert "10 pairs" not in flat


def test_a_fully_unmarked_sheet_says_so_instead_of_reporting_zero():
    _, record = build_pairs.build(items(), KEY, seed="test")
    rendered = score_pairs.render(score_pairs.score({"P1": "", "P2": ""}, record))
    assert "no result here" in rendered
