"""Contract for the three parser policies.

The load-bearing property is the nesting one: `strict ⊆ lenient ⊆ prose`. If it ever breaks,
a run's "unparsed" count stops meaning what the report says it means.
"""

import pytest

from continuation_parser import (
    POLICIES,
    extract,
    extract_lenient,
    extract_prose,
    extract_strict,
    first_object,
    strip_fences,
)

CANONICAL = '{"continuation": "Mara turned, her grey eyes catching the lamplight."}'

# Every shape a free-decoding model has actually been observed to emit, plus the ones that
# would break a naive scanner.
SAMPLES = {
    "canonical": CANONICAL,
    "fenced": f"```json\n{CANONICAL}\n```",
    "alias-key": '{"text": "Mara turned."}',
    "single-unusual-key": '{"result": "Mara turned."}',
    "bare-string": '"Mara turned."',
    "preamble": 'Here is the continuation:\n{"continuation": "Mara turned."}',
    "trailing-chatter": '{"continuation": "Mara turned."}\n\nLet me know if you want more.',
    "brace-in-string": '{"continuation": "He said {hi} and left."}',
    "escaped-quote": '{"continuation": "She said \\"grey\\" once."}',
    "two-objects": '{"continuation": "First."} {"continuation": "Second."}',
    "prose": "Mara turned, her grey eyes catching the lamplight.",
    "empty": "",
    "whitespace": "   \n  ",
    "truncated": '{"continuation": "Mara turn',
    "wrong-typed": '{"continuation": ["Mara", "turned"]}',
}


def test_strict_reads_the_constrained_shape():
    result = extract_strict(CANONICAL)

    assert result.text == "Mara turned, her grey eyes catching the lamplight."
    assert result.unparsed is False


def test_strict_refuses_an_alias_key_that_lenient_accepts():
    assert extract_strict('{"text": "Mara turned."}').unparsed is True
    assert extract_lenient('{"text": "Mara turned."}').text == "Mara turned."


def test_strict_refuses_a_non_string_continuation():
    assert extract_strict('{"continuation": ["Mara"]}').unparsed is True


def test_lenient_reads_a_bare_json_string():
    assert extract_lenient('"Mara turned."').text == "Mara turned."


def test_a_brace_inside_a_string_does_not_end_the_object():
    """The masking bug: `first {` … `first }` would cut here and lose valid JSON."""
    assert first_object('{"continuation": "He said {hi} and left."}') == (
        '{"continuation": "He said {hi} and left."}'
    )


def test_an_escaped_quote_does_not_end_the_string():
    assert extract_strict('{"continuation": "She said \\"grey\\" once."}').text == (
        'She said "grey" once.'
    )


def test_two_objects_yield_the_first():
    assert extract_strict('{"continuation": "First."} {"continuation": "Second."}').text == "First."


def test_trailing_chatter_is_ignored():
    text = '{"continuation": "Mara turned."}\n\nLet me know if you want more.'
    assert extract_strict(text).text == "Mara turned."


def test_a_truncated_object_is_unparsed_but_prose_keeps_the_text():
    assert extract_strict(SAMPLES["truncated"]).unparsed is True
    result = extract_prose(SAMPLES["truncated"])
    assert result.text == SAMPLES["truncated"]
    assert result.unparsed is False


def test_prose_never_reports_unparsed():
    for name, text in SAMPLES.items():
        assert extract_prose(text).unparsed is False, name


def test_a_blank_completion_is_an_empty_prose_not_a_missing_one():
    assert extract_prose("   \n ").text == ""
    assert extract_prose("").text == ""


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_acceptance_nests_strict_then_lenient_then_prose(name):
    text = SAMPLES[name]
    strict, lenient, prose = extract(text, "strict"), extract(text, "lenient"), extract(text, "prose")

    assert not (strict.unparsed is False and lenient.unparsed is True)
    assert not (lenient.unparsed is False and prose.unparsed is True)
    if not strict.unparsed:
        assert lenient.text == strict.text


def test_strip_fences_leaves_the_body_alone():
    assert strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_an_unknown_policy_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown parser policy"):
        extract(CANONICAL, "guess")


def test_the_policy_list_is_ordered_from_strictest():
    assert POLICIES == ("strict", "lenient", "prose")
