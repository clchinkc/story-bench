"""The character audit: the author's text must come out unchanged, and the model's must come out corrected.

The negative control is the load-bearing test. A repair that "fixes" the author's own prose is worse than no
repair at all, because the damage is invisible — a wrong Traditional character reads as correct. So the first
test asserts byte-identity on prose containing the exact characters a naive converter would break.
"""

from __future__ import annotations

from story_rl.traditional import audit, defects, repair


AUTHOR_PROSE = (
    "他把手掌按在耀文後腰上，微微發燙。外頭雷滾得像一塊懸在半空。"
    "櫃台的瓷痕裡積著一層薄薄的灰，指尖劃過一圈細微的缺角。"
    "岩屑在輪胎下鬆散地滑開，遠處的臺階上沒有人。"
    "畫面裡那張制服的胸前別著一枚舊徽章，面前的人卻先低下了頭。"
    "松樹的影子落在紙張邊緣，他把紙攤平，雨水順著纖維洇出一圈深色。"
)


def test_the_authors_own_prose_is_never_changed():
    """Negative control. Every character here is hers; a repair that touches it is a defect."""
    cleaned, findings = repair(AUTHOR_PROSE)
    assert cleaned == AUTHOR_PROSE
    assert findings == []
    assert defects(AUTHOR_PROSE) == []


def test_a_word_rule_does_not_lengthen_the_text():
    """The first implementation patched characters by index, so `冲动` -> `衝動` inserted a second character
    and produced `衝动動`. Length is the cheapest witness that a multi-character rule was applied correctly."""
    cleaned, _ = repair("難以抗拒的衝动。")
    assert cleaned == "難以抗拒的衝動。"
    assert len(cleaned) == len("難以抗拒的衝动。")


def test_an_unambiguous_simplified_character_is_repaired():
    cleaned, findings = repair("风从巷口吹过来，带点河面返上来的湿冷。")
    assert "风" not in cleaned and "湿" not in cleaned and "来" not in cleaned
    assert "風" in cleaned and "濕" in cleaned
    assert all(f.kind in {"simplified", "context"} for f in findings)


def test_the_author_form_wins_over_the_generic_form():
    """She writes 裏 (311) not 裡 (1), 後 (210) not 后 (0), 著 (450) not 着 (0)."""
    cleaned, _ = repair("風口里呼出幾聲斷續的藍，最后勺子里只剩一層薄薄的白。")
    assert "裏" in cleaned and "裡" not in cleaned
    assert "後" in cleaned
    assert "里" not in cleaned


def test_a_character_that_is_already_traditional_is_left_alone():
    """松樹 is not 鬆樹, 岩 is not 巖, 台 is her form and not 臺."""
    for text in ("松樹的影子", "岩屑在輪胎下打滑", "櫃台上放著一杯水"):
        cleaned, findings = repair(text)
        assert cleaned == text, f"{text!r} was rewritten to {cleaned!r}"
        assert findings == []


def test_a_verb_use_of_the_same_character_is_repaired():
    """The same character can be right in one word and wrong in another."""
    cleaned, _ = repair("他慢慢松開手指，又把卷起的袖口放下。")
    assert "鬆開" in cleaned
    assert "捲起" in cleaned


def test_the_reward_vetoes_a_simplified_character_and_pays_correct_text():
    """The defect the reward could not see. Both texts are otherwise identical in length and shape."""
    from story_rl.prose import guard

    bad = "他把手掌按在耀文后腰上，指尖划过一圈细微的缺角，纸张边缘被磨得起了毛。"
    good = "他把手掌按在耀文後腰上，指尖劃過一圈細微的缺角，紙張邊緣被磨得起了毛。"

    assert any(r.startswith("simplified") for r in guard(bad, "zh"))
    assert guard(good, "zh") == []


def test_audit_reports_position_and_excerpt_for_a_repair():
    findings = audit("前面风很大，他没说话。")
    assert findings and findings[0].char == "风"
    assert findings[0].position == 2
    assert findings[0].replacement == "風"
    assert "风" in findings[0].excerpt
