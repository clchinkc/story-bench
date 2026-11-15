from src.function_ledger import Annotation, diagnose


def _a(function: str, scene: int) -> Annotation:
    return Annotation(function=function, scene=scene)


def test_canonical_sequence_is_clean():
    seq = ["8a", "9", "10", "11", "12", "13", "14", "15", "16", "18", "19", "20",
           "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    d = diagnose([_a(f, i) for i, f in enumerate(seq, start=1)])
    assert d.branch == "8a"
    assert d.findings == ()
    assert d.applicable and d.clean


def test_out_of_order_function_is_reported():
    d = diagnose([_a("8a", 1), _a("18", 2), _a("16", 3)])
    assert any(f.kind == "order" and "18" in f.detail for f in d.findings)


def test_missing_prerequisite_is_reported():
    d = diagnose([_a("8b", 1), _a("18", 2)])
    assert any(f.kind == "missing-prerequisite" and "16" in f.detail for f in d.findings)


def test_lack_branch_is_recognised():
    d = diagnose([_a("8b", 1), _a("9", 2)])
    assert d.branch == "8b"
    assert d.applicable


def test_no_branch_means_not_applicable():
    d = diagnose([_a("27", 1)])
    assert d.branch is None
    assert not d.applicable
    assert any(f.kind == "no-branch" for f in d.findings)


def test_unknown_function_is_flagged_not_crashed():
    d = diagnose([_a("8a", 1), _a("99", 2)])
    assert any(f.kind == "unknown-function" for f in d.findings)
    assert d.branch == "8a"


def test_repeated_function_uses_first_scene():
    d = diagnose([_a("21", 1), _a("20", 5), _a("20", 9)])
    assert any(f.kind == "order" and "21" in f.detail for f in d.findings)
