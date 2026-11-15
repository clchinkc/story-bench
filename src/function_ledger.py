"""Function-ledger diagnostics for a story annotation.

The ledger is a *sequence* check, not a coverage score. Propp's functions may be
omitted but do not reorder, so the only findings this module reports are order,
missing-prerequisite, and which opening branch the story used (8a villainy or 8b
lack). Coverage is reported for completeness and is never scored.

Vocabulary: 31 functions numbered 1..31. Function 8 has two branches, 8a and 8b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

Branch = Literal["8a", "8b"]


@dataclass(frozen=True)
class Function:
    id: str
    zh: str
    en: str


def _f(id: str, zh: str, en: str) -> Function:
    return Function(id=id, zh=zh, en=en)


FUNCTIONS: tuple[Function, ...] = (
    _f("1", "缺席", "Absentation"),
    _f("2", "禁令", "Interdiction"),
    _f("3", "違背禁令", "Violation"),
    _f("4", "搜索勘查", "Reconnaissance"),
    _f("5", "惡徒發現主角", "Delivery"),
    _f("6", "詐騙", "Trickery"),
    _f("7", "共犯", "Complicity"),
    _f("8a", "惡行", "Villainy"),
    _f("8b", "缺失", "Lack"),
    _f("9", "調解", "Mediation"),
    _f("10", "反擊開始", "Counteraction"),
    _f("11", "出發離去", "Departure"),
    _f("12", "第一次測試", "First donor function"),
    _f("13", "主角的反動", "Hero's reaction"),
    _f("14", "取得法力", "Provision of agent"),
    _f("15", "引導／遷移", "Spatial transference"),
    _f("16", "奮鬥", "Struggle"),
    _f("17", "烙印", "Branding"),
    _f("18", "勝利", "Victory"),
    _f("19", "不幸清除", "Liquidation"),
    _f("20", "返回", "Return"),
    _f("21", "追擊", "Pursuit"),
    _f("22", "救援", "Rescue"),
    _f("23", "無人認識者出現", "Unrecognised arrival"),
    _f("24", "無根據的要求", "Unfounded claims"),
    _f("25", "艱難工作", "Difficult task"),
    _f("26", "任務完成", "Task resolved"),
    _f("27", "認識", "Recognition"),
    _f("28", "曙光／揭穿", "Exposure"),
    _f("29", "變容", "Transfiguration"),
    _f("30", "處罰", "Punishment"),
    _f("31", "婚禮", "Wedding"),
)

BY_ID: dict[str, Function] = {f.id: f for f in FUNCTIONS}

#: The two opening branches. A story uses one; using neither means the ledger
#: does not apply (it is not a task-shaped story).
OPENING_BRANCHES: tuple[str, ...] = ("8a", "8b")

#: Functions whose presence implies an earlier function in any told sequence.
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "10": ("9",),
    "11": ("10",),
    "14": ("12", "13"),
    "16": ("15",),
    "18": ("16",),
    "19": ("18",),
    "21": ("20",),
    "22": ("21",),
    "24": ("23",),
    "26": ("25",),
    "27": ("26", "24"),
    "28": ("27",),
    "29": ("28",),
    "30": ("28",),
    "31": ("29", "30"),
}


#: Functions satisfied by ANY one of several predecessors (function 9 follows
#: villainy or lack, never both).
ANY_OF: dict[str, tuple[str, ...]] = {"9": ("8a", "8b")}


@dataclass(frozen=True)
class Annotation:
    """One observed function at one scene.

    A function may be observed more than once (Propp allows repetition); pass one
    annotation per occurrence and the module uses the first scene for ordering.
    """

    function: str
    scene: int
    evidence: str = ""


@dataclass(frozen=True)
class Finding:
    kind: Literal["order", "missing-prerequisite", "no-branch", "unknown-function"]
    detail: str


@dataclass(frozen=True)
class Diagnosis:
    functions_present: tuple[str, ...]
    branch: Branch | None
    coverage: float
    findings: tuple[Finding, ...]

    @property
    def applicable(self) -> bool:
        return self.branch is not None

    @property
    def clean(self) -> bool:
        return not self.findings


def _first_scene(annotations: Iterable[Annotation]) -> dict[str, int]:
    first: dict[str, int] = {}
    for a in annotations:
        if a.function not in BY_ID:
            continue
        if a.function not in first or a.scene < first[a.function]:
            first[a.function] = a.scene
    return first


def _branch(first: dict[str, int]) -> Branch | None:
    if "8a" in first and "8b" in first:
        return "8a" if first["8a"] <= first["8b"] else "8b"
    if "8a" in first:
        return "8a"
    if "8b" in first:
        return "8b"
    return None


def _unknown_findings(annotations: Iterable[Annotation]) -> list[Finding]:
    unknown = sorted({a.function for a in annotations if a.function not in BY_ID})
    return [Finding("unknown-function", f"unknown function id: {u}") for u in unknown]

def _findings_for(later: str, earlier: str, first: dict[str, int]) -> list[Finding]:
    if earlier not in first:
        return [
            Finding(
                "missing-prerequisite",
                f"function {later} appears without its prerequisite {earlier}",
            )
        ]
    if first[earlier] > first[later]:
        return [
            Finding(
                "order",
                f"function {later} (scene {first[later]}) precedes {earlier} "
                f"(scene {first[earlier]})",
            )
        ]
    return []


def _any_of_findings(first: dict[str, int]) -> list[Finding]:
    findings: list[Finding] = []
    for later, options in ANY_OF.items():
        if later in first and not any(o in first for o in options):
            findings.append(
                Finding(
                    "missing-prerequisite",
                    f"function {later} appears without any of {options}",
                )
            )
    return findings


def _prerequisite_findings(first: dict[str, int]) -> list[Finding]:
    findings: list[Finding] = []
    for later, earlier_ids in PREREQUISITES.items():
        if later not in first:
            continue
        for earlier in earlier_ids:
            findings.extend(_findings_for(later, earlier, first))
    return findings


def load_annotations(payload: dict) -> list[Annotation]:
    """Read {"annotations": [{"function": "8a", "scene": 1, "evidence": "..."}]}."""
    return [
        Annotation(
            function=item["function"],
            scene=int(item["scene"]),
            evidence=item.get("evidence", ""),
        )
        for item in payload.get("annotations", [])
    ]


def diagnose(annotations: Sequence[Annotation]) -> Diagnosis:
    """Report order and prerequisite findings for a scene-ordered annotation."""
    first = _first_scene(annotations)
    branch = _branch(first)
    findings = (
        _unknown_findings(annotations)
        + _any_of_findings(first)
        + _prerequisite_findings(first)
    )
    if branch is None:
        findings.append(Finding("no-branch", "neither opening branch (8a or 8b) is present"))
    return Diagnosis(
        functions_present=tuple(sorted(first, key=lambda fid: first[fid])),
        branch=branch,
        coverage=round(len(first) / len(FUNCTIONS), 4),
        findings=tuple(findings),
    )

def main(argv: Sequence[str] | None = None) -> int:
    """CLI: read an annotation JSON file, print the diagnosis as JSON."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Function-ledger diagnostics")
    parser.add_argument("--annotations", required=True, help="path to annotation JSON")
    args = parser.parse_args(argv)

    with open(args.annotations, encoding="utf-8") as handle:
        payload = json.load(handle)

    diagnosis = diagnose(load_annotations(payload))
    print(
        json.dumps(
            {
                "branch": diagnosis.branch,
                "applicable": diagnosis.applicable,
                "coverage": diagnosis.coverage,
                "functions_present": list(diagnosis.functions_present),
                "findings": [
                    {"kind": f.kind, "detail": f.detail} for f in diagnosis.findings
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if diagnosis.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
