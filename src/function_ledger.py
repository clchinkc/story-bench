"""Function-ledger diagnostics for a story annotation.

The ledger is a *sequence* check, not a coverage score. Propp's functions may be
omitted but do not reorder, so the only findings this module reports are order,
missing-predecessor, and which opening branch the story used (8a villainy or 8b
lack). Coverage is reported for completeness and is never scored.

Vocabulary: 31 numbered functions, with function 8 split into two branches (8a
and 8b). A sequence uses one branch, so it occupies 31 slots and carries 32 ids.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal, Sequence

Branch = Literal["8a", "8b"]


@dataclass(frozen=True)
class Function:
    id: str
    zh: str
    en: str


FUNCTION_SEQUENCE: tuple[Function, ...] = (
    Function("1", "缺席", "Absentation"),
    Function("2", "禁令", "Interdiction"),
    Function("3", "違背禁令", "Violation"),
    Function("4", "搜索勘查", "Reconnaissance"),
    Function("5", "惡徒發現主角", "Delivery"),
    Function("6", "詐騙", "Trickery"),
    Function("7", "共犯", "Complicity"),
    Function("8a", "惡行", "Villainy"),
    Function("8b", "缺失", "Lack"),
    Function("9", "調解", "Mediation"),
    Function("10", "反擊開始", "Counteraction"),
    Function("11", "出發離去", "Departure"),
    Function("12", "第一次測試", "First donor function"),
    Function("13", "主角的反動", "Hero's reaction"),
    Function("14", "取得法力", "Provision of agent"),
    Function("15", "引導／遷移", "Spatial transference"),
    Function("16", "奮鬥", "Struggle"),
    Function("17", "烙印", "Branding"),
    Function("18", "勝利", "Victory"),
    Function("19", "不幸清除", "Liquidation"),
    Function("20", "返回", "Return"),
    Function("21", "追擊", "Pursuit"),
    Function("22", "救援", "Rescue"),
    Function("23", "無人認識者出現", "Unrecognised arrival"),
    Function("24", "無根據的要求", "Unfounded claims"),
    Function("25", "艱難工作", "Difficult task"),
    Function("26", "任務完成", "Task resolved"),
    Function("27", "認識", "Recognition"),
    Function("28", "曙光／揭穿", "Exposure"),
    Function("29", "變容", "Transfiguration"),
    Function("30", "處罰", "Punishment"),
    Function("31", "婚禮", "Wedding"),
)

BY_ID: dict[str, Function] = {function.id: function for function in FUNCTION_SEQUENCE}

#: The two opening branches. A story uses one; using neither means the ledger
#: does not apply (it is not a task-shaped story).
OPENING_BRANCHES: tuple[Branch, ...] = ("8a", "8b")

#: Slots in one sequence: functions 1..31, with 8 counted once because a story
#: takes 8a or 8b and never both. Coverage divides by this, not by `len(FUNCTION_SEQUENCE)`,
#: which counts both branches and so could never reach 1.
SEQUENCE_SLOTS = 31

#: Functions whose presence implies every listed predecessor.
REQUIRES_ALL: dict[str, tuple[str, ...]] = {
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
REQUIRES_ANY: dict[str, tuple[str, ...]] = {"9": ("8a", "8b")}


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
    kind: Literal["order", "missing-predecessor", "no-branch", "unknown-function"]
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


def _first_scene(annotations: Sequence[Annotation]) -> dict[str, int]:
    first: dict[str, int] = {}
    for annotation in annotations:
        if annotation.function not in BY_ID:
            continue
        if annotation.function not in first or annotation.scene < first[annotation.function]:
            first[annotation.function] = annotation.scene
    return first


def _branch(first: dict[str, int]) -> Branch | None:
    """The opening branch the story used; a tie on scene goes to villainy (8a)."""
    if "8a" in first and "8b" in first:
        return "8a" if first["8a"] <= first["8b"] else "8b"
    if "8a" in first:
        return "8a"
    if "8b" in first:
        return "8b"
    return None


def _unknown_findings(annotations: Sequence[Annotation]) -> list[Finding]:
    unknown = sorted({annotation.function for annotation in annotations if annotation.function not in BY_ID})
    return [Finding("unknown-function", f"unknown function id: {fid}") for fid in unknown]


def _predecessor_findings(later: str, earlier: str, first: dict[str, int]) -> list[Finding]:
    if earlier not in first:
        return [Finding("missing-predecessor", f"function {later} appears without its predecessor {earlier}")]
    if first[earlier] > first[later]:
        return [
            Finding(
                "order",
                f"function {later} (scene {first[later]}) precedes {earlier} (scene {first[earlier]})",
            )
        ]
    return []


def _all_predecessor_findings(first: dict[str, int]) -> list[Finding]:
    findings: list[Finding] = []
    for later, earlier_ids in REQUIRES_ALL.items():
        if later not in first:
            continue
        for earlier in earlier_ids:
            findings.extend(_predecessor_findings(later, earlier, first))
    return findings


def _any_predecessor_findings(first: dict[str, int]) -> list[Finding]:
    findings: list[Finding] = []
    for later, options in REQUIRES_ANY.items():
        if later in first and not any(option in first for option in options):
            findings.append(Finding("missing-predecessor", f"function {later} appears without any of {options}"))
    return findings


def load_annotations(document: dict) -> list[Annotation]:
    """Read {"annotations": [{"function": "8a", "scene": 1, "evidence": "..."}]}."""
    return [
        Annotation(function=item["function"], scene=int(item["scene"]), evidence=item.get("evidence", ""))
        for item in document.get("annotations", [])
    ]


def diagnose(annotations: Sequence[Annotation]) -> Diagnosis:
    """Report order and predecessor findings for a scene-ordered annotation."""
    first = _first_scene(annotations)
    branch = _branch(first)
    findings = (
        _unknown_findings(annotations) + _any_predecessor_findings(first) + _all_predecessor_findings(first)
    )
    if branch is None:
        findings.append(Finding("no-branch", "neither opening branch (8a or 8b) is present"))
    return Diagnosis(
        functions_present=tuple(sorted(first, key=lambda fid: first[fid])),
        branch=branch,
        coverage=round(len(first) / SEQUENCE_SLOTS, 4),
        findings=tuple(findings),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: read an annotation JSON file, print the diagnosis as JSON."""
    parser = argparse.ArgumentParser(description="Function-ledger diagnostics")
    parser.add_argument("--annotations", required=True, help="path to annotation JSON")
    args = parser.parse_args(argv)

    with open(args.annotations, encoding="utf-8") as handle:
        document = json.load(handle)

    diagnosis = diagnose(load_annotations(document))
    print(
        json.dumps(
            {
                "branch": diagnosis.branch,
                "applicable": diagnosis.applicable,
                "coverage": diagnosis.coverage,
                "functions_present": list(diagnosis.functions_present),
                "findings": [{"kind": finding.kind, "detail": finding.detail} for finding in diagnosis.findings],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if diagnosis.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
