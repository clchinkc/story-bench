"""Closed A/B/C/D arm manifest resolution and the frozen harness spec.

config/arms.yaml is the single arm-manifest authority. There is no default arm
and no partial manifest: an unknown arm, an unknown flag, or an unequal basic
tool set is refused, not defaulted (shared-interface.md S1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from environment_contract import ContractViolation, digest

ARMS: tuple[str, ...] = ("A", "B", "C", "D")
EXPECTED_FLAGS: Mapping[str, tuple[bool, bool]] = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}
FROZEN_ARGV_TEMPLATE: tuple[str, ...] = (
    "claude", "-p", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
    "--mcp-config", "{mcp_config}", "--no-session-persistence", "--model", "{model}",
)
ARM_ENTRY_KEYS: frozenset[str] = frozenset({"instruction_treatment", "specialized_tools", "treatment_packet"})

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "config" / "arms.yaml"


class ArmManifestError(ContractViolation):
    code = "arm-manifest"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArmManifestError(message)


def load_arm_document(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArmManifestError(f"arm manifest is missing: {path}") from exc
    except yaml.YAMLError as exc:
        raise ArmManifestError(f"arm manifest is not valid YAML: {exc}") from exc
    _require(isinstance(data, dict), "arm manifest must be a mapping")
    return data


@dataclass(frozen=True)
class ArmManifest:
    arm: str
    instruction_treatment: bool
    specialized_tools: bool
    treatment_packet: str | None
    harness: Mapping[str, Any]
    basic_mcp: Mapping[str, Any]
    specialized_mcp: Mapping[str, Any]
    equalisation: Mapping[str, Any]
    logging: Mapping[str, Any]
    document: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def harness_argv(self, *, mcp_config_path: str | Path, model_id: str) -> list[str]:
        template = self.harness.get("argv_template")
        _require(
            tuple(template) == FROZEN_ARGV_TEMPLATE,
            f"harness argv template is not the frozen ADR-023 set: {template!r}",
        )
        return [
            str(part).replace("{mcp_config}", str(mcp_config_path)).replace("{model}", str(model_id))
            for part in template
        ]

    def tool_manifest_digest(self, *, specialized: bool | None = None) -> str:
        from protected_mcp import basic_inventory

        specialized = self.specialized_tools if specialized is None else specialized
        return digest(basic_inventory(specialized=specialized))


def validate_arm_document(document: Mapping[str, Any]) -> None:
    arms = document.get("arms")
    _require(isinstance(arms, dict), "arm manifest requires an 'arms' mapping")
    _require(set(arms) == set(ARMS), f"arm manifest must declare exactly {list(ARMS)}, got {sorted(arms)}")
    for arm, entry in arms.items():
        _require(isinstance(entry, dict), f"arm {arm} entry must be a mapping")
        unknown = sorted(set(entry) - ARM_ENTRY_KEYS)
        _require(not unknown, f"arm {arm} carries unknown flags: {unknown}")
        instruction = entry.get("instruction_treatment")
        specialized = entry.get("specialized_tools")
        _require(instruction in ("off", "arm-b-v1"), f"arm {arm} has an unknown instruction flag {instruction!r}")
        _require(specialized in ("on", "off"), f"arm {arm} has an unknown specialized flag {specialized!r}")
        expected = EXPECTED_FLAGS[arm]
        _require(
            (instruction != "off", specialized == "on") == expected,
            f"arm {arm} flags {(instruction, specialized)!r} disagree with the frozen A/B/C/D table {expected!r}",
        )
    harness = document.get("harness")
    _require(isinstance(harness, dict), "arm manifest requires a harness block")
    _require(
        tuple(harness.get("argv_template", ())) == FROZEN_ARGV_TEMPLATE,
        "the harness argv template must be the frozen ADR-023 set",
    )
    _require(isinstance(harness.get("latency_budget_ms"), int), "the basic latency budget must be one numeric constant")
    for block in ("basic_mcp", "specialized_mcp", "equalisation", "logging"):
        _require(isinstance(document.get(block), dict), f"arm manifest requires a {block} block")


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    document = load_arm_document(path)
    validate_arm_document(document)
    return document


def resolve_arm(arm: str, path: str | Path | None = None) -> ArmManifest:
    document = load_manifest(path)
    if arm not in ARMS:
        raise ArmManifestError(f"unknown arm {arm!r}; the closed set is {list(ARMS)}")
    entry = document["arms"][arm]
    return ArmManifest(
        arm=arm,
        instruction_treatment=entry["instruction_treatment"] != "off",
        specialized_tools=entry["specialized_tools"] == "on",
        treatment_packet=entry.get("treatment_packet"),
        harness=document["harness"],
        basic_mcp=document["basic_mcp"],
        specialized_mcp=document["specialized_mcp"],
        equalisation=document["equalisation"],
        logging=document["logging"],
        document=document,
    )


def harness_argv(arm: str, *, mcp_config_path: str | Path, model_id: str, path: str | Path | None = None) -> list[str]:
    return resolve_arm(arm, path).harness_argv(mcp_config_path=mcp_config_path, model_id=model_id)


def model_id(path: str | Path | None = None) -> str:
    return str(load_manifest(path)["harness"]["model_snapshot"])


def arm_b_instruction_path(document: Mapping[str, Any] | None = None) -> str:
    document = document if document is not None else load_manifest()
    return str(document["treatment_packet"]["arm_b_instruction_file"])


def compose_treatment_packet(*, arm: str, task_brief: str, arm_b_instruction: str, path: str | Path | None = None) -> str:
    """The declared non-settings treatment-packet channel (S4 rule 3).

    The same host-composed mechanism for every arm. For the instruction-treated
    arms (B and D) the arm-B instruction text is prepended to the brief; for A/C
    the packet is the bare task brief.
    """
    manifest = resolve_arm(arm, path)
    if manifest.instruction_treatment:
        return arm_b_instruction.rstrip("\n") + "\n\n---\n\n" + task_brief
    return task_brief


def equalisation_record(*, episode_inputs: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """The equal-across-arms record (S11/T7). Arm is deliberately absent."""
    document = load_manifest(path)
    return {
        "source_facts_digest": episode_inputs["source_facts_digest"],
        "brief_digest": digest(episode_inputs["task_brief"]),
        "memory_init_digest": episode_inputs["memory_init_digest"],
        "information_entitlement_digest": episode_inputs["information_entitlement_digest"],
        "model_snapshot": document["harness"]["model_snapshot"],
        "reasoning_effort": document["harness"]["reasoning_effort"],
        "temperature": document["harness"]["temperature"],
        "seed": document["harness"]["seed"],
        "max_turns": document["harness"]["max_turns"],
        "max_retries": document["harness"]["max_retries"],
        "max_output_tokens": document["harness"]["max_output_tokens"],
        "wall_time_s": document["harness"]["wall_time_s"],
        "dollar_cap_usd": document["harness"]["dollar_cap_usd"],
        "base_transport_digest": digest(
            {
                "server_id": document["basic_mcp"]["server_id"],
                "server_version": document["basic_mcp"]["server_version"],
                "transport": document["basic_mcp"]["transport"],
                "latency_budget_ms": document["harness"]["latency_budget_ms"],
            }
        ),
        "native_binding_digest": episode_inputs["native_binding_digest"],
    }


def assert_equalisation(records: Mapping[str, Mapping[str, Any]]) -> None:
    if set(records) != set(ARMS):
        raise ArmManifestError(f"equalisation requires records for exactly {list(ARMS)}")
    baseline = dict(records["A"])
    for arm in ARMS:
        if dict(records[arm]) != baseline:
            differing = sorted(
                key for key in set(baseline) | set(records[arm]) if baseline.get(key) != records[arm].get(key)
            )
            raise ArmManifestError(f"arm {arm} is not equalised on {differing}")


def required_log_channels(path: str | Path | None = None) -> tuple[str, ...]:
    document = load_manifest(path)
    return tuple(document["logging"]["required_channels"])
