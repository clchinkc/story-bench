"""Loader/validator for the versioned portable arm-B instruction package.

Renamed from arm_b_package.py (F-08) so the data package and the loader cannot
collide. The loader anchors resources with importlib.resources.files(
"arm_b_package"), so manifest.json and the manifest instructionFile resolve to
the real package directory (risk row B6), verifies the digest bindings and the
closed capabilityDelta acceptance rule (F-09), and refuses an unversioned,
undocumented or capability-losing package.

The source skill is never edited: this module only reads the installed package
digest as a provenance pin.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from environment_contract import ContractViolation, canonical_json, parse_canonical_json
from native_binding import PINNED_COMMIT, PINNED_INSTALLED_SKILL_SHA256

PACKAGE_NAME = "arm_b_package"
MANIFEST_NAME = "manifest.json"
INSTRUCTION_FILE = "v1/SKILL.md"
PACKAGE_VERSION = "arm-b-v1"
DELIVERY_CHANNEL = "host-treatment-packet"

OWNER_WAVE_SERVICE = "owner-wave answering by the labeled simulated-author fixture"
GENESIS_SERVICE = "genesis authorization"
PACKET_SERVICE = "host-side storage of prepared packets"
HOST_SERVICES: frozenset[str] = frozenset({OWNER_WAVE_SERVICE, GENESIS_SERVICE, PACKET_SERVICE})

MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "schemaVersion",
        "packageVersion",
        "armBSkillOnlyClaim",
        "sourceSkill",
        "instructionFile",
        "instructionSha256",
        "deliveryChannel",
        "hostServicesEqualInAllArms",
        "derivation",
        "excluded",
    }
)
SOURCE_SKILL_KEYS: frozenset[str] = frozenset({"identity", "path", "sha256", "installedAt", "nativeBinding"})
DERIVATION_KEYS: frozenset[str] = frozenset(
    {"id", "kind", "source", "basicOperation", "change", "mechanismPreserved", "capabilityDelta"}
)
DELTA_KEYS: frozenset[str] = frozenset(
    {"kind", "reference", "reason", "hostServiceEqualInAllArms", "removesMatchedTaskStep", "parentAcceptanceRef"}
)
EXCLUDED_KEYS: frozenset[str] = frozenset({"kind", "reference", "reason"})
DELTA_KINDS: frozenset[str] = frozenset({"none", "host-service", "participant-loss"})

SEPARABLE_BY_CONTRACT = "SEPARABLE_BY_CONTRACT"
BLOCKED = "BLOCKED"


class PackageRefused(ContractViolation):
    code = "arm-b-package-refused"


class PackageLayoutError(PackageRefused):
    code = "arm-b-package-layout"


@dataclass(frozen=True)
class ArmBPackage:
    anchor: Path
    manifest: Mapping[str, Any]
    instruction_text: str
    instruction_sha256: str
    arm_b_skill_only_claim: bool
    separability_verdict: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_package_anchor(package: str = PACKAGE_NAME) -> Path:
    """Resolve the data package directory, refusing the sibling-module layout.

    The preflight collision (a sibling arm_b_package.py) makes the anchor resolve
    to the containing directory as a module rather than to a real package
    directory; that layout is refused here.
    """
    spec = importlib.util.find_spec(package)
    if spec is None:
        raise PackageLayoutError(f"{package!r} does not resolve to an importable package")
    if spec.submodule_search_locations is None:
        raise PackageLayoutError(
            f"{package!r} resolves to a module file, not a package directory; "
            "the documented package layout cannot load manifest.json"
        )
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise PackageLayoutError(f"{package!r} resolves to multiple locations: {locations!r}")
    anchor = Path(locations[0])
    if not anchor.is_dir() or anchor.name != package:
        raise PackageLayoutError(f"{package!r} anchor {anchor} is not a directory named {package!r}")
    return anchor


def _read_package_text(package: str, rel: str) -> str:
    resource = importlib.resources.files(package).joinpath(rel)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PackageLayoutError(f"package resource {rel!r} is missing from {package!r}") from exc


def validate_capability_delta(delta: Any, *, entry_id: str) -> str:
    """Apply the closed capabilityDelta acceptance rule (shared-interface S12.3).

    Returns the delta kind. Raises PackageRefused for any malformed object, a
    missing field, an unknown kind, an unknown host service, a participant loss
    without a parent acceptance reference, or removesMatchedTaskStep=true.
    """
    if not isinstance(delta, Mapping):
        raise PackageRefused(f"derivation {entry_id}: capabilityDelta must be a closed object, not {type(delta).__name__}")
    missing = sorted(DELTA_KEYS - set(delta))
    extra = sorted(set(delta) - DELTA_KEYS)
    if missing or extra:
        raise PackageRefused(f"derivation {entry_id}: capabilityDelta has missing {missing} or extra {extra} fields")
    kind = delta["kind"]
    if kind not in DELTA_KINDS:
        raise PackageRefused(f"derivation {entry_id}: unknown capabilityDelta kind {kind!r}")
    if delta["hostServiceEqualInAllArms"] is not True:
        raise PackageRefused(f"derivation {entry_id}: hostServiceEqualInAllArms must be true")
    if delta["removesMatchedTaskStep"] is True:
        raise PackageRefused(
            f"derivation {entry_id}: removesMatchedTaskStep=true; arm-B separability verdict is {BLOCKED}"
        )
    if not isinstance(delta["reason"], str) or not delta["reason"].strip():
        raise PackageRefused(f"derivation {entry_id}: capabilityDelta.reason is required")
    if kind == "none":
        if delta["reference"] is not None or delta["parentAcceptanceRef"] is not None:
            raise PackageRefused(f"derivation {entry_id}: kind=none must have reference=null and parentAcceptanceRef=null")
    elif kind == "host-service":
        if delta["reference"] not in HOST_SERVICES:
            raise PackageRefused(
                f"derivation {entry_id}: kind=host-service must name a host service in {sorted(HOST_SERVICES)}"
            )
    elif kind == "participant-loss":
        if delta["parentAcceptanceRef"] in (None, ""):
            raise PackageRefused(
                f"derivation {entry_id}: kind=participant-loss requires a non-null parentAcceptanceRef"
            )
        if delta["removesMatchedTaskStep"] is not False:
            raise PackageRefused(f"derivation {entry_id}: participant-loss must not remove a matched-task step")
    return str(kind)


def validate_manifest(manifest: Any, *, instruction_text: str) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise PackageRefused("manifest.json must be a JSON object")
    missing = sorted(MANIFEST_KEYS - set(manifest))
    extra = sorted(set(manifest) - MANIFEST_KEYS)
    if missing or extra:
        raise PackageRefused(f"manifest.json is not closed: missing {missing}, extra {extra}")
    if manifest["schemaVersion"] != 1:
        raise PackageRefused(f"unsupported manifest schemaVersion {manifest['schemaVersion']!r}")
    if manifest["packageVersion"] != PACKAGE_VERSION:
        raise PackageRefused(f"unversioned or unexpected packageVersion {manifest['packageVersion']!r}")
    if manifest["deliveryChannel"] != DELIVERY_CHANNEL:
        raise PackageRefused(f"deliveryChannel must be {DELIVERY_CHANNEL!r}, got {manifest['deliveryChannel']!r}")
    if manifest["instructionFile"] != INSTRUCTION_FILE:
        raise PackageRefused(f"instructionFile must be {INSTRUCTION_FILE!r}")
    if not isinstance(manifest["armBSkillOnlyClaim"], bool):
        raise PackageRefused("armBSkillOnlyClaim must be a boolean")

    source = manifest["sourceSkill"]
    if not isinstance(source, Mapping) or set(source) != SOURCE_SKILL_KEYS:
        raise PackageRefused("sourceSkill must be a closed object with identity/path/sha256/installedAt/nativeBinding")
    if source["sha256"] != PINNED_INSTALLED_SKILL_SHA256:
        raise PackageRefused(
            f"sourceSkill.sha256 {source['sha256']!r} is not the installed packaged skill digest "
            f"{PINNED_INSTALLED_SKILL_SHA256!r}"
        )
    if source["nativeBinding"] != PINNED_COMMIT:
        raise PackageRefused(f"sourceSkill.nativeBinding is not the pinned native commit {PINNED_COMMIT!r}")

    actual_instruction = _sha256_text(instruction_text)
    if manifest["instructionSha256"] != actual_instruction:
        raise PackageRefused(
            f"instructionSha256 {manifest['instructionSha256']!r} does not match the package bytes {actual_instruction!r}"
        )

    derivation = manifest["derivation"]
    if not isinstance(derivation, list) or not derivation:
        raise PackageRefused("derivation must be a non-empty list")
    seen_ids: set[str] = set()
    participant_loss = False
    for entry in derivation:
        if not isinstance(entry, Mapping):
            raise PackageRefused("each derivation entry must be an object")
        entry_missing = sorted(DERIVATION_KEYS - set(entry))
        entry_extra = sorted(set(entry) - DERIVATION_KEYS)
        if entry_missing or entry_extra:
            raise PackageRefused(f"derivation entry has missing {entry_missing} or extra {entry_extra} fields")
        entry_id = str(entry["id"])
        if entry_id in seen_ids:
            raise PackageRefused(f"duplicate derivation id {entry_id!r}")
        seen_ids.add(entry_id)
        for field in ("kind", "source", "basicOperation", "change", "mechanismPreserved"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise PackageRefused(f"derivation {entry_id}: {field} must be a non-empty string")
        kind = validate_capability_delta(entry["capabilityDelta"], entry_id=entry_id)
        participant_loss = participant_loss or kind == "participant-loss"
    if participant_loss and manifest["armBSkillOnlyClaim"] is not False:
        raise PackageRefused("a participant-loss derivation forces armBSkillOnlyClaim to false")

    excluded = manifest["excluded"]
    if not isinstance(excluded, list) or not excluded:
        raise PackageRefused("excluded must be a non-empty list")
    for entry in excluded:
        if not isinstance(entry, Mapping) or not EXCLUDED_KEYS.issubset(entry):
            raise PackageRefused("each excluded entry must carry kind/reference/reason")

    services = manifest["hostServicesEqualInAllArms"]
    if not isinstance(services, list) or set(services) != set(HOST_SERVICES):
        raise PackageRefused("hostServicesEqualInAllArms must enumerate the three host services of S12.4")

    return {
        "armBSkillOnlyClaim": bool(manifest["armBSkillOnlyClaim"]),
        "separabilityVerdict": SEPARABLE_BY_CONTRACT,
        "derivationCount": len(derivation),
    }


def load_package(package: str = PACKAGE_NAME) -> ArmBPackage:
    anchor = resolve_package_anchor(package)
    manifest_text = _read_package_text(package, MANIFEST_NAME)
    manifest = parse_canonical_json(manifest_text)
    instruction_file = str(manifest.get("instructionFile", INSTRUCTION_FILE))
    instruction_text = _read_package_text(package, instruction_file)
    verdict = validate_manifest(manifest, instruction_text=instruction_text)
    return ArmBPackage(
        anchor=anchor,
        manifest=manifest,
        instruction_text=instruction_text,
        instruction_sha256=_sha256_text(instruction_text),
        arm_b_skill_only_claim=verdict["armBSkillOnlyClaim"],
        separability_verdict=verdict["separabilityVerdict"],
    )


def instruction_sha256() -> str:
    return load_package().instruction_sha256


def package_digest() -> str:
    package = load_package()
    return canonical_json(
        {
            "packageVersion": package.manifest["packageVersion"],
            "instructionSha256": package.instruction_sha256,
            "derivationCount": len(package.manifest["derivation"]),
        }
    )
