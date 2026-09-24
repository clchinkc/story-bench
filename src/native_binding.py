"""Explicit pinned native artifact/executable resolution.

The trusted host resolves the native narrative engine through THIS module.
The protected MCP server does NOT resolve it: src/protected_mcp.py imports no
native binding and receives the already-resolved engine (or its cooperative
test double) by injection. Implicit PATH/PYTHONPATH discovery is refused
(risk row N1). The native_binding_digest produced here is a member of the
closed 16-member environment-ID record (shared-interface.md S5.1 #11).

Pin basis: the accepted W2b-I native checkpoint at
286b006f9712d3084b83c111bc012963088b7afc with its accepted distributions
(bindings/execution/w2b/parent/assessment-protocol-acceptance.json).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from environment_contract import ContractViolation, digest

PINNED_REPOSITORY = "/Users/clchinkc/Documents/GitHub/story-evaluation-candidates/2026-09-22/narrative-craft"
PINNED_COMMIT = "286b006f9712d3084b83c111bc012963088b7afc"
PINNED_PACKAGE_VERSION = "0.8.0"
PINNED_WHEEL_SHA256 = "b4050fb5de38b2ad0fa579b204cd454059a3ae15a730c7c2ab6a67b30779cf8f"
PINNED_SDIST_SHA256 = "fb3a2e8ba9c9888633ce99323eb37ba7e18ea352891301c4bdd9c05ffba07757"
PINNED_CLI_ABSOLUTE_PATH = PINNED_REPOSITORY + "/.venv/bin/narrative-craft"
PINNED_INSTALLED_SKILL_SHA256 = "3f4ddcf4245f232400485ccedf5931f59916793e62b0a15a3c16601ac7b42830"

PINNED_BINDING_RECORD: dict[str, Any] = {
    "repository": PINNED_REPOSITORY,
    "commit": PINNED_COMMIT,
    "package_version": PINNED_PACKAGE_VERSION,
    "wheel_sha256": PINNED_WHEEL_SHA256,
    "sdist_sha256": PINNED_SDIST_SHA256,
}


class NativeBindingRefused(ContractViolation):
    code = "native-binding-refused"


@dataclass(frozen=True)
class NativeBinding:
    repository: str
    commit: str
    package_version: str
    wheel_sha256: str
    sdist_sha256: str
    cli_path: str
    cli_present: bool

    @property
    def digest(self) -> str:
        return digest(PINNED_BINDING_RECORD)

    def as_record(self) -> dict[str, Any]:
        return dict(PINNED_BINDING_RECORD)


def native_binding_digest() -> str:
    return digest(PINNED_BINDING_RECORD)


def _pythonpath_would_decide_native(environ: Mapping[str, str]) -> str | None:
    raw = environ.get("PYTHONPATH")
    if not raw:
        return None
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry)
        if (candidate / "narrative_craft").is_dir() or (candidate / "narrative_craft.py").is_file():
            return entry
        if candidate.name == "narrative_craft" and (candidate / "src" / "narrative_craft").is_dir():
            return entry
    return None


def resolve_native_binding(
    *,
    environ: Mapping[str, str] | None = None,
    which=shutil.which,
    cli_path: str | Path | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> NativeBinding:
    """Resolve the ONE pinned native binding or refuse.

    Refusal cases: a PATH-resolved narrative-craft that is not the pinned
    executable; a PYTHONPATH that would decide which native module resolves; a
    supplied artifact digest that disagrees with the pin.
    """
    env = dict(os.environ if environ is None else environ)
    requested = str(cli_path) if cli_path is not None else env.get("NC_NATIVE_CLI", PINNED_CLI_ABSOLUTE_PATH)
    resolved = Path(requested).expanduser()

    decided_by_pythonpath = _pythonpath_would_decide_native(env)
    if decided_by_pythonpath is not None:
        raise NativeBindingRefused(
            "PYTHONPATH would supply a native module instead of the pinned artifact: "
            f"{decided_by_pythonpath!r}"
        )

    discovered = which("narrative-craft", path=env.get("PATH", ""))
    if discovered is not None:
        try:
            same = Path(discovered).resolve() == resolved.resolve()
        except OSError:
            same = False
        if not same:
            raise NativeBindingRefused(
                f"PATH discovery resolves an unpinned native executable {discovered!r}; "
                f"the pinned binding is {str(resolved)!r}"
            )

    if artifacts:
        wheel = artifacts.get("wheel_sha256")
        sdist = artifacts.get("sdist_sha256")
        if wheel is not None and wheel != PINNED_WHEEL_SHA256:
            raise NativeBindingRefused(f"wheel digest {wheel!r} is not the pinned {PINNED_WHEEL_SHA256!r}")
        if sdist is not None and sdist != PINNED_SDIST_SHA256:
            raise NativeBindingRefused(f"sdist digest {sdist!r} is not the pinned {PINNED_SDIST_SHA256!r}")

    return NativeBinding(
        repository=PINNED_REPOSITORY,
        commit=PINNED_COMMIT,
        package_version=PINNED_PACKAGE_VERSION,
        wheel_sha256=PINNED_WHEEL_SHA256,
        sdist_sha256=PINNED_SDIST_SHA256,
        cli_path=str(resolved),
        cli_present=resolved.is_file(),
    )


def verify_artifact(path: str | Path, expected_sha256: str) -> str:
    import hashlib

    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise NativeBindingRefused(f"artifact {path} digest {actual} != pinned {expected_sha256}")
    return actual


def assert_pinned_name_is_absent_from_path(path_value: str) -> None:
    """T4 helper: the PARTICIPANT child PATH must not resolve the native CLI.

    The destination is the workspace copy, not a PATH; T4's PATH requirement is
    about the participant child environment (environment_contract.participant_child_env).
    Each entry is checked for the bare unpinned name narrative-craft, so a PATH
    that would shadow the pinned absolute binding is refused.
    """
    for entry in path_value.split(os.pathsep):
        if not entry:
            continue
        if (Path(entry) / "narrative-craft").exists():
            raise NativeBindingRefused(f"native CLI is reachable on the participant PATH at {entry!r}")
