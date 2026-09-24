"""Driver-side blocked-capability probes (shared-interface.md S14, F-07 / N-04).

The probe performs REAL child operations (entitlement-scoped path access, spawn,
environment read, import, tool discovery) in a child process launched with the
participant's exact sanitized environment and cwd discipline. It does not run
inside the participant process, because a tools-empty participant has no
capability to spawn or open.

Behavioural path channels (conditions repair F-01 / AC-1a..e). The path channels
(filesystem_escape, source_inspection, inherited_settings) are observed in three
driver-constructible worlds, and the channel outcome is decided by the SHIPPED
enforcement seam, never by whether a fixture happened to create a file:

  * W_GRANT  the target EXISTS and the entitlement permits it         -> granted
  * W_DENY   the target EXISTS but the entitlement refuses it         -> denied
             (the detail names the enforcement refusal, never a
             missing-file/FileNotFoundError absence)
  * W_ABSENT the target does not exist and the entitlement permits it -> absent
             (an explicit third outcome that denied_channels() does not count)

The child imports authority_boundary.guard_read (the same
Entitlement.permits_read -> resolve_inside order as
ProtectedMcpServer.tool_read) from a driver-supplied source path and reports that
seam's refusal as the authoritative channel outcome, while separately recording
the raw OS read result in raw_os. In W_DENY the raw OS read succeeds, so the
denial provably comes from the entitlement and not from the file's absence.

The probe's secret-key set (F-02 / AC-2c) is an independently declared literal,
deliberately wider than the environment-contract strip policy, so a strip-list
gap stays observable instead of being self-confirming.

Honest limit: this is runtime discipline plus explicit denial at the shipped
Python seam, NOT a kernel or OS sandbox (see docs/environment-contract.md). A
child that does not route through the seam is not constrained by the probe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROBE_CHANNELS: tuple[str, ...] = (
    "filesystem_escape",
    "source_inspection",
    "cli_execution",
    "environment_secrets",
    "inherited_settings",
    "tool_discovery",
)
PATH_CHANNELS: tuple[str, ...] = ("filesystem_escape", "source_inspection", "inherited_settings")
OUTCOMES: tuple[str, ...] = ("granted", "denied", "absent")

# F-02 / AC-2c: independent of environment_contract; includes the probe-only
# W2C_PROBE_SENTINEL_CREDENTIAL, which the strip policy deliberately omits.
DEFAULT_SECRET_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "NC_EVAL_ROOT",
    "NC_EVAL_CASE",
    "W2C_PROBE_SENTINEL_CREDENTIAL",
)
CLI_NAME = "narrative-craft"

DEFAULT_PATH_TARGETS: Mapping[str, str] = {
    "filesystem_escape": "blocked/escape-target.txt",
    "source_inspection": "blocked/native-src/cli.py",
    "inherited_settings": "blocked/CLAUDE.md",
}
ABSENT_PATH_TARGETS: Mapping[str, str] = {
    "filesystem_escape": "allowed/absent-escape-target.txt",
    "source_inspection": "allowed/absent-native-src-cli.py",
    "inherited_settings": "allowed/absent-CLAUDE.md",
}
WORLD_ENTITLEMENTS: Mapping[str, Mapping[str, Any]] = {
    "grant": {"read": ["**"], "denied": []},
    "deny": {"read": ["allowed"], "denied": ["blocked"]},
    "absent": {"read": ["**"], "denied": []},
}
SEAM_DEFAULT = "authority_boundary:guard_read"


def world_targets(world: str) -> dict[str, str]:
    """The target rel paths a world observes. W_ABSENT targets do not exist."""
    if world == "absent":
        return dict(ABSENT_PATH_TARGETS)
    return dict(DEFAULT_PATH_TARGETS)


# --------------------------------------------------------------------------
# Child mode: stdlib-only, launched as python -P <this file> --probe
# --------------------------------------------------------------------------
def _observed(present: bool, detail: str, *, outcome: str | None = None, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "present": present,
        "detail": detail,
        "outcome": outcome or ("granted" if present else "denied"),
    }
    entry.update(extra)
    return entry


def _attempt(operation) -> dict[str, Any]:
    try:
        operation()
    except Exception as exc:  # noqa: BLE001
        return _observed(False, f"{type(exc).__name__}: {exc}"[:200])
    return _observed(True, "operation succeeded")


def _raw_read(path: Path) -> str:
    """The honest raw-OS observation of the same target, recorded separately."""
    try:
        path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:200]
    return "readable"


def _observe_path(*, workspace_root: Path, rel: str, entitlement: Any, guard: Any) -> dict[str, Any]:
    """Score one path channel from the shipped seam, in one of the three worlds."""
    target = Path(workspace_root) / rel
    try:
        guard(workspace_root, rel, entitlement)
    except Exception as exc:  # noqa: BLE001
        return _observed(
            False,
            f"{type(exc).__name__}: {exc}"[:200],
            outcome="denied",
            enforcement=f"seam refused rel {rel!r}",
            raw_os=_raw_read(target),
        )
    raw = _raw_read(target)
    if raw == "readable":
        return _observed(
            True,
            "operation succeeded",
            outcome="granted",
            enforcement=f"seam granted rel {rel!r}",
            raw_os=raw,
        )
    return _observed(
        False,
        raw,
        outcome="absent",
        enforcement=f"seam granted rel {rel!r}",
        raw_os=raw,
    )


def _load_seam(spec: Mapping[str, Any]):
    """Import the driver-supplied enforcement seam (default authority_boundary)."""
    import importlib

    src_dir = str(Path(__file__).resolve().parent)
    for entry in [src_dir, *[str(path) for path in spec.get("seam_paths", [])]]:
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    module_name, _, attr = str(spec.get("seam", SEAM_DEFAULT)).partition(":")
    module = importlib.import_module(module_name or "authority_boundary")
    guard = getattr(module, attr or "guard_read")
    entitlement_type = getattr(module, "Entitlement", None)
    if entitlement_type is None:
        from authority_boundary import Entitlement as entitlement_type  # noqa: PLC0415
    return guard, entitlement_type


def _child_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    import importlib.util
    import shutil

    report: dict[str, Any] = {"channels": {}, "cwd": os.getcwd(), "path": os.environ.get("PATH", "")}
    guard, entitlement_type = _load_seam(spec)
    entitlement_data = spec.get("entitlement") or {}
    entitlement = entitlement_type(
        read_allow=tuple(entitlement_data.get("read", ("**",))),
        write_scope=(),
        denied=tuple(entitlement_data.get("denied", ())),
    )
    workspace_root = Path(spec["workspace_root"])
    targets = dict(spec.get("targets") or DEFAULT_PATH_TARGETS)
    for channel in PATH_CHANNELS:
        report["channels"][channel] = _observe_path(
            workspace_root=workspace_root,
            rel=str(targets[channel]),
            entitlement=entitlement,
            guard=guard,
        )

    report["channels"]["cli_execution"] = _attempt(
        lambda: subprocess.run([CLI_NAME, "--version"], capture_output=True, check=True, timeout=20)
    )

    secret_keys = list(spec.get("secret_keys", DEFAULT_SECRET_KEYS))
    upper_secret_keys = {key.upper() for key in secret_keys}
    present_secrets = sorted(
        name for name in os.environ if name in secret_keys or name.upper() in upper_secret_keys
    )
    report["channels"]["environment_secrets"] = _observed(
        bool(present_secrets),
        f"visible secret keys: {present_secrets}" if present_secrets else "no secret key visible",
    )

    discovered: list[str] = []
    if shutil.which(CLI_NAME):
        discovered.append("which:" + CLI_NAME)
    if importlib.util.find_spec("narrative_craft") is not None:
        discovered.append("import:narrative_craft")
    ambient = Path(spec["probe_root"]) / ".mcp.json"
    if ambient.exists():
        try:
            servers = sorted(json.loads(ambient.read_text(encoding="utf-8")).get("mcpServers", {}))
        except Exception:  # noqa: BLE001
            servers = ["unreadable"]
        discovered.extend("mcp:" + name for name in servers)
    report["channels"]["tool_discovery"] = _observed(
        bool(discovered),
        f"discovered {discovered}" if discovered else "only the protected server is reachable",
    )
    return report

# --------------------------------------------------------------------------
# Driver mode
# --------------------------------------------------------------------------
def run_probe(
    *,
    spec: Mapping[str, Any],
    cwd: Path,
    env: Mapping[str, str],
    python_executable: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    python_executable = python_executable or sys.executable
    proc = subprocess.run(
        [python_executable, "-P", str(Path(__file__).resolve()), "--probe", "--spec", "-"],
        input=json.dumps(spec),
        text=True,
        capture_output=True,
        cwd=str(cwd),
        env=dict(env),
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"probe child failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def build_probe_spec(
    *,
    probe_root: str | Path,
    workspace_root: str | Path | None = None,
    entitlement: Mapping[str, Any] | None = None,
    targets: Mapping[str, str] | None = None,
    world: str = "deny",
    seam: str | None = None,
    seam_paths: Sequence[str | Path] | None = None,
    secret_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    probe_root = Path(probe_root)
    workspace_root = Path(workspace_root) if workspace_root is not None else probe_root / "workspace"
    return {
        "probe_root": str(probe_root),
        "workspace_root": str(workspace_root),
        "world": world,
        "entitlement": dict(entitlement if entitlement is not None else WORLD_ENTITLEMENTS["deny"]),
        "targets": dict(targets if targets is not None else world_targets(world)),
        "seam": seam or SEAM_DEFAULT,
        "seam_paths": [
            str(path) for path in (seam_paths if seam_paths is not None else [Path(__file__).resolve().parent])
        ],
        "secret_keys": list(secret_keys if secret_keys is not None else DEFAULT_SECRET_KEYS),
    }


def build_probe_world(probe_root: str | Path, *, open_world: bool = False) -> Path:
    """Build the driver-side disk layout shared by the three worlds.

    The W_GRANT and W_DENY worlds observe the SAME existing blocked/ targets and
    therefore differ only in the entitlement the driver supplies. W_ABSENT
    observes permitted-but-missing allowed/ targets. open_world additionally
    populates the permissive control surface (fake CLI, importable package,
    ambient MCP config, memory file) used by the red control.
    """
    root = Path(probe_root)
    workspace = root / "workspace"
    (workspace / "allowed").mkdir(parents=True, exist_ok=True)
    (workspace / "blocked" / "native-src").mkdir(parents=True, exist_ok=True)
    (workspace / "allowed" / "grant.txt").write_text("granted bytes\n", encoding="utf-8")
    (workspace / "blocked" / "escape-target.txt").write_text("outside-entitlement bytes\n", encoding="utf-8")
    (workspace / "blocked" / "native-src" / "cli.py").write_text("# native source\n", encoding="utf-8")
    (workspace / "blocked" / "CLAUDE.md").write_text("# inherited memory\n", encoding="utf-8")
    if open_world:
        (workspace / ".claude").mkdir(exist_ok=True)
        (workspace / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
        (root / ".mcp.json").write_text('{"mcpServers":{"linear-server":{}}}', encoding="utf-8")
        (workspace / "CLAUDE.md").write_text("# inherited memory\n", encoding="utf-8")
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        fake_cli = bin_dir / CLI_NAME
        fake_cli.write_text("#!/bin/sh" + chr(10) + "echo fake-native 0.8.0" + chr(10), encoding="utf-8")
        fake_cli.chmod(0o755)
        pysrc = root / "pysrc"
        (pysrc / "narrative_craft").mkdir(parents=True, exist_ok=True)
        (pysrc / "narrative_craft" / "__init__.py").write_text("", encoding="utf-8")
    return workspace


def permissive_env(base_env: Mapping[str, str], probe_root: str | Path) -> dict[str, str]:
    probe_root = Path(probe_root)
    env = dict(base_env)
    env["PATH"] = str(probe_root / "bin")
    env["PYTHONPATH"] = str(probe_root / "pysrc")
    env["ANTHROPIC_API_KEY"] = "probe-permissive-secret"
    return env


def _channel_outcome(entry: Mapping[str, Any]) -> str:
    outcome = entry.get("outcome")
    if outcome in OUTCOMES:
        return str(outcome)
    return "granted" if entry.get("present", False) else "denied"


def denied_channels(report: Mapping[str, Any]) -> list[str]:
    channels = report.get("channels", {})
    return [name for name in PROBE_CHANNELS if _channel_outcome(channels.get(name, {})) == "denied"]


def present_channels(report: Mapping[str, Any]) -> list[str]:
    channels = report.get("channels", {})
    return [name for name in PROBE_CHANNELS if _channel_outcome(channels.get(name, {})) == "granted"]


def assert_report_complete(report: Mapping[str, Any]) -> None:
    channels = report.get("channels", {})
    missing = sorted(set(PROBE_CHANNELS) - set(channels))
    if missing:
        raise RuntimeError(f"probe report omits channels {missing}")
    for name in PROBE_CHANNELS:
        entry = channels[name]
        if not isinstance(entry, Mapping) or "present" not in entry or "detail" not in entry:
            raise RuntimeError(f"probe channel {name} is not a typed observation")
        outcome = entry.get("outcome")
        if outcome not in OUTCOMES:
            raise RuntimeError(f"probe channel {name} has no three-world outcome: {outcome!r}")
        if bool(entry["present"]) != (outcome == "granted"):
            raise RuntimeError(
                f"probe channel {name} outcome {outcome!r} disagrees with present={entry['present']!r}"
            )
        if entry["present"] and not str(entry["detail"]).strip():
            raise RuntimeError(f"probe channel {name} claims presence with no evidence")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--probe" not in argv:
        print("capability_probe is a driver-side probe; --probe is required", file=sys.stderr)
        return 2
    spec: dict[str, Any]
    if "--spec" in argv and argv[argv.index("--spec") + 1] == "-":
        spec = json.loads(sys.stdin.read())
    else:
        spec_path = argv[argv.index("--spec") + 1]
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    report = _child_report(spec)
    sys.stdout.write(json.dumps(report, sort_keys=True) + chr(10))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
