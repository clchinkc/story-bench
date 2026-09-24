"""Driver-side blocked-capability probes (shared-interface.md S14, F-07 / N-04).

The probe performs REAL child operations (open, spawn, environment read, import,
tool discovery) in a child process launched with the participant's exact
sanitized environment and cwd discipline. It does not run inside the participant
process, because a tools-empty participant has no capability to spawn or open.

Every channel has a permissive red control: the same probe against a
deliberately granted world must observe the capability as PRESENT, proving the
probe is behavioral and can detect a grant (risk rows P1, P2).

Honest limit: W2c is runtime discipline plus explicit denial, not a kernel
sandbox (see docs/environment-contract.md).
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
DEFAULT_SECRET_KEYS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
CLI_NAME = "narrative-craft"

# --------------------------------------------------------------------------
# Child mode: stdlib-only, launched as python -P <this file> --probe
# --------------------------------------------------------------------------
def _attempt(operation) -> dict[str, Any]:
    try:
        operation()
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "detail": f"{type(exc).__name__}: {exc}"[:200]}
    return {"present": True, "detail": "operation succeeded"}


def _child_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    import importlib.util
    import shutil

    root = Path(spec["probe_root"])
    report: dict[str, Any] = {"channels": {}, "cwd": os.getcwd(), "path": os.environ.get("PATH", "")}

    report["channels"]["filesystem_escape"] = _attempt(lambda: Path(spec["escape_path"]).read_bytes())
    report["channels"]["source_inspection"] = _attempt(lambda: Path(spec["source_path"]).read_bytes())
    report["channels"]["cli_execution"] = _attempt(
        lambda: subprocess.run([CLI_NAME, "--version"], capture_output=True, check=True, timeout=20)
    )

    secret_keys = list(spec.get("secret_keys", DEFAULT_SECRET_KEYS))
    present_secrets = sorted(key for key in secret_keys if os.environ.get(key))
    report["channels"]["environment_secrets"] = {
        "present": bool(present_secrets),
        "detail": f"visible secret keys: {present_secrets}" if present_secrets else "no secret key visible",
    }

    settings_candidates = [
        root / ".claude" / "settings.json",
        root / ".claude" / "settings.local.json",
        root / ".mcp.json",
        root / "CLAUDE.md",
        root / "AGENTS.md",
        Path(os.getcwd()) / "CLAUDE.md",
        Path(os.getcwd()) / "AGENTS.md",
    ]
    found = sorted(str(path) for path in settings_candidates if path.exists())
    report["channels"]["inherited_settings"] = {
        "present": bool(found),
        "detail": f"found {found}" if found else "no project setting source or memory file reachable",
    }

    discovered: list[str] = []
    if shutil.which(CLI_NAME):
        discovered.append("which:" + CLI_NAME)
    if importlib.util.find_spec("narrative_craft") is not None:
        discovered.append("import:narrative_craft")
    ambient = root / ".mcp.json"
    if ambient.exists():
        try:
            servers = sorted(json.loads(ambient.read_text(encoding="utf-8")).get("mcpServers", {}))
        except Exception:  # noqa: BLE001
            servers = ["unreadable"]
        discovered.extend("mcp:" + name for name in servers)
    report["channels"]["tool_discovery"] = {
        "present": bool(discovered),
        "detail": f"discovered {discovered}" if discovered else "only the protected server is reachable",
    }
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


def build_probe_spec(*, probe_root: str | Path) -> dict[str, Any]:
    probe_root = Path(probe_root)
    return {
        "probe_root": str(probe_root),
        "escape_path": str(probe_root / "escape-target.txt"),
        "source_path": str(probe_root / "native-src" / "cli.py"),
        "secret_keys": list(DEFAULT_SECRET_KEYS),
    }


def grant_permissive_world(probe_root: str | Path) -> Path:
    """Populate a deliberately permissive world for the red control."""
    probe_root = Path(probe_root)
    (probe_root / "native-src").mkdir(parents=True, exist_ok=True)
    (probe_root / "escape-target.txt").write_text("reachable outside the entitlement", encoding="utf-8")
    (probe_root / "native-src" / "cli.py").write_text("# native source", encoding="utf-8")
    (probe_root / ".claude").mkdir(exist_ok=True)
    (probe_root / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    (probe_root / ".mcp.json").write_text('{"mcpServers":{"linear-server":{}}}', encoding="utf-8")
    (probe_root / "CLAUDE.md").write_text("# inherited memory", encoding="utf-8")
    bin_dir = probe_root / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_cli = bin_dir / CLI_NAME
    fake_cli.write_text("#!/bin/sh" + chr(10) + "echo fake-native 0.8.0" + chr(10), encoding="utf-8")
    fake_cli.chmod(0o755)
    pysrc = probe_root / "pysrc"
    (pysrc / "narrative_craft").mkdir(parents=True, exist_ok=True)
    (pysrc / "narrative_craft" / "__init__.py").write_text("", encoding="utf-8")
    return probe_root


def permissive_env(base_env: Mapping[str, str], probe_root: str | Path) -> dict[str, str]:
    probe_root = Path(probe_root)
    env = dict(base_env)
    env["PATH"] = str(probe_root / "bin")
    env["PYTHONPATH"] = str(probe_root / "pysrc")
    env["ANTHROPIC_API_KEY"] = "probe-permissive-secret"
    return env


def denied_channels(report: Mapping[str, Any]) -> list[str]:
    return [name for name in PROBE_CHANNELS if not report.get("channels", {}).get(name, {}).get("present", False)]


def present_channels(report: Mapping[str, Any]) -> list[str]:
    return [name for name in PROBE_CHANNELS if report.get("channels", {}).get(name, {}).get("present", False)]


def assert_report_complete(report: Mapping[str, Any]) -> None:
    channels = report.get("channels", {})
    missing = sorted(set(PROBE_CHANNELS) - set(channels))
    if missing:
        raise RuntimeError(f"probe report omits channels {missing}")
    for name in PROBE_CHANNELS:
        entry = channels[name]
        if not isinstance(entry, Mapping) or "present" not in entry or "detail" not in entry:
            raise RuntimeError(f"probe channel {name} is not a typed observation")
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


