#!/usr/bin/env python3
"""Story continuity repair — command line for the GRPO harness.

    python scripts/story_rl.py estimate                 the spend bound, no side effect
    python scripts/story_rl.py train                    the paid run (needs an approval figure)
    python scripts/story_rl.py sample --plan            what sampling would send and cost
    python scripts/story_rl.py sample --out base.jsonl  sample the base model
    python scripts/story_rl.py evaluate --generations trained.jsonl --against base.jsonl
    python scripts/story_rl.py skip-rate --run ~/tinker-runs/story_repair_grpo

Every subcommand is a thin call into `src/story_rl/`; the library holds the behaviour, this file holds
the argument parsing. `sys.path` is extended the same way the repo's other scripts do it, because the
package is not installed — `pythonpath = ["src"]` covers pytest, and this line covers the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SUBCOMMANDS = {
    "estimate": ("story_rl.train", "print the spend bound for the default configuration"),
    "train": ("story_rl.train", "run GRPO against the repair reward"),
    "sample": ("story_rl.sample", "sample the base model or a checkpoint"),
    "evaluate": ("story_rl.evaluate", "score generations, optionally paired against a baseline"),
    "skip-rate": ("story_rl.skip_rate", "the skip rate the training metrics cannot report"),
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        print("subcommands:")
        for name, (_module, description) in SUBCOMMANDS.items():
            print(f"  {name:10s} {description}")
        return 0 if args else 2

    name, rest = args[0], args[1:]
    if name not in SUBCOMMANDS:
        raise SystemExit(f"unknown subcommand {name!r}; try --help")

    module_name, _description = SUBCOMMANDS[name]
    module = __import__(module_name, fromlist=["main"])
    if name == "estimate":
        rest = ["--estimate", *rest]
    return module.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
