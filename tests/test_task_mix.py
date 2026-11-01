"""The declared task mix must agree with the code and the frozen task set, or the study is mispriced.

`studies/revision-v1.yaml` now carries `task_mix`, and the reservation ceiling is derived from it. A
declaration that has drifted from `TASK_TOKEN_CONFIG` or from the task counts would still look
plausible and would silently misprice every arm-run — in whichever direction the drift happened to go.
These controls make the drift fail here instead.
"""
import glob
import pathlib

import pytest
import yaml

from llm_client import TASK_TOKEN_CONFIG

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mix():
    study = yaml.safe_load((ROOT / "studies" / "revision-v1.yaml").read_text())
    assert "task_mix" in study, "the study no longer declares a task mix"
    return study["task_mix"]


def test_every_declared_max_tokens_matches_the_code(mix):
    for family, declared in mix["max_tokens_by_family"].items():
        assert TASK_TOKEN_CONFIG[family]["max_tokens"] == declared, family


def test_every_declared_family_is_a_code_family(mix):
    assert set(mix["families"]) <= set(TASK_TOKEN_CONFIG), set(mix["families"]) - set(TASK_TOKEN_CONFIG)


def test_every_weight_is_the_frozen_task_count(mix):
    for family, weight in mix["families"].items():
        counted = len(glob.glob(str(ROOT / "dataset" / "tasks" / family / "*.yaml")))
        assert counted == weight, f"{family}: declared {weight}, {counted} task file(s) on disk"


def test_no_family_is_both_declared_and_excluded(mix):
    assert not set(mix["families"]) & set(mix["excluded_multicall"])


def test_every_code_family_is_either_declared_or_explicitly_excluded(mix):
    """The dangerous omission is a family that is silently in neither list."""
    neither = set(TASK_TOKEN_CONFIG) - set(mix["families"]) - set(mix["excluded_multicall"])
    assert not neither, f"unpriced and unexcluded: {sorted(neither)}"


def test_the_agentic_families_are_the_ones_excluded(mix):
    agentic = {name for name in TASK_TOKEN_CONFIG if name.startswith("agentic_")}
    assert agentic == set(mix["excluded_multicall"])


def test_the_mix_ceiling_is_below_the_longest_family():
    """A mix that priced at or above the longest family would be decorative."""
    mix = yaml.safe_load((ROOT / "studies" / "revision-v1.yaml").read_text())["task_mix"]
    weights = mix["families"]
    mean_out = sum(mix["max_tokens_by_family"][f] * w for f, w in weights.items()) / sum(weights.values())
    longest = max(mix["max_tokens_by_family"].values())
    assert mean_out < longest, f"mean {mean_out} not below longest {longest}"
