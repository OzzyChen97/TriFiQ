from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_budget_projection_comparison import exact_ranked_removals, stable_write


def test_exact_ranked_removals_matches_exhaustive_priority():
    order = ["a", "b", "c", "d", "e"]
    costs = dict(zip(order, [6, 4, 3, 2, 1]))
    for target in range(sum(costs.values()) + 1):
        choices = [bits for bits in itertools.product((0, 1), repeat=len(order))
                   if sum(costs[n] * b for n, b in zip(order, bits)) == target]
        if not choices:
            with pytest.raises(ValueError):
                exact_ranked_removals(order, costs, target)
            continue
        best = max(choices)
        assert exact_ranked_removals(order, costs, target) == [n for n, b in zip(order, best) if b]


def test_matching_skips_preferred_removal_when_suffix_infeasible():
    assert exact_ranked_removals(["a", "b", "c"], {"a": 5, "b": 3, "c": 3}, 6) == ["b", "c"]


@pytest.mark.parametrize("order,costs,target", [
    (["a", "a"], {"a": 1}, 1),
    (["a"], {"a": -1}, 0),
    (["a"], {"a": 2}, 1),
    (["a"], {"a": 2}, 4),
    ([], {}, 1),
    (["a"], {"a": 2}, -1),
])
def test_invalid_matching_rejected(order, costs, target):
    with pytest.raises(ValueError):
        exact_ranked_removals(order, costs, target)


def test_empty_and_zero_target():
    assert exact_ranked_removals([], {}, 0) == []
    assert exact_ranked_removals(["a"], {"a": 3}, 0) == []


def test_immutable_write_and_check(tmp_path):
    path = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError):
        stable_write(path, {"value": 1}, check=True)
    stable_write(path, {"value": 1}, check=False)
    original = path.read_bytes()
    stable_write(path, {"value": 1}, check=True)
    with pytest.raises(ValueError):
        stable_write(path, {"value": 2}, check=False)
    assert path.read_bytes() == original
    assert json.loads(original) == {"value": 1}