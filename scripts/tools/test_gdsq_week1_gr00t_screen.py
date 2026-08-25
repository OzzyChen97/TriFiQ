from __future__ import annotations

import pytest

import gdsq_week1_gr00t_screen as screen


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/tmp/mask_00.plan.json", 0),
        ("candidate_09.plan.json", 9),
        ("candidate_25.plan.json", 25),
    ],
)
def test_candidate_index_from_double_suffix(path: str, expected: int) -> None:
    assert screen.candidate_index_from_path(path) == expected


@pytest.mark.parametrize(
    "path",
    ["mask_00.json", "mask_x.plan.json", "mask_00.plan.json.bak"],
)
def test_candidate_index_rejects_malformed_names(path: str) -> None:
    with pytest.raises(ValueError, match="candidate"):
        screen.candidate_index_from_path(path)
