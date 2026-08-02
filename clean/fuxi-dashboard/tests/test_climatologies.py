"""Tests for sampling rules and unexchangeable climatology identities."""

from __future__ import annotations

import numpy as np
import pytest

from science.contracts import BASELINES
from science.formulas import calendar_interpolation, climatology_spread
from science.validators import validate_baseline_identity
from scripts.build_prototype_data import climatology_bracket


def test_baselines_are_distinct_and_locked() -> None:
    assert BASELINES == {
        "fuxi": "2002-2021",
        "imd": "1991-2020",
        "imerg": "2001-2022",
    }
    for source, baseline in BASELINES.items():
        validate_baseline_identity(source, baseline)


@pytest.mark.parametrize(
    ("source", "incorrect"),
    [
        ("fuxi", "1991-2020"),
        ("imd", "2001-2022"),
        ("imerg", "1991-2020"),
    ],
)
def test_imd_imerg_and_fuxi_baselines_cannot_be_exchanged(
    source: str, incorrect: str
) -> None:
    with pytest.raises(ValueError, match="requires baseline"):
        validate_baseline_identity(source, incorrect)


def test_twenty_years_receive_equal_weight_after_member_mean() -> None:
    yearly_member_means = np.arange(1.0, 21.0)
    assert yearly_member_means.mean() == pytest.approx(10.5)
    assert climatology_spread(yearly_member_means) == pytest.approx(
        np.std(yearly_member_means, ddof=1)
    )


def test_july_27_alignment_uses_documented_slots_and_weight() -> None:
    slots = np.asarray(["0721", "0725", "0728", "0801"])
    left, right, weight = climatology_bracket(slots, "0727")
    assert (left, right) == ("0725", "0728")
    assert weight == pytest.approx(2.0 / 3.0)
    np.testing.assert_allclose(
        calendar_interpolation([1.0], [4.0], weight), [3.0]
    )


def test_exact_calendar_slot_needs_no_interpolation() -> None:
    assert climatology_bracket(np.asarray(["0725", "0728"]), "0728") == (
        "0728",
        "0728",
        0.0,
    )
