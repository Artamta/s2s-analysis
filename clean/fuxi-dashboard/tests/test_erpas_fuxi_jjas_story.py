from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from scripts.build_erpas_fuxi_jjas_story import (
    mean_selected_members,
    validate_remapped_support,
)


def test_remapped_validation_allows_nan_outside_india_support() -> None:
    values = np.asarray([[[1.0, np.nan], [2.0, np.nan]]], dtype=float)
    support = np.asarray([[True, False], [True, False]])

    validate_remapped_support(values, support, "forecast")


def test_remapped_validation_rejects_nan_inside_india_support() -> None:
    values = np.asarray([[[1.0, np.nan], [2.0, np.nan]]], dtype=float)
    support = np.asarray([[True, True], [True, False]])

    with pytest.raises(ValueError, match="non-finite on verified India support"):
        validate_remapped_support(values, support, "forecast")


def test_member_mean_uses_only_requested_leading_subset() -> None:
    values = xr.DataArray(
        np.arange(10, dtype=float).reshape(5, 2), dims=("member", "lead_day")
    )

    selected = mean_selected_members(values, 2)

    np.testing.assert_allclose(selected.values, [1.0, 2.0])


def test_member_mean_rejects_oversubscription() -> None:
    values = xr.DataArray(np.ones((5, 2)), dims=("member", "lead_day"))

    with pytest.raises(ValueError, match="requested 11 members"):
        mean_selected_members(values, 11)
