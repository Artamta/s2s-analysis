import numpy as np

from fuxi_adapter.baselines import (
    apply_log_bias_correction,
    fit_log_bias_correction,
    verification_midpoint_months,
)


def test_midpoint_months_follow_lead_windows():
    # Week 1 covers period starts 26 Feb--3 Mar; its fourth period starts on
    # 29 February.  An erroneous period-end/+4 convention would return March.
    dates = np.asarray(["2024-02-26"], dtype="datetime64[D]")
    assert verification_midpoint_months(dates, 2).tolist() == [[2, 3]]


def test_log_bias_recovers_constant_multiplicative_error():
    dates = np.arange("2020-01-02", "2021-01-01", 14, dtype="datetime64[D]")
    fuxi = np.ones((len(dates), 2, 3, 3), dtype=np.float32)
    truth = np.full_like(fuxi, 3.0)
    mask = np.ones((3, 3), dtype=bool)
    fitted = fit_log_bias_correction(fuxi, truth, dates, mask, shrinkage=0.0)
    corrected = apply_log_bias_correction(fuxi, dates, fitted)
    np.testing.assert_allclose(corrected, truth, rtol=1e-6, atol=1e-6)


def test_outside_mask_does_not_affect_fit():
    dates = np.asarray(["2020-06-01", "2020-06-15"], dtype="datetime64[D]")
    fuxi = np.ones((2, 1, 2, 2), dtype=np.float32)
    truth = np.full_like(fuxi, 2.0)
    truth[..., 0, 0] = 999.0
    mask = np.asarray([[False, True], [True, True]])
    fitted = fit_log_bias_correction(fuxi, truth, dates, mask, shrinkage=1.0)
    assert np.isnan(fitted.lead_month_residual[..., 0, 0]).all()
