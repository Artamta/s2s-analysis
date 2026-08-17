import numpy as np

from india_s2s_bench.metrics import case_metrics, spatial_acc


def test_spatial_acc_is_area_weighted_and_centered():
    truth = np.array([[1.0, 2.0], [4.0, 8.0]])
    prediction = 3.0 * truth + 10.0
    weight = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(spatial_acc(prediction, truth, weight), 1.0)


def test_case_metrics_absolute_errors_and_shared_anomalies():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = truth + 1.0
    climatology = np.zeros_like(truth)
    weight = np.ones_like(truth)
    result = case_metrics(prediction, truth, climatology, weight, wet_threshold=1.0)
    assert np.isclose(result["acc"], 1.0)
    assert np.isclose(result["rmse"], 1.0)
    assert np.isclose(result["mae"], 1.0)
    assert np.isclose(result["bias"], 1.0)
