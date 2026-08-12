"""Unit tests for storage-free v3 auxiliary-feature assembly."""

import json
from pathlib import Path

import numpy as np
import pytest

from fuxi_adapter.data import (
    FuxiTPDistributionData,
    FuxiTPDistributionSplitArrays,
    ModelArrays,
)
from fuxi_adapter.v3_workflow import (
    _append_tp_distribution_features,
    _use_tp_distribution_features,
)


def _model_arrays(initializations: np.ndarray, valid_mask: np.ndarray) -> ModelArrays:
    sample_count = initializations.size
    target = np.zeros((sample_count, 6, 2, 2), dtype=np.float32)
    return ModelArrays(
        inputs=np.zeros((sample_count, 6, 2, 2, 2), dtype=np.float32),
        target=target,
        mask=valid_mask.copy(),
        weight=valid_mask.astype(np.float32),
        initializations=initializations.copy(),
        channel_names=("existing_feature", "fuxi_t2m_mean_weekly"),
    )


def _distribution_split(
    name: str,
    initializations: np.ndarray,
    base_values: np.ndarray,
) -> FuxiTPDistributionSplitArrays:
    return FuxiTPDistributionSplitArrays(
        name=name,
        initializations=initializations.copy(),
        member_log_median_anomaly=base_values.astype(np.float32),
        member_log_iqr=(base_values + 10.0).astype(np.float32),
        probability_exceeds_imerg_climatology=(0.1 + 0.005 * base_values).astype(
            np.float32
        ),
    )


def _distribution_data(
    train_initializations: np.ndarray,
    validation_initializations: np.ndarray,
) -> FuxiTPDistributionData:
    lead_offset = 10.0 * np.arange(6, dtype=np.float32)[None, :, None, None]
    train_case = np.asarray(
        [
            [[[0.0, 2.0], [100.0, 100.0]]],
            [[[4.0, 6.0], [100.0, 100.0]]],
        ],
        dtype=np.float32,
    )
    train_values = np.broadcast_to(train_case, (2, 6, 2, 2)) + lead_offset
    validation_values = np.full((1, 6, 2, 2), 1000.0, dtype=np.float32)
    test_initializations = np.asarray(["2024-01-01"], dtype="datetime64[D]")
    test_values = np.zeros((1, 6, 2, 2), dtype=np.float32)
    return FuxiTPDistributionData(
        train=_distribution_split(
            "train", train_initializations, train_values
        ),
        validation=_distribution_split(
            "validation", validation_initializations, validation_values
        ),
        test=_distribution_split("test", test_initializations, test_values),
        latitude=np.asarray([1.0, 0.0]),
        longitude=np.asarray([70.0, 71.0]),
        climatology_threshold_support=np.asarray(
            [[True, True], [False, False]], dtype=bool
        ),
        source_manifest={"feature_contract": "synthetic"},
    )


def test_distribution_features_append_after_t2m_with_train_only_scaling() -> None:
    train_initializations = np.asarray(
        ["2020-01-02", "2020-01-06"], dtype="datetime64[D]"
    )
    validation_initializations = np.asarray(
        ["2023-01-02"], dtype="datetime64[D]"
    )
    train_mask = np.broadcast_to(
        np.asarray([[True, True], [False, False]])[None, None],
        (2, 6, 2, 2),
    ).copy()
    validation_mask = np.broadcast_to(
        np.asarray([[True, True], [False, False]])[None, None],
        (1, 6, 2, 2),
    ).copy()
    train_arrays = _model_arrays(train_initializations, train_mask)
    validation_arrays = _model_arrays(validation_initializations, validation_mask)
    distribution = _distribution_data(
        train_initializations, validation_initializations
    )
    area_weights = np.asarray([[1.0, 3.0], [0.0, 0.0]], dtype=np.float64)

    train_result, validation_result, normalization = (
        _append_tp_distribution_features(
            train_arrays,
            validation_arrays,
            distribution,
            train_mask,
            validation_mask,
            area_weights,
        )
    )

    expected_names = (
        "existing_feature",
        "fuxi_t2m_mean_weekly",
        "member_log_median_anomaly",
        "member_log_iqr",
        "probability_exceeds_imerg_climatology",
    )
    assert train_result.channel_names == expected_names
    assert validation_result.channel_names == expected_names
    assert train_result.inputs.shape == (2, 6, 5, 2, 2)
    assert validation_result.inputs.shape == (1, 6, 5, 2, 2)
    assert train_result.inputs.dtype == np.float32
    np.testing.assert_array_equal(train_result.inputs[:, :, :2], train_arrays.inputs)
    np.testing.assert_array_equal(
        validation_result.inputs[:, :, :2], validation_arrays.inputs
    )
    # Unsupported cells are the normalized mean (zero), not transformed fill.
    assert np.all(train_result.inputs[:, :, 2:, 1, :] == 0.0)
    assert np.all(validation_result.inputs[:, :, 2:, 1, :] == 0.0)

    median_record = normalization["features"]["member_log_median_anomaly"]
    expected_mean = 3.5 + 10.0 * np.arange(6, dtype=np.float32)
    np.testing.assert_allclose(median_record["mean_by_lead"], expected_mean)
    # Extreme validation values cannot alter train-fitted moments.
    assert median_record["mean_by_lead"][0] == 3.5
    assert normalization["feature_order"] == list(expected_names[-3:])
    assert normalization["fitted_split"] == "train"


def test_distribution_feature_assembly_rejects_initialization_drift() -> None:
    train_initializations = np.asarray(
        ["2020-01-02", "2020-01-06"], dtype="datetime64[D]"
    )
    validation_initializations = np.asarray(
        ["2023-01-02"], dtype="datetime64[D]"
    )
    mask_train = np.ones((2, 6, 2, 2), dtype=bool)
    mask_validation = np.ones((1, 6, 2, 2), dtype=bool)
    train_arrays = _model_arrays(train_initializations, mask_train)
    validation_arrays = _model_arrays(validation_initializations, mask_validation)
    distribution = _distribution_data(
        train_initializations, validation_initializations
    )
    distribution.train.initializations[0] = np.datetime64("2020-01-09")

    with pytest.raises(ValueError, match="training initializations are misaligned"):
        _append_tp_distribution_features(
            train_arrays,
            validation_arrays,
            distribution,
            mask_train,
            mask_validation,
            np.ones((2, 2), dtype=np.float64),
        )


def test_distribution_flag_defaults_false_and_requires_boolean() -> None:
    assert _use_tp_distribution_features({}) is False
    assert _use_tp_distribution_features({"use_tp_distribution_features": True})
    with pytest.raises(ValueError, match="must be a boolean"):
        _use_tp_distribution_features({"use_tp_distribution_features": "true"})


def test_distribution_config_changes_only_experiment_and_input_contract() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    hybrid = json.loads(
        (config_root / "fuxi_imerg_late_acc_v3_hybrid.json").read_text()
    )
    distribution = json.loads(
        (config_root / "fuxi_imerg_late_acc_v3_distribution.json").read_text()
    )
    assert distribution["use_tp_distribution_features"] is True
    assert distribution["experiment_name"] == "fuxi_imerg_late_acc_v3_distribution"
    assert distribution["input_contract"] != hybrid["input_contract"]
    for key in ("experiment_name", "input_contract", "use_tp_distribution_features"):
        hybrid.pop(key, None)
        distribution.pop(key, None)
    assert distribution == hybrid
