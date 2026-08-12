"""Smoke and validation tests for non-interactive evaluation plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fuxi_adapter.plotting import (
    plot_mean_maps,
    plot_metric_by_lead,
    plot_training_history,
)


def _assert_written_and_closed(path, figures_before):
    assert path.is_file()
    assert path.stat().st_size > 1_000
    assert plt.get_fignums() == figures_before


def test_plot_metric_by_lead_writes_wide_summary(tmp_path):
    summary = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "region": ["india"] * 6,
            "model": ["raw_fuxi"] * 3 + ["residual_unet"] * 3,
            "lead_week": [1, 2, 3, 1, 2, 3],
            "acc": [0.6, 0.4, 0.2, 0.65, 0.46, 0.28],
        }
    )
    destination = tmp_path / "nested" / "acc_by_lead.png"
    figures_before = plt.get_fignums()

    result = plot_metric_by_lead(summary, "acc", destination)

    assert result == destination
    _assert_written_and_closed(destination, figures_before)


def test_plot_metric_by_lead_accepts_tidy_summary(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["raw_fuxi", "adapter"],
            "week": [1, 1],
            "metric": ["rmse", "rmse"],
            "value": [3.4, 3.1],
        }
    )
    destination = tmp_path / "rmse.png"

    plot_metric_by_lead(summary, "RMSE", destination)

    assert destination.stat().st_size > 1_000


def test_plot_training_history_writes_and_closes_figure(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4],
            "train_loss": [1.0, 0.7, 0.5, 0.4],
            "validation_loss": [1.1, 0.8, 0.6, 0.65],
        }
    )
    destination = tmp_path / "history.png"
    figures_before = plt.get_fignums()

    plot_training_history(history, destination)

    _assert_written_and_closed(destination, figures_before)


@pytest.mark.parametrize("lead_index", [None, 2])
def test_plot_mean_maps_supports_all_leads_and_one_lead(tmp_path, lead_index):
    generator = np.random.default_rng(4)
    predictions = {
        "raw_fuxi": generator.gamma(2.0, 1.0, size=(5, 6, 7, 8)),
        "corrected": generator.gamma(2.2, 1.0, size=(5, 6, 7, 8)),
        "imerg_truth": generator.gamma(2.1, 1.0, size=(5, 6, 7, 8)),
    }
    latitude = np.linspace(7.0, 35.0, 7)
    longitude = np.linspace(67.0, 98.0, 8)
    mask = np.ones((7, 8), dtype=bool)
    mask[0, :2] = False
    destination = tmp_path / f"maps_{lead_index}.png"
    figures_before = plt.get_fignums()

    plot_mean_maps(
        predictions,
        latitude,
        longitude,
        mask,
        destination,
        lead_index=lead_index,
    )

    _assert_written_and_closed(destination, figures_before)


def test_plot_mean_maps_validates_shapes(tmp_path):
    with pytest.raises(ValueError, match="same shape"):
        plot_mean_maps(
            {
                "a": np.ones((2, 6, 4, 5)),
                "b": np.ones((2, 5, 4, 5)),
            },
            np.arange(4),
            np.arange(5),
            np.ones((4, 5), dtype=bool),
            tmp_path / "unused.png",
        )
