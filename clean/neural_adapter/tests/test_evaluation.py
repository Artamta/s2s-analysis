import numpy as np

from fuxi_adapter.evaluation import season_labels


def test_season_uses_week_midpoint():
    dates = np.asarray(
        [[
            np.arange("2024-02-27", "2024-03-05", dtype="datetime64[D]"),
            np.arange("2024-06-01", "2024-06-08", dtype="datetime64[D]"),
        ]]
    )
    assert season_labels(dates).tolist() == [["MAM", "JJA"]]

