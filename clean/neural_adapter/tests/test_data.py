"""Tests for the leakage-safe FuXi--IMERG data contract."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from fuxi_adapter.data import (
    DataIntegrityError,
    DataPaths,
    calendar_positions_for_dates,
    collapse_observation_fraction,
    derive_valid_dates,
    fit_normalization,
    load_adapter_data,
    load_fuxi_t2m_data,
    load_fuxi_tp_distribution_data,
    make_model_arrays,
    purged_split_indices,
    reconstruct_precipitation,
)


def test_valid_dates_are_six_complete_weeks() -> None:
    initializations = np.asarray(["2024-12-30"], dtype="datetime64[D]")
    dates = derive_valid_dates(initializations)
    assert dates.shape == (1, 6, 7)
    # Dates label IMERG period starts.  The corresponding FuXi period-end
    # labels are each one day later.
    assert dates[0, 0, 0] == np.datetime64("2024-12-30")
    assert dates[0, 0, -1] == np.datetime64("2025-01-05")
    assert dates[0, -1, -1] == np.datetime64("2025-02-09")


def test_observation_fraction_accepts_2d_and_identical_repeats() -> None:
    fraction = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
    assert np.array_equal(
        collapse_observation_fraction(fraction, (2, 2)), fraction
    )
    repeated_first = np.repeat(fraction[None, :, :], 3, axis=0)
    repeated_last = np.repeat(fraction[:, :, None], 3, axis=2)
    assert np.array_equal(
        collapse_observation_fraction(repeated_first, (2, 2)), fraction
    )
    assert np.array_equal(
        collapse_observation_fraction(repeated_last, (2, 2)), fraction
    )


def test_observation_fraction_rejects_time_varying_values() -> None:
    values = np.ones((3, 2, 2), dtype=np.float32)
    values[2, 0, 0] = 0.5
    with pytest.raises(DataIntegrityError, match="changes"):
        collapse_observation_fraction(values, (2, 2))


def test_climatology_calendar_handles_leap_day() -> None:
    climate_dates = np.arange(
        np.datetime64("2000-01-01"), np.datetime64("2001-01-01")
    )
    dates = np.asarray(
        [[np.datetime64("2023-02-28"), np.datetime64("2024-02-29")]]
    )
    positions = calendar_positions_for_dates(dates, climate_dates)
    assert positions.shape == dates.shape
    assert positions.tolist() == [[58, 59]]


def _twice_weekly_dates(year: int) -> np.ndarray:
    days = np.arange(
        np.datetime64("%04d-01-01" % year),
        np.datetime64("%04d-01-01" % (year + 1)),
    )
    # 1970-01-01 was Thursday (weekday 3 with Monday=0).
    weekday = (days.astype(np.int64) + 3) % 7
    return days[(weekday == 0) | (weekday == 3)]


def test_purged_split_has_frozen_counts_and_no_boundary_overlap() -> None:
    dates = np.concatenate([_twice_weekly_dates(year) for year in range(2020, 2025)])
    missing_2024 = np.asarray(
        ["2024-06-06", "2024-06-10", "2024-06-13", "2024-06-17", "2024-08-05"],
        dtype="datetime64[D]",
    )
    dates = dates[~np.isin(dates, missing_2024)]
    split = purged_split_indices(dates)
    assert {name: len(index) for name, index in split.items()} == {
        "train": 302,
        "validation": 93,
        "test": 100,
    }
    assert dates[split["train"]][-1] == np.datetime64("2022-11-21")
    assert dates[split["train"]][-1] + np.timedelta64(42, "D") <= dates[
        split["validation"]
    ][0]
    assert dates[split["validation"]][-1] == np.datetime64("2023-11-20")
    assert dates[split["validation"]][-1] + np.timedelta64(42, "D") <= dates[
        split["test"]
    ][0]


def _adapter_stub(
    latitude: np.ndarray,
    longitude: np.ndarray,
    train: tuple[str, ...] = ("2020-01-02", "2020-01-06"),
    validation: tuple[str, ...] = ("2023-01-02",),
    test: tuple[str, ...] = ("2024-01-01",),
    climatology_values: tuple[float, float, float] = (24.5, 39.5, 9.5),
) -> SimpleNamespace:
    split_dates = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    support = np.ones((27, 27), dtype=np.float32)
    support[0, 0] = 0.0
    splits = {}
    for (name, dates), climatology_value in zip(
        split_dates.items(), climatology_values
    ):
        climatology = np.full(
            (len(dates), 6, 27, 27), climatology_value, dtype=np.float32
        )
        climatology[..., 0, 0] = np.nan
        splits[name] = SimpleNamespace(
            initializations=np.asarray(dates, dtype="datetime64[D]"),
            imerg_climatology=climatology,
        )
    climatology_store = "/synthetic/imerg_2001_2019.zarr"
    return SimpleNamespace(
        splits=splits,
        latitude=np.asarray(latitude),
        longitude=np.asarray(longitude),
        observation_fraction=support,
        audit=SimpleNamespace(
            climatology_baseline="2001-2019", climatology_day_count=366
        ),
        source_manifest={
            "imerg_climatology_store": climatology_store,
            "zmetadata_sha256": {climatology_store: "c" * 64},
        },
    )


def _t2m_dataset(
    dates: tuple[str, ...],
    latitude: np.ndarray,
    longitude: np.ndarray,
    value_offset: float,
) -> xr.Dataset:
    initializations = np.asarray(dates, dtype="datetime64[ns]")
    shape = (len(initializations), 6, 27, 27)
    case_value = value_offset + np.arange(len(initializations), dtype=np.float32)
    mean = np.broadcast_to(case_value[:, None, None, None], shape).copy()
    spread = np.full(shape, 1.25, dtype=np.float32)
    member_count = np.full(shape, 50, dtype=np.int16)
    field_dims = ("init", "lead_week", "latitude", "longitude")
    return xr.Dataset(
        data_vars={
            "ensemble_mean_weekly": (
                field_dims,
                mean,
                {
                    "units": "degC",
                    "temporal_statistic": "mean_of_complete_7_day_block",
                },
            ),
            "ensemble_std_weekly": (
                field_dims,
                spread,
                {
                    "units": "degC",
                    "temporal_statistic": "mean_of_complete_7_day_block",
                },
            ),
            "ensemble_member_count_weekly": (field_dims, member_count),
        },
        coords={
            "init": initializations,
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": latitude,
            "longitude": longitude,
            "member": np.arange(50, dtype=np.int16),
        },
        attrs={
            "archive_id": "india_s2s_benchmark_v1",
            "model": "fuxi_s2s",
            "experiment_id": (
                "model-run/fuxi/"
                "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
            ),
            "variable": "t2m",
            "grid_id": "common_1p5",
            "source_grid_equals_common": True,
            "distribution_representation": "members",
            "ensemble_std_ddof": 0,
        },
    )


def _fake_t2m_archive(
    tmp_path,
    monkeypatch,
    annual_dates: dict[int, tuple[str, ...]],
) -> tuple[DataPaths, dict[str, xr.Dataset], np.ndarray, np.ndarray]:
    latitude = np.linspace(39.0, 0.0, 27, dtype=np.float64)
    longitude = np.linspace(60.0, 99.0, 27, dtype=np.float64)
    paths = DataPaths(tmp_path)
    datasets = {}
    for year, dates in annual_dates.items():
        store = paths.t2m_forecast_root / (str(year) + ".zarr")
        store.mkdir(parents=True)
        (store / ".zmetadata").write_text("synthetic-t2m-%d\n" % year)
        datasets[str(store)] = _t2m_dataset(
            dates, latitude, longitude, float(year - 2000)
        )

    def open_zarr(path, consolidated=True):
        assert consolidated is True
        return datasets[str(path)].copy(deep=True)

    monkeypatch.setattr(xr, "open_zarr", open_zarr)
    return paths, datasets, latitude, longitude


def test_t2m_loader_aligns_existing_splits_and_records_sources(
    tmp_path, monkeypatch
) -> None:
    paths, _, latitude, longitude = _fake_t2m_archive(
        tmp_path,
        monkeypatch,
        {
            2020: ("2020-01-02", "2020-01-06", "2020-01-09"),
            2023: ("2023-01-02", "2023-01-05"),
            2024: ("2024-01-01", "2024-01-04"),
        },
    )
    adapter = _adapter_stub(latitude, longitude)
    result = load_fuxi_t2m_data(adapter, paths)

    assert tuple(result.splits) == ("train", "validation", "test")
    assert result.train.ensemble_mean_weekly.shape == (2, 6, 27, 27)
    assert result.validation.ensemble_std_weekly.shape == (1, 6, 27, 27)
    np.testing.assert_array_equal(
        result.train.initializations, adapter.splits["train"].initializations
    )
    np.testing.assert_allclose(result.train.ensemble_mean_weekly[:, 0, 0, 0], [20, 21])
    assert result.source_manifest["ensemble_member_count"] == 50
    hashes = result.source_manifest["zmetadata_sha256"]
    assert len(hashes) == 3
    assert all(len(value) == 64 for value in hashes.values())
    assert set(result.source_manifest["aligned_initialization_sha256"]) == {
        "train",
        "validation",
        "test",
    }


def test_t2m_loader_rejects_a_missing_adapter_initialization(
    tmp_path, monkeypatch
) -> None:
    paths, _, latitude, longitude = _fake_t2m_archive(
        tmp_path,
        monkeypatch,
        {
            2020: ("2020-01-02", "2020-01-06"),
            2023: ("2023-01-02",),
            2024: ("2024-01-01",),
        },
    )
    adapter = _adapter_stub(
        latitude, longitude, validation=("2023-01-05",)
    )
    with pytest.raises(DataIntegrityError, match="missing validation initialization"):
        load_fuxi_t2m_data(adapter, paths)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("grid", "longitude grid differs"),
        ("lead", "lead_week coordinate"),
        ("members", "members 0..49"),
        ("member_count", "do not use all 50 members"),
        ("metadata", "variable='tp'; expected 't2m'"),
        ("nonfinite", "contains non-finite values"),
    ],
)
def test_t2m_loader_enforces_archive_contract(
    tmp_path, monkeypatch, corruption, message
) -> None:
    paths, datasets, latitude, longitude = _fake_t2m_archive(
        tmp_path,
        monkeypatch,
        {
            2020: ("2020-01-02", "2020-01-06"),
            2023: ("2023-01-02",),
            2024: ("2024-01-01",),
        },
    )
    key = str(paths.t2m_forecast_root / "2023.zarr")
    dataset = datasets[key]
    if corruption == "grid":
        changed = longitude.copy()
        changed[-1] += 0.25
        datasets[key] = dataset.assign_coords(longitude=changed)
    elif corruption == "lead":
        datasets[key] = dataset.isel(lead_week=slice(0, 5))
    elif corruption == "members":
        datasets[key] = dataset.isel(member=slice(0, 49))
    elif corruption == "member_count":
        dataset["ensemble_member_count_weekly"].values[0, 0, 0, 0] = 49
    elif corruption == "metadata":
        dataset.attrs["variable"] = "tp"
    elif corruption == "nonfinite":
        dataset["ensemble_mean_weekly"].values[0, 0, 0, 0] = np.nan
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError("unknown corruption")

    with pytest.raises(DataIntegrityError, match=message):
        load_fuxi_t2m_data(_adapter_stub(latitude, longitude), paths)


def _tp_member_dataset(
    dates: tuple[str, ...],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> xr.Dataset:
    initializations = np.asarray(dates, dtype="datetime64[ns]")
    member_values = np.arange(50, dtype=np.float32)
    member_shape = (len(initializations), 50, 6, 27, 27)
    forecast = np.broadcast_to(
        member_values[None, :, None, None, None], member_shape
    ).copy()
    summary_shape = (len(initializations), 6, 27, 27)
    member_count = np.full(summary_shape, 50, dtype=np.int16)
    member_available = np.ones((len(initializations), 50), dtype=bool)
    member_dims = ("init", "member", "lead_week", "latitude", "longitude")
    summary_dims = ("init", "lead_week", "latitude", "longitude")
    return xr.Dataset(
        data_vars={
            "forecast_weekly_mean": (
                member_dims,
                forecast,
                {
                    "units": "mm day-1",
                    "temporal_statistic": "mean_of_complete_7_day_block",
                },
            ),
            "ensemble_member_count_weekly": (summary_dims, member_count),
            "member_available": (("init", "member"), member_available),
        },
        coords={
            "init": initializations,
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": latitude,
            "longitude": longitude,
            "member": np.arange(50, dtype=np.int16),
        },
        attrs={
            "archive_id": "india_s2s_benchmark_v1",
            "model": "fuxi_s2s",
            "experiment_id": (
                "model-run/fuxi/"
                "fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
            ),
            "variable": "tp",
            "grid_id": "common_1p5",
            "source_grid_equals_common": True,
            "distribution_representation": "members",
            "ensemble_std_ddof": 0,
        },
    )


def _fake_tp_archive(
    tmp_path,
    monkeypatch,
    annual_dates: dict[int, tuple[str, ...]],
) -> tuple[DataPaths, dict[str, xr.Dataset], np.ndarray, np.ndarray]:
    latitude = np.linspace(39.0, 0.0, 27, dtype=np.float64)
    longitude = np.linspace(60.0, 99.0, 27, dtype=np.float64)
    paths = DataPaths(tmp_path)
    datasets = {}
    for year, dates in annual_dates.items():
        store = paths.forecast_root / (str(year) + ".zarr")
        store.mkdir(parents=True)
        (store / ".zmetadata").write_text("synthetic-tp-%d\n" % year)
        datasets[str(store)] = _tp_member_dataset(dates, latitude, longitude)

    def open_zarr(path, consolidated=True):
        assert consolidated is True
        return datasets[str(path)].copy(deep=True)

    monkeypatch.setattr(xr, "open_zarr", open_zarr)
    return paths, datasets, latitude, longitude


def _record_tp_store_hashes(
    adapter: SimpleNamespace, paths: DataPaths, years: tuple[int, ...]
) -> None:
    hashes = adapter.source_manifest["zmetadata_sha256"]
    for year in years:
        store = paths.forecast_root / (str(year) + ".zarr")
        hashes[str(store)] = hashlib.sha256(
            (store / ".zmetadata").read_bytes()
        ).hexdigest()


def test_tp_distribution_loader_computes_frozen_log_features(
    tmp_path, monkeypatch
) -> None:
    annual_dates = {
        2020: ("2020-01-02", "2020-01-06", "2020-01-09"),
        2023: ("2023-01-02", "2023-01-05"),
        2024: ("2024-01-01", "2024-01-04"),
    }
    paths, _, latitude, longitude = _fake_tp_archive(
        tmp_path, monkeypatch, annual_dates
    )
    adapter = _adapter_stub(latitude, longitude)
    _record_tp_store_hashes(adapter, paths, tuple(annual_dates))
    result = load_fuxi_tp_distribution_data(adapter, paths)

    log_members = np.log1p(np.arange(50, dtype=np.float32))
    expected_median_anomaly = np.quantile(
        log_members, 0.5, method="linear"
    ) - np.log1p(np.float32(24.5))
    expected_iqr = np.quantile(
        log_members, 0.75, method="linear"
    ) - np.quantile(log_members, 0.25, method="linear")
    assert result.train.member_log_median_anomaly.shape == (2, 6, 27, 27)
    assert result.train.member_log_iqr.shape == (2, 6, 27, 27)
    assert result.train.member_log_median_anomaly.dtype == np.float32
    assert result.train.member_log_iqr.dtype == np.float32
    assert (
        result.train.probability_exceeds_imerg_climatology.dtype == np.float32
    )
    np.testing.assert_allclose(
        result.train.member_log_median_anomaly[:, 0, 1, 1],
        expected_median_anomaly,
        rtol=1e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.train.member_log_iqr[:, 0, 1, 1], expected_iqr, rtol=1e-6
    )
    np.testing.assert_allclose(
        result.train.probability_exceeds_imerg_climatology[:, 0, 1, 1], 0.5
    )
    np.testing.assert_allclose(
        result.validation.probability_exceeds_imerg_climatology[:, 0, 1, 1],
        0.2,
    )
    np.testing.assert_allclose(
        result.test.probability_exceeds_imerg_climatology[:, 0, 1, 1], 0.8
    )
    assert not result.climatology_threshold_support[0, 0]
    assert result.train.member_log_median_anomaly[0, 0, 0, 0] == 0.0
    assert (
        result.train.probability_exceeds_imerg_climatology[0, 0, 0, 0]
        == 0.0
    )
    assert result.source_manifest["feature_contract"] == [
        "member_log_median_anomaly",
        "member_log_iqr",
        "probability_exceeds_imerg_climatology",
    ]
    assert result.source_manifest["climatology_threshold_baseline"] == "2001-2019"
    assert len(result.source_manifest["zmetadata_sha256"]) == 3


def test_tp_distribution_loader_rejects_missing_initialization(
    tmp_path, monkeypatch
) -> None:
    annual_dates = {
        2020: ("2020-01-02", "2020-01-06"),
        2023: ("2023-01-02",),
        2024: ("2024-01-01",),
    }
    paths, _, latitude, longitude = _fake_tp_archive(
        tmp_path, monkeypatch, annual_dates
    )
    adapter = _adapter_stub(latitude, longitude, validation=("2023-01-05",))
    _record_tp_store_hashes(adapter, paths, tuple(annual_dates))
    with pytest.raises(DataIntegrityError, match="missing validation initialization"):
        load_fuxi_tp_distribution_data(adapter, paths)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("grid", "latitude grid differs"),
        ("lead", "lead_week coordinate"),
        ("members", "members 0..49"),
        ("member_available", "member_available is not complete"),
        ("member_count", "do not use all 50 members"),
        ("metadata", "variable='t2m'; expected 'tp'"),
        ("nonfinite", "non-finite or negative"),
    ],
)
def test_tp_distribution_loader_enforces_member_contract(
    tmp_path, monkeypatch, corruption, message
) -> None:
    annual_dates = {
        2020: ("2020-01-02", "2020-01-06"),
        2023: ("2023-01-02",),
        2024: ("2024-01-01",),
    }
    paths, datasets, latitude, longitude = _fake_tp_archive(
        tmp_path, monkeypatch, annual_dates
    )
    adapter = _adapter_stub(latitude, longitude)
    _record_tp_store_hashes(adapter, paths, tuple(annual_dates))
    key = str(paths.forecast_root / "2023.zarr")
    dataset = datasets[key]
    if corruption == "grid":
        changed = latitude.copy()
        changed[-1] -= 0.25
        datasets[key] = dataset.assign_coords(latitude=changed)
    elif corruption == "lead":
        datasets[key] = dataset.isel(lead_week=slice(0, 5))
    elif corruption == "members":
        datasets[key] = dataset.isel(member=slice(0, 49))
    elif corruption == "member_available":
        dataset["member_available"].values[0, 0] = False
    elif corruption == "member_count":
        dataset["ensemble_member_count_weekly"].values[0, 0, 0, 0] = 49
    elif corruption == "metadata":
        dataset.attrs["variable"] = "t2m"
    elif corruption == "nonfinite":
        dataset["forecast_weekly_mean"].values[0, 0, 0, 0, 0] = np.nan
    else:  # pragma: no cover
        raise AssertionError("unknown corruption")

    with pytest.raises(DataIntegrityError, match=message):
        load_fuxi_tp_distribution_data(adapter, paths)


def test_tp_distribution_loader_requires_fixed_climatology_and_source_hash(
    tmp_path, monkeypatch
) -> None:
    annual_dates = {
        2020: ("2020-01-02", "2020-01-06"),
        2023: ("2023-01-02",),
        2024: ("2024-01-01",),
    }
    paths, _, latitude, longitude = _fake_tp_archive(
        tmp_path, monkeypatch, annual_dates
    )
    adapter = _adapter_stub(latitude, longitude)
    _record_tp_store_hashes(adapter, paths, tuple(annual_dates))
    adapter.audit.climatology_baseline = "2020-2024"
    with pytest.raises(DataIntegrityError, match="fixed 366-day IMERG 2001-2019"):
        load_fuxi_tp_distribution_data(adapter, paths)

    adapter.audit.climatology_baseline = "2001-2019"
    adapter.source_manifest["zmetadata_sha256"][
        str(paths.forecast_root / "2023.zarr")
    ] = "0" * 64
    with pytest.raises(DataIntegrityError, match="source hash differs"):
        load_fuxi_tp_distribution_data(adapter, paths)


LIVE_FORECAST_METADATA = (
    DataPaths().forecast_root / "2020.zarr" / ".zmetadata"
)


@pytest.mark.skipif(
    not LIVE_FORECAST_METADATA.is_file(), reason="frozen India S2S archive is unavailable"
)
def test_live_archive_audit_and_model_arrays() -> None:
    data = load_adapter_data(strict=True)
    assert data.audit.split_counts == {"train": 302, "validation": 93, "test": 100}
    assert data.audit.finite_target_fraction_on_support == 1.0
    assert data.audit.imerg_support_cell_count == 174
    assert data.test.valid_dates[-1, -1, -1] == np.datetime64("2025-02-09")
    assert data.test.verification_end[-1] == np.datetime64("2025-02-10")

    stats = fit_normalization(data.train, data.area_weight_km2)
    assert np.array_equal(stats.target_mean, np.zeros(6, dtype=np.float32))
    arrays = make_model_arrays(
        data.train, stats, data.latitude, data.longitude, data.area_weight_km2
    )
    assert arrays.inputs.shape == (302, 6, 9, 27, 27)
    assert arrays.target.shape == arrays.mask.shape == arrays.weight.shape
    assert np.isfinite(arrays.inputs).all()
    assert np.isfinite(arrays.target).all()
    assert np.all(arrays.weight[~arrays.mask] == 0.0)

    zero_residual = np.zeros_like(data.validation.target_log_residual)
    identity = reconstruct_precipitation(data.validation, zero_residual, stats)
    np.testing.assert_allclose(identity, data.validation.fuxi_mean, rtol=2e-6, atol=1e-6)

    distribution = load_fuxi_tp_distribution_data(data)
    assert int(distribution.climatology_threshold_support.sum()) == 174
    for name, split in distribution.splits.items():
        assert split.grid_shape == (27, 27)
        assert split.lead_count == 6
        np.testing.assert_array_equal(
            split.initializations, data.splits[name].initializations
        )
        assert split.member_log_median_anomaly.dtype == np.float32
        assert split.member_log_iqr.dtype == np.float32
        assert split.probability_exceeds_imerg_climatology.dtype == np.float32
        assert np.isfinite(split.member_log_median_anomaly).all()
        assert np.isfinite(split.member_log_iqr).all()
        assert np.isfinite(split.probability_exceeds_imerg_climatology).all()
    assert distribution.source_manifest["feature_contract"] == [
        "member_log_median_anomaly",
        "member_log_iqr",
        "probability_exceeds_imerg_climatology",
    ]
