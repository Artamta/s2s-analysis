#!/usr/bin/env python3
"""Fair common-date IMD benchmark for the trained FuXi rainfall adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr


HERE = Path(__file__).resolve().parent
CLEAN = HERE.parent.parent
EVALUATION = HERE.parent / "india_s2s_evaluation_v1"
CLIMATOLOGY = HERE.parent / "india_s2s_climatology_v1"
BIAS = CLEAN / "bias-correction"
NEURAL_SRC = CLEAN / "neural_adapter" / "src"
for source in (EVALUATION, CLIMATOLOGY, BIAS, NEURAL_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import fuxi_imerg_a100_big_temporal as engine  # noqa: E402
import fuxi_imerg_full_archive_latelead as common  # noqa: E402
import fuxi_imd_attention_climatology as imd_experiment  # noqa: E402
import imd_tp_diagnostic as diagnostic  # noqa: E402
from fuxi_adapter.baselines import (  # noqa: E402
    LogBiasCorrection,
    apply_log_bias_correction,
)
from s2s_climatology.core import (  # noqa: E402
    circular_window_mean,
    fixed_climatology_day,
)


base = engine.base
YEARS = tuple(range(2020, 2025))
TRAIN_YEARS = tuple(range(2002, 2018))
REGIONS = ("all_india", *diagnostic.REGION_ORDER)
PHYSICAL_MODELS = tuple(diagnostic.CORE_MODELS)
DERIVED_MODELS = ("fuxi_log_bias", "fuxi_imd_adapter")
MODEL_ORDER = (
    "cma",
    "dlesym_v0",
    "ecmwf",
    "fuxi_s2s",
    "fuxi_log_bias",
    "fuxi_imd_adapter",
    "ncep",
    "neuralgcm",
    "ukmo",
)
MODEL_LABELS = {
    **diagnostic.DISPLAY_NAMES,
    "fuxi_log_bias": "FuXi + log-bias",
    "fuxi_imd_adapter": "FuXi + IMD adapter",
}
MODEL_COLORS = {
    "cma": "#7F7F7F",
    "dlesym_v0": "#9467BD",
    "ecmwf": "#8C564B",
    "fuxi_s2s": "#0072B2",
    "fuxi_log_bias": "#CC79A7",
    "fuxi_imd_adapter": "#D55E00",
    "ncep": "#E69F00",
    "neuralgcm": "#009E73",
    "ukmo": "#56B4E9",
}
MODEL_MARKERS = {
    "cma": "o",
    "dlesym_v0": "v",
    "ecmwf": "x",
    "fuxi_s2s": "s",
    "fuxi_log_bias": "D",
    "fuxi_imd_adapter": "P",
    "ncep": "^",
    "neuralgcm": "*",
    "ukmo": "<",
}
METRICS = (
    ("acc", "Model-specific LOYO ACC", "higher"),
    ("rmse_mm_day", "RMSE (mm day$^{-1}$)", "lower"),
    ("mae_mm_day", "MAE (mm day$^{-1}$)", "lower"),
    ("bias_mm_day", "Bias (mm day$^{-1}$)", "zero"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def catalog_records(variable: str) -> list[dict[str, Any]]:
    payload = json.loads(diagnostic.FORECAST_CATALOG.read_text(encoding="utf-8"))
    return [
        record
        for record in payload["records"]
        if record["variable"] == variable and record["grid"] == "common_1p5"
    ]


def find_record(
    records: list[dict[str, Any]], model: str, year: int
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record["model"] == model and int(record["year"]) == year
    ]
    if len(matches) != 1:
        raise ValueError(f"record lookup failed for {model}/{year}: {len(matches)}")
    return matches[0]


def load_common_dates(
    tp_records: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[int, pd.DatetimeIndex]]:
    experiments, paired = diagnostic.paired_initializations(
        tp_records, list(PHYSICAL_MODELS), list(YEARS)
    )
    expected = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 100}
    counts = {year: len(values) for year, values in paired.items()}
    if counts != expected:
        raise ValueError(f"common-date counts changed: {counts}")
    return experiments, paired


def load_adapter_support(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as dataset:
        weights = dataset.area_weight_km2.load().values.astype(np.float64)
        latitude = dataset.latitude.values.astype(np.float64)
        longitude = dataset.longitude.values.astype(np.float64)
    support = weights > 0.0
    if int(support.sum()) != 171:
        raise ValueError(f"adapter support changed: {int(support.sum())} cells")
    return support, latitude, longitude


def load_operational_fuxi(
    tp_records: list[dict[str, Any]],
    t2m_records: list[dict[str, Any]],
    paired: Mapping[int, pd.DatetimeIndex],
) -> tuple[base.ForecastData, np.ndarray, tuple[str, ...]]:
    means: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    initializations: list[np.ndarray] = []
    source_stores: list[str] = []
    latitude = None
    longitude = None
    for year in YEARS:
        inits = paired[year]
        tp_record = find_record(tp_records, "fuxi_s2s", year)
        t2m_record = find_record(t2m_records, "fuxi_s2s", year)
        with xr.open_zarr(tp_record["store"], consolidated=True) as dataset:
            means.append(
                dataset.ensemble_mean_weekly.sel(init=inits).load().values.astype(np.float32)
            )
            spreads.append(
                dataset.ensemble_std_weekly.sel(init=inits).load().values.astype(np.float32)
            )
            current_latitude = dataset.latitude.values.astype(np.float64)
            current_longitude = dataset.longitude.values.astype(np.float64)
        with xr.open_zarr(t2m_record["store"], consolidated=True) as dataset:
            # The benchmark stores temperature in degrees C; training used kelvin.
            temperatures.append(
                (
                    dataset.ensemble_mean_weekly.sel(init=inits).load().values
                    + np.float32(273.15)
                ).astype(np.float32)
            )
        if latitude is None:
            latitude = current_latitude
            longitude = current_longitude
        elif not np.array_equal(latitude, current_latitude) or not np.array_equal(
            longitude, current_longitude
        ):
            raise ValueError("operational FuXi grid changed between years")
        initializations.append(inits.values.astype("datetime64[D]"))
        source_stores.extend((tp_record["store"], t2m_record["store"]))
    assert latitude is not None and longitude is not None
    inits = np.concatenate(initializations)
    mean = np.concatenate(means)
    spread = np.concatenate(spreads)
    t2m = np.concatenate(temperatures)
    if mean.shape != (517, 6, 27, 27) or spread.shape != mean.shape or t2m.shape != mean.shape:
        raise ValueError("unexpected operational FuXi arrays")
    if np.any(mean < 0.0) or np.any(spread < 0.0) or not np.isfinite(t2m).all():
        raise ValueError("operational FuXi inputs are invalid")
    forecast = base.ForecastData(
        initializations=inits,
        valid_dates=base.derive_valid_dates(inits),
        ensemble_mean=mean,
        ensemble_spread=spread,
        latitude=latitude,
        longitude=longitude,
        source_files=tuple(source_stores),
    )
    return forecast, t2m, tuple(source_stores)


def build_training_climatology(support: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    base.TRAIN_YEARS = TRAIN_YEARS
    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    stores: list[str] = []
    for year in TRAIN_YEARS:
        path = diagnostic.OBS_ROOT / f"daily/imd/tp/india_1p5_27x27_v1/{year}.zarr"
        year_dates, year_values, _ = diagnostic.annual_observation(year)
        dates.append(year_dates.values.astype("datetime64[D]"))
        values.append(year_values)
        stores.append(str(path))
    climatology = base.build_training_climatology(
        np.concatenate(dates), np.concatenate(values), support
    )
    return climatology, tuple(stores)


def normalize_dynamic(
    values: np.ndarray,
    statistics: Mapping[str, Any],
    support: np.ndarray,
    *,
    preserve_full_domain: bool = False,
) -> np.ndarray:
    mean = np.asarray(statistics["mean_by_lead"], dtype=np.float32)
    std = np.asarray(statistics["std_by_lead"], dtype=np.float32)
    normalized = (values - mean[None, :, None, None]) / std[None, :, None, None]
    valid = (
        np.isfinite(normalized)
        if preserve_full_domain
        else support[None, None] & np.isfinite(normalized)
    )
    return np.where(valid, normalized, 0.0).astype(np.float32)


def build_features(
    forecast: base.ForecastData,
    t2m_weekly: np.ndarray,
    climatology_daily: np.ndarray,
    normalization: Mapping[str, Any],
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = normalization.get("spatial_context", {})
    preserve_fuxi_context = bool(context.get("enabled", False))
    if preserve_fuxi_context and context.get("full_domain_channels") != [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "fuxi_t2m_weekly",
    ]:
        raise ValueError("saved full-domain FuXi context contract changed")
    weekly_climatology = np.mean(
        climatology_daily[base.calendar_positions(forecast.valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    channels: list[np.ndarray] = [
        normalize_dynamic(
            np.log1p(forecast.ensemble_mean).astype(np.float32),
            normalization["log_fuxi_mean"],
            support,
            preserve_full_domain=preserve_fuxi_context,
        ),
        normalize_dynamic(
            np.log1p(forecast.ensemble_spread).astype(np.float32),
            normalization["log_fuxi_spread"],
            support,
            preserve_full_domain=preserve_fuxi_context,
        ),
        normalize_dynamic(
            np.log1p(weekly_climatology).astype(np.float32),
            normalization["log_imd_climatology"],
            support,
        ),
    ]

    cases, leads, height, width = forecast.ensemble_mean.shape
    latitude = forecast.latitude.astype(np.float32)
    longitude = forecast.longitude.astype(np.float32)
    lat_scaled = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    lon_scaled = 2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    channels.extend(
        [
            np.broadcast_to(lat_scaled[None, None, :, None], (cases, leads, height, width)),
            np.broadcast_to(lon_scaled[None, None, None, :], (cases, leads, height, width)),
        ]
    )
    midpoint = pd.DatetimeIndex(forecast.valid_dates[:, :, 3].reshape(-1))
    angle = 2.0 * np.pi * (midpoint.dayofyear.to_numpy() - 1) / 365.2425
    angle = angle.reshape(cases, leads)
    channels.extend(
        [
            np.broadcast_to(np.sin(angle)[:, :, None, None], (cases, leads, height, width)),
            np.broadcast_to(np.cos(angle)[:, :, None, None], (cases, leads, height, width)),
            np.broadcast_to(
                np.linspace(-1.0, 1.0, leads, dtype=np.float32)[None, :, None, None],
                (cases, leads, height, width),
            ),
            np.broadcast_to(support[None, None], (cases, leads, height, width)).astype(np.float32),
        ]
    )
    raw_anomaly = np.log1p(forecast.ensemble_mean) - np.log1p(weekly_climatology)
    channels.append(
        normalize_dynamic(
            raw_anomaly.astype(np.float32),
            normalization["explicit_log_fuxi_anomaly"],
            support,
        )
    )
    channels.append(
        normalize_dynamic(
            t2m_weekly,
            normalization["fuxi_t2m_weekly"],
            support,
            preserve_full_domain=preserve_fuxi_context,
        )
    )
    standard = np.stack(channels, axis=2).astype(np.float32)

    climatology_candidates = []
    for offset in imd_experiment.OFFSETS_DAYS:
        shifted = forecast.valid_dates + np.timedelta64(offset, "D")
        candidate = np.mean(
            climatology_daily[base.calendar_positions(shifted)],
            axis=2,
            dtype=np.float64,
        )
        climatology_candidates.append(candidate.astype(np.float32))
    bank = np.stack(climatology_candidates, axis=2)
    climo_stats = normalization["log_imd_climatology"]
    climo_mean = np.asarray(climo_stats["mean_by_lead"], dtype=np.float32)
    climo_std = np.asarray(climo_stats["std_by_lead"], dtype=np.float32)
    normalized_bank = (
        np.log1p(bank).astype(np.float32)
        - climo_mean[None, :, None, None, None]
    ) / climo_std[None, :, None, None, None]
    anomaly_stats = normalization["explicit_log_fuxi_anomaly"]
    anomaly_mean = np.asarray(anomaly_stats["mean_by_lead"], dtype=np.float32)
    anomaly_std = np.asarray(anomaly_stats["std_by_lead"], dtype=np.float32)
    bank_anomaly = np.log1p(forecast.ensemble_mean)[:, :, None] - np.log1p(bank)
    normalized_bank_anomaly = (
        bank_anomaly - anomaly_mean[None, :, None, None, None]
    ) / anomaly_std[None, :, None, None, None]
    valid = support[None, None, None]
    normalized_bank = np.where(valid, normalized_bank, 0.0).astype(np.float32)
    normalized_bank_anomaly = np.where(
        valid, normalized_bank_anomaly, 0.0
    ).astype(np.float32)
    features = np.concatenate(
        (standard, normalized_bank, normalized_bank_anomaly), axis=2
    ).astype(np.float32)
    expected_shape = (len(forecast.initializations), 6, 29, 27, 27)
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(f"invalid adapter feature array: {features.shape}")
    if preserve_fuxi_context:
        for channel_index, name in (
            (0, "log_fuxi_mean"),
            (1, "log_fuxi_spread"),
            (10, "fuxi_t2m_weekly"),
        ):
            outside = features[:, :, channel_index, ~support]
            if not np.isfinite(outside).all() or not np.any(outside != 0.0):
                raise ValueError(f"benchmark omitted full-domain context for {name}")
        for channel_index in (2, 9, *range(11, 29)):
            if np.any(features[:, :, channel_index, ~support] != 0.0):
                raise ValueError("benchmark leaked an IMD channel outside support")
    if list(normalization["input_channels"]) != [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_calendar_climatology",
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "support",
        "explicit_log_fuxi_minus_imd_climatology",
        "fuxi_t2m_weekly",
        *[
            f"imd_climatology_offset_{offset:+d}d"
            for offset in imd_experiment.OFFSETS_DAYS
        ],
        *[
            f"fuxi_minus_imd_climatology_offset_{offset:+d}d"
            for offset in imd_experiment.OFFSETS_DAYS
        ],
    ]:
        raise ValueError("saved input-channel contract changed")
    return features, weekly_climatology


def candidate_from_metadata(metadata: Mapping[str, Any]) -> engine.Candidate:
    return engine.Candidate(
        name=str(metadata["name"]),
        label=str(metadata["label"]),
        architecture=str(metadata["architecture"]),
        batch_size=int(metadata["batch_size"]),
        learning_rate=float(metadata["learning_rate"]),
        weight_decay=float(metadata["weight_decay"]),
        dropout=float(metadata["dropout"]),
        color=str(metadata["color"]),
    )


def infer_adapter(
    run: Path,
    forecast: base.ForecastData,
    features: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    selected = str(selection["selected_model"])
    alpha = float(selection["selected_alpha"])
    anchor = np.load(run / "models" / "log_bias_anchor.npz")
    correction = LogBiasCorrection(
        anchor["lead_month_residual"], float(anchor["shrinkage"])
    )
    log_bias = apply_log_bias_correction(
        forecast.ensemble_mean, forecast.initializations, correction
    )
    if selected == "log_bias":
        return log_bias, log_bias.copy(), selection
    training = manifest["training"][selected]
    candidate = candidate_from_metadata(training)
    residual = engine.predict_candidate(
        candidate,
        training,
        features,
        np.arange(len(features)),
        run,
        inactive_lead_count=0,
    )
    adapter = common.reconstruct(
        log_bias,
        (alpha * residual).astype(np.float32),
        anchor["target_scale"].astype(np.float32),
        support,
    )
    return log_bias, adapter, selection


def save_prediction_store(
    path: Path,
    forecast: base.ForecastData,
    log_bias: np.ndarray,
    adapter: np.ndarray,
    support: np.ndarray,
    selection: Mapping[str, Any],
) -> None:
    prediction = np.stack((forecast.ensemble_mean, log_bias, adapter)).astype(np.float32)
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                prediction,
            ),
            "adapter_support": (("latitude", "longitude"), support),
        },
        coords={
            "method": ["raw_fuxi", "fuxi_log_bias", "fuxi_imd_adapter"],
            "init": forecast.initializations.astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": "FuXi IMD adapter on common 2020-2024 operational benchmark dates",
            "selected_model": str(selection["selected_model"]),
            "selected_alpha": float(selection["selected_alpha"]),
            "training_season": "June-September initializations only",
            "units": "mm day-1",
        },
    )
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={"prediction": {"chunks": (1, 32, 1, 27, 27)}},
    )


def loyo_model_climatology(
    prediction: np.ndarray,
    initializations: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    years = pd.DatetimeIndex(initializations).year.to_numpy()
    yearly = []
    for year in YEARS:
        selected = years == year
        values = prediction[selected][..., support]
        rolled, _ = circular_window_mean(
            values,
            fixed_climatology_day(initializations[selected]),
            half_width_days=15,
            calendar_days=366,
        )
        yearly.append(rolled)
    yearly_array = np.stack(yearly).astype(np.float32)
    output = np.full(prediction.shape, np.nan, dtype=np.float32)
    for year_index, year in enumerate(YEARS):
        selected = years == year
        loyo = (
            (yearly_array.sum(axis=0, dtype=np.float64) - yearly_array[year_index])
            / (len(YEARS) - 1)
        ).astype(np.float32)
        days = fixed_climatology_day(initializations[selected]) - 1
        positions = np.flatnonzero(selected)
        for local, position in enumerate(positions):
            field = output[position]
            field[:, support] = loyo[days[local]]
            output[position] = field
    if not np.isfinite(output[..., support]).all():
        raise ValueError("derived LOYO forecast climatology is incomplete")
    return output


def load_spatial_areas(support: np.ndarray) -> dict[str, np.ndarray]:
    with xr.open_zarr(diagnostic.SPATIAL_SUPPORT, consolidated=True) as dataset:
        cell_area = dataset.cell_area_km2.load().values.astype(np.float64)
        areas = {
            "all_india": dataset.india_area_weight_km2.load().values.astype(np.float64),
            **{
                region: cell_area
                * dataset[f"{region}_fraction"].load().values.astype(np.float64)
                for region in diagnostic.REGION_ORDER
            },
        }
    return {name: np.where(support, values, 0.0) for name, values in areas.items()}


def score_field(
    rows: list[dict[str, Any]],
    *,
    model: str,
    experiment: str,
    year: int,
    initializations: pd.DatetimeIndex,
    forecast: np.ndarray,
    model_climatology: np.ndarray,
    truth: np.ndarray,
    truth_climatology: np.ndarray,
    coverage: np.ndarray,
    areas: Mapping[str, np.ndarray],
) -> None:
    for case_index, init in enumerate(initializations):
        for lead_index in range(6):
            for region, region_area in areas.items():
                weights = region_area * coverage[case_index, lead_index]
                raw = diagnostic.weighted_metrics(
                    forecast[case_index, lead_index],
                    truth[case_index, lead_index],
                    weights,
                )
                anomaly = diagnostic.weighted_metrics(
                    forecast[case_index, lead_index]
                    - model_climatology[case_index, lead_index],
                    truth[case_index, lead_index]
                    - truth_climatology[case_index, lead_index],
                    weights,
                )
                supported = weights > 0.0
                field = forecast[case_index, lead_index]
                rows.append(
                    {
                        "cohort": "adapter_common_support_2020_2024",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "experiment_id": experiment,
                        "verification_year": year,
                        "init": str(init.date()),
                        "season": diagnostic.season_for_month(init.month),
                        "training_domain": (
                            "in_domain" if init.month in {6, 7, 8, 9} else "out_of_training_season"
                        ),
                        "region": region,
                        "region_label": diagnostic.REGION_LABELS[region],
                        "lead_week": lead_index + 1,
                        "acc": anomaly["acc"],
                        "mae_mm_day": raw["mae_mm_day"],
                        "rmse_mm_day": raw["rmse_mm_day"],
                        "bias_mm_day": raw["bias_mm_day"],
                        "valid_cell_count": raw["valid_cell_count"],
                        "effective_area_km2": raw["effective_area_km2"],
                        "minimum_forecast_supported_mm_day": float(
                            np.nanmin(np.where(supported, field, np.nan))
                        ),
                    }
                )


def score_all_models(
    tp_records: list[dict[str, Any]],
    climo_records: list[dict[str, Any]],
    experiments: Mapping[str, str],
    paired: Mapping[int, pd.DatetimeIndex],
    forecast: base.ForecastData,
    log_bias: np.ndarray,
    adapter: np.ndarray,
    support: np.ndarray,
) -> pd.DataFrame:
    print("Building IMD 1991-2019 verification climatology...", flush=True)
    observation_climatology, _ = diagnostic.build_imd_climatology()
    observation_dates, observation_values, observation_coverage = (
        diagnostic.load_verification_observations()
    )
    areas = load_spatial_areas(support)
    log_bias_climo = loyo_model_climatology(
        log_bias, forecast.initializations, support
    )
    adapter_climo = loyo_model_climatology(
        adapter, forecast.initializations, support
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for year in YEARS:
        inits = paired[year]
        count = len(inits)
        truth, truth_climo, coverage = diagnostic.weekly_observation_fields(
            inits,
            observation_dates,
            observation_values,
            observation_coverage,
            observation_climatology,
        )
        for model in PHYSICAL_MODELS:
            experiment = experiments[model]
            forecast_record = diagnostic.record_for(
                tp_records, model, experiment, year
            )
            climo_record = diagnostic.record_for(
                climo_records, model, experiment
            )
            with xr.open_zarr(forecast_record["store"], consolidated=True) as dataset:
                values = (
                    dataset.ensemble_mean_weekly.sel(init=inits).load().values.astype(np.float32)
                )
            with xr.open_zarr(climo_record["store"], consolidated=True) as dataset:
                indexer = xr.DataArray(diagnostic.fixed_day(inits), dims="case")
                model_climo = (
                    dataset.climatology_weekly_mean_loyo.sel(verification_year=year)
                    .sel(climatology_day=indexer)
                    .load()
                    .values.astype(np.float32)
                )
            score_field(
                rows,
                model=model,
                experiment=experiment,
                year=year,
                initializations=inits,
                forecast=values,
                model_climatology=model_climo,
                truth=truth,
                truth_climatology=truth_climo,
                coverage=coverage,
                areas=areas,
            )
        for model, values, climatology in (
            (
                "fuxi_log_bias",
                log_bias[offset : offset + count],
                log_bias_climo[offset : offset + count],
            ),
            (
                "fuxi_imd_adapter",
                adapter[offset : offset + count],
                adapter_climo[offset : offset + count],
            ),
        ):
            score_field(
                rows,
                model=model,
                experiment="trained_imd_adapter_2002_2019",
                year=year,
                initializations=inits,
                forecast=values,
                model_climatology=climatology,
                truth=truth,
                truth_climatology=truth_climo,
                coverage=coverage,
                areas=areas,
            )
        offset += count
        print(f"Scored {year}: {count} common initializations", flush=True)
    cases = pd.DataFrame(rows)
    expected = 517 * 6 * len(REGIONS) * len(MODEL_ORDER)
    if len(cases) != expected or cases.duplicated(
        ["model", "init", "region", "lead_week"]
    ).any():
        raise ValueError(f"case-score contract failed: {len(cases)} rows")
    return cases


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["acc", "mae_mm_day", "rmse_mm_day", "bias_mm_day"]
    weekly = (
        cases.groupby(
            ["region", "region_label", "model", "model_label", "lead_week"],
            as_index=False,
        )[metrics]
        .mean()
    )
    seasonal = (
        cases.groupby(
            [
                "season",
                "region",
                "region_label",
                "model",
                "model_label",
                "lead_week",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )
    rankings = (
        cases.groupby(
            ["season", "region", "region_label", "model", "model_label"],
            as_index=False,
        )[metrics]
        .mean()
    )
    rankings["rmse_rank"] = rankings.groupby(["season", "region"])[
        "rmse_mm_day"
    ].rank(method="min")
    rankings["acc_rank"] = rankings.groupby(["season", "region"])["acc"].rank(
        method="min", ascending=False
    )
    return weekly, seasonal, rankings


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)


def plot_lines(
    axis: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    *,
    include_ecmwf: bool = True,
) -> None:
    for model in MODEL_ORDER:
        if model == "ecmwf" and not include_ecmwf:
            continue
        selected = table.loc[table.model.eq(model)].sort_values("lead_week")
        if selected.empty:
            continue
        highlight = model in {"fuxi_s2s", "fuxi_log_bias", "fuxi_imd_adapter"}
        axis.plot(
            selected.lead_week,
            selected[metric],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=2.5 if model == "fuxi_imd_adapter" else (1.8 if highlight else 1.1),
            markersize=6 if highlight else 4,
            alpha=1.0 if highlight else 0.72,
            linestyle="--" if model == "ecmwf" else "-",
        )
    axis.set_xticks(range(1, 7), [f"W{week}" for week in range(1, 7)])
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def make_overall_figure(weekly: pd.DataFrame, output: Path) -> None:
    selected = weekly.loc[weekly.region.eq("all_india")]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for axis, (metric, label, goal) in zip(axes.flat, METRICS):
        plot_lines(axis, selected, metric)
        axis.set_title(f"{label} · {goal} is better")
        axis.set_xlabel("Lead week")
        axis.set_ylabel(label)
        if metric == "bias_mm_day":
            axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Common-date India S2S rainfall benchmark\n"
        "IMD verification · 517 paired starts · identical 171-cell adapter support",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(top=0.77, hspace=0.38, wspace=0.25)
    save_figure(figure, output / "01_all_india_all_models")


def make_stratified_figure(
    summary: pd.DataFrame,
    output: Path,
    *,
    metric: str,
    dimension: str,
) -> None:
    if dimension == "season":
        values = diagnostic.SEASON_ORDER
        selected = summary.loc[summary.region.eq("all_india")]
        title = "Initialization-season comparison · All India"
        labels = {value: value for value in values}
        stem_dimension = "season"
    else:
        values = list(diagnostic.REGION_ORDER)
        selected = summary
        title = "IMD homogeneous-region comparison · all seasons"
        labels = diagnostic.REGION_LABELS
        stem_dimension = "region"
    metric_label = dict((name, label) for name, label, _ in METRICS)[metric]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=True)
    for axis, value in zip(axes.flat, values):
        panel = selected.loc[selected[dimension].eq(value)]
        plot_lines(axis, panel, metric, include_ecmwf=False)
        count = panel.loc[panel.model.eq("fuxi_s2s"), "lead_week"].size
        axis.set_title(f"{labels[value]}" + (" · adapter training domain" if value == "JJA" else ""))
        axis.set_xlabel("Lead week")
        axis.set_ylabel(metric_label)
        if metric == "bias_mm_day":
            axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"{title}\n{metric_label} · ECMWF excluded from primary panels after archived TP QC failure",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.subplots_adjust(top=0.76, hspace=0.38, wspace=0.25)
    save_figure(figure, output / f"02_{stem_dimension}_{metric}")


def make_skill_heatmap(cases: pd.DataFrame, output: Path) -> None:
    selected = cases.loc[
        cases.model.isin(("fuxi_s2s", "fuxi_imd_adapter"))
    ]
    summary = (
        selected.groupby(["season", "region", "model"], as_index=False)[
            ["rmse_mm_day", "acc"]
        ].mean()
    )
    pivot_rmse = summary.pivot_table(
        index=["season", "region"], columns="model", values="rmse_mm_day"
    )
    pivot_acc = summary.pivot_table(
        index=["season", "region"], columns="model", values="acc"
    )
    rmse_skill = 100.0 * (
        pivot_rmse["fuxi_s2s"] - pivot_rmse["fuxi_imd_adapter"]
    ) / pivot_rmse["fuxi_s2s"]
    acc_change = pivot_acc["fuxi_imd_adapter"] - pivot_acc["fuxi_s2s"]
    index = pd.MultiIndex.from_product(
        [diagnostic.SEASON_ORDER, REGIONS], names=["season", "region"]
    )
    rmse_matrix = rmse_skill.reindex(index).to_numpy().reshape(4, 5)
    acc_matrix = acc_change.reindex(index).to_numpy().reshape(4, 5)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for axis, matrix, title, cmap, limits, fmt in (
        (
            axes[0],
            rmse_matrix,
            "RMSE reduction vs raw FuXi (%)",
            "RdYlGn",
            max(1.0, float(np.nanmax(np.abs(rmse_matrix)))),
            ".1f",
        ),
        (
            axes[1],
            acc_matrix,
            "Model-climatology ACC change vs raw FuXi",
            "RdBu_r",
            max(0.01, float(np.nanmax(np.abs(acc_matrix)))),
            ".3f",
        ),
    ):
        image = axis.imshow(matrix, cmap=cmap, vmin=-limits, vmax=limits, aspect="auto")
        axis.set_xticks(
            range(5), [diagnostic.REGION_LABELS[value].replace(" India", "") for value in REGIONS], rotation=25, ha="right"
        )
        axis.set_yticks(range(4), diagnostic.SEASON_ORDER)
        axis.set_title(title, fontweight="semibold")
        for row in range(4):
            for column in range(5):
                value = matrix[row, column]
                axis.text(column, row, format(value, fmt), ha="center", va="center", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(
        "Where the IMD adapter changes FuXi skill\n"
        "Positive values are improvements · JJA is the fully in-training season",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )
    figure.tight_layout()
    save_figure(figure, output / "03_adapter_skill_by_season_region")


def write_readout(
    output: Path,
    cases: pd.DataFrame,
    rankings: pd.DataFrame,
    run: Path,
    selection: Mapping[str, Any],
) -> None:
    jja = rankings.loc[
        rankings.season.eq("JJA") & rankings.region.eq("all_india")
    ].sort_values("rmse_mm_day")
    adapter = jja.loc[jja.model.eq("fuxi_imd_adapter")].iloc[0]
    raw = jja.loc[jja.model.eq("fuxi_s2s")].iloc[0]
    skill = 100.0 * (raw.rmse_mm_day - adapter.rmse_mm_day) / raw.rmse_mm_day
    lines = [
        "# FuXi IMD adapter: common-date multi-model benchmark",
        "",
        "## Fairness contract",
        "",
        "- 517 identical initialization dates for every model, 2020–2024.",
        "- Six identical seven-day lead windows and the same IMD daily truth.",
        "- Identical 171-cell adapter support, observation coverage, cell-area weights, and four frozen IMD homogeneous-region masks.",
        "- ACC uses a separate 2020–2024 leave-one-year-out 31-day model climatology for every forecast system, including the adapter.",
        "- The adapter was trained on 2002–2017 and selected on 2018–2019; benchmark dates were not used to fit its weights.",
        "",
        "## Domain warning",
        "",
        "The adapter was trained only on June–September initializations. JJA is fully in-domain; SON mixes September with out-of-domain October–November; DJF and MAM are out-of-training-season diagnostics.",
        "",
        "## JJA all-India result, pooled W1–W6",
        "",
        f"Adapter RMSE is **{adapter.rmse_mm_day:.3f} mm/day**, compared with raw FuXi **{raw.rmse_mm_day:.3f} mm/day**: **{skill:+.2f}%**.",
        f"Adapter ACC is **{adapter.acc:.3f}**, compared with raw FuXi **{raw.acc:.3f}**.",
        "",
        "| Model | RMSE | ACC | MAE | Bias | RMSE rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in jja.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.rmse_mm_day:.3f} | {row.acc:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:+.3f} | {int(row.rmse_rank)} |"
        )
    all_india = rankings.loc[
        rankings.region.eq("all_india")
        & rankings.model.isin(("fuxi_s2s", "fuxi_imd_adapter"))
    ].pivot(index="season", columns="model", values=["rmse_mm_day", "acc"])
    lines.extend(
        [
            "",
            "## Season-wise FuXi change, pooled W1–W6",
            "",
            "| Season | Domain | Raw RMSE | Adapter RMSE | RMSE reduction | Raw ACC | Adapter ACC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for season in diagnostic.SEASON_ORDER:
        row = all_india.loc[season]
        raw_rmse = row[("rmse_mm_day", "fuxi_s2s")]
        adapter_rmse = row[("rmse_mm_day", "fuxi_imd_adapter")]
        reduction = 100.0 * (raw_rmse - adapter_rmse) / raw_rmse
        domain = "in-domain" if season == "JJA" else ("mixed" if season == "SON" else "out-of-season")
        lines.append(
            f"| {season} | {domain} | {raw_rmse:.3f} | {adapter_rmse:.3f} | "
            f"{reduction:+.2f}% | {row[('acc', 'fuxi_s2s')]:.3f} | "
            f"{row[('acc', 'fuxi_imd_adapter')]:.3f} |"
        )
    jja_regions = rankings.loc[
        rankings.season.eq("JJA")
        & rankings.model.isin(("fuxi_s2s", "fuxi_imd_adapter"))
    ].pivot(index="region", columns="model", values=["rmse_mm_day", "acc"])
    lines.extend(
        [
            "",
            "## JJA homogeneous-region change, pooled W1–W6",
            "",
            "| Region | Raw RMSE | Adapter RMSE | RMSE reduction | Raw ACC | Adapter ACC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        row = jja_regions.loc[region]
        raw_rmse = row[("rmse_mm_day", "fuxi_s2s")]
        adapter_rmse = row[("rmse_mm_day", "fuxi_imd_adapter")]
        reduction = 100.0 * (raw_rmse - adapter_rmse) / raw_rmse
        lines.append(
            f"| {diagnostic.REGION_LABELS[region]} | {raw_rmse:.3f} | {adapter_rmse:.3f} | "
            f"{reduction:+.2f}% | {row[('acc', 'fuxi_s2s')]:.3f} | "
            f"{row[('acc', 'fuxi_imd_adapter')]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- ECMWF is retained in the complete tables, but its archived precipitation has a documented accumulation reset near lead day 16; primary plots exclude it.",
            "- A good JJA result supports in-domain generalization to new operational start dates. It does not establish year-round calibration.",
            "- Raw RMSE/MAE/bias and model-climatology ACC are comparable because they are recomputed from fields under one common contract; the earlier incompatible score tables were not merged.",
            "",
            "## Adapter provenance",
            "",
            f"- Run: `{run}`",
            f"- Frozen selected model: `{selection['selected_model']}`",
            f"- Frozen residual alpha: `{float(selection['selected_alpha']):.3f}`",
            "- Three validation-selected checkpoint ensembles are preserved in the run directory.",
        ]
    )
    (output / "RESULTS_READOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError("adapter run is not a completed full experiment")
    support, latitude, longitude = load_adapter_support(run)
    tp_records, climo_records = diagnostic.load_catalog_records()
    t2m_records = catalog_records("t2m")
    experiments, paired = load_common_dates(tp_records)
    print("Loading common-date operational FuXi predictors...", flush=True)
    forecast, t2m, source_stores = load_operational_fuxi(
        tp_records, t2m_records, paired
    )
    if not np.array_equal(forecast.latitude, latitude) or not np.array_equal(
        forecast.longitude, longitude
    ):
        raise ValueError("training and benchmark grids differ")
    print("Rebuilding the frozen 2002-2017 IMD feature climatology...", flush=True)
    training_climatology, imd_stores = build_training_climatology(support)
    normalization = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    features, _ = build_features(
        forecast, t2m, training_climatology, normalization, support
    )
    print("Running the three-checkpoint adapter ensemble...", flush=True)
    log_bias, adapter, selection = infer_adapter(run, forecast, features, support)
    prediction_store = output / "predictions.zarr"
    save_prediction_store(
        prediction_store, forecast, log_bias, adapter, support, selection
    )
    del features

    print("Scoring all models with one common IMD and regional contract...", flush=True)
    cases = score_all_models(
        tp_records,
        climo_records,
        experiments,
        paired,
        forecast,
        log_bias,
        adapter,
        support,
    )
    weekly, seasonal, rankings = summarize(cases)
    cases.to_csv(output / "case_metrics.csv", index=False)
    weekly.to_csv(output / "weekly_summary.csv", index=False)
    seasonal.to_csv(output / "seasonal_regional_weekly_summary.csv", index=False)
    rankings.to_csv(output / "seasonal_regional_rankings.csv", index=False)

    configure_plotting()
    make_overall_figure(weekly, figures)
    for metric in ("rmse_mm_day", "acc"):
        make_stratified_figure(
            seasonal, figures, metric=metric, dimension="season"
        )
        make_stratified_figure(
            weekly, figures, metric=metric, dimension="region"
        )
    make_skill_heatmap(cases, figures)
    write_readout(output, cases, rankings, run, selection)

    source_copy = output / "code"
    source_copy.mkdir()
    sources = {
        "run_benchmark.py": Path(__file__),
        "imd_tp_diagnostic.py": EVALUATION / "imd_tp_diagnostic.py",
        "models.py": NEURAL_SRC / "fuxi_adapter" / "models.py",
    }
    code_hashes = {}
    for name, source in sources.items():
        shutil.copy2(source, source_copy / name)
        code_hashes[name] = sha256_file(source_copy / name)
    result_manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_run": str(run),
        "adapter_manifest_sha256": sha256_file(run / "manifest.json"),
        "adapter_selection_sha256": sha256_file(run / "selection.json"),
        "selected_model": selection["selected_model"],
        "selected_alpha": selection["selected_alpha"],
        "years": list(YEARS),
        "paired_initialization_counts": {
            str(year): len(paired[year]) for year in YEARS
        },
        "paired_initialization_total": 517,
        "models": list(MODEL_ORDER),
        "regions": list(REGIONS),
        "lead_weeks": list(range(1, 7)),
        "adapter_support_cells": int(support.sum()),
        "adapter_training_years": list(TRAIN_YEARS),
        "adapter_training_initialization_months": [6, 7, 8, 9],
        "metric_contract": {
            "raw": "case-wise area-weighted RMSE, MAE, and bias against IMD",
            "acc": "case-wise area-weighted spatial anomaly correlation using each model's 2020-2024 LOYO 31-day climatology and IMD 1991-2019 climatology",
            "weights": "frozen area x IMD homogeneous-region fraction x weekly observation coverage x 171-cell adapter support",
        },
        "ecmwf_warning": "archived TP accumulation reset near lead day 16; retained in complete tables, excluded from primary stratified plots",
        "operational_fuxi_source_stores": list(source_stores),
        "training_imd_source_stores": list(imd_stores),
        "forecast_catalog": str(diagnostic.FORECAST_CATALOG),
        "forecast_catalog_sha256": sha256_file(diagnostic.FORECAST_CATALOG),
        "forecast_climatology_catalog": str(diagnostic.FORECAST_CLIMO_CATALOG),
        "forecast_climatology_catalog_sha256": sha256_file(
            diagnostic.FORECAST_CLIMO_CATALOG
        ),
        "code_sha256": code_hashes,
        "outputs": {},
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
        if prediction_store in artifact.parents:
            continue
        result_manifest["outputs"][str(artifact.relative_to(output))] = sha256_file(
            artifact
        )
    result_manifest["outputs"]["predictions.zarr"] = sha256_tree(prediction_store)
    (output / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: fair 517-date, 9-model, 5-region IMD benchmark: {output}")


if __name__ == "__main__":
    main()
