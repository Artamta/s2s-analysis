#!/usr/bin/env python3
"""Calculate the audited 31-case FuXi/ERPAS IMD+IMERG review dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
BUILD_SCRIPT = (
    WORKSPACE
    / "deliverables/fuxi_erpas_acc_multiseason_2023_2024/scripts/build_acc_csv.py"
)
IMERG_OBS = (
    WORKSPACE
    / "deliverables/imd_acc_1p5_full_2023_2024/data/observations/"
    "imerg_final_v07b_jjas_2023_2024_india.nc"
)
IMERG_CLIMO = (
    HERE
    / "data/imerg_climatology/"
    "imerg_final_v07b_climatology_2001_2022_1p5_daily.nc"
)
PROCESSED = HERE / "data/processed/review_fields_2023_2024.nc"
METRICS = HERE / "metrics"
LOGS = HERE / "logs"
MODELS = ("FuXi-S2S", "ERPAS")
REFERENCES = ("IMD", "IMERG Final V07B")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = import_file("review_metric_build_support", BUILD_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object):
    """Convert scientific scalar/container types without changing values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def load_imerg_observations(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    india_fraction: np.ndarray,
    weight: np.ndarray,
    land_support: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[pd.Timestamp, np.ndarray], dict]:
    with xr.open_dataset(IMERG_OBS) as source:
        if (
            source.attrs.get("product") != "GPM_3IMERGDF.07"
            or source.attrs.get("revision") != "V07B"
            or source.attrs.get("doi") != "10.5067/GPM/IMERGDF/DAY/07"
        ):
            raise ValueError("IMERG observation provenance contract failed")
        dates = pd.DatetimeIndex(source.period_start.values)
        values = source.precipitation.load().values.astype(np.float64)
        counts = source.precipitation_cnt.load().values
        source_lat = source.latitude.values.astype(float)
        source_lon = source.longitude.values.astype(float)
        attrs = dict(source.attrs)
    if len(dates) != 280 or not dates.is_unique:
        raise ValueError("IMERG observations are not the expected 280 unique days")
    expected = pd.DatetimeIndex(
        list(pd.date_range("2023-06-08", "2023-10-25"))
        + list(pd.date_range("2024-06-06", "2024-10-23"))
    )
    if not dates.equals(expected):
        raise ValueError("IMERG observation date coverage differs from the 31-case union")
    if int(counts.min()) != 48 or int(counts.max()) != 48:
        raise ValueError("IMERG observation half-hour count is not exactly 48")
    if not np.isfinite(values).all() or float(values.min()) < 0:
        raise ValueError("IMERG observations are non-finite or negative")

    support, support_lat, support_lon = land_support
    source_support = build.core.remap_support_fraction(
        support, support_lat, support_lon, source_lat, source_lon
    )
    remapped, denominator, target_area, checks = build.core.remap_conservative(
        values,
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        support=source_support,
    )
    mask = weight > 0
    if not np.isfinite(remapped[:, mask]).all() or float(np.min(remapped[:, mask])) < 0:
        raise ValueError("remapped IMERG observations failed finite/nonnegative checks")
    support_audit = build.core.support_representation_audit(
        denominator, target_area, india_fraction
    )
    return (
        {date: remapped[index].astype(np.float32) for index, date in enumerate(dates)},
        {
            "path": str(IMERG_OBS),
            "sha256": sha256_file(IMERG_OBS),
            "source_attrs": attrs,
            "date_count": len(dates),
            "first_date": str(dates.min().date()),
            "last_date": str(dates.max().date()),
            "minimum_half_hour_count": int(counts.min()),
            "maximum_half_hour_count": int(counts.max()),
            "remap_checks": checks,
            "support_representation": support_audit,
        },
    )


def load_imerg_climatology(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    india_fraction: np.ndarray,
    weight: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    if not IMERG_CLIMO.is_file():
        raise FileNotFoundError(
            f"missing fixed IMERG climatology {IMERG_CLIMO}; run the Slurm staging workflow"
        )
    manifest_path = IMERG_CLIMO.with_suffix(".manifest.json")
    with xr.open_dataset(IMERG_CLIMO) as source:
        if (
            source.attrs.get("product") != "GPM_3IMERGDF.07"
            or source.attrs.get("revision") != "V07B"
            or source.attrs.get("baseline_years") != "2001-2022"
            or int(source.attrs.get("baseline_year_count", -1)) != 22
        ):
            raise ValueError("fixed IMERG climatology provenance contract failed")
        if not (
            np.array_equal(source.latitude.values, target_lat)
            and np.array_equal(source.longitude.values, target_lon)
            and np.allclose(source.india_fraction.values, india_fraction, rtol=0, atol=1e-7)
            and np.allclose(source.spatial_weight.values, weight, rtol=0, atol=1e-12)
        ):
            raise ValueError("fixed IMERG climatology grid/support differs from verification")
        keys = source.calendar_month_day.values.astype(str)
        daily = source.daily_precipitation_climatology.load().values.astype(np.float32)
        counts = source.daily_sample_count.values
        attrs = dict(source.attrs)
    expected_keys = np.asarray(
        [stamp.strftime("%m-%d") for stamp in pd.date_range("2000-06-06", "2000-10-25")]
    )
    if not np.array_equal(keys, expected_keys) or len(np.unique(keys)) != 142:
        raise ValueError("IMERG climatology calendar-day keys are incomplete")
    mask = weight > 0
    if not (np.asarray(counts) == 22).all():
        raise ValueError("IMERG climatology does not contain exactly 22 samples per day")
    if not np.isfinite(daily[:, mask]).all() or float(np.min(daily[:, mask])) < 0:
        raise ValueError("IMERG climatology failed finite/nonnegative checks")
    return (
        {key: daily[index] for index, key in enumerate(keys)},
        {
            "path": str(IMERG_CLIMO),
            "sha256": sha256_file(IMERG_CLIMO),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "attrs": attrs,
            "sample_count_min": int(np.min(counts)),
            "sample_count_max": int(np.max(counts)),
        },
    )


def imerg_weekly_reference(
    case: dict,
    observed: dict[pd.Timestamp, np.ndarray],
    climatology: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(case["valid_start"]).tz_localize(None)
    end = pd.Timestamp(case["valid_end"]).tz_localize(None)
    dates = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    if len(dates) != 28:
        raise ValueError(f"{case['case_id']} does not have a 28-day window")
    missing = [stamp for stamp in dates if stamp not in observed]
    if missing:
        raise ValueError(f"IMERG observations missing {case['case_id']}: {missing}")
    raw = np.stack([observed[stamp] for stamp in dates])
    clim = np.stack([climatology[stamp.strftime("%m-%d")] for stamp in dates])
    shape = (4, 7) + raw.shape[1:]
    return raw.reshape(shape).mean(axis=1), clim.reshape(shape).mean(axis=1)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (reference, model, week), group in metrics.groupby(
        ["reference", "model", "week"], sort=True
    ):
        row = {
            "reference": reference,
            "model": model,
            "week": int(week),
            "n_cases": int(group.case_id.nunique()),
        }
        for metric in ("acc", "mae_mm_day", "rmse_mm_day", "bias_mm_day"):
            values = group[metric].to_numpy(dtype=float)
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_q25": float(np.percentile(values, 25)),
                    f"{metric}_q75": float(np.percentile(values, 75)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["reference", "model", "week"])


def moving_block_bootstrap(
    ordered: pd.DataFrame,
    value_column: str,
    samples: int,
    seed: int,
    block_length: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for year in (2023, 2024):
        values = ordered.loc[ordered.year == year, value_column].to_numpy(dtype=float)
        n = len(values)
        if n not in (17, 14):
            raise ValueError(f"unexpected {year} paired sample {n}")
        blocks = int(np.ceil(n / block_length))
        starts = rng.integers(0, n, size=(samples, blocks))
        offsets = np.arange(block_length)[None, None, :]
        indices = ((starts[:, :, None] + offsets) % n).reshape(samples, -1)[:, :n]
        draws.append(values[indices])
    return np.concatenate(draws, axis=1).mean(axis=1)


def bootstrap_differences(
    metrics: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for reference_index, reference in enumerate(REFERENCES):
        for week in range(1, 5):
            subset = metrics[
                (metrics.reference == reference) & (metrics.week == week)
            ].copy()
            for metric_index, metric in enumerate(("acc", "mae_mm_day", "rmse_mm_day")):
                pivot = subset.pivot(
                    index=["case_id", "year", "erpas_init"],
                    columns="model",
                    values=metric,
                ).reset_index()
                pivot = pivot.sort_values(["year", "erpas_init"])
                if len(pivot) != 31 or pivot[list(MODELS)].isna().any().any():
                    raise ValueError(f"paired bootstrap input failed for {reference}/{week}/{metric}")
                if metric == "acc":
                    pivot["oriented_difference"] = pivot["FuXi-S2S"] - pivot["ERPAS"]
                    definition = "FuXi-S2S minus ERPAS; positive favors FuXi-S2S"
                else:
                    pivot["oriented_difference"] = pivot["ERPAS"] - pivot["FuXi-S2S"]
                    definition = f"ERPAS minus FuXi-S2S {metric}; positive favors FuXi-S2S"
                distribution = moving_block_bootstrap(
                    pivot,
                    "oriented_difference",
                    samples,
                    seed + reference_index * 100 + week * 10 + metric_index,
                )
                rows.append(
                    {
                        "reference": reference,
                        "week": week,
                        "metric": metric,
                        "n_cases": 31,
                        "block_length_initializations": 4,
                        "bootstrap_samples": samples,
                        "seed": seed + reference_index * 100 + week * 10 + metric_index,
                        "difference_definition": definition,
                        "point_mean_difference": float(pivot.oriented_difference.mean()),
                        "bootstrap_ci_lower_2p5": float(np.percentile(distribution, 2.5)),
                        "bootstrap_ci_upper_97p5": float(np.percentile(distribution, 97.5)),
                    }
                )
    return pd.DataFrame(rows)


def select_initializations(metrics: pd.DataFrame) -> pd.DataFrame:
    imerg = metrics[metrics.reference == "IMERG Final V07B"]
    pivot = imerg.pivot(
        index=["case_id", "erpas_init", "fuxi_init", "year"],
        columns=["model", "week"],
        values="acc",
    )
    delta = np.mean(
        np.column_stack(
            [pivot[("FuXi-S2S", week)] - pivot[("ERPAS", week)] for week in range(1, 5)]
        ),
        axis=1,
    )
    metadata = pivot.index.to_frame(index=False)
    candidates = pd.DataFrame(
        {
            "case_id": metadata["case_id"].astype(str),
            "erpas_init": metadata["erpas_init"].astype(str),
            "fuxi_init": metadata["fuxi_init"].astype(str),
            "year": metadata["year"].astype(int),
        }
    )
    candidates["four_week_mean_delta_acc"] = delta
    candidates = candidates.sort_values(["four_week_mean_delta_acc", "erpas_init"]).reset_index(drop=True)
    candidates["delta_rank"] = np.arange(1, len(candidates) + 1)
    chosen: list[dict] = []
    used: set[str] = set()
    labels = ((0.25, "challenging"), (0.50, "typical"), (0.75, "fuxi_favoring"))
    for percentile, label in labels:
        target = float(np.quantile(candidates.four_week_mean_delta_acc, percentile))
        ranked = candidates.assign(
            distance=(candidates.four_week_mean_delta_acc - target).abs()
        ).sort_values(["distance", "erpas_init"])
        selected = next(row for row in ranked.itertuples() if row.case_id not in used)
        used.add(selected.case_id)
        chosen.append(
            {
                "selection": label,
                "target_percentile": percentile,
                "target_delta_acc": target,
                "case_id": selected.case_id,
                "erpas_init": selected.erpas_init,
                "fuxi_init": selected.fuxi_init,
                "year": int(selected.year),
                "four_week_mean_delta_acc": float(selected.four_week_mean_delta_acc),
                "delta_rank_of_31": int(selected.delta_rank),
                "selection_rule": "nearest percentile of four-week mean FuXi-minus-ERPAS IMERG ACC; ties earliest ERPAS IC",
            }
        )
    return pd.DataFrame(chosen)


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("bootstrap-samples must be at least 1000")
    for directory in (PROCESSED.parent, METRICS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)

    all_cases, excluded = build.build_cases()
    cases = [
        case
        for case in all_cases
        if pd.Timestamp(case["erpas_init"]).month in build.SEASON_WINDOWS["JJAS"]
    ]
    if len(all_cases) != 101 or len(cases) != 31:
        raise ValueError(f"unexpected paired sample all={len(all_cases)}, JJAS={len(cases)}")
    year_counts = {
        year: sum(pd.Timestamp(case["erpas_init"]).year == year for case in cases)
        for year in (2023, 2024)
    }
    if year_counts != {2023: 17, 2024: 14}:
        raise ValueError(f"unexpected JJAS year counts {year_counts}")

    config = json.loads(build.BASE_CONFIG.read_text(encoding="utf-8"))
    config["cases"] = all_cases
    config["model_roots"]["erpas"] = str(build.ERPAS_ROOT)
    config["model_roots"]["fuxi"] = str(build.FUXI_ROOT)
    with xr.open_dataset(build.SOURCE_DATA) as source:
        reference = source.load()
    target_lat, target_lon, _, original_weight, original_land_support = (
        build.accmod.load_land_support(reference)
    )
    observed_imd, imd_climo, india_fraction, weight, land_support, imd_audit = (
        build.remap_imd(
            all_cases,
            target_lat,
            target_lon,
            original_land_support,
            original_weight > 0,
        )
    )
    mask = weight > 0
    if int(mask.sum()) != 169:
        raise ValueError(f"fixed India support is {int(mask.sum())}, expected 169")
    observed_imerg, imerg_obs_audit = load_imerg_observations(
        target_lat, target_lon, india_fraction, weight, land_support
    )
    imerg_climo, imerg_climo_audit = load_imerg_climatology(
        target_lat, target_lon, india_fraction, weight
    )
    model_climatology = build.system_climatologies(
        all_cases, target_lat, target_lon, land_support
    )
    if set(model_climatology) != {case["case_id"] for case in cases}:
        raise ValueError("system climatology cases differ from the 31 paired JJAS sample")

    observed_weekly_fields = {reference_name: [] for reference_name in REFERENCES}
    observed_anomaly_fields = {reference_name: [] for reference_name in REFERENCES}
    forecast_weekly_fields = {model: [] for model in MODELS}
    forecast_anomaly_fields = {model: [] for model in MODELS}
    week_starts: list[list[np.datetime64]] = []
    week_ends: list[list[np.datetime64]] = []
    metric_rows: list[dict] = []
    source_qc: dict[str, dict] = {}

    for case_index, case in enumerate(cases, start=1):
        erpas_init = pd.Timestamp(case["erpas_init"]).tz_localize(None)
        fuxi_init = pd.Timestamp(case["comparison_init"]).tz_localize(None)
        imd_raw, imd_baseline = build.weekly_reference(case, observed_imd, imd_climo)
        imerg_raw, imerg_baseline = imerg_weekly_reference(
            case, observed_imerg, imerg_climo
        )
        references = {
            "IMD": (imd_raw, imd_baseline),
            "IMERG Final V07B": (imerg_raw, imerg_baseline),
        }

        erpas_weekly, erpas_qc = build.load_erpas_variable_count(
            config,
            case,
            target_lat,
            target_lon,
            land_support,
            india_fraction,
        )
        fuxi_weekly, fuxi_qc = build.core.load_standard_model(
            "FuXi-S2S",
            config,
            case,
            target_lat,
            target_lon,
            land_support,
            india_fraction,
        )
        forecasts = {
            "FuXi-S2S": fuxi_weekly / 7.0,
            "ERPAS": erpas_weekly / 7.0,
        }
        source_qc[case["case_id"]] = {"FuXi-S2S": fuxi_qc, "ERPAS": erpas_qc}

        starts = [erpas_init + pd.Timedelta(days=1 + 7 * week) for week in range(4)]
        ends = [stamp + pd.Timedelta(days=7) for stamp in starts]
        week_starts.append([stamp.to_datetime64() for stamp in starts])
        week_ends.append([stamp.to_datetime64() for stamp in ends])

        for reference_name, (truth, baseline) in references.items():
            truth_anomaly = truth - baseline
            truth[:, ~mask] = np.nan
            truth_anomaly[:, ~mask] = np.nan
            observed_weekly_fields[reference_name].append(truth.astype(np.float32))
            observed_anomaly_fields[reference_name].append(truth_anomaly.astype(np.float32))

        for model in MODELS:
            forecast = forecasts[model]
            forecast_anomaly = forecast - model_climatology[case["case_id"]][model]
            forecast[:, ~mask] = np.nan
            forecast_anomaly[:, ~mask] = np.nan
            if not (
                np.isfinite(forecast[:, mask]).all()
                and np.isfinite(forecast_anomaly[:, mask]).all()
            ):
                raise ValueError(f"non-finite forecast field for {case['case_id']}/{model}")
            forecast_weekly_fields[model].append(forecast.astype(np.float32))
            forecast_anomaly_fields[model].append(forecast_anomaly.astype(np.float32))
            for reference_name, (truth, baseline) in references.items():
                truth_anomaly = truth - baseline
                for week in range(4):
                    error = forecast[week] - truth[week]
                    scores = build.engine.error_metrics(error, weight)
                    metric_rows.append(
                        {
                            "case_id": case["case_id"],
                            "erpas_init": erpas_init.strftime("%Y-%m-%d"),
                            "fuxi_init": fuxi_init.strftime("%Y-%m-%d"),
                            "year": erpas_init.year,
                            "week": week + 1,
                            "valid_period_start": starts[week].strftime("%Y-%m-%d"),
                            "valid_period_end_exclusive": ends[week].strftime("%Y-%m-%d"),
                            "reference": reference_name,
                            "model": model,
                            "erpas_source_count": int(erpas_qc["source_count"]),
                            "acc": build.engine.anomaly_correlation(
                                forecast_anomaly[week], truth_anomaly[week], weight
                            ),
                            "mae_mm_day": scores["mae"],
                            "rmse_mm_day": scores["rmse"],
                            "bias_mm_day": scores["bias"],
                        }
                    )
        print(f"processed paired initialization {case_index}/31", flush=True)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["reference", "erpas_init", "model", "week"]
    )
    summary = summarize(metrics)
    bootstrap = bootstrap_differences(
        metrics, args.bootstrap_samples, args.bootstrap_seed
    )
    selected = select_initializations(metrics)

    case_ids = [case["case_id"] for case in cases]
    erpas_dates = [pd.Timestamp(case["erpas_init"]).tz_localize(None).to_datetime64() for case in cases]
    fuxi_dates = [pd.Timestamp(case["comparison_init"]).tz_localize(None).to_datetime64() for case in cases]
    dataset = xr.Dataset(
        data_vars={
            "observed_weekly_rainfall": (
                ("reference", "case", "week", "latitude", "longitude"),
                np.stack([np.stack(observed_weekly_fields[name]) for name in REFERENCES]),
                {"units": "mm day-1"},
            ),
            "observed_weekly_anomaly": (
                ("reference", "case", "week", "latitude", "longitude"),
                np.stack([np.stack(observed_anomaly_fields[name]) for name in REFERENCES]),
                {"units": "mm day-1"},
            ),
            "forecast_weekly_rainfall": (
                ("model", "case", "week", "latitude", "longitude"),
                np.stack([np.stack(forecast_weekly_fields[name]) for name in MODELS]),
                {"units": "mm day-1"},
            ),
            "forecast_weekly_anomaly": (
                ("model", "case", "week", "latitude", "longitude"),
                np.stack([np.stack(forecast_anomaly_fields[name]) for name in MODELS]),
                {"units": "mm day-1"},
            ),
            "india_fraction": (
                ("latitude", "longitude"), india_fraction.astype(np.float32)
            ),
            "spatial_weight": (("latitude", "longitude"), weight.astype(np.float64)),
        },
        coords={
            "reference": list(REFERENCES),
            "model": list(MODELS),
            "case": case_ids,
            "week": np.arange(1, 5, dtype=np.int8),
            "latitude": target_lat,
            "longitude": target_lon,
            "erpas_initialization": (("case",), np.asarray(erpas_dates)),
            "fuxi_initialization": (("case",), np.asarray(fuxi_dates)),
            "week_start": (("case", "week"), np.asarray(week_starts)),
            "week_end_exclusive": (("case", "week"), np.asarray(week_ends)),
        },
        attrs={
            "title": "FuXi-S2S and ERPAS rainfall review fields against IMD and IMERG",
            "sample": "31 paired JJAS initializations: 17 in 2023 and 14 in 2024",
            "pairing": "ERPAS Wednesday IC paired with preceding-Monday FuXi IC; identical Thursday-Wednesday valid weeks",
            "grid": "common FuXi-native 1.5-degree grid; 169-cell fixed India support",
            "forecast_climatologies": "FuXi native 2002-2021 lead/init climatology; ERPAS provider 20-source climatology",
            "imd_climatology": "IMD 1991-2020 calendar-day climatology",
            "imerg_climatology": "IMERG Final V07B fixed 2001-2022 calendar-day climatology",
            "acc_definition": "case-wise area-weighted spatial Pearson anomaly correlation",
            "error_definition": "case-wise area-weighted raw weekly rainfall error in mm/day",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    temporary = PROCESSED.with_name(f".{PROCESSED.name}.{os.getpid()}.tmp")
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={
            name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
            for name in (
                "observed_weekly_rainfall",
                "observed_weekly_anomaly",
                "forecast_weekly_rainfall",
                "forecast_weekly_anomaly",
            )
        },
    )
    os.replace(temporary, PROCESSED)
    metrics_path = METRICS / "per_case_metrics_2023_2024.csv"
    summary_path = METRICS / "summary_metrics_2023_2024.csv"
    bootstrap_path = METRICS / "paired_block_bootstrap_differences_2023_2024.csv"
    selected_path = METRICS / "selected_spatial_initializations_2023_2024.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    selected.to_csv(selected_path, index=False)

    reproduced_max = 0.0
    for row in metrics.itertuples():
        forecast_anomaly = dataset.forecast_weekly_anomaly.sel(
            model=row.model, case=row.case_id, week=row.week
        ).values
        observed_anomaly = dataset.observed_weekly_anomaly.sel(
            reference=row.reference, case=row.case_id, week=row.week
        ).values
        forecast_raw = dataset.forecast_weekly_rainfall.sel(
            model=row.model, case=row.case_id, week=row.week
        ).values
        observed_raw = dataset.observed_weekly_rainfall.sel(
            reference=row.reference, case=row.case_id, week=row.week
        ).values
        acc = build.engine.anomaly_correlation(forecast_anomaly, observed_anomaly, weight)
        error = build.engine.error_metrics(forecast_raw - observed_raw, weight)
        reproduced_max = max(
            reproduced_max,
            abs(acc - row.acc),
            abs(error["mae"] - row.mae_mm_day),
            abs(error["rmse"] - row.rmse_mm_day),
            abs(error["bias"] - row.bias_mm_day),
        )

    checks = {
        "paired_case_count_31": metrics.case_id.nunique() == 31,
        "year_counts_17_14": year_counts == {2023: 17, 2024: 14},
        "exactly_496_primary_rows": len(metrics) == 31 * 2 * 2 * 4,
        "two_models_every_reference_case_week": bool(
            (metrics.groupby(["reference", "case_id", "week"]).model.nunique() == 2).all()
        ),
        "four_lead_weeks": set(metrics.week.unique()) == {1, 2, 3, 4},
        "all_acc_in_bounds": bool(metrics.acc.between(-1, 1).all()),
        "all_errors_finite": bool(
            np.isfinite(metrics[["mae_mm_day", "rmse_mm_day", "bias_mm_day"]]).all().all()
        ),
        "mae_rmse_nonnegative": bool(
            (metrics[["mae_mm_day", "rmse_mm_day"]] >= 0).all().all()
        ),
        "fixed_support_169_cells": int(mask.sum()) == 169,
        "three_distinct_selected_initializations": selected.case_id.nunique() == 3,
        "saved_metrics_reproduced": reproduced_max <= 2e-6,
    }
    audit = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "31 paired FuXi-S2S/ERPAS JJAS starts in 2023-2024, verified independently against IMD and IMERG Final V07B",
        "case_pairing": "ERPAS Wednesday with preceding-Monday FuXi; four identical Thursday-Wednesday seven-day windows",
        "aggregation": "arithmetic mean of case-wise area-weighted spatial ACC; IQR is descriptive",
        "bootstrap": {
            "method": "paired circular moving-block bootstrap within each year",
            "block_length_initializations": 4,
            "samples": args.bootstrap_samples,
            "base_seed": args.bootstrap_seed,
            "purpose": "robustness supplement only; no headline significance claim",
        },
        "forecast_climatologies": {
            "FuXi-S2S": str(build.accmod.FUXI_CLIMO),
            "ERPAS": str(build.accmod.ERPAS_CLIMO),
        },
        "imd": imd_audit,
        "imerg_observations": imerg_obs_audit,
        "imerg_climatology": imerg_climo_audit,
        "excluded_missing_pairs": excluded,
        "source_qc": source_qc,
        "maximum_metric_reproduction_difference": reproduced_max,
        "checks": checks,
        "outputs": [
            str(PROCESSED),
            str(metrics_path),
            str(summary_path),
            str(bootstrap_path),
            str(selected_path),
        ],
    }
    (LOGS / "method_audit.json").write_text(
        json.dumps(audit, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "checks": checks}, indent=2))
    print(f"wrote {PROCESSED}")
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {bootstrap_path}")
    print(f"wrote {selected_path}")
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
