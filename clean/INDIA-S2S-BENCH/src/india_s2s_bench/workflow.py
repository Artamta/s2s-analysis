"""Frozen India S2S benchmark workflow.

The only operation that reads 2025 forecast values is ``run_experiment``.
``preflight`` restricts itself to paths, coordinates, dimensions, and dates.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .bootstrap import paired_interval
from .metrics import aggregate_case_metrics, case_metrics


ARCHIVE = Path("/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1")
TRUTH_ROOT = ARCHIVE / "observations/ground_truth_v1/daily/imd/tp/india_1p5_27x27_v1"
CLIMATOLOGY = ARCHIVE / "observations/ground_truth_v1/climatologies/imd_1991_2019.zarr"
SPATIAL = ARCHIVE / "spatial/spatial_support.zarr"
NATIVE_2025 = {
    "dlesym_v0": Path("/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/dlesym/dlesym_v0_isccp_era5_tpdiag_t2m_00z_2025_ens1/forecasts/2025"),
    "neuralgcm": Path("/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2025_ens10/forecasts/2025"),
}
EXPERIMENTS = {
    "cma": "physics__cma_operational_2020_2025",
    "dlesym_v0": "model-run__dlesym__dlesym_v0_isccp_era5_tpdiag_t2m_00z_2020_2024_ens1",
    "ecmwf": "physics__ecmwf_operational_2020_2025",
    "fuxi_s2s": "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50",
    "ncep": "physics__ncep_operational_2020_2025",
    "neuralgcm": "model-run__neural-gcm__neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10",
    "ukmo": "physics__ukmo_operational_2020_2025",
}
REGIONS = {
    "india": "india_fraction",
    "northwest_india": "northwest_india_fraction",
    "central_india": "central_india_fraction",
    "south_peninsula": "south_peninsula_fraction",
    "east_northeast_india": "east_northeast_india_fraction",
}
SPREAD_MODELS = ("cma", "ecmwf", "fuxi_s2s", "ncep", "neuralgcm", "ukmo")


def load_protocol(root: Path) -> dict:
    return json.loads((root / "protocol.json").read_text(encoding="utf-8"))


def standard_store(model: str, year: int) -> Path:
    return ARCHIVE / "forecasts" / model / EXPERIMENTS[model] / "tp/common_1p5" / f"{year}.zarr"


def standard_manifest(model: str, year: int) -> Path:
    return ARCHIVE / "manifests" / model / EXPERIMENTS[model] / "tp/common_1p5" / f"{year}.json"


def _native_dates(model: str) -> set[np.datetime64]:
    dates = set()
    for path in NATIVE_2025[model].glob("*.nc"):
        token = path.stem
        if len(token) != 8 or not token.isdigit():
            raise ValueError(f"unexpected native forecast filename: {path.name}")
        dates.add(np.datetime64(f"{token[:4]}-{token[4:6]}-{token[6:]}", "D"))
    return dates


def available_dates(model: str, year: int) -> set[np.datetime64]:
    if year == 2025 and model in NATIVE_2025:
        return _native_dates(model)
    manifest = json.loads(standard_manifest(model, year).read_text(encoding="utf-8"))
    return {np.datetime64(value, "D") for value in manifest["initializations"]}


def common_dates(models: Iterable[str], year: int, months: Iterable[int]) -> list[np.datetime64]:
    common = set.intersection(*(available_dates(model, year) for model in models))
    wanted = set(int(month) for month in months)
    return sorted(date for date in common if pd.Timestamp(date).month in wanted)


def _metadata_record(path: Path, kind: str) -> dict:
    record = {"kind": kind, "path": str(path), "exists": path.exists()}
    if path.suffix == ".zarr" and path.exists():
        meta = path / ".zmetadata"
        record["zmetadata_sha256"] = sha256_file(meta)
        # Parse consolidated metadata directly. Opening xarray here can read one
        # coordinate chunk per initialization on the archive filesystem and is
        # unnecessary for a values-blind preflight.
        consolidated = json.loads(meta.read_text(encoding="utf-8"))["metadata"]
        arrays = {}
        for key, value in consolidated.items():
            if key.endswith("/.zarray"):
                arrays[key.removesuffix("/.zarray")] = value["shape"]
        record["arrays"] = arrays
    return record


def preflight(root: Path, output: Path) -> dict:
    protocol = load_protocol(root)
    models = protocol["models"]
    years = sorted(set(protocol["splits"]["train_years"] + protocol["splits"]["validation_years"] + protocol["splits"]["test_years"]))
    records = []
    coverage = []
    for year in years:
        dates_by_model = {}
        for model in models:
            if year == 2025 and model in NATIVE_2025:
                path = NATIVE_2025[model]
                records.append({"kind": "native_forecast_directory", "path": str(path), "exists": path.exists(), "files": len(list(path.glob("*.nc")))})
            else:
                path = standard_store(model, year)
                records.append(_metadata_record(path, "standardized_forecast"))
                records.append({"kind": "standardized_manifest", "path": str(standard_manifest(model, year)), "exists": standard_manifest(model, year).exists(), "sha256": sha256_file(standard_manifest(model, year))})
            dates_by_model[model] = available_dates(model, year)
        common = common_dates(models, year, protocol["season"]["initialization_months"])
        for model in models:
            jjas = sorted(date for date in dates_by_model[model] if pd.Timestamp(date).month in protocol["season"]["initialization_months"])
            coverage.append({"year": year, "model": model, "jjas_available": len(jjas), "common_jjas": len(common), "missing_from_common": len(set(common) - set(jjas)), "common_dates": " ".join(map(str, common))})
        truth = TRUTH_ROOT / f"{year}.zarr"
        records.append(_metadata_record(truth, "imd_truth"))
    records.extend([_metadata_record(CLIMATOLOGY, "imd_climatology"), _metadata_record(SPATIAL, "spatial_support")])
    failures = [record["path"] for record in records if not record["exists"]]
    if failures:
        raise FileNotFoundError(f"preflight missing inputs: {failures}")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(output / "coverage_preflight.csv", index=False)
    report = {
        "status": "passed",
        "forecast_values_read": False,
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "records": records,
    }
    write_json(output / "preflight.json", report)
    return report


@dataclass
class YearData:
    dates: np.ndarray
    forecasts: np.ndarray  # case, week, model, lat, lon
    spreads: np.ndarray  # case, week, spread-model, lat, lon
    truth: np.ndarray  # case, week, lat, lon
    climatology: np.ndarray  # case, week, lat, lon
    latitude: np.ndarray
    longitude: np.ndarray


def _load_standard(model: str, year: int, dates: list[np.datetime64]) -> tuple[np.ndarray, np.ndarray | None]:
    manifest = json.loads(standard_manifest(model, year).read_text(encoding="utf-8"))
    positions = {np.datetime64(value, "D"): index for index, value in enumerate(manifest["initializations"])}
    indices = [positions[date] for date in dates]
    with xr.open_zarr(standard_store(model, year), consolidated=True) as ds:
        if ds["ensemble_mean_weekly"].attrs.get("units") != "mm day-1":
            raise ValueError(f"{model}/{year}: unexpected weekly precipitation units")
        if ds.sizes.get("lead_week") != 6 or ds.sizes.get("latitude") != 27 or ds.sizes.get("longitude") != 27:
            raise ValueError(f"{model}/{year}: standardized shape contract changed")
        selected = ds.isel(init=indices)
        mean = selected["ensemble_mean_weekly"].load().values.astype(np.float32)
        spread = selected["ensemble_std_weekly"].load().values.astype(np.float32) if model in SPREAD_MODELS else None
    return mean, spread


def _load_native(model: str, dates: list[np.datetime64]) -> tuple[np.ndarray, np.ndarray | None]:
    means, spreads = [], []
    for date in dates:
        path = NATIVE_2025[model] / f"{pd.Timestamp(date):%Y%m%d}.nc"
        with xr.open_dataset(path) as ds:
            if ds["tp"].attrs.get("units") != "mm day-1":
                raise ValueError(f"{path}: expected tp units mm day-1")
            if not np.array_equal(ds.lead_day.values, np.arange(1, 43)):
                raise ValueError(f"{path}: lead days are not 1...42")
            if not np.array_equal(ds.latitude.values, np.arange(39.0, -0.01, -1.5)) or not np.array_equal(ds.longitude.values, np.arange(60.0, 99.01, 1.5)):
                raise ValueError(f"{path}: native output is not on the canonical grid")
            expected_end = date + np.arange(1, 43).astype("timedelta64[D]")
            if not np.array_equal(ds.forecast_period_end.values.astype("datetime64[D]"), expected_end):
                raise ValueError(f"{path}: forecast period endpoints are misaligned")
            daily = ds["tp"].transpose("member", "lead_day", "latitude", "longitude").load().values.astype(np.float32)
        weekly = daily[:, :42].reshape(daily.shape[0], 6, 7, 27, 27).mean(axis=2, dtype=np.float64)
        means.append(weekly.mean(axis=0, dtype=np.float64).astype(np.float32))
        if model in SPREAD_MODELS:
            spreads.append(weekly.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32))
    return np.stack(means), np.stack(spreads) if spreads else None


def _load_truth(year: int, dates: list[np.datetime64]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.stack([date + np.arange(1, 43).astype("timedelta64[D]") for date in dates])
    with xr.open_zarr(TRUTH_ROOT / f"{year}.zarr", consolidated=True) as ds:
        truth_daily = ds["observation"].sel(time=valid.ravel().astype("datetime64[ns]")).load().values
        latitude = ds.latitude.values
        longitude = ds.longitude.values
    truth = truth_daily.reshape(len(dates), 6, 7, 27, 27).mean(axis=2, dtype=np.float64).astype(np.float32)
    with xr.open_zarr(CLIMATOLOGY, consolidated=True) as ds:
        lookup = {pd.Timestamp(value).strftime("%m-%d"): index for index, value in enumerate(ds.climatology_date.values)}
        indices = np.array([lookup[pd.Timestamp(value).strftime("%m-%d")] for value in valid.ravel()])
        clim_daily = ds["climatology_mean"].isel(climatology_day=xr.DataArray(indices, dims="sample")).load().values
    clim = clim_daily.reshape(len(dates), 6, 7, 27, 27).mean(axis=2, dtype=np.float64).astype(np.float32)
    return truth, clim, latitude, longitude


def load_year(protocol: dict, year: int) -> YearData:
    models = protocol["models"]
    dates = common_dates(models, year, protocol["season"]["initialization_months"])
    if not dates:
        raise ValueError(f"no common JJAS dates for {year}")
    forecasts, spread_by_model = [], {}
    for model in models:
        if year == 2025 and model in NATIVE_2025:
            mean, spread = _load_native(model, dates)
        else:
            mean, spread = _load_standard(model, year, dates)
        forecasts.append(mean)
        if spread is not None:
            spread_by_model[model] = spread
    truth, climatology, latitude, longitude = _load_truth(year, dates)
    return YearData(
        dates=np.asarray(dates),
        forecasts=np.stack(forecasts, axis=2),
        spreads=np.stack([spread_by_model[model] for model in SPREAD_MODELS], axis=2),
        truth=truth,
        climatology=climatology,
        latitude=latitude,
        longitude=longitude,
    )


def _weights_and_regions() -> dict[str, np.ndarray]:
    with xr.open_zarr(SPATIAL, consolidated=True) as spatial, xr.open_zarr(TRUTH_ROOT / "2025.zarr", consolidated=True) as truth:
        area = spatial["cell_area_km2"].load().values
        observation_fraction = truth["observation_fraction"].load().values
        return {name: area * spatial[variable].load().values * observation_fraction for name, variable in REGIONS.items()}


def _feature_cube(data: YearData, week: int, variant: str, models: list[str]) -> tuple[np.ndarray, list[str]]:
    anomaly = data.forecasts[:, week] - data.climatology[:, week, None]
    parts, names = [], []
    if variant in {"piggycast_forecast_only", "piggycast_full"}:
        parts.append(anomaly)
        names.extend([f"forecast_anomaly_{model}" for model in models])
        parts.append(data.spreads[:, week])
        names.extend([f"spread_{model}" for model in SPREAD_MODELS])
    if variant in {"piggycast_full", "location_calendar_only"}:
        ncase = len(data.dates)
        lat2d, lon2d = np.meshgrid(data.latitude, data.longitude, indexing="ij")
        midpoint = data.dates + np.timedelta64(week * 7 + 4, "D")
        angle = 2 * np.pi * np.array([pd.Timestamp(date).dayofyear for date in midpoint]) / 365.25
        static = np.stack([
            np.broadcast_to(lat2d, (ncase, 27, 27)),
            np.broadcast_to(lon2d, (ncase, 27, 27)),
            np.broadcast_to(np.sin(angle)[:, None, None], (ncase, 27, 27)),
            np.broadcast_to(np.cos(angle)[:, None, None], (ncase, 27, 27)),
        ], axis=1)
        parts.append(static)
        names.extend(["latitude", "longitude", "verification_doy_sin", "verification_doy_cos"])
    return np.concatenate(parts, axis=1), names


def _flatten_features(cube: np.ndarray, target: np.ndarray, support: np.ndarray):
    # cube: case, feature, lat, lon
    x = cube.transpose(0, 2, 3, 1).reshape(-1, cube.shape[1])
    y = target.reshape(-1)
    w = np.broadcast_to(support, target.shape).reshape(-1)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    return x[mask], y[mask], w[mask], mask


def validation_baselines(data: YearData, models: list[str], india_weight: np.ndarray) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    weights = np.empty((6, len(models)), dtype=np.float64)
    selected = []
    rows = []
    for week in range(6):
        model_rmse = []
        support = np.broadcast_to(india_weight, data.truth[:, week].shape)
        for index, model in enumerate(models):
            error = data.forecasts[:, week, index] - data.truth[:, week]
            mask = np.isfinite(error) & (support > 0)
            rmse = float(np.sqrt(np.average(error[mask] ** 2, weights=support[mask])))
            model_rmse.append(rmse)
        inverse = 1 / np.asarray(model_rmse)
        weights[week] = inverse / inverse.sum()
        selected.append(models[int(np.argmin(model_rmse))])
        for model, rmse, value in zip(models, model_rmse, weights[week]):
            rows.append({"lead_week": week + 1, "model": model, "validation_rmse": rmse, "inverse_rmse_weight": value, "selected": model == selected[-1]})
    return weights, selected, pd.DataFrame(rows)


def _fit_predict_xgb(protocol: dict, train: list[YearData], validation: YearData, test: YearData, variant: str, india_weight: np.ndarray, model_dir: Path):
    from xgboost import XGBRegressor

    config = protocol["xgboost"]
    predictions = np.empty_like(test.truth, dtype=np.float32)
    metadata = []
    for week in range(6):
        train_cube_parts, train_targets = [], []
        for data in train:
            cube, feature_names = _feature_cube(data, week, variant, protocol["models"])
            train_cube_parts.append(cube)
            train_targets.append(data.truth[:, week] - data.climatology[:, week])
        train_cube = np.concatenate(train_cube_parts)
        train_target = np.concatenate(train_targets)
        validation_cube, _ = _feature_cube(validation, week, variant, protocol["models"])
        validation_target = validation.truth[:, week] - validation.climatology[:, week]
        x_train, y_train, w_train, _ = _flatten_features(train_cube, train_target, india_weight)
        x_val, y_val, w_val, _ = _flatten_features(validation_cube, validation_target, india_weight)
        params = {key: value for key, value in config.items() if key not in {"early_stopping_rounds", "seed"}}
        params["random_state"] = config["seed"]
        selector = XGBRegressor(**params, early_stopping_rounds=config["early_stopping_rounds"], n_jobs=max(1, min(8, os.cpu_count() or 1)))
        selector.fit(
            x_train,
            y_train,
            sample_weight=w_train / np.mean(w_train),
            eval_set=[(x_val, y_val)],
            sample_weight_eval_set=[w_val / np.mean(w_val)],
            verbose=False,
        )
        best_trees = int(selector.best_iteration) + 1
        # Frozen refit uses the validation-selected tree count and all pre-2025 rows.
        refit_cube = np.concatenate([train_cube, validation_cube])
        refit_target = np.concatenate([train_target, validation_target])
        x_refit, y_refit, w_refit, _ = _flatten_features(refit_cube, refit_target, india_weight)
        refit_params = dict(params)
        refit_params["n_estimators"] = best_trees
        final = XGBRegressor(**refit_params, n_jobs=max(1, min(8, os.cpu_count() or 1)))
        final.fit(x_refit, y_refit, sample_weight=w_refit / np.mean(w_refit), verbose=False)
        test_cube, _ = _feature_cube(test, week, variant, protocol["models"])
        flat = test_cube.transpose(0, 2, 3, 1).reshape(-1, test_cube.shape[1])
        valid = np.all(np.isfinite(flat), axis=1)
        anomaly_prediction = np.full(len(flat), np.nan, dtype=np.float32)
        anomaly_prediction[valid] = final.predict(flat[valid]).astype(np.float32)
        predictions[:, week] = anomaly_prediction.reshape(len(test.dates), 27, 27) + test.climatology[:, week]
        path = model_dir / f"{variant}_week{week + 1}.json"
        final.save_model(path)
        metadata.append({"method": variant, "lead_week": week + 1, "features": feature_names, "best_trees": best_trees, "train_rows": len(x_train), "validation_rows": len(x_val), "refit_rows": len(x_refit), "model_path": str(path)})
    return predictions, metadata


def _evaluate(predictions: dict[str, np.ndarray], test: YearData, regions: dict[str, np.ndarray], wet_threshold: float) -> pd.DataFrame:
    rows = []
    for method, fields in predictions.items():
        for case, date in enumerate(test.dates):
            for week in range(6):
                for region, weight in regions.items():
                    values = case_metrics(fields[case, week], test.truth[case, week], test.climatology[case, week], weight, wet_threshold)
                    rows.append({"method": method, "initialization": str(date), "lead_week": week + 1, "region": region, **values})
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = [("all", ["method", "region"]), ("by_lead", ["method", "region", "lead_week"])]
    for aggregation, keys in groupings:
        for values, group in frame.groupby(keys, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = dict(zip(keys, values))
            row.update({"aggregation": aggregation, **aggregate_case_metrics(group)})
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_tables(frame: pd.DataFrame, protocol: dict):
    india = frame[frame.region == "india"]
    comparisons = [
        ("piggycast_full", "equal_weight"),
        ("piggycast_full", "validation_selected_model"),
        ("piggycast_forecast_only", "location_calendar_only"),
    ]
    for model in protocol["models"]:
        comparisons.append((model, "equal_weight"))
    rows, sensitivity = [], []
    for metric in ("acc", "rmse", "mae", "bias", "wet_fraction_error"):
        pivot = india.pivot(index=["initialization", "lead_week"], columns="method", values=metric).sort_index()
        initialization = pivot.index.get_level_values("initialization").to_numpy()
        for a, b in comparisons:
            for block in protocol["uncertainty"]["sensitivity_block_lengths"]:
                interval = paired_interval(pivot[a].to_numpy(), pivot[b].to_numpy(), initialization, draws=protocol["uncertainty"]["draws"], block_length=block, seed=protocol["uncertainty"]["seed"] + block)
                row = {"method_a": a, "method_b": b, "metric": metric, **interval}
                sensitivity.append(row)
                if block == protocol["uncertainty"]["primary_block_length_initializations"]:
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(sensitivity)


def regional_acc_intervals(frame: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        subset = frame[frame.region == region]
        pivot = subset.pivot(index=["initialization", "lead_week"], columns="method", values="acc").sort_index()
        initialization = pivot.index.get_level_values("initialization").to_numpy()
        interval = paired_interval(
            pivot["piggycast_full"].to_numpy(),
            pivot["equal_weight"].to_numpy(),
            initialization,
            draws=protocol["uncertainty"]["draws"],
            block_length=protocol["uncertainty"]["primary_block_length_initializations"],
            seed=protocol["uncertainty"]["seed"] + 101,
        )
        rows.append({"region": region, "method_a": "piggycast_full", "method_b": "equal_weight", "metric": "acc", **interval})
    return pd.DataFrame(rows)


def _gate_report(aggregate: pd.DataFrame, paired: pd.DataFrame, case_frame: pd.DataFrame, protocol: dict) -> dict:
    primary = paired.set_index(["method_a", "method_b", "metric"])
    india = aggregate[(aggregate.aggregation == "all") & (aggregate.region == "india")].set_index("method")
    checks = {}
    for name, a, b in [
        ("full_acc_vs_equal", "piggycast_full", "equal_weight"),
        ("full_acc_vs_selected", "piggycast_full", "validation_selected_model"),
        ("forecast_acc_vs_location", "piggycast_forecast_only", "location_calendar_only"),
    ]:
        interval = primary.loc[(a, b, "acc")]
        checks[name] = {"passed": bool(interval.ci_low > 0), "effect": float(interval.effect), "ci_low": float(interval.ci_low), "ci_high": float(interval.ci_high)}
    equal_rmse = float(india.loc["equal_weight", "rmse"])
    full_rmse = float(india.loc["piggycast_full", "rmse"])
    checks["rmse_guard"] = {"passed": bool(full_rmse <= equal_rmse * (1 + protocol["claim_gates"]["maximum_relative_rmse_regression"])), "full": full_rmse, "equal": equal_rmse}
    bias_delta = abs(float(india.loc["piggycast_full", "bias"])) - abs(float(india.loc["equal_weight", "bias"]))
    checks["bias_guard"] = {"passed": bool(bias_delta <= protocol["claim_gates"]["maximum_absolute_bias_regression_mm_day"]), "absolute_bias_regression": bias_delta}
    regional = aggregate[(aggregate.aggregation == "all") & (aggregate.region != "india")].pivot(index="region", columns="method", values="acc")
    deltas = regional["piggycast_full"] - regional["equal_weight"]
    positive = int((deltas > 0).sum())
    checks["regional_guard"] = {"passed": bool(positive >= protocol["claim_gates"]["minimum_regions_with_positive_acc_delta_vs_equal_weight"]), "positive_regions": positive, "deltas": {key: float(value) for key, value in deltas.items()}}
    passed = all(check["passed"] for check in checks.values())
    return {"headline_claim_allowed": passed, "required_wording": "confirmatory improvement" if passed else "mixed or negative confirmatory result", "checks": checks}


def run_experiment(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    protocol = load_protocol(root)
    shutil.copy2(root / "protocol.json", output / "protocol.frozen.json")
    preflight(root, output / "preflight")
    train = [load_year(protocol, year) for year in protocol["splits"]["train_years"]]
    validation = load_year(protocol, protocol["splits"]["validation_years"][0])
    # This is the single confirmatory opening of forecast values.
    test = load_year(protocol, protocol["splits"]["test_years"][0])
    regions = _weights_and_regions()
    weights, selected, weight_table = validation_baselines(validation, protocol["models"], regions["india"])
    weight_table.to_csv(output / "validation_weights.csv", index=False)
    predictions = {model: test.forecasts[:, :, index] for index, model in enumerate(protocol["models"])}
    predictions["equal_weight"] = np.mean(test.forecasts, axis=2, dtype=np.float64).astype(np.float32)
    predictions["validation_weighted"] = np.einsum("cwmxy,wm->cwxy", test.forecasts, weights).astype(np.float32)
    predictions["validation_selected_model"] = np.stack([test.forecasts[:, week, protocol["models"].index(selected[week])] for week in range(6)], axis=1)
    model_dir = output / "models"
    model_dir.mkdir()
    fit_metadata = []
    for variant in ("piggycast_forecast_only", "piggycast_full", "location_calendar_only"):
        predictions[variant], records = _fit_predict_xgb(protocol, train, validation, test, variant, regions["india"], model_dir)
        fit_metadata.extend(records)
    pd.DataFrame(fit_metadata).to_json(output / "fit_metadata.json", orient="records", indent=2)
    frame = _evaluate(predictions, test, regions, protocol["metrics"]["wet_threshold_mm_day"])
    frame.to_csv(output / "case_metrics.csv", index=False, float_format="%.10g")
    aggregate = aggregate_metrics(frame)
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False, float_format="%.10g")
    paired, sensitivity = _paired_tables(frame, protocol)
    paired.to_csv(output / "paired_intervals.csv", index=False, float_format="%.10g")
    sensitivity.to_csv(output / "bootstrap_sensitivity.csv", index=False, float_format="%.10g")
    regional_intervals = regional_acc_intervals(frame, protocol)
    regional_intervals.to_csv(output / "regional_acc_intervals.csv", index=False, float_format="%.10g")
    gates = _gate_report(aggregate, paired, frame, protocol)
    write_json(output / "gate_report.json", gates)
    coverage_rows = [{"year": year, "initialization": str(date), "included": True} for year_data, year in [(item, int(item.dates[0].astype("datetime64[Y]").astype(int) + 1970)) for item in [*train, validation, test]] for date in year_data.dates]
    pd.DataFrame(coverage_rows).to_csv(output / "coverage.csv", index=False)
    generate_reporting(output, aggregate, paired, regional_intervals, gates)
    manifest = build_manifest(root, output, protocol, test, selected)
    write_json(output / "manifest.json", manifest)
    return gates


def generate_reporting(output: Path, aggregate: pd.DataFrame, paired: pd.DataFrame, regional_intervals: pd.DataFrame, gates: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    table_dir = output / "tables"
    figure_dir.mkdir(exist_ok=True); table_dir.mkdir(exist_ok=True)
    india = aggregate[(aggregate.aggregation == "by_lead") & (aggregate.region == "india")]
    preferred = ["cma", "dlesym_v0", "ecmwf", "fuxi_s2s", "ncep", "neuralgcm", "ukmo", "equal_weight", "validation_weighted", "piggycast_forecast_only", "piggycast_full", "location_calendar_only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method in preferred:
        subset = india[india.method == method].sort_values("lead_week")
        style = {"linewidth": 2.5} if method in {"piggycast_full", "equal_weight"} else {"linewidth": 1.1, "alpha": .75}
        axes[0].plot(subset.lead_week, subset.acc, marker="o", label=method, **style)
        axes[1].plot(subset.lead_week, subset.rmse, marker="o", label=method, **style)
    axes[0].set(xlabel="Lead week", ylabel="Spatial ACC", title="India anomaly correlation")
    axes[1].set(xlabel="Lead week", ylabel="RMSE (mm day$^{-1}$)", title="India rainfall error")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
    fig.savefig(figure_dir / "skill_by_lead.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "skill_by_lead.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    stages = [
        (0.08, "Seven systems\n42 daily leads"),
        (0.30, "Common 1.5° grid\nIMD 1991–2019 anomalies"),
        (0.53, "2020–23 train\n2024 validation"),
        (0.75, "Frozen XGBoost\n3 feature variants"),
        (0.93, "Untouched 2025\n35 initializations"),
    ]
    for index, (x, label) in enumerate(stages):
        ax.text(x, .5, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=.45", "facecolor": "#e8f0f7", "edgecolor": "#345"})
        if index < len(stages) - 1:
            ax.annotate("", xy=(stages[index + 1][0] - .09, .5), xytext=(x + .09, .5), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(.53, .12, "All forecasts and observations share init+1…+42 valid-day support; test fitting is forbidden", ha="center", fontsize=8)
    ax.set_axis_off()
    fig.savefig(figure_dir / "pipeline.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    comparison = paired[(paired.metric == "acc") & (paired.method_a.str.startswith("piggycast"))]
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    labels = [f"{a} − {b}" for a, b in zip(comparison.method_a, comparison.method_b)]
    ax.errorbar(comparison.effect, np.arange(len(comparison)), xerr=[comparison.effect - comparison.ci_low, comparison.ci_high - comparison.effect], fmt="o")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Paired mean spatial-ACC difference", title="2025 moving-block intervals")
    fig.savefig(figure_dir / "paired_acc.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "paired_acc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    regional = regional_intervals.sort_values("effect")
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    y = np.arange(len(regional))
    ax.errorbar(regional.effect, y, xerr=[regional.effect - regional.ci_low, regional.ci_high - regional.effect], fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=y, yticklabels=regional.region.str.replace("_", " "), xlabel="PiggyCast full − equal-weight spatial ACC", title="2025 regional paired differences")
    fig.savefig(figure_dir / "regional_acc.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "regional_acc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    overall = aggregate[(aggregate.aggregation == "all") & (aggregate.region == "india")][["method", "acc", "rmse", "mae", "bias", "wet_fraction_error"]].sort_values("acc", ascending=False)
    overall.to_csv(table_dir / "overall_2025.csv", index=False, float_format="%.4f")
    latex = overall.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True, caption="Confirmatory 2025 JJAS-initialized India scores.", label="tab:overall")
    (table_dir / "overall_2025.tex").write_text(latex, encoding="utf-8")
    write_json(table_dir / "claim_status.json", gates)


def build_manifest(root: Path, output: Path, protocol: dict, test: YearData, selected: list[str]) -> dict:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "audit_report.json"}:
            files.append({"path": str(path.relative_to(output)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except Exception:
        commit = None
        dirty = None
    source_paths = [root / "run.py", root / "protocol.json", *sorted((root / "src/india_s2s_bench").glob("*.py"))]
    sources = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in source_paths]
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "repository_commit": commit,
        "repository_working_tree_dirty": dirty,
        "source_files": sources,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "test_initializations": [str(value) for value in test.dates],
        "test_initialization_count": len(test.dates),
        "validation_selected_models_by_week": selected,
        "neural_correction_included": False,
        "files": files,
    }


def audit(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    failures = []
    for item in manifest["files"]:
        path = output / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            failures.append(item["path"])
    case_frame = pd.read_csv(output / "case_metrics.csv")
    regenerated = aggregate_metrics(case_frame).sort_values(["aggregation", "region", "method", "lead_week"], na_position="first").reset_index(drop=True)
    saved = pd.read_csv(output / "aggregate_metrics.csv").sort_values(["aggregation", "region", "method", "lead_week"], na_position="first").reset_index(drop=True)
    columns = ["acc", "rmse", "mae", "bias", "wet_fraction_error", "negative_fraction"]
    aggregate_match = len(saved) == len(regenerated) and all(np.allclose(saved[column], regenerated[column], equal_nan=True, rtol=2e-9, atol=2e-9) for column in columns)
    if not aggregate_match:
        failures.append("aggregate_metrics.csv: independent recomputation mismatch")
    report = {"status": "passed" if not failures else "failed", "hashes_checked": len(manifest["files"]), "aggregate_recomputed": aggregate_match, "failures": failures}
    write_json(output / "audit_report.json", report)
    if failures:
        raise ValueError(json.dumps(report, indent=2))
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
