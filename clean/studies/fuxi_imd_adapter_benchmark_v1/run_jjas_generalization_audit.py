#!/usr/bin/env python3
"""Audit a frozen FuXi--IMD adapter on 2022--2024 JJAS initializations.

This is a development generalization audit, not a model-selection stage.  It
loads only 2022--2024 operational FuXi and IMD verification data, keeps 2025
initializations out of scope, and applies the already frozen validation choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr

import run_benchmark as benchmark


base = benchmark.base
diagnostic = benchmark.diagnostic
AUDIT_YEARS = (2022, 2023, 2024)
INITIALIZATION_MONTHS = (6, 7, 8, 9)
EXPECTED_COUNTS = {2022: 35, 2023: 35, 2024: 30}
METHODS = ("raw_fuxi", "log_bias", "selected_adapter")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "FuXi + log-bias",
    "selected_adapter": "Frozen validation-selected forecast",
}


def latest_full_context_run() -> Path:
    root = (
        benchmark.BIAS
        / "results"
        / "fuxi_imd_full_context_compact_allweeks"
    )
    runs = sorted(
        candidate
        for candidate in root.glob("full_*")
        if (candidate / "manifest.json").is_file()
    )
    if not runs:
        raise FileNotFoundError("no completed full-context training run found")
    return runs[-1]


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


def load_operational_fuxi_jjas(
    tp_records: list[dict[str, Any]],
    t2m_records: list[dict[str, Any]],
) -> tuple[base.ForecastData, np.ndarray, dict[int, pd.DatetimeIndex], tuple[str, ...]]:
    means: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    initializations: list[np.ndarray] = []
    paired: dict[int, pd.DatetimeIndex] = {}
    source_stores: list[str] = []
    latitude = None
    longitude = None
    for year in AUDIT_YEARS:
        tp_record = benchmark.find_record(tp_records, "fuxi_s2s", year)
        t2m_record = benchmark.find_record(t2m_records, "fuxi_s2s", year)
        with xr.open_zarr(tp_record["store"], consolidated=True) as dataset:
            all_inits = pd.DatetimeIndex(dataset.init.values)
            inits = all_inits[all_inits.month.isin(INITIALIZATION_MONTHS)]
            if len(inits) != EXPECTED_COUNTS[year]:
                raise ValueError(
                    f"{year}: expected {EXPECTED_COUNTS[year]} JJAS starts, "
                    f"found {len(inits)}"
                )
            if dataset.ensemble_mean_weekly.attrs.get("units") != "mm day-1":
                raise ValueError(f"{year}: unexpected operational TP units")
            means.append(
                dataset.ensemble_mean_weekly.sel(init=inits).load().values.astype(
                    np.float32
                )
            )
            spreads.append(
                dataset.ensemble_std_weekly.sel(init=inits).load().values.astype(
                    np.float32
                )
            )
            current_latitude = dataset.latitude.values.astype(np.float64)
            current_longitude = dataset.longitude.values.astype(np.float64)
        with xr.open_zarr(t2m_record["store"], consolidated=True) as dataset:
            if dataset.ensemble_mean_weekly.attrs.get("units") != "degC":
                raise ValueError(f"{year}: unexpected operational T2M units")
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
            raise ValueError("operational FuXi grid changed between audit years")
        paired[year] = inits
        initializations.append(inits.values.astype("datetime64[D]"))
        source_stores.extend((tp_record["store"], t2m_record["store"]))

    assert latitude is not None and longitude is not None
    inits = np.concatenate(initializations)
    mean = np.concatenate(means)
    spread = np.concatenate(spreads)
    t2m = np.concatenate(temperatures)
    expected_shape = (sum(EXPECTED_COUNTS.values()), 6, 27, 27)
    if mean.shape != expected_shape or spread.shape != mean.shape or t2m.shape != mean.shape:
        raise ValueError(
            f"unexpected operational FuXi JJAS shapes: {mean.shape}, "
            f"{spread.shape}, {t2m.shape}"
        )
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(spread).all()
        or not np.isfinite(t2m).all()
        or np.any(mean < 0.0)
        or np.any(spread < 0.0)
    ):
        raise ValueError("operational FuXi JJAS predictors are invalid")
    if np.any(pd.DatetimeIndex(inits).year == 2025):
        raise ValueError("2025 initialization entered the development audit")
    forecast = base.ForecastData(
        initializations=inits,
        valid_dates=base.derive_valid_dates(inits),
        ensemble_mean=mean,
        ensemble_spread=spread,
        latitude=latitude,
        longitude=longitude,
        source_files=tuple(source_stores),
    )
    return forecast, t2m, paired, tuple(source_stores)


def load_imd_audit_fields(
    forecast: base.ForecastData,
    training_climatology: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    dates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    stores: list[str] = []
    required_years = sorted(
        set(pd.DatetimeIndex(forecast.valid_dates.reshape(-1)).year.tolist())
    )
    if required_years != list(AUDIT_YEARS):
        raise ValueError(f"unexpected IMD verification years: {required_years}")
    for year in required_years:
        year_dates, year_values, year_coverage = diagnostic.annual_observation(year)
        dates.append(year_dates.values.astype("datetime64[D]"))
        values.append(year_values)
        coverage.append(year_coverage)
        stores.append(
            str(
                diagnostic.OBS_ROOT
                / f"daily/imd/tp/india_1p5_27x27_v1/{year}.zarr"
            )
        )
    all_dates = np.concatenate(dates)
    all_values = np.concatenate(values)
    all_coverage = np.concatenate(coverage)
    requested = forecast.valid_dates.reshape(-1)
    positions = np.searchsorted(all_dates, requested)
    if np.any(positions >= len(all_dates)) or not np.array_equal(
        all_dates[positions], requested
    ):
        raise ValueError("one or more audit IMD verification dates are missing")
    daily = all_values[positions].reshape(*forecast.valid_dates.shape, 27, 27)
    daily_coverage = all_coverage[positions].reshape(
        *forecast.valid_dates.shape, 27, 27
    )
    truth = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    weekly_coverage = np.min(daily_coverage, axis=2).astype(np.float32)
    climatology = np.mean(
        training_climatology[base.calendar_positions(forecast.valid_dates)],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    return truth, climatology, weekly_coverage, tuple(stores)


def score(
    forecast: base.ForecastData,
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    climatology: np.ndarray,
    weekly_coverage: np.ndarray,
    support: np.ndarray,
) -> pd.DataFrame:
    areas = benchmark.load_spatial_areas(support)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        prediction = predictions[method]
        for case_index, init in enumerate(
            pd.DatetimeIndex(forecast.initializations)
        ):
            for lead_index in range(6):
                for region, region_area in areas.items():
                    weights = region_area * weekly_coverage[case_index, lead_index]
                    absolute = diagnostic.weighted_metrics(
                        prediction[case_index, lead_index],
                        truth[case_index, lead_index],
                        weights,
                    )
                    anomaly = diagnostic.weighted_metrics(
                        prediction[case_index, lead_index]
                        - climatology[case_index, lead_index],
                        truth[case_index, lead_index]
                        - climatology[case_index, lead_index],
                        weights,
                    )
                    rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "init": str(init.date()),
                            "year": int(init.year),
                            "lead_week": lead_index + 1,
                            "region": region,
                            "region_label": diagnostic.REGION_LABELS[region],
                            "rmse_mm_day": absolute["rmse_mm_day"],
                            "mae_mm_day": absolute["mae_mm_day"],
                            "bias_mm_day": absolute["bias_mm_day"],
                            "acc": anomaly["acc"],
                            "valid_cell_count": absolute["valid_cell_count"],
                            "effective_area_km2": absolute["effective_area_km2"],
                        }
                    )
    result = pd.DataFrame(rows)
    expected = len(METHODS) * len(forecast.initializations) * 6 * len(areas)
    if len(result) != expected or result.duplicated(
        ["method", "init", "lead_week", "region"]
    ).any():
        raise ValueError(f"audit score contract failed: {len(result)} rows")
    return result


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]
    by_lead = (
        cases.groupby(
            ["region", "region_label", "method", "method_label", "lead_week"],
            as_index=False,
        )[metrics]
        .mean()
    )
    regional = (
        cases.groupby(
            ["region", "region_label", "method", "method_label"],
            as_index=False,
        )[metrics]
        .mean()
    )
    yearly = (
        cases.groupby(
            ["year", "region", "region_label", "method", "method_label", "lead_week"],
            as_index=False,
        )[metrics]
        .mean()
    )
    return by_lead, regional, yearly


def generalization_guards(
    by_lead: pd.DataFrame, regional: pd.DataFrame, yearly: pd.DataFrame
) -> dict[str, Any]:
    all_india = by_lead.loc[by_lead.region.eq("all_india")]
    lead_rmse = all_india.pivot(index="lead_week", columns="method", values="rmse_mm_day")
    lead_acc = all_india.pivot(index="lead_week", columns="method", values="acc")
    region_rmse = regional.pivot(index="region", columns="method", values="rmse_mm_day")
    region_acc = regional.pivot(index="region", columns="method", values="acc")
    yearly_all = (
        yearly.loc[yearly.region.eq("all_india")]
        .groupby(["year", "method"], as_index=False)[["rmse_mm_day", "acc"]]
        .mean()
    )
    year_rmse = yearly_all.pivot(index="year", columns="method", values="rmse_mm_day")
    pooled = regional.loc[regional.region.eq("all_india")].set_index("method")
    guards = {
        "primary_pooled_rmse_better_than_raw": bool(
            pooled.loc["selected_adapter", "rmse_mm_day"]
            < pooled.loc["raw_fuxi", "rmse_mm_day"]
        ),
        "primary_pooled_rmse_better_than_log_bias": bool(
            pooled.loc["selected_adapter", "rmse_mm_day"]
            < pooled.loc["log_bias", "rmse_mm_day"]
        ),
        "all_six_leads_rmse_better_than_raw": bool(
            (lead_rmse.selected_adapter < lead_rmse.raw_fuxi).all()
        ),
        "all_six_leads_acc_not_worse_than_raw": bool(
            (lead_acc.selected_adapter >= lead_acc.raw_fuxi).all()
        ),
        "all_five_regions_rmse_better_than_raw": bool(
            (region_rmse.selected_adapter < region_rmse.raw_fuxi).all()
        ),
        "all_five_regions_acc_not_worse_than_raw": bool(
            (region_acc.selected_adapter >= region_acc.raw_fuxi).all()
        ),
        "all_three_years_rmse_better_than_log_bias": bool(
            (year_rmse.selected_adapter < year_rmse.log_bias).all()
        ),
    }
    guards["strict_spatiotemporal_generalization_pass"] = bool(all(guards.values()))
    guards["lead_rmse_skill_vs_raw_pct"] = {
        f"W{lead}": float(
            100.0
            * (
                lead_rmse.loc[lead, "raw_fuxi"]
                - lead_rmse.loc[lead, "selected_adapter"]
            )
            / lead_rmse.loc[lead, "raw_fuxi"]
        )
        for lead in lead_rmse.index
    }
    guards["lead_acc_delta_vs_raw"] = {
        f"W{lead}": float(
            lead_acc.loc[lead, "selected_adapter"]
            - lead_acc.loc[lead, "raw_fuxi"]
        )
        for lead in lead_acc.index
    }
    return guards


def write_results(
    output: Path,
    by_lead: pd.DataFrame,
    regional: pd.DataFrame,
    guards: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    all_india = by_lead.loc[by_lead.region.eq("all_india")]
    lead = all_india.pivot(index="lead_week", columns="method")
    pooled = regional.loc[regional.region.eq("all_india")].set_index("method")
    raw = pooled.loc["raw_fuxi"]
    selected = pooled.loc["selected_adapter"]
    skill = 100.0 * (raw.rmse_mm_day - selected.rmse_mm_day) / raw.rmse_mm_day
    lines = [
        "# Frozen FuXi–IMD adapter: 2022–2024 JJAS generalization audit",
        "",
        "> Development audit only. The model and residual gate were frozen on 2018–2019; no 2025 FuXi initialization is used here.",
        "",
        "## Contract",
        "",
        "- 100 operational FuXi JJAS starts: 35 in 2022, 35 in 2023, 30 in 2024",
        "- IMD target and verification on the frozen 171-cell India support",
        "- W1 uses initialization day through day +6, matching the reforecast training contract",
        "- ACC subtracts the same fixed, training-only 2002–2017 IMD climatology from every method",
        "- Selection is not reopened after looking at this audit",
        "",
        "## Pooled all-India W1–W6",
        "",
        f"Frozen selection: `{selection['selected_model']}`, α={float(selection['selected_alpha']):.3f}.",
        f"RMSE: **{selected.rmse_mm_day:.3f} mm/day** vs raw FuXi **{raw.rmse_mm_day:.3f} mm/day** ({skill:+.2f}%).",
        f"ACC: **{selected.acc:.3f}** vs raw FuXi **{raw.acc:.3f}**.",
        "",
        "| Week | Raw RMSE | Log-bias RMSE | Selected RMSE | Raw ACC | Log-bias ACC | Selected ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for week in range(1, 7):
        lines.append(
            f"| W{week} | {lead.loc[week, ('rmse_mm_day', 'raw_fuxi')]:.3f} | "
            f"{lead.loc[week, ('rmse_mm_day', 'log_bias')]:.3f} | "
            f"{lead.loc[week, ('rmse_mm_day', 'selected_adapter')]:.3f} | "
            f"{lead.loc[week, ('acc', 'raw_fuxi')]:.3f} | "
            f"{lead.loc[week, ('acc', 'log_bias')]:.3f} | "
            f"{lead.loc[week, ('acc', 'selected_adapter')]:.3f} |"
        )
    lines.extend(["", "## Predeclared generalization guards", ""])
    for name, value in guards.items():
        if isinstance(value, bool):
            lines.append(f"- {name}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(
        [
            "",
            "A failed strict guard means this candidate is not yet an all-lead, all-region replacement for raw FuXi; it does not justify tuning on these audit years.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_predictions(
    path: Path,
    forecast: base.ForecastData,
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    climatology: np.ndarray,
    support: np.ndarray,
    selection: Mapping[str, Any],
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack([predictions[method] for method in METHODS]).astype(np.float32),
            ),
            "truth_imd": (
                ("init", "lead_week", "latitude", "longitude"),
                truth.astype(np.float32),
            ),
            "fixed_imd_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                climatology.astype(np.float32),
            ),
            "adapter_support": (("latitude", "longitude"), support),
        },
        coords={
            "method": list(METHODS),
            "init": forecast.initializations.astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "scope": "development generalization audit; 2022-2024 JJAS initializations",
            "selected_model": str(selection["selected_model"]),
            "selected_alpha": float(selection["selected_alpha"]),
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "acc_reference": "common fixed training-only 2002-2017 IMD climatology",
            "units": "mm day-1",
        },
    )
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={"prediction": {"chunks": (1, 25, 1, 27, 27)}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = (args.run or latest_full_context_run()).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError("adapter run is not a completed full experiment")
    if selection.get("status") != "frozen" or selection.get("smoke") is not False:
        raise ValueError("adapter selection is not frozen")
    support, latitude, longitude = benchmark.load_adapter_support(run)
    tp_records, _ = diagnostic.load_catalog_records()
    t2m_records = benchmark.catalog_records("t2m")
    forecast, t2m, paired, fuxi_stores = load_operational_fuxi_jjas(
        tp_records, t2m_records
    )
    if not np.array_equal(forecast.latitude, latitude) or not np.array_equal(
        forecast.longitude, longitude
    ):
        raise ValueError("training and audit grids differ")

    print("Rebuilding frozen 2002-2017 IMD predictors...", flush=True)
    training_climatology, training_imd_stores = benchmark.build_training_climatology(
        support
    )
    normalization = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    features, rebuilt_climatology = benchmark.build_features(
        forecast, t2m, training_climatology, normalization, support
    )
    print("Applying the frozen validation selection...", flush=True)
    log_bias, selected_adapter, frozen_selection = benchmark.infer_adapter(
        run, forecast, features, support
    )
    del features
    truth, fixed_climatology, weekly_coverage, verification_imd_stores = (
        load_imd_audit_fields(forecast, training_climatology)
    )
    if not np.allclose(
        rebuilt_climatology,
        fixed_climatology,
        rtol=0.0,
        atol=2.0e-6,
        equal_nan=True,
    ):
        raise ValueError("feature and verification IMD climatologies differ")
    predictions = {
        "raw_fuxi": forecast.ensemble_mean,
        "log_bias": log_bias,
        "selected_adapter": selected_adapter,
    }
    cases = score(
        forecast,
        predictions,
        truth,
        fixed_climatology,
        weekly_coverage,
        support,
    )
    by_lead, regional, yearly = summarize(cases)
    guards = generalization_guards(by_lead, regional, yearly)
    cases.to_csv(output / "case_metrics.csv", index=False)
    by_lead.to_csv(output / "summary_by_lead_region.csv", index=False)
    regional.to_csv(output / "summary_by_region.csv", index=False)
    yearly.to_csv(output / "summary_by_year_lead_region.csv", index=False)
    (output / "generalization_guards.json").write_text(
        json.dumps(guards, indent=2) + "\n", encoding="utf-8"
    )
    write_results(output, by_lead, regional, guards, frozen_selection)
    prediction_store = output / "predictions.zarr"
    save_predictions(
        prediction_store,
        forecast,
        predictions,
        truth,
        fixed_climatology,
        support,
        frozen_selection,
    )

    code = output / "code"
    code.mkdir()
    sources = {
        "run_jjas_generalization_audit.py": Path(__file__),
        "run_benchmark.py": Path(benchmark.__file__),
        "models.py": benchmark.NEURAL_SRC / "fuxi_adapter" / "models.py",
    }
    code_hashes = {}
    for name, source in sources.items():
        shutil.copy2(source, code / name)
        code_hashes[name] = sha256_file(code / name)
    result_manifest = {
        "status": "complete",
        "scientific_status": "development generalization audit; not a final untouched test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_run": str(run),
        "adapter_manifest_sha256": sha256_file(run / "manifest.json"),
        "adapter_selection_sha256": sha256_file(run / "selection.json"),
        "selected_model": frozen_selection["selected_model"],
        "selected_alpha": frozen_selection["selected_alpha"],
        "audit_initialization_years": list(AUDIT_YEARS),
        "audit_initialization_months": list(INITIALIZATION_MONTHS),
        "audit_counts": {str(year): len(paired[year]) for year in AUDIT_YEARS},
        "audit_case_count": len(forecast.initializations),
        "final_initialization_year_quarantined": 2025,
        "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
        "support_cells": int(support.sum()),
        "metric_contract": {
            "errors": "mean case-wise area-weighted RMSE, MAE, and bias against IMD",
            "acc": "case-wise area-weighted spatial ACC after subtracting one common fixed 2002-2017 IMD climatology",
            "primary": "all-India equal-case mean across W1-W6",
        },
        "generalization_guards": guards,
        "operational_fuxi_source_stores": list(fuxi_stores),
        "training_imd_source_stores": list(training_imd_stores),
        "verification_imd_source_stores": list(verification_imd_stores),
        "code_sha256": code_hashes,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "outputs": {},
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
    print((output / "RESULTS.md").read_text(encoding="utf-8"), flush=True)
    print(f"PASS: frozen 100-case 2022-2024 JJAS audit: {output}", flush=True)


if __name__ == "__main__":
    main()
