#!/usr/bin/env python3
"""Frozen-split categorical PBC baseline for FuXi-S2S precipitation over India.

The driver reuses the verified 51-member cache, IMD loader, area weights, and
purged 2002--2017 / 2018--2019 / 2020--2021 splits from the canonical
all-season ensemble experiment.  It never opens 2025 observations.

This is a leakage-safe *frozen-split adaptation* of Guan et al. (2026), not a
claim to reproduce their rolling operational implementation.  The precise
divergences are written to every run README and manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import PROJECT_ROOT

import fuxi_allseason_ensemble_calibration as frozen
from fuxi_pbc_core import (
    CalendarQuantiles,
    DebiasFit,
    IssueTimeLags,
    PersistenceFit,
    apply_persistence,
    apply_selected_debias,
    build_daily_issue_time_lags,
    calendar_fields,
    cdf_projection_diagnostics,
    combine_projected_components,
    ensemble_cdf,
    fit_calendar_quantiles,
    fit_debias,
    fit_persistence,
    is_valid_cdf,
    lag_observation_cdf,
    observation_cdf,
    probability_bias,
    project_cdf,
    ranked_probability_score,
    select_debias_spans,
    upper_tail_brier_score,
    verification_end_dates,
    verification_midpoints,
    weighted_spatial_mean,
)


EXPERIMENT = "fuxi_allseason_pbc_baseline_v1"
DEFAULT_CACHE = frozen.DEFAULT_CACHE
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv2" / "fuxi_allseason_pbc_baseline"
QUINTILE_LEVELS = tuple(np.arange(0.2, 1.0, 0.2).tolist())
SEMIDECILE_LEVELS = tuple(np.arange(0.05, 1.0, 0.05).tolist())
METHODS = (
    "raw_fuxi_categorical",
    "debias_plus_plus",
    "persistence_plus_plus",
    "pbc_combined",
)
METHOD_LABELS = {
    "raw_fuxi_categorical": "Raw FuXi categorical",
    "debias_plus_plus": "Projected Debias++",
    "persistence_plus_plus": "Projected Persistence++",
    "pbc_combined": "Combined PBC",
}
DEFAULT_DEBIAS_SPANS = (14, 28, 35)
DEFAULT_RIDGE = 1.0e-3
DEFAULT_CALENDAR_WINDOW_DAYS = 31
DEFAULT_MINIMUM_SAMPLES = 8
DEFAULT_CDF_CHUNK_SIZE = 16
DEFAULT_BOOTSTRAP_SAMPLES = 2000
SMOKE_BOOTSTRAP_SAMPLES = 100


class BaselineContractError(ValueError):
    """Raised when the frozen PBC experiment contract is violated."""


@dataclass
class FamilyResult:
    """Fitted artifacts and held-out CDFs for one categorical resolution."""

    name: str
    model: CalendarQuantiles
    debias_fits: tuple[DebiasFit, ...]
    selected_debias_spans: np.ndarray
    validation_rps_by_span: dict[int, np.ndarray]
    persistence_fit: PersistenceFit
    methods: dict[str, np.ndarray]
    observed_test_cdf: np.ndarray
    empirical_climatology_test_cdf: np.ndarray
    projection_diagnostics: dict[str, Mapping[str, float]]
    cdf_validity: dict[str, Mapping[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".temporary", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "src/fuxi_allseason_pbc_baseline.py": Path(__file__).resolve(),
        "src/fuxi_pbc_core.py": PROJECT_ROOT / "src/fuxi_pbc_core.py",
        "src/fuxi_allseason_ensemble_calibration.py": PROJECT_ROOT
        / "src/fuxi_allseason_ensemble_calibration.py",
        "src/fuxi_allseason_member_cache.py": PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
    }
    optional_sources = {
        "slurm/run_allseason_pbc_baseline.sbatch": PROJECT_ROOT
        / "slurm/run_allseason_pbc_baseline.sbatch",
        "plan/CAPACITY_PBC_STUDY_20260822.md": PROJECT_ROOT
        / "plan/CAPACITY_PBC_STUDY_20260822.md",
    }
    sources.update(
        {
            relative: path
            for relative, path in optional_sources.items()
            if path.is_file()
        }
    )
    checksums: dict[str, str] = {}
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksums[str(destination.relative_to(output))] = frozen.sha256_file(
            destination
        )
    return checksums


def _output_checksums(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative in {"manifest.json", "failure.json"}:
            continue
        result[relative] = frozen.sha256_file(path)
    return result


def parse_spans(value: str) -> tuple[int, ...]:
    try:
        spans = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Debias spans must be comma-separated integers"
        ) from error
    if (
        not spans
        or len(set(spans)) != len(spans)
        or any(span < 0 or span > 183 for span in spans)
    ):
        raise argparse.ArgumentTypeError(
            "Debias spans must be unique values in [0,183]"
        )
    return spans


def assert_temporal_contract(
    initializations: np.ndarray,
    splits: frozen.SplitIndices,
    lags: IssueTimeLags | None = None,
) -> dict[str, str]:
    """Prove the purge and issue-time predictor availability contracts."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    ends = verification_end_dates(starts, 6)[:, -1]
    if np.any(ends[splits.train] >= np.datetime64("2018-01-01")):
        raise BaselineContractError("training outcome crosses the validation boundary")
    if np.any(ends[splits.validation] >= np.datetime64("2020-01-01")):
        raise BaselineContractError("validation outcome crosses the test boundary")
    if np.any(starts[splits.test] < np.datetime64("2020-01-01")) or np.any(
        starts[splits.test] >= np.datetime64("2022-01-01")
    ):
        raise BaselineContractError(
            "test initializations are not confined to 2020--2021"
        )
    if np.any(pd.DatetimeIndex(starts).year == 2025):
        raise BaselineContractError("2025 appears in the forecast cache")
    if lags is not None:
        issue = np.broadcast_to(starts[:, None], lags.window_end.shape)
        available = lags.source_indices >= 0
        if np.any(lags.window_end[available] >= issue[available]):
            raise BaselineContractError("Persistence++ lag reaches or follows issuance")
        if not np.all(lags.available[np.concatenate((splits.validation, splits.test))]):
            raise BaselineContractError(
                "held-out issues lack exact init-7/init-14 W1 observation lags; "
                "use the full canonical cache"
            )
    return {
        "training_latest_outcome_end": np.datetime_as_string(
            ends[splits.train].max(), unit="D"
        ),
        "validation_latest_outcome_end": np.datetime_as_string(
            ends[splits.validation].max(), unit="D"
        ),
        "test_latest_initialization": np.datetime_as_string(
            starts[splits.test].max(), unit="D"
        ),
    }


def load_daily_imd_for_lags(
    cache: frozen.MemberCache,
    source_stores: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Reload the audited daily IMD calendar needed by issue-time lag windows.

    The forecast cache contains only selected initialization dates and restarts its
    cadence at calendar boundaries.  It therefore cannot represent every valid
    ``init-7``/``init-14`` observation window.  The already-audited IMD source
    stores are complete daily calendars, which is the operationally correct source
    for those predictors.
    """

    if not source_stores:
        raise BaselineContractError("no audited IMD stores are available for lags")
    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    for raw_store in source_stores:
        store = Path(raw_store)
        if "2025" in store.name or "2025" in str(store):
            raise BaselineContractError("daily lag loader was given a sealed 2025 store")
        if not (store / ".zmetadata").is_file():
            raise FileNotFoundError(store)
        with xr.open_zarr(store, consolidated=True) as dataset:
            if (
                dataset.attrs.get("source") != "imd"
                or dataset.attrs.get("units") != "mm day-1"
            ):
                raise BaselineContractError(f"unexpected daily IMD metadata in {store}")
            if not np.array_equal(dataset.latitude.values, cache.latitude) or not np.array_equal(
                dataset.longitude.values, cache.longitude
            ):
                raise BaselineContractError(f"daily IMD grid differs in {store}")
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
        if values.shape != (dates.size, *cache.members.shape[-2:]):
            raise BaselineContractError(f"unexpected daily IMD shape in {store}")
        if dates.size not in (365, 366) or not np.all(
            np.diff(dates) == np.timedelta64(1, "D")
        ):
            raise BaselineContractError(f"incomplete daily IMD calendar in {store}")
        all_dates.append(dates)
        all_values.append(values)

    dates = np.concatenate(all_dates)
    values = np.concatenate(all_values)
    order = np.argsort(dates)
    dates = dates[order]
    values = values[order]
    if np.unique(dates).size != dates.size or np.any(
        np.diff(dates) != np.timedelta64(1, "D")
    ):
        raise BaselineContractError("audited IMD lag calendar is not continuous")
    return dates, values


def _fit_family(
    name: str,
    levels: Sequence[float],
    members: np.ndarray,
    truth: np.ndarray,
    initializations: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    lags: IssueTimeLags,
    support: np.ndarray,
    weights: np.ndarray,
    *,
    calendar_window_days: int,
    minimum_samples: int,
    debias_spans: Sequence[int],
    ridge: float,
    cdf_chunk_size: int,
) -> FamilyResult:
    """Fit both PBC components and produce held-out projected ablations."""

    model = fit_calendar_quantiles(
        truth,
        initializations,
        train_indices,
        levels,
        support,
        window_radius_days=(calendar_window_days - 1) // 2,
        minimum_samples=minimum_samples,
    )
    active_indices = np.concatenate(
        (train_indices, validation_indices, test_indices)
    ).astype(np.int64)
    if np.unique(active_indices).size != active_indices.size:
        raise BaselineContractError("train, validation, and test indices overlap")
    local_train = np.arange(len(train_indices), dtype=np.int64)
    local_validation = np.arange(
        len(train_indices), len(train_indices) + len(validation_indices), dtype=np.int64
    )
    local_test = np.arange(
        len(train_indices) + len(validation_indices),
        active_indices.size,
        dtype=np.int64,
    )
    active_initializations = initializations[active_indices]
    thresholds = calendar_fields(model, active_initializations, truth.shape[1])
    raw = ensemble_cdf(
        members,
        thresholds,
        chunk_size=cdf_chunk_size,
        case_indices=active_indices,
    )
    observed = observation_cdf(truth[active_indices], thresholds)
    empirical_climatology = calendar_fields(
        model, active_initializations, truth.shape[1], empirical=True
    )
    lag_cdf = lag_observation_cdf(lags, model, active_indices)
    active_lag_available = lags.available[active_indices]

    debias_fits = tuple(
        fit_debias(
            raw,
            observed,
            active_initializations,
            local_train,
            support,
            half_window_days=int(span),
            minimum_samples=minimum_samples,
        )
        for span in debias_spans
    )
    selected_spans, validation_scores = select_debias_spans(
        debias_fits,
        raw[local_validation],
        observed[local_validation],
        active_initializations[local_validation],
        weights,
    )
    persistence_fit = fit_persistence(
        raw,
        observed,
        empirical_climatology,
        lag_cdf,
        active_lag_available,
        local_train,
        support,
        ridge=ridge,
    )

    raw_test = raw[local_test].copy()
    observed_test = observed[local_test].copy()
    climate_test = empirical_climatology[local_test].copy()
    debias_unprojected = apply_selected_debias(
        raw_test,
        active_initializations[local_test],
        debias_fits,
        selected_spans,
        project=False,
    )
    debias_projected = project_cdf(debias_unprojected, axis=2)
    persistence_unprojected = apply_persistence(
        raw_test,
        climate_test,
        lag_cdf[local_test],
        active_lag_available[local_test],
        persistence_fit,
        project=False,
    )
    persistence_projected = project_cdf(persistence_unprojected, axis=2)
    combined = combine_projected_components(debias_projected, persistence_projected)
    methods = {
        "raw_fuxi_categorical": raw_test,
        "debias_plus_plus": debias_projected,
        "persistence_plus_plus": persistence_projected,
        "pbc_combined": combined,
    }
    diagnostics = {
        "debias_plus_plus": cdf_projection_diagnostics(
            debias_unprojected, debias_projected
        ),
        "persistence_plus_plus": cdf_projection_diagnostics(
            persistence_unprojected, persistence_projected
        ),
    }
    validity: dict[str, Mapping[str, Any]] = {}
    for method, cdf in methods.items():
        supported = cdf[..., support]
        unsupported = cdf[..., ~support]
        validity[method] = {
            "valid_probability_cdf": bool(is_valid_cdf(cdf, axis=2)),
            "all_supported_values_finite": bool(np.isfinite(supported).all()),
            "all_supported_values_bounded_0_1": bool(
                np.all((supported >= 0.0) & (supported <= 1.0))
            ),
            "all_supported_rows_nondecreasing": bool(
                np.all(np.diff(supported, axis=2) >= -1.0e-7)
            ),
            "all_unsupported_values_nan": bool(
                unsupported.size == 0 or np.isnan(unsupported).all()
            ),
            "supported_cdf_rows": int(
                cdf.shape[0] * cdf.shape[1] * np.count_nonzero(support)
            ),
        }
        if not all(
            validity[method][key]
            for key in (
                "valid_probability_cdf",
                "all_supported_values_finite",
                "all_supported_values_bounded_0_1",
                "all_supported_rows_nondecreasing",
                "all_unsupported_values_nan",
            )
        ):
            raise BaselineContractError(
                f"invalid final CDF artifact for {name}/{method}"
            )
    del thresholds, raw, observed, empirical_climatology, lag_cdf
    gc.collect()
    return FamilyResult(
        name=name,
        model=model,
        debias_fits=debias_fits,
        selected_debias_spans=selected_spans,
        validation_rps_by_span=validation_scores,
        persistence_fit=persistence_fit,
        methods=methods,
        observed_test_cdf=observed_test,
        empirical_climatology_test_cdf=climate_test,
        projection_diagnostics=diagnostics,
        cdf_validity=validity,
    )


def _seasons(midpoints: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(midpoints.reshape(-1)).month.to_numpy()
    labels = np.asarray(
        [
            (
                "DJF"
                if month in (12, 1, 2)
                else (
                    "MAM"
                    if month in (3, 4, 5)
                    else "JJA" if month in (6, 7, 8) else "SON"
                )
            )
            for month in months
        ],
        dtype=object,
    )
    return labels.reshape(midpoints.shape)


def quintile_case_scores(
    result: FamilyResult,
    test_initializations: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """One paired RPS row per method, initialization, and lead."""

    observed = result.observed_test_cdf
    nominal = np.broadcast_to(
        result.model.levels[None, None, :, None, None], observed.shape
    )
    nominal_rps = weighted_spatial_mean(
        ranked_probability_score(nominal, observed), weights
    )
    empirical_rps = weighted_spatial_mean(
        ranked_probability_score(result.empirical_climatology_test_cdf, observed),
        weights,
    )
    midpoints = verification_midpoints(test_initializations, observed.shape[1])
    starts = (
        test_initializations[:, None]
        + (7 * np.arange(observed.shape[1])).astype("timedelta64[D]")[None]
    )
    ends = starts + np.timedelta64(6, "D")
    seasons = _seasons(midpoints)
    records: list[dict[str, Any]] = []
    for method, cdf in result.methods.items():
        scores = weighted_spatial_mean(ranked_probability_score(cdf, observed), weights)
        for case in range(len(test_initializations)):
            for lead in range(observed.shape[1]):
                nominal_reference = float(nominal_rps[case, lead])
                empirical_reference = float(empirical_rps[case, lead])
                score = float(scores[case, lead])
                records.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "initialization": np.datetime_as_string(
                            test_initializations[case], unit="D"
                        ),
                        "lead_week": lead + 1,
                        "verification_start": np.datetime_as_string(
                            starts[case, lead], unit="D"
                        ),
                        "verification_midpoint": np.datetime_as_string(
                            midpoints[case, lead], unit="D"
                        ),
                        "verification_end": np.datetime_as_string(
                            ends[case, lead], unit="D"
                        ),
                        "season": seasons[case, lead],
                        "rps": score,
                        "nominal_climatology_rps": nominal_reference,
                        "training_empirical_climatology_rps": empirical_reference,
                        "rpss_vs_nominal_climatology_case": (
                            1.0 - score / nominal_reference
                            if nominal_reference > 0.0
                            else np.nan
                        ),
                        "rpss_vs_training_empirical_climatology_case": (
                            1.0 - score / empirical_reference
                            if empirical_reference > 0.0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _aggregate_scores(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    grouped = frame.groupby(
        [*group_columns, "method", "method_label"], as_index=False
    ).agg(
        rps=("rps", "mean"),
        nominal_climatology_rps=("nominal_climatology_rps", "mean"),
        training_empirical_climatology_rps=(
            "training_empirical_climatology_rps",
            "mean",
        ),
        n_initializations=("initialization", "nunique"),
    )
    grouped["rpss_vs_nominal_climatology"] = (
        1.0 - grouped.rps / grouped.nominal_climatology_rps
    )
    grouped["rpss_vs_training_empirical_climatology"] = (
        1.0 - grouped.rps / grouped.training_empirical_climatology_rps
    )
    return grouped


def probability_bias_table(
    result: FamilyResult,
    test_initializations: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seasons = _seasons(
        verification_midpoints(test_initializations, result.observed_test_cdf.shape[1])
    )
    for method, cdf in result.methods.items():
        case_bias = probability_bias(cdf, result.observed_test_cdf, weights)
        for lead in range(case_bias.shape[1]):
            for cut, nominal in enumerate(result.model.levels):
                records.append(
                    {
                        "family": result.name,
                        "method": method,
                        "lead_week": lead + 1,
                        "season": "ALL",
                        "nominal_cumulative_probability": float(nominal),
                        "probability_bias": float(np.mean(case_bias[:, lead, cut])),
                        "mean_absolute_case_probability_bias": float(
                            np.mean(np.abs(case_bias[:, lead, cut]))
                        ),
                    }
                )
                for season in ("DJF", "MAM", "JJA", "SON"):
                    chosen = seasons[:, lead] == season
                    if not np.any(chosen):
                        continue
                    records.append(
                        {
                            "family": result.name,
                            "method": method,
                            "lead_week": lead + 1,
                            "season": season,
                            "nominal_cumulative_probability": float(nominal),
                            "probability_bias": float(
                                np.mean(case_bias[chosen, lead, cut])
                            ),
                            "mean_absolute_case_probability_bias": float(
                                np.mean(np.abs(case_bias[chosen, lead, cut]))
                            ),
                        }
                    )
    return pd.DataFrame.from_records(records)


def upper_extreme_scores(
    result: FamilyResult,
    test_initializations: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Score the nondegenerate upper semidecile; flag the lower tail honestly."""

    nominal_event_probability = float(1.0 - result.model.levels[-1])
    observed = result.observed_test_cdf
    empirical_reference = upper_tail_brier_score(
        result.empirical_climatology_test_cdf, observed, weights
    )
    observed_high = 1.0 - observed[:, :, -1]
    nominal_reference = weighted_spatial_mean(
        (nominal_event_probability - observed_high) ** 2, weights
    )
    midpoints = verification_midpoints(test_initializations, observed.shape[1])
    seasons = _seasons(midpoints)
    records: list[dict[str, Any]] = []
    for method, cdf in result.methods.items():
        scores = upper_tail_brier_score(cdf, observed, weights)
        for lead in range(scores.shape[1]):
            groups = [("ALL", np.ones(scores.shape[0], dtype=bool))]
            groups.extend(
                (season, seasons[:, lead] == season)
                for season in ("DJF", "MAM", "JJA", "SON")
            )
            for season, chosen in groups:
                if not np.any(chosen):
                    continue
                bs = float(np.mean(scores[chosen, lead]))
                nominal_bs = float(np.mean(nominal_reference[chosen, lead]))
                empirical_bs = float(np.mean(empirical_reference[chosen, lead]))
                records.append(
                    {
                        "event": "upper_5pct",
                        "status": "scored",
                        "method": method,
                        "lead_week": lead + 1,
                        "season": season,
                        "brier_score": bs,
                        "nominal_climatology_brier_score": nominal_bs,
                        "training_empirical_climatology_brier_score": empirical_bs,
                        "bss_vs_nominal_climatology": (
                            1.0 - bs / nominal_bs if nominal_bs > 0.0 else np.nan
                        ),
                        "bss_vs_training_empirical_climatology": (
                            1.0 - bs / empirical_bs if empirical_bs > 0.0 else np.nan
                        ),
                        "n_initializations": int(np.count_nonzero(chosen)),
                    }
                )
    records.append(
        {
            "event": "lower_5pct",
            "status": "not_scored_partially_degenerate_zero_tied_threshold",
            "method": "not_applicable",
            "lead_week": 0,
            "season": "ALL",
            "brier_score": np.nan,
            "nominal_climatology_brier_score": np.nan,
            "training_empirical_climatology_brier_score": np.nan,
            "bss_vs_nominal_climatology": np.nan,
            "bss_vs_training_empirical_climatology": np.nan,
            "n_initializations": len(test_initializations),
        }
    )
    return pd.DataFrame.from_records(records)


def threshold_diagnostics(name: str, model: CalendarQuantiles) -> pd.DataFrame:
    """Expose zero/duplicate quantile cuts rather than hiding rainfall ties."""

    records: list[dict[str, Any]] = []
    supported = model.support
    previous: np.ndarray | None = None
    for cut, nominal in enumerate(model.levels):
        values = model.thresholds[:, cut][:, supported]
        empirical = model.empirical_cdf[:, cut][:, supported]
        duplicate = (
            np.zeros_like(values, dtype=bool)
            if previous is None
            else values == previous
        )
        records.append(
            {
                "family": name,
                "cut_index": cut + 1,
                "nominal_cumulative_probability": float(nominal),
                "zero_threshold_fraction": float(np.mean(values == 0.0)),
                "duplicate_previous_threshold_fraction": float(np.mean(duplicate)),
                "mean_training_empirical_strict_cdf_probability": float(
                    np.mean(empirical)
                ),
                "min_training_empirical_strict_cdf_probability": float(
                    np.min(empirical)
                ),
                "max_training_empirical_strict_cdf_probability": float(
                    np.max(empirical)
                ),
                "minimum_threshold_mm_day": float(np.min(values)),
                "median_threshold_mm_day": float(np.median(values)),
                "maximum_threshold_mm_day": float(np.max(values)),
                "minimum_calendar_training_samples": int(
                    model.sample_count_by_day.min()
                ),
                "unique_training_verification_windows": int(
                    model.unique_fit_window_count
                ),
                "deduplicated_forecast_lead_replicates": int(
                    model.duplicate_fit_window_count
                ),
            }
        )
        previous = values
    return pd.DataFrame.from_records(records)


def paired_block_bootstrap(
    case_scores: pd.DataFrame,
    *,
    samples: int,
    block_length: int = 13,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Paired moving-block uncertainty for RPS reductions between methods."""

    comparisons = (
        ("pbc_combined", "raw_fuxi_categorical"),
        ("debias_plus_plus", "raw_fuxi_categorical"),
        ("persistence_plus_plus", "raw_fuxi_categorical"),
        ("pbc_combined", "debias_plus_plus"),
        ("pbc_combined", "persistence_plus_plus"),
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, int, pd.DataFrame]] = []
    pooled = case_scores.groupby(["initialization", "method"], as_index=False).agg(
        rps=("rps", "mean")
    )
    scopes.append(("W1-W6", 0, pooled))
    scopes.extend(
        (f"W{int(lead)}", int(lead), case_scores.loc[case_scores.lead_week == lead])
        for lead in sorted(case_scores.lead_week.unique())
    )
    for lead_scope, lead, selected in scopes:
        pivot = selected.pivot(
            index="initialization", columns="method", values="rps"
        ).sort_index()
        count = len(pivot)
        if count < 2:
            continue
        draw_count = int(np.ceil(count / block_length))
        draws = np.empty((samples, count), dtype=np.int64)
        for sample in range(samples):
            starts = rng.integers(0, count, size=draw_count)
            draw = np.concatenate(
                [(start + np.arange(block_length)) % count for start in starts]
            )[:count]
            draws[sample] = draw
        for method, baseline in comparisons:
            method_values = pivot[method].to_numpy(dtype=np.float64)
            baseline_values = pivot[baseline].to_numpy(dtype=np.float64)
            reductions = 1.0 - np.mean(method_values[draws], axis=1) / np.mean(
                baseline_values[draws], axis=1
            )
            point = 1.0 - method_values.mean() / baseline_values.mean()
            records.append(
                {
                    "lead_scope": lead_scope,
                    "lead_week": int(lead),
                    "method": method,
                    "baseline": baseline,
                    "rps_reduction_fraction": float(point),
                    "ci_lower_95": float(np.quantile(reductions, 0.025)),
                    "ci_upper_95": float(np.quantile(reductions, 0.975)),
                    "bootstrap_probability_improvement": float(
                        np.mean(reductions > 0.0)
                    ),
                    "bootstrap_samples": int(samples),
                    "block_length_initializations": int(block_length),
                }
            )
    return pd.DataFrame.from_records(records)


def _save_models(path: Path, quintile: FamilyResult, semidecile: FamilyResult) -> None:
    arrays: dict[str, np.ndarray] = {}
    for prefix, result in (("quintile", quintile), ("semidecile", semidecile)):
        arrays[f"{prefix}_levels"] = result.model.levels
        arrays[f"{prefix}_thresholds_mm_day"] = result.model.thresholds
        arrays[f"{prefix}_training_empirical_strict_cdf"] = result.model.empirical_cdf
        arrays[f"{prefix}_fit_indices"] = result.model.fit_indices
        arrays[f"{prefix}_calendar_sample_count"] = result.model.sample_count_by_day
        arrays[f"{prefix}_unique_fit_window_count"] = np.asarray(
            result.model.unique_fit_window_count, dtype=np.int64
        )
        arrays[f"{prefix}_duplicate_fit_window_count"] = np.asarray(
            result.model.duplicate_fit_window_count, dtype=np.int64
        )
        arrays[f"{prefix}_selected_debias_span_by_lead"] = result.selected_debias_spans
        for fit in result.debias_fits:
            arrays[f"{prefix}_debias_correction_span_{fit.half_window_days}"] = (
                fit.correction
            )
        arrays[f"{prefix}_persistence_coefficients"] = (
            result.persistence_fit.coefficients
        )
        arrays[f"{prefix}_persistence_usable_fit_indices"] = (
            result.persistence_fit.usable_fit_indices
        )
    _atomic_npz(path, **arrays)


def build_readme(
    weekwise: pd.DataFrame,
    extremes: pd.DataFrame,
    *,
    smoke: bool,
) -> str:
    pooled = weekwise.groupby("method", as_index=False).agg(
        rps=("rps", "mean"),
        nominal_climatology_rps=("nominal_climatology_rps", "mean"),
        training_empirical_climatology_rps=(
            "training_empirical_climatology_rps",
            "mean",
        ),
    )
    pooled["rpss_nominal"] = 1.0 - pooled.rps / pooled.nominal_climatology_rps
    pooled["rpss_tie_aware"] = (
        1.0 - pooled.rps / pooled.training_empirical_climatology_rps
    )
    pooled = pooled[
        [
            "method",
            "rps",
            "training_empirical_climatology_rps",
            "rpss_tie_aware",
            "nominal_climatology_rps",
            "rpss_nominal",
        ]
    ]
    lines = [
        "# FuXi-on-IMD categorical PBC baseline",
        "",
        (
            "**NON-SCIENTIFIC SMOKE:** plumbing only; do not quote these scores."
            if smoke
            else (
                "**Scientific status:** retrospective 2020–2021 development "
                "evaluation; not an untouched final test."
            )
        ),
        "",
        (
            "The 2020–2021 evaluation period is retrospective development data and "
            "is not an untouched final test."
        ),
        "",
        (
            "This run applies a compact categorical probability correction to all "
            "51 FuXi members over the frozen India/IMD experiment support."
        ),
        "",
        "## Frozen-split divergence from Guan et al. (2026)",
        "",
        (
            "The paper updates its correction prequentially at every forecast date "
            "using up to 20 prior years, adaptively scores spans on the preceding "
            "three years, can average neighboring issuance dates, and has access to "
            "a complete observation calendar. This implementation instead fits "
            "thresholds and both corrections once on purged 2002–2017, selects the "
            "Debias++ half-window (14/28/35 days) per lead on 2018–2019, and scores "
            "2020–2021. Its climatological thresholds use a centered 31-day window "
            "over unique training verification weeks rather than the paper's "
            "0/2/4-day offsets over 20 rolling years. It uses exact W1 IMD windows "
            "starting at issue−7 and issue−14; both end before issuance. Therefore "
            "this is an inspired baseline, not a numerical reproduction of the "
            "paper."
        ),
        "",
        (
            "CDFs use strict `P(Y < q)`. Zero precipitation creates duplicate "
            "thresholds, so the **scientific headline is training-empirical "
            "tie-aware RPSS**. Paper-compatible nominal equiprobable RPSS is "
            "retained only as a comparison sensitivity. The nominal lower 5% cut "
            "is zero for a substantial, seasonally heterogeneous subset of "
            "calendar-day/grid thresholds and is therefore not presented as a "
            "valid homogeneous lower-semidecile event; upper-5% BSS is reported."
        ),
        "",
        "## Pooled quintile scores",
        "",
        "```text",
        pooled.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
        "## Outputs",
        "",
        "- `metrics/quintile_case_scores.csv`: paired initialization/lead RPS values.",
        "- `metrics/weekwise_metrics.csv` and `seasonal_weekwise_metrics.csv`: RPSS summaries.",
        "- `metrics/component_ablation_metrics.csv`: raw, projected components, and combined PBC.",
        "- `metrics/paired_block_bootstrap.csv`: paired moving-block uncertainty.",
        "- `metrics/semidecile_extreme_metrics.csv`: upper-5% Brier scores and BSS.",
        "- `metrics/probability_bias.csv`: signed cumulative-probability bias.",
        "- `metrics/threshold_diagnostics.csv`: zero and duplicate threshold evidence.",
        "- `models/pbc_fit.npz`: training-only thresholds and fitted corrections.",
        "- `manifest.json`: protocol, provenance, selected spans, and hashes.",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    snapshot = _source_snapshot(output)
    # Even a smoke run needs the full forecast archive for the canonical split
    # inventory. Only 32/16/16 active CDF cases are materialized below, so this
    # remains a computational smoke without weakening the issue-time contract.
    cache = frozen.load_member_cache(Path(args.cache), allow_partial=False)
    splits = frozen.make_split_indices(cache.initializations)
    split_counts = {name: len(value) for name, value in splits.as_dict().items()}
    train = splits.train
    validation = splits.validation
    test = splits.test
    if args.smoke:
        train = frozen.select_evenly(train, min(32, len(train)))
        validation = frozen.select_evenly(validation, min(16, len(validation)))
        test = frozen.select_evenly(test, min(16, len(test)))
    if min(len(train), len(validation), len(test)) == 0:
        raise BaselineContractError("train, validation, and test must all be non-empty")
    observations = frozen.load_imd_observations(cache)
    if any("2025" in Path(path).name for path in observations.source_stores):
        raise BaselineContractError("the observation loader opened a sealed 2025 store")
    support = observations.weights > 0.0
    daily_dates, daily_values = load_daily_imd_for_lags(
        cache, observations.source_stores
    )
    lags = build_daily_issue_time_lags(
        cache.initializations, daily_dates, daily_values
    )
    del daily_dates, daily_values
    temporal_evidence = assert_temporal_contract(cache.initializations, splits, lags)
    temporal_evidence["persistence_lag_source"] = (
        "complete daily IMD calendar; exact seven-day means"
    )
    temporal_evidence["persistence_usable_training_cases"] = int(
        np.count_nonzero(lags.available[splits.train])
    )
    temporal_evidence["persistence_usable_validation_cases"] = int(
        np.count_nonzero(lags.available[splits.validation])
    )
    temporal_evidence["persistence_usable_development_cases"] = int(
        np.count_nonzero(lags.available[splits.test])
    )
    if not np.all(lags.available[np.concatenate((validation, test))]):
        raise BaselineContractError("selected held-out cases lack issue-time lags")

    evaluation = output / "evaluation"
    metrics_directory = output / "metrics"
    models_directory = output / "models"
    evaluation.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)
    normalized_weights = observations.weights / observations.weights.sum(
        dtype=np.float64
    )
    _atomic_npz(
        evaluation / "scoring_support.npz",
        latitude=cache.latitude.astype(np.float64),
        longitude=cache.longitude.astype(np.float64),
        observation_fraction=observations.observation_fraction.astype(np.float32),
        support_mask=support,
        scoring_weight=observations.weights.astype(np.float64),
        normalized_scoring_weight=normalized_weights,
    )

    print(
        "Fitting training-only climatological quintiles and PBC components...",
        flush=True,
    )
    quintile = _fit_family(
        "quintile",
        QUINTILE_LEVELS,
        cache.members,
        observations.weekly_truth,
        cache.initializations,
        train,
        validation,
        test,
        lags,
        support,
        observations.weights,
        calendar_window_days=args.calendar_window_days,
        minimum_samples=args.minimum_calendar_samples,
        debias_spans=args.debias_spans,
        ridge=args.ridge,
        cdf_chunk_size=args.cdf_chunk_size,
    )
    case_scores = quintile_case_scores(
        quintile, cache.initializations[test], observations.weights
    )
    weekwise = _aggregate_scores(case_scores, ("lead_week",))
    seasonal = _aggregate_scores(case_scores, ("season", "lead_week"))
    component_ablation = weekwise.copy()
    component_ablation["ablation_role"] = component_ablation.method.map(
        {
            "raw_fuxi_categorical": "uncorrected_reference",
            "debias_plus_plus": "projected_debias_component",
            "persistence_plus_plus": "projected_persistence_component",
            "pbc_combined": "equal_weight_projected_components",
        }
    )
    bootstrap = paired_block_bootstrap(
        case_scores,
        samples=args.bootstrap_samples,
        block_length=13,
    )
    quintile_bias = probability_bias_table(
        quintile, cache.initializations[test], observations.weights
    )

    print("Fitting training-only semideciles for upper-tail Brier skill...", flush=True)
    semidecile = _fit_family(
        "semidecile",
        SEMIDECILE_LEVELS,
        cache.members,
        observations.weekly_truth,
        cache.initializations,
        train,
        validation,
        test,
        lags,
        support,
        observations.weights,
        calendar_window_days=args.calendar_window_days,
        minimum_samples=args.minimum_calendar_samples,
        debias_spans=args.debias_spans,
        ridge=args.ridge,
        cdf_chunk_size=args.cdf_chunk_size,
    )
    extremes = upper_extreme_scores(
        semidecile, cache.initializations[test], observations.weights
    )
    semidecile_bias = probability_bias_table(
        semidecile, cache.initializations[test], observations.weights
    )
    diagnostics = pd.concat(
        (
            threshold_diagnostics("quintile", quintile.model),
            threshold_diagnostics("semidecile", semidecile.model),
        ),
        ignore_index=True,
    )

    case_scores.to_csv(metrics_directory / "quintile_case_scores.csv", index=False)
    weekwise.to_csv(metrics_directory / "weekwise_metrics.csv", index=False)
    seasonal.to_csv(metrics_directory / "seasonal_weekwise_metrics.csv", index=False)
    component_ablation.to_csv(
        metrics_directory / "component_ablation_metrics.csv", index=False
    )
    bootstrap.to_csv(metrics_directory / "paired_block_bootstrap.csv", index=False)
    extremes.to_csv(metrics_directory / "semidecile_extreme_metrics.csv", index=False)
    pd.concat((quintile_bias, semidecile_bias), ignore_index=True).to_csv(
        metrics_directory / "probability_bias.csv", index=False
    )
    diagnostics.to_csv(metrics_directory / "threshold_diagnostics.csv", index=False)
    _save_models(models_directory / "pbc_fit.npz", quintile, semidecile)
    (output / "README.md").write_text(
        build_readme(weekwise, extremes, smoke=args.smoke), encoding="utf-8"
    )

    lower_row = diagnostics.loc[
        (diagnostics.family == "semidecile")
        & np.isclose(diagnostics.nominal_cumulative_probability, 0.05)
    ].iloc[0]
    upper_row = diagnostics.loc[
        (diagnostics.family == "semidecile")
        & np.isclose(diagnostics.nominal_cumulative_probability, 0.95)
    ].iloc[0]
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "smoke": bool(args.smoke),
        "scientific_status": (
            "non-scientific plumbing smoke test"
            if args.smoke
            else "retrospective 2020-2021 development evaluation; not an untouched final test"
        ),
        "created_utc": utc_now(),
        "output_path": str(Path(args.output).resolve()),
        "command_line": [sys.executable, *sys.argv],
        "contract": {
            "forecast": "FuXi native weekly TP empirical CDF from all 51 members",
            "target": "IMD weekly mean precipitation, mm day-1",
            "region": "39N-0N, 60E-99E, 27x27 India box; weighted IMD support",
            "train_years": list(frozen.TRAIN_YEARS),
            "validation_years": list(frozen.VALIDATION_YEARS),
            "test_years": list(frozen.TEST_YEARS),
            "sealed_unopened_years": list(frozen.SEALED_YEARS),
            "sealed_2025_target_opened": False,
            "cdf_definition": "strict empirical P(Y < q)",
            "weekly_target_convention": (
                "W1=[init,init+6], Wj advances by 7 days; calendar label is "
                "weekly midpoint"
            ),
            "persistence_lags": "W1 IMD truth at init-7 and init-14; windows end init-1 and init-8",
            "boundary_purge": "42-day outcomes crossing 2018 or 2020 boundary are excluded",
            "test_reuse_warning": "2020-2021 has prior development exposure and is not independent",
            "paper_protocol_divergence": (
                "static train-only 2002-2017 fit and 2018-2019 per-lead span selection; "
                "not the paper's per-issue rolling 20-year fit, preceding-3-year adaptive "
                "selection, 0/2/4-day threshold calendar, or multi-issuance ensemble"
            ),
        },
        "temporal_evidence": temporal_evidence,
        "split_counts_archive": split_counts,
        "split_counts_selected": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "retained_initializations": {
            name: [
                np.datetime_as_string(value, unit="D")
                for value in cache.initializations[index]
            ]
            for name, index in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "methods": list(METHODS),
        "component_ablations": {
            "raw": "uncorrected categorical empirical FuXi CDF",
            "debias_plus_plus": "additive probability bias, clip, isotonic projection",
            "persistence_plus_plus": "ridge regression, clip, isotonic projection",
            "combined": "equal-weight average of the two projected components",
        },
        "quantile_definitions": {
            "quintile_levels": list(QUINTILE_LEVELS),
            "semidecile_levels": list(SEMIDECILE_LEVELS),
            "threshold_fit_scope": (
                "selected training split only; unique observed weekly verification "
                "windows pooled by midpoint calendar window (duplicate issue/lead "
                "views removed)"
            ),
            "calendar_window_days": args.calendar_window_days,
            "minimum_samples_with_nearest_calendar_fallback": args.minimum_calendar_samples,
            "primary_tie_aware_reference": (
                "training empirical strict P(Y<q) for each calendar threshold"
            ),
            "nominal_paper_comparison_sensitivity": "fixed cumulative probabilities k/K",
            "lower_5pct_status": (
                "not scored because zero-rain ties make the nominal 5% event "
                "partially degenerate and spatially/calendar heterogeneous"
            ),
            "lower_5pct_zero_threshold_fraction": float(
                lower_row.zero_threshold_fraction
            ),
            "upper_5pct_zero_threshold_fraction": float(
                upper_row.zero_threshold_fraction
            ),
        },
        "fitting": {
            "debias_candidate_half_spans_days": list(args.debias_spans),
            "quintile_selected_half_span_by_lead": quintile.selected_debias_spans.tolist(),
            "semidecile_selected_half_span_by_lead": semidecile.selected_debias_spans.tolist(),
            "quintile_validation_rps_by_span": {
                str(span): values.tolist()
                for span, values in quintile.validation_rps_by_span.items()
            },
            "semidecile_validation_rps_by_span": {
                str(span): values.tolist()
                for span, values in semidecile.validation_rps_by_span.items()
            },
            "persistence_ridge": args.ridge,
            "persistence_feature_names": list(quintile.persistence_fit.feature_names),
            "persistence_usable_training_cases": int(
                quintile.persistence_fit.usable_fit_indices.size
            ),
            "blend_debias_weight": 0.5,
            "projection": "exact equal-weight pool-adjacent-violators isotonic regression",
            "projection_diagnostics": {
                "quintile": quintile.projection_diagnostics,
                "semidecile": semidecile.projection_diagnostics,
            },
            "cdf_validity": {
                "quintile": quintile.cdf_validity,
                "semidecile": semidecile.cdf_validity,
            },
        },
        "evaluation": {
            "primary_metric": (
                "quintile RPSS vs training empirical strict-CDF climatology "
                "(tie-aware)"
            ),
            "paper_comparison_sensitivity": (
                "quintile RPSS vs nominal equal-probability climatology"
            ),
            "rps_definition": (
                "sum of squared cumulative-probability errors over the four finite "
                "quintile cuts"
            ),
            "extreme_metric": (
                "upper-bin Brier skill score for event Y >= training q95 under "
                "the strict-CDF convention"
            ),
            "area_weighting": "india_area_weight_km2 x IMD observation_fraction",
            "support_cells": int(np.count_nonzero(support)),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_block_length_initializations": 13,
        },
        "cache": frozen.cache_provenance(cache),
        "observation_stores": list(observations.source_stores),
        "source_snapshot_sha256": snapshot,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest["artifact_sha256"] = _output_checksums(output)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/evaluate the frozen-split FuXi-on-IMD categorical PBC baseline."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--calendar-window-days", type=int, default=DEFAULT_CALENDAR_WINDOW_DAYS
    )
    parser.add_argument(
        "--minimum-calendar-samples", type=int, default=DEFAULT_MINIMUM_SAMPLES
    )
    parser.add_argument(
        "--debias-spans", type=parse_spans, default=DEFAULT_DEBIAS_SPANS
    )
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--cdf-chunk-size", type=int, default=DEFAULT_CDF_CHUNK_SIZE)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.calendar_window_days < 1 or args.calendar_window_days % 2 != 1:
        raise ValueError("--calendar-window-days must be a positive odd integer")
    if args.minimum_calendar_samples < 1 or args.cdf_chunk_size < 1:
        raise ValueError("sample and chunk sizes must be positive")
    if args.ridge < 0.0 or not np.isfinite(args.ridge):
        raise ValueError("--ridge must be finite and nonnegative")
    if args.bootstrap_samples is None:
        args.bootstrap_samples = (
            SMOKE_BOOTSTRAP_SAMPLES if args.smoke else DEFAULT_BOOTSTRAP_SAMPLES
        )
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    requested = (
        default_output() if args.output is None else Path(args.output)
    ).resolve()
    args.output = requested
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {requested}")
    staging = requested.parent / f".{requested.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        run_experiment(args, staging)
        os.replace(staging, requested)
    except Exception as error:
        _atomic_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "started_utc": started,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "requested_output": str(requested),
            },
        )
        print(f"FAILED; diagnostics retained in {staging}", file=sys.stderr, flush=True)
        raise
    print(f"PASS: completed {'smoke' if args.smoke else 'full'} PBC run at {requested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
