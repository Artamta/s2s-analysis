#!/usr/bin/env python3
"""Build five auditable CCAI figures for the frozen FuXi-to-IMD adapter.

The figures use only the frozen 2022--2024 development-generalization audit.
They deliberately do not mix this adapter evaluation (W1 = init+0..6) with
INDIA-S2S-BENCH (W1 = init+1..7), and they do not claim independent 2025
confirmation.  New bootstrap and intensity diagnostics are post-hoc,
descriptive additions derived from the frozen case table and prediction cube.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


HERE = Path(__file__).resolve().parent
CLEAN_ROOT = HERE.parent.parent
BIAS_CORRECTION = CLEAN_ROOT / "bias-correction"
if str(BIAS_CORRECTION) not in sys.path:
    sys.path.insert(0, str(BIAS_CORRECTION))

from plot_physical_validation_results import load_india_boundary  # noqa: E402


DEFAULT_AUDIT = HERE / "results/full_context_jjas_2022_2024_job91439"
DEFAULT_RUN = (
    BIAS_CORRECTION
    / "results/fuxi_imd_full_context_compact_allweeks/full_20260811T152024Z"
)
DEFAULT_BOUNDARY = CLEAN_ROOT / "fuxi-dashboard/public/data/india-admin.json"
DEFAULT_OUTPUT = HERE / "results/full_context_jjas_2022_2024_ccai_figures_v1"

METHODS = ("raw_fuxi", "log_bias", "selected_adapter")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi",
    "log_bias": "Training-only log-bias",
    "selected_adapter": "Anchored neural adapter",
}
METHOD_SHORT = {
    "raw_fuxi": "Raw",
    "log_bias": "Log-bias",
    "selected_adapter": "Adapter",
}
METHOD_COLORS = {
    "raw_fuxi": "#4B5563",
    "log_bias": "#B24C8A",
    "selected_adapter": "#0072B2",
}
METHOD_MARKERS = {"raw_fuxi": "o", "log_bias": "s", "selected_adapter": "D"}
BASELINE_COLORS = {"raw_fuxi": "#4B5563", "log_bias": "#B24C8A"}
BASELINE_LABELS = {"raw_fuxi": "vs Raw FuXi", "log_bias": "vs log-bias"}
INK = "#17232E"
MUTED = "#5E6B75"
GRID = "#AAB5C0"
POSITIVE = "#087F5B"
NEGATIVE = "#B42318"
PANEL_BG = "#F7F9FA"

EXPECTED_YEARS = (2022, 2023, 2024)
EXPECTED_YEAR_COUNTS = {2022: 35, 2023: 35, 2024: 30}
EXPECTED_LEADS = (1, 2, 3, 4, 5, 6)
EXPECTED_REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
EXPECTED_AUDIT_MANIFEST_SHA256 = (
    "260fd5e344dead5359c482e74479ccbe1e3775fc5f0d040a4bbffdd412ac83e9"
)
EXPECTED_RUN_MANIFEST_SHA256 = (
    "cb5b2b6fa43f76e7d4e8eba22a185ee4084a9b4964b2de5978b3a527505566fa"
)
EXPECTED_SELECTION_SHA256 = (
    "d579aad90822ff504881f8334f18bd3962fd2c4a9bc75fb25e43455e9bbe0978"
)
REGION_LABELS = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast",
}

BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_BLOCK_LENGTH = 13
BOOTSTRAP_SEED = 20_260_818
THRESHOLDS_MM_DAY = (1.0, 5.0, 10.0, 20.0)
INTENSITY_STRATA = (
    ("dry_lt1", "<1", 0.0, 1.0),
    ("light_1_5", "1–5", 1.0, 5.0),
    ("moderate_5_10", "5–10", 5.0, 10.0),
    ("heavy_10_20", "10–20", 10.0, 20.0),
    ("extreme_ge20", "≥20", 20.0, np.inf),
)

FIGURE_STEMS = (
    "01_lead_skill_and_correction_decomposition",
    "02_year_region_robustness",
    "03_paired_case_gains_and_failures",
    "04_native_grid_spatial_footprint",
    "05_intensity_and_extremes_stress_test",
)
TABLE_FILENAMES = (
    "bootstrap_indices_two_stage.npy",
    "bootstrap_indices_within_year.npy",
    "case_win_rates.csv",
    "intensity_strata_metrics.csv",
    "lead_summary.csv",
    "paired_block_bootstrap_effects.csv",
    "paired_case_effects.csv",
    "spatial_diagnostics.nc",
    "spatial_improvement_summary.csv",
    "threshold_metrics.csv",
    "weighting_diagnostic.csv",
)
EXPECTED_SOURCE_RECORDS = {
    "renderer",
    "audit_manifest",
    "audit_case_metrics",
    "audit_summary_by_lead_region",
    "audit_summary_by_region",
    "audit_summary_by_year_lead_region",
    "audit_generalization_guards",
    "audit_predictions",
    "run_manifest",
    "frozen_selection",
    "run_predictions_area_weights",
    "india_boundary",
    "boundary_loader",
}
EXPECTED_OUTPUT_FILES = frozenset(
    {f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")}
    | {"README.md", "CAPTIONS.md"}
    | {f"tables/{name}" for name in TABLE_FILENAMES}
)


class FigureContractError(ValueError):
    """Raised when a frozen input or derived figure contract is violated."""


@dataclass(frozen=True)
class SourceBundle:
    audit: Path
    run: Path
    boundary_path: Path
    cases: pd.DataFrame
    summary_by_lead_region: pd.DataFrame
    predictions: xr.Dataset
    area_weight_km2: np.ndarray
    boundary_segments: tuple[np.ndarray, ...]
    boundary_provenance: Mapping[str, Any]
    audit_manifest: Mapping[str, Any]
    run_manifest: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in Path(path).rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(CLEAN_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _verify_recorded_hash(path: Path, expected: str) -> None:
    actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
    if actual != expected:
        raise FigureContractError(
            f"source hash mismatch for {path}: expected {expected}, got {actual}"
        )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.2,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#9AA7B1",
            "xtick.color": "#344451",
            "ytick.color": "#344451",
            "legend.fontsize": 9.2,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.set_facecolor("white")
    axis.grid(axis=grid_axis, color=GRID, alpha=0.26, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9.3)


def add_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    figure.text(
        0.055,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.2,
        color=MUTED,
    )
    figure.text(
        0.945,
        0.957,
        "DEVELOPMENT AUDIT",
        ha="right",
        va="top",
        fontsize=9.0,
        weight="bold",
        color="#8A360E",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF4E6",
            "edgecolor": "#E7B36A",
            "linewidth": 0.8,
        },
    )


def add_footer(figure: plt.Figure, text: str) -> None:
    figure.text(
        0.055,
        0.025,
        text,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )


def save_figure(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(
        png,
        bbox_inches="tight",
        dpi=320,
        metadata={"Software": "make_ccai_neural_adapter_figures.py"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "make_ccai_neural_adapter_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return png, pdf


def validate_case_table(cases: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "method_label",
        "init",
        "year",
        "lead_week",
        "region",
        "region_label",
        "rmse_mm_day",
        "mae_mm_day",
        "bias_mm_day",
        "acc",
        "valid_cell_count",
        "effective_area_km2",
    }
    missing = required.difference(cases.columns)
    if missing:
        raise FigureContractError(f"case table lacks columns: {sorted(missing)}")
    frame = cases.copy()
    frame["init"] = pd.to_datetime(frame["init"], errors="raise")
    if len(frame) != 9_000:
        raise FigureContractError(f"expected 9,000 case rows, found {len(frame)}")
    keys = ["method", "init", "lead_week", "region"]
    if frame.duplicated(keys).any():
        raise FigureContractError(
            "case table contains duplicate method/init/lead/region keys"
        )
    if tuple(sorted(frame.method.unique())) != tuple(sorted(METHODS)):
        raise FigureContractError("case table method set changed")
    if tuple(sorted(frame.year.unique())) != EXPECTED_YEARS:
        raise FigureContractError("case table year set changed")
    if tuple(sorted(frame.lead_week.unique())) != EXPECTED_LEADS:
        raise FigureContractError("case table lead set changed")
    if tuple(sorted(frame.region.unique())) != tuple(sorted(EXPECTED_REGIONS)):
        raise FigureContractError("case table region set changed")
    counts = frame[["init", "year"]].drop_duplicates().groupby("year").size().to_dict()
    if counts != EXPECTED_YEAR_COUNTS:
        raise FigureContractError(f"initialization counts changed: {counts}")
    numeric = ["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise FigureContractError("case table contains non-finite metrics")
    expected_per_method_region = 100 * len(EXPECTED_LEADS)
    cell_counts = frame.groupby(["method", "region"]).size()
    if not cell_counts.eq(expected_per_method_region).all():
        raise FigureContractError("case table is incomplete by method and region")
    return frame.sort_values(keys).reset_index(drop=True)


def reconcile_saved_summary(cases: pd.DataFrame, summary: pd.DataFrame) -> None:
    metric_columns = ["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]
    rebuilt = (
        cases.groupby(
            ["region", "region_label", "method", "method_label", "lead_week"],
            as_index=False,
        )[metric_columns]
        .mean()
        .sort_values(["region", "method", "lead_week"])
        .reset_index(drop=True)
    )
    stored = summary.sort_values(["region", "method", "lead_week"]).reset_index(
        drop=True
    )
    if list(stored.columns) != list(rebuilt.columns):
        raise FigureContractError("saved lead/region summary columns changed")
    try:
        pd.testing.assert_frame_equal(
            rebuilt,
            stored,
            check_dtype=False,
            check_exact=False,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
    except AssertionError as exc:
        raise FigureContractError("saved lead/region summary does not rebuild") from exc


def load_and_verify_sources(
    audit: Path, run: Path, boundary_path: Path
) -> SourceBundle:
    audit = Path(audit).expanduser().resolve()
    run = Path(run).expanduser().resolve()
    boundary_path = Path(boundary_path).expanduser().resolve()
    audit_manifest_path = audit / "manifest.json"
    run_manifest_path = run / "manifest.json"
    selection_path = run / "selection.json"
    for required in (audit_manifest_path, run_manifest_path, selection_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for path, expected in (
        (audit_manifest_path, EXPECTED_AUDIT_MANIFEST_SHA256),
        (run_manifest_path, EXPECTED_RUN_MANIFEST_SHA256),
        (selection_path, EXPECTED_SELECTION_SHA256),
    ):
        _verify_recorded_hash(path, expected)
    if audit_manifest.get("status") != "complete":
        raise FigureContractError("audit manifest is not complete")
    if run_manifest.get("status") != "complete":
        raise FigureContractError("adapter run manifest is not complete")
    if selection.get("status") != "frozen":
        raise FigureContractError("adapter selection is not frozen")
    if selection.get("selected_model") != "normal_climo_model":
        raise FigureContractError("unexpected selected adapter model")
    if audit_manifest.get("selected_model") != "normal_climo_model":
        raise FigureContractError("audit selected model changed")
    if audit_manifest.get("adapter_selection_sha256") != sha256_file(selection_path):
        raise FigureContractError(
            "audit is not linked to the supplied frozen selection"
        )

    required_audit_outputs = (
        "case_metrics.csv",
        "summary_by_lead_region.csv",
        "summary_by_region.csv",
        "summary_by_year_lead_region.csv",
        "generalization_guards.json",
        "predictions.zarr",
    )
    recorded_outputs = audit_manifest.get("outputs", {})
    for relative in required_audit_outputs:
        expected = recorded_outputs.get(relative)
        if not isinstance(expected, str):
            raise FigureContractError(f"audit manifest lacks hash for {relative}")
        _verify_recorded_hash(audit / relative, expected)

    run_artifacts = run_manifest.get("artifacts", {})
    for relative in ("selection.json", "predictions.zarr"):
        expected = run_artifacts.get(relative)
        if not isinstance(expected, str):
            raise FigureContractError(f"run manifest lacks hash for {relative}")
        _verify_recorded_hash(run / relative, expected)

    cases = validate_case_table(pd.read_csv(audit / "case_metrics.csv"))
    summary = pd.read_csv(audit / "summary_by_lead_region.csv")
    reconcile_saved_summary(cases, summary)

    predictions = xr.open_zarr(audit / "predictions.zarr", consolidated=True).load()
    expected_sizes = {
        "method": 3,
        "init": 100,
        "lead_week": 6,
        "latitude": 27,
        "longitude": 27,
    }
    if dict(predictions.sizes) != expected_sizes:
        raise FigureContractError(
            f"prediction dimensions changed: {dict(predictions.sizes)}"
        )
    if tuple(predictions.method.values.tolist()) != METHODS:
        raise FigureContractError("prediction method order changed")
    if tuple(int(value) for value in predictions.lead_week.values) != EXPECTED_LEADS:
        raise FigureContractError("prediction lead coordinates changed")
    support = np.asarray(predictions.adapter_support.values, dtype=bool)
    if support.shape != (27, 27) or int(support.sum()) != 171:
        raise FigureContractError("prediction support differs from 171-cell contract")
    init_years, init_counts = np.unique(
        pd.DatetimeIndex(predictions.init.values).year, return_counts=True
    )
    if dict(zip(init_years.tolist(), init_counts.tolist())) != EXPECTED_YEAR_COUNTS:
        raise FigureContractError("prediction initialization years/counts changed")

    with xr.open_zarr(run / "predictions.zarr", consolidated=True) as training:
        area_weight = np.asarray(
            training.area_weight_km2.load().values, dtype=np.float64
        )
    if area_weight.shape != support.shape:
        raise FigureContractError("area-weight grid differs from prediction grid")
    if int(np.count_nonzero(area_weight)) != 171:
        raise FigureContractError("area weights differ from 171-cell support")
    if not np.array_equal(area_weight > 0.0, support):
        raise FigureContractError("positive area weights differ from adapter support")
    if not np.isfinite(area_weight).all() or np.any(area_weight < 0.0):
        raise FigureContractError("area weights are invalid")

    boundary, boundary_provenance = load_india_boundary(boundary_path)
    if int(boundary_provenance.get("feature_count", 0)) < 30:
        raise FigureContractError("India boundary coverage is incomplete")
    return SourceBundle(
        audit=audit,
        run=run,
        boundary_path=boundary_path,
        cases=cases,
        summary_by_lead_region=summary,
        predictions=predictions,
        area_weight_km2=area_weight,
        boundary_segments=tuple(boundary),
        boundary_provenance=boundary_provenance,
        audit_manifest=audit_manifest,
        run_manifest=run_manifest,
    )


def ordered_initializations(cases: pd.DataFrame) -> pd.DatetimeIndex:
    values = (
        cases[["init"]]
        .drop_duplicates()
        .sort_values("init")
        .init.to_numpy(dtype="datetime64[ns]")
    )
    if len(values) != 100:
        raise FigureContractError("expected 100 unique initialization dates")
    return pd.DatetimeIndex(values)


def metric_cube(cases: pd.DataFrame, region: str, metric: str) -> np.ndarray:
    if metric not in {"rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"}:
        raise ValueError(f"unsupported metric: {metric}")
    dates = ordered_initializations(cases)
    cube = np.empty((len(METHODS), len(dates), len(EXPECTED_LEADS)), dtype=np.float64)
    selected = cases.loc[cases.region.eq(region)]
    for method_index, method in enumerate(METHODS):
        pivot = selected.loc[selected.method.eq(method)].pivot(
            index="init", columns="lead_week", values=metric
        )
        try:
            values = pivot.loc[dates, list(EXPECTED_LEADS)].to_numpy(dtype=np.float64)
        except KeyError as exc:
            raise FigureContractError(
                f"incomplete metric cube for {method}/{region}"
            ) from exc
        cube[method_index] = values
    if not np.isfinite(cube).all():
        raise FigureContractError(f"non-finite values in {metric}/{region} cube")
    return cube


def year_stratified_circular_indices(
    initializations: pd.DatetimeIndex,
    draws: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, dict[int, slice]]:
    """Sample circular blocks within each year, retaining all leads per start."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    years = np.asarray(initializations.year, dtype=np.int64)
    if tuple(sorted(np.unique(years).tolist())) != EXPECTED_YEARS:
        raise FigureContractError("bootstrap initialization years changed")
    rng = np.random.default_rng(seed)
    sampled_segments: list[np.ndarray] = []
    slices: dict[int, slice] = {}
    cursor = 0
    offsets = np.arange(block_length, dtype=np.int64)
    for year in EXPECTED_YEARS:
        positions = np.flatnonzero(years == year)
        n_year = len(positions)
        if n_year != EXPECTED_YEAR_COUNTS[year] or block_length > n_year:
            raise FigureContractError(f"invalid bootstrap contract for {year}")
        n_blocks = math.ceil(n_year / block_length)
        starts = rng.integers(0, n_year, size=(draws, n_blocks), endpoint=False)
        local = (starts[:, :, None] + offsets[None, None, :]) % n_year
        local = local.reshape(draws, -1)[:, :n_year]
        sampled_segments.append(positions[local])
        slices[year] = slice(cursor, cursor + n_year)
        cursor += n_year
    indices = np.concatenate(sampled_segments, axis=1)
    if indices.shape != (draws, len(initializations)):
        raise FigureContractError("bootstrap index matrix has the wrong shape")
    return indices, slices


def two_stage_circular_indices(
    initializations: pd.DatetimeIndex,
    draws: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample audit years, then circular blocks, retaining all leads.

    The three season slots retain their observed sizes (35, 35, and 30 starts).
    A resampled source year can fill any slot; circular wrapping permits a
    30-start source season to fill a 35-start slot without dropping the slot.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    years = np.asarray(initializations.year, dtype=np.int64)
    unique_years = np.asarray(sorted(np.unique(years)), dtype=np.int64)
    if tuple(unique_years.tolist()) != EXPECTED_YEARS:
        raise FigureContractError("two-stage bootstrap initialization years changed")
    groups = {year: np.flatnonzero(years == year) for year in EXPECTED_YEARS}
    if {year: len(group) for year, group in groups.items()} != EXPECTED_YEAR_COUNTS:
        raise FigureContractError("two-stage bootstrap year counts changed")
    slot_counts = [EXPECTED_YEAR_COUNTS[year] for year in EXPECTED_YEARS]
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_length, dtype=np.int64)
    result = np.empty((draws, len(initializations)), dtype=np.int16)
    for draw in range(draws):
        sampled_years = rng.choice(unique_years, size=len(unique_years), replace=True)
        pieces: list[np.ndarray] = []
        for source_year, target_count in zip(sampled_years, slot_counts):
            group = groups[int(source_year)]
            n_blocks = math.ceil(target_count / block_length)
            starts = rng.integers(0, len(group), size=n_blocks, endpoint=False)
            local = (starts[:, None] + offsets[None, :]) % len(group)
            pieces.append(group[local.reshape(-1)[:target_count]])
        result[draw] = np.concatenate(pieces).astype(np.int16)
    return result


def _effect_from_means(
    candidate_mean: np.ndarray, baseline_mean: np.ndarray, metric: str
) -> np.ndarray:
    if metric == "rmse_skill_pct":
        if np.any(baseline_mean <= 0.0):
            raise FigureContractError("RMSE baseline must be positive")
        return 100.0 * (baseline_mean - candidate_mean) / baseline_mean
    if metric == "acc_delta":
        return candidate_mean - baseline_mean
    raise ValueError(f"unknown paired effect metric: {metric}")


def summarize_effect(
    candidate: np.ndarray,
    baseline: np.ndarray,
    sample_indices: np.ndarray,
    metric: str,
) -> tuple[float, float, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 2:
        raise ValueError("paired arrays must have shape [initialization, lead]")
    point = float(
        _effect_from_means(
            np.asarray(candidate.mean()), np.asarray(baseline.mean()), metric
        )
    )
    candidate_draw = candidate[sample_indices].mean(axis=(1, 2))
    baseline_draw = baseline[sample_indices].mean(axis=(1, 2))
    effects = _effect_from_means(candidate_draw, baseline_draw, metric)
    lower, upper = np.quantile(effects, (0.025, 0.975))
    return point, float(lower), float(upper)


def build_bootstrap_effects(
    cases: pd.DataFrame,
    two_stage_indices: np.ndarray,
    within_year_indices: np.ndarray,
    year_slices: Mapping[int, slice],
) -> pd.DataFrame:
    method_index = {method: index for index, method in enumerate(METHODS)}
    rows: list[dict[str, Any]] = []

    def append_scope(
        *,
        scope_type: str,
        scope: str,
        region: str,
        lead_indices: Sequence[int],
        sample_columns: np.ndarray | slice,
        n_starts: int,
        bootstrap_indices: np.ndarray,
    ) -> None:
        rmse = metric_cube(cases, region, "rmse_mm_day")
        acc = metric_cube(cases, region, "acc")
        sample = bootstrap_indices[:, sample_columns]
        candidate_positions = (
            np.arange(100)[sample_columns]
            if isinstance(sample_columns, slice)
            else np.asarray(sample_columns, dtype=np.int64)
        )
        for baseline in ("raw_fuxi", "log_bias"):
            baseline_index = method_index[baseline]
            candidate_index = method_index["selected_adapter"]
            for metric, cube in (("rmse_skill_pct", rmse), ("acc_delta", acc)):
                candidate = cube[candidate_index][candidate_positions][:, lead_indices]
                reference = cube[baseline_index][candidate_positions][:, lead_indices]
                remap = np.empty(100, dtype=np.int64)
                remap.fill(-1)
                remap[candidate_positions] = np.arange(len(candidate_positions))
                local_sample = remap[sample]
                if np.any(local_sample < 0):
                    raise FigureContractError("bootstrap scope remapping failed")
                point, lower, upper = summarize_effect(
                    candidate, reference, local_sample, metric
                )
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "region": region,
                        "baseline": baseline,
                        "metric": metric,
                        "effect": point,
                        "ci_lower_2p5": lower,
                        "ci_upper_97p5": upper,
                        "n_starts": n_starts,
                        "n_case_leads": n_starts * len(lead_indices),
                        "bootstrap_draws": bootstrap_indices.shape[0],
                        "block_length_starts": BOOTSTRAP_BLOCK_LENGTH,
                        "interval": (
                            "paired within-year circular moving-block percentile 95%; "
                            "descriptive conditional on the named audit season"
                            if scope_type == "year"
                            else "paired two-stage percentile 95%: resample audit years, "
                            "then circular 13-start blocks; descriptive"
                        ),
                    }
                )

    all_positions = np.arange(100, dtype=np.int64)
    append_scope(
        scope_type="pooled",
        scope="W1-W6",
        region="all_india",
        lead_indices=np.arange(6),
        sample_columns=all_positions,
        n_starts=100,
        bootstrap_indices=two_stage_indices,
    )
    for lead_index, lead in enumerate(EXPECTED_LEADS):
        append_scope(
            scope_type="lead",
            scope=f"W{lead}",
            region="all_india",
            lead_indices=[lead_index],
            sample_columns=all_positions,
            n_starts=100,
            bootstrap_indices=two_stage_indices,
        )
    for year in EXPECTED_YEARS:
        append_scope(
            scope_type="year",
            scope=str(year),
            region="all_india",
            lead_indices=np.arange(6),
            sample_columns=year_slices[year],
            n_starts=EXPECTED_YEAR_COUNTS[year],
            bootstrap_indices=within_year_indices,
        )
    for region in EXPECTED_REGIONS[1:]:
        append_scope(
            scope_type="region",
            scope=REGION_LABELS[region],
            region=region,
            lead_indices=np.arange(6),
            sample_columns=all_positions,
            n_starts=100,
            bootstrap_indices=two_stage_indices,
        )
    result = pd.DataFrame(rows)
    expected_rows = 2 * 2 * (1 + 6 + 3 + 4)
    if len(result) != expected_rows:
        raise FigureContractError("bootstrap effect table is incomplete")
    return result


def build_lead_summary(cases: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse_mm_day", "mae_mm_day", "bias_mm_day", "acc"]
    india = cases.loc[cases.region.eq("all_india")]
    by_lead = india.groupby(["method", "method_label", "lead_week"], as_index=False)[
        metrics
    ].mean()
    pooled = india.groupby(["method", "method_label"], as_index=False)[metrics].mean()
    pooled["lead_week"] = "ALL"
    result = pd.concat([by_lead, pooled], ignore_index=True)
    return result[["method", "method_label", "lead_week", *metrics]]


def build_paired_case_effects(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    india = cases.loc[cases.region.eq("all_india")]
    keys = ["init", "year", "lead_week"]
    metrics = ["rmse_mm_day", "bias_mm_day", "acc"]
    wide = india.pivot(index=keys, columns="method", values=metrics)
    result = wide.index.to_frame(index=False)
    raw_rmse = wide[("rmse_mm_day", "raw_fuxi")].to_numpy()
    adapter_rmse = wide[("rmse_mm_day", "selected_adapter")].to_numpy()
    raw_acc = wide[("acc", "raw_fuxi")].to_numpy()
    adapter_acc = wide[("acc", "selected_adapter")].to_numpy()
    raw_bias = wide[("bias_mm_day", "raw_fuxi")].to_numpy()
    adapter_bias = wide[("bias_mm_day", "selected_adapter")].to_numpy()
    result["rmse_improvement_mm_day"] = raw_rmse - adapter_rmse
    result["acc_improvement"] = adapter_acc - raw_acc
    result["abs_bias_improvement_mm_day"] = np.abs(raw_bias) - np.abs(adapter_bias)
    result["signed_bias_change_mm_day"] = adapter_bias - raw_bias
    if len(result) != 600 or not np.isfinite(result.iloc[:, 3:].to_numpy()).all():
        raise FigureContractError("paired case-effect table is invalid")

    metric_columns = (
        ("rmse_improvement_mm_day", "RMSE"),
        ("acc_improvement", "ACC"),
        ("abs_bias_improvement_mm_day", "absolute bias"),
    )
    win_rows: list[dict[str, Any]] = []
    for lead in ("ALL", *EXPECTED_LEADS):
        selected = result if lead == "ALL" else result.loc[result.lead_week.eq(lead)]
        for column, label in metric_columns:
            values = selected[column].to_numpy(dtype=np.float64)
            win_rows.append(
                {
                    "lead_week": lead,
                    "metric": column,
                    "metric_label": label,
                    "wins": int(np.count_nonzero(values > 0.0)),
                    "ties": int(np.count_nonzero(values == 0.0)),
                    "total": len(values),
                    "win_fraction_pct": float(100.0 * np.mean(values > 0.0)),
                    "mean_effect": float(values.mean()),
                }
            )
    return result, pd.DataFrame(win_rows)


def build_threshold_metrics(
    predictions: xr.Dataset, area_weight_km2: np.ndarray
) -> pd.DataFrame:
    truth = np.asarray(predictions.truth_imd.values, dtype=np.float64)
    forecast = np.asarray(predictions.prediction.values, dtype=np.float64)
    weights = np.broadcast_to(area_weight_km2[None, None], truth.shape)
    common_valid = (
        (weights > 0.0) & np.isfinite(truth) & np.all(np.isfinite(forecast), axis=0)
    )
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        method_forecast = forecast[method_index]
        valid_weight = weights[common_valid]
        observed_values = truth[common_valid]
        forecast_values = method_forecast[common_valid]
        total = float(valid_weight.sum(dtype=np.float64))
        for threshold in THRESHOLDS_MM_DAY:
            observed_event = observed_values >= threshold
            forecast_event = forecast_values >= threshold
            hits = float(valid_weight[observed_event & forecast_event].sum())
            misses = float(valid_weight[observed_event & ~forecast_event].sum())
            false_alarms = float(valid_weight[~observed_event & forecast_event].sum())
            correct_negatives = float(
                valid_weight[~observed_event & ~forecast_event].sum()
            )
            random_hits = (hits + misses) * (hits + false_alarms) / total
            ets_denominator = hits + misses + false_alarms - random_hits
            ets = (hits - random_hits) / ets_denominator
            csi = hits / (hits + misses + false_alarms)
            pod = hits / (hits + misses)
            far = false_alarms / (hits + false_alarms)
            frequency_bias = (hits + false_alarms) / (hits + misses)
            brier = (misses + false_alarms) / total
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "threshold_mm_day": threshold,
                    "ets": ets,
                    "csi": csi,
                    "pod": pod,
                    "far": far,
                    "frequency_bias": frequency_bias,
                    "brier_score": brier,
                    "hits_area_weight_sum_km2_case_lead": hits,
                    "misses_area_weight_sum_km2_case_lead": misses,
                    "false_alarms_area_weight_sum_km2_case_lead": false_alarms,
                    "correct_negatives_area_weight_sum_km2_case_lead": correct_negatives,
                    "total_area_weight_sum_km2_case_lead": total,
                    "definition": (
                        "pooled deterministic weekly-mean cell-lead threshold contingency; "
                        "static cell-area x IMD-support weighted with common finite mask; "
                        "not daily extreme-event verification"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_intensity_metrics(
    predictions: xr.Dataset, area_weight_km2: np.ndarray
) -> pd.DataFrame:
    truth = np.asarray(predictions.truth_imd.values, dtype=np.float64)
    forecast = np.asarray(predictions.prediction.values, dtype=np.float64)
    weights = np.broadcast_to(area_weight_km2[None, None], truth.shape)
    common_valid = (
        (weights > 0.0) & np.isfinite(truth) & np.all(np.isfinite(forecast), axis=0)
    )
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        method_forecast = forecast[method_index]
        for key, label, lower, upper in INTENSITY_STRATA:
            mask = common_valid & (truth >= lower) & (truth < upper)
            selected_weight = weights[mask]
            error = (method_forecast - truth)[mask]
            observed = truth[mask]
            predicted = method_forecast[mask]
            denominator = float(selected_weight.sum(dtype=np.float64))
            if denominator <= 0.0:
                raise FigureContractError(f"empty intensity stratum: {key}")
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "stratum": key,
                    "stratum_label": label,
                    "lower_mm_day": lower,
                    "upper_mm_day": upper,
                    "cell_case_lead_count": int(np.count_nonzero(mask)),
                    "area_weight_sum_km2_case_lead": denominator,
                    "rmse_mm_day": float(
                        np.sqrt(np.sum(selected_weight * error**2) / denominator)
                    ),
                    "mae_mm_day": float(
                        np.sum(selected_weight * np.abs(error)) / denominator
                    ),
                    "bias_mm_day": float(np.sum(selected_weight * error) / denominator),
                    "truth_mean_mm_day": float(
                        np.sum(selected_weight * observed) / denominator
                    ),
                    "prediction_mean_mm_day": float(
                        np.sum(selected_weight * predicted) / denominator
                    ),
                    "definition": (
                        "pooled cell-lead errors stratified by verifying weekly-mean IMD; "
                        "static cell-area x IMD-support weighted with common finite mask; "
                        "not daily extreme-event verification"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_weighting_diagnostic(
    cases: pd.DataFrame,
    predictions: xr.Dataset,
    area_weight_km2: np.ndarray,
) -> pd.DataFrame:
    """Quantify static-map weighting versus the audit's dynamic case weights."""

    truth = np.asarray(predictions.truth_imd.values, dtype=np.float64)
    forecast = np.asarray(predictions.prediction.values, dtype=np.float64)
    spatial_weight = np.broadcast_to(area_weight_km2[None, None], truth.shape)
    common_valid = (
        (spatial_weight > 0.0)
        & np.isfinite(truth)
        & np.all(np.isfinite(forecast), axis=0)
    )
    static_by_case_lead = np.where(common_valid, spatial_weight, 0.0).sum(
        axis=(2, 3), dtype=np.float64
    )
    audit = (
        cases.loc[
            cases.region.eq("all_india") & cases.method.eq("selected_adapter"),
            ["init", "lead_week", "effective_area_km2"],
        ]
        .pivot(index="init", columns="lead_week", values="effective_area_km2")
        .loc[pd.DatetimeIndex(predictions.init.values), list(EXPECTED_LEADS)]
        .to_numpy(dtype=np.float64)
    )
    if audit.shape != static_by_case_lead.shape:
        raise FigureContractError("weighting diagnostic case grid changed")
    difference = static_by_case_lead - audit
    static_total = float(static_by_case_lead.sum(dtype=np.float64))
    audit_total = float(audit.sum(dtype=np.float64))
    relative_excess = float(100.0 * (static_total - audit_total) / audit_total)
    return pd.DataFrame(
        [
            {
                "static_area_weight_sum_km2_case_lead": static_total,
                "audit_effective_area_sum_km2_case_lead": audit_total,
                "static_minus_audit_km2_case_lead": static_total - audit_total,
                "static_relative_excess_pct": relative_excess,
                "case_leads_with_fractional_coverage_difference": int(
                    np.count_nonzero(np.abs(difference) > 1.0e-6)
                ),
                "total_case_leads": int(difference.size),
                "maximum_case_lead_difference_km2": float(np.max(np.abs(difference))),
                "static_weight_definition": (
                    "frozen cell-area x IMD-support weights with common finite truth/forecast mask"
                ),
                "audit_weight_definition": (
                    "region area x saved-at-scoring-time fractional weekly coverage"
                ),
                "reason_for_static_posthoc_weighting": (
                    "fractional weekly_coverage was not saved in the frozen prediction cube"
                ),
            }
        ]
    )


def positive_area_fraction(
    field: np.ndarray, area_weight_km2: np.ndarray, support: np.ndarray
) -> float:
    valid = support & np.isfinite(field) & (area_weight_km2 > 0.0)
    weight = np.where(valid, area_weight_km2, 0.0)
    if weight.sum() <= 0.0:
        raise FigureContractError("spatial field has no positive evaluation weight")
    return float(100.0 * weight[field > 0.0].sum() / weight.sum())


def build_spatial_diagnostics(
    predictions: xr.Dataset, area_weight_km2: np.ndarray
) -> tuple[xr.Dataset, pd.DataFrame]:
    truth = np.asarray(predictions.truth_imd.values, dtype=np.float64)
    forecast = np.asarray(predictions.prediction.values, dtype=np.float64)
    support = np.asarray(predictions.adapter_support.values, dtype=bool)
    truth_support = truth[..., support]
    forecast_support = forecast[..., support]
    common_valid = np.isfinite(truth_support) & np.all(
        np.isfinite(forecast_support), axis=0
    )
    errors = np.where(
        common_valid[None], forecast_support - truth_support[None], np.nan
    )
    rmse = np.full((len(METHODS), *support.shape), np.nan, dtype=np.float64)
    bias = np.full_like(rmse, np.nan)
    lead_rmse = np.full(
        (len(METHODS), len(EXPECTED_LEADS), *support.shape),
        np.nan,
        dtype=np.float64,
    )
    rmse[:, support] = np.sqrt(np.nanmean(errors**2, axis=(1, 2)))
    bias[:, support] = np.nanmean(errors, axis=(1, 2))
    lead_rmse[:, :, support] = np.sqrt(np.nanmean(errors**2, axis=1))
    method_index = {method: index for index, method in enumerate(METHODS)}
    adapter = method_index["selected_adapter"]
    skill = np.stack(
        [
            rmse[method_index["raw_fuxi"]] - rmse[adapter],
            rmse[method_index["log_bias"]] - rmse[adapter],
        ]
    )
    lead_skill = np.stack(
        [
            lead_rmse[method_index["raw_fuxi"]] - lead_rmse[adapter],
            lead_rmse[method_index["log_bias"]] - lead_rmse[adapter],
        ]
    )
    diagnostics = xr.Dataset(
        data_vars={
            "rmse": (("method", "latitude", "longitude"), rmse),
            "bias": (("method", "latitude", "longitude"), bias),
            "lead_rmse": (
                ("method", "lead_week", "latitude", "longitude"),
                lead_rmse,
            ),
            "rmse_reduction": (
                ("baseline", "latitude", "longitude"),
                skill,
            ),
            "lead_rmse_reduction": (
                ("baseline", "lead_week", "latitude", "longitude"),
                lead_skill,
            ),
            "adapter_support": (("latitude", "longitude"), support),
            "area_weight_km2": (
                ("latitude", "longitude"),
                area_weight_km2,
            ),
        },
        coords={
            "method": list(METHODS),
            "baseline": ["raw_fuxi", "log_bias"],
            "lead_week": list(EXPECTED_LEADS),
            "latitude": predictions.latitude.values,
            "longitude": predictions.longitude.values,
        },
        attrs={
            "scope": "2022-2024 development generalization audit",
            "local_rmse_definition": "sqrt(mean cellwise squared error across starts and leads)",
            "local_reduction_definition": "baseline local RMSE minus adapter local RMSE",
            "spatial_inference": "descriptive; no pixel-wise significance assessment",
            "grid": "native 1.5-degree evaluation grid; no interpolation",
        },
    )
    rows: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(("raw_fuxi", "log_bias")):
        pooled = skill[baseline_index]
        rows.append(
            {
                "baseline": baseline,
                "scope": "pooled_W1-W6",
                "lead_week": "ALL",
                "area_fraction_lower_rmse_pct": positive_area_fraction(
                    pooled, area_weight_km2, support
                ),
                "spatial_significance": "not assessed",
            }
        )
        for lead_index, lead in enumerate(EXPECTED_LEADS):
            rows.append(
                {
                    "baseline": baseline,
                    "scope": f"W{lead}",
                    "lead_week": lead,
                    "area_fraction_lower_rmse_pct": positive_area_fraction(
                        lead_skill[baseline_index, lead_index], area_weight_km2, support
                    ),
                    "spatial_significance": "not assessed",
                }
            )
    return diagnostics, pd.DataFrame(rows)


def _method_lines(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    title: str,
) -> None:
    leads = np.asarray(EXPECTED_LEADS)
    for method in METHODS:
        values = summary.loc[
            summary.method.eq(method) & summary.lead_week.ne("ALL")
        ].copy()
        values["lead_week"] = values.lead_week.astype(int)
        values = values.sort_values("lead_week")
        axis.plot(
            leads,
            values[metric],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=3.0 if method == "selected_adapter" else 2.2,
            markersize=7.0 if method == "selected_adapter" else 6.2,
            markeredgecolor="white",
            markeredgewidth=0.75,
            label=METHOD_LABELS[method],
            zorder=5 if method == "selected_adapter" else 4,
        )
    axis.set_xticks(leads, [f"W{lead}" for lead in leads])
    axis.set_xlim(0.7, 6.3)
    axis.set_xlabel("Lead week")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", color=INK, pad=9)
    style_axis(axis)


def plot_lead_scorecard(
    lead_summary: pd.DataFrame, effects: pd.DataFrame, output: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 9.5))
    figure.subplots_adjust(
        left=0.075, right=0.965, top=0.855, bottom=0.12, hspace=0.37, wspace=0.25
    )
    add_header(
        figure,
        "Anchored neural postprocessor: lead-wise skill and correction anatomy",
        "FuXi → IMD · 100 JJAS starts in 2022–2024 · W1 = initialization day through +6",
    )

    _method_lines(
        axes[0, 0],
        lead_summary,
        "rmse_mm_day",
        ylabel="Mean case-wise RMSE (mm day$^{-1}$)",
        title="a  Error magnitude",
    )
    _method_lines(
        axes[0, 1],
        lead_summary,
        "acc",
        ylabel="Common-reference spatial ACC",
        title="b  Anomaly pattern skill",
    )
    axes[0, 1].legend(frameon=False, loc="upper right", handlelength=2.4)

    effect_axis = axes[1, 0]
    lead_effects = effects.loc[
        effects.scope_type.eq("lead") & effects.metric.eq("rmse_skill_pct")
    ]
    positions = np.asarray(EXPECTED_LEADS, dtype=float)
    offsets = {"raw_fuxi": -0.13, "log_bias": 0.13}
    for baseline in ("raw_fuxi", "log_bias"):
        values = lead_effects.loc[lead_effects.baseline.eq(baseline)].copy()
        values["lead"] = values.scope.str.removeprefix("W").astype(int)
        values = values.sort_values("lead")
        effect = values.effect.to_numpy()
        lower = values.ci_lower_2p5.to_numpy()
        upper = values.ci_upper_97p5.to_numpy()
        effect_axis.errorbar(
            positions + offsets[baseline],
            effect,
            yerr=np.vstack((effect - lower, upper - effect)),
            color=BASELINE_COLORS[baseline],
            marker="o" if baseline == "raw_fuxi" else "s",
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="none",
            elinewidth=1.7,
            capsize=4,
            label=BASELINE_LABELS[baseline],
            zorder=4,
        )
    effect_axis.axhline(0.0, color="#65727C", linewidth=1.0, linestyle="--")
    effect_axis.set_xticks(positions, [f"W{lead}" for lead in EXPECTED_LEADS])
    effect_axis.set_xlim(0.65, 6.35)
    effect_axis.set_xlabel("Lead week")
    effect_axis.set_ylabel("Adapter RMSE reduction (%)")
    effect_axis.set_title("c  Paired RMSE reduction", loc="left", color=INK, pad=9)
    effect_axis.legend(frameon=False, loc="upper right")
    style_axis(effect_axis)

    bias_axis = axes[1, 1]
    _method_lines(
        bias_axis,
        lead_summary,
        "bias_mm_day",
        ylabel="Mean signed bias (mm day$^{-1}$)",
        title="d  Mean-bias trade-off",
    )
    bias_axis.axhline(0.0, color="#65727C", linewidth=1.0, linestyle="--", zorder=2)

    pooled = lead_summary.loc[lead_summary.lead_week.eq("ALL")].set_index("method")
    raw_rmse = pooled.loc["raw_fuxi", "rmse_mm_day"]
    adapter_rmse = pooled.loc["selected_adapter", "rmse_mm_day"]
    raw_acc = pooled.loc["raw_fuxi", "acc"]
    adapter_acc = pooled.loc["selected_adapter", "acc"]
    add_footer(
        figure,
        (
            f"Pooled W1–W6: RMSE {raw_rmse:.3f} → {adapter_rmse:.3f} mm day⁻¹ "
            f"({100*(raw_rmse-adapter_rmse)/raw_rmse:.2f}% lower); ACC {raw_acc:.3f} → "
            f"{adapter_acc:.3f} (Δ {adapter_acc-raw_acc:+.3f}). Error bars: 2,000 paired, "
            "two-stage year + circular block-13 percentile intervals; descriptive, not p-values."
        ),
    )
    return list(save_figure(figure, output / FIGURE_STEMS[0]))


def plot_robustness(effects: pd.DataFrame, output: Path) -> list[Path]:
    groups = [
        ("year", "2022", "2022"),
        ("year", "2023", "2023"),
        ("year", "2024", "2024"),
        ("region", "Northwest India", "Northwest"),
        ("region", "Central India", "Central"),
        ("region", "South Peninsula", "South peninsula"),
        ("region", "East & Northeast", "East & Northeast"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 8.2), sharey=True)
    figure.subplots_adjust(left=0.18, right=0.965, top=0.84, bottom=0.15, wspace=0.16)
    add_header(
        figure,
        "The raw-to-adapter gain repeats across years and regional masks",
        "Neural increment over the log-bias anchor is smaller and heterogeneous",
    )
    y = np.arange(len(groups), dtype=float)
    offsets = {"raw_fuxi": -0.13, "log_bias": 0.13}
    metric_specs = (
        ("rmse_skill_pct", "a  RMSE reduction", "Reduction in mean case RMSE (%)"),
        ("acc_delta", "b  ACC change", "Adapter − baseline ACC"),
    )
    for axis, (metric, title, xlabel) in zip(axes, metric_specs):
        axis.axvline(0.0, color="#65727C", linewidth=1.1, linestyle="--", zorder=1)
        axis.axhline(2.5, color="#C9D1D7", linewidth=1.0, zorder=1)
        for baseline in ("raw_fuxi", "log_bias"):
            points: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for scope_type, scope, _ in groups:
                match = effects.loc[
                    effects.scope_type.eq(scope_type)
                    & effects.scope.eq(scope)
                    & effects.baseline.eq(baseline)
                    & effects.metric.eq(metric)
                ]
                if len(match) != 1:
                    raise FigureContractError(
                        f"missing robustness effect: {scope_type}/{scope}/{baseline}/{metric}"
                    )
                row = match.iloc[0]
                points.append(float(row.effect))
                lower.append(float(row.ci_lower_2p5))
                upper.append(float(row.ci_upper_97p5))
            point_array = np.asarray(points)
            lower_array = np.asarray(lower)
            upper_array = np.asarray(upper)
            axis.errorbar(
                point_array,
                y + offsets[baseline],
                xerr=np.vstack((point_array - lower_array, upper_array - point_array)),
                color=BASELINE_COLORS[baseline],
                marker="o" if baseline == "raw_fuxi" else "s",
                markersize=7.0,
                markeredgecolor="white",
                markeredgewidth=0.75,
                linestyle="none",
                elinewidth=1.8,
                capsize=4,
                label=BASELINE_LABELS[baseline],
                zorder=4,
            )
        axis.set_title(title, loc="left", color=INK, pad=10)
        axis.set_xlabel(xlabel)
        axis.set_yticks(y, [label for _, _, label in groups])
        axis.grid(axis="x", color=GRID, alpha=0.26, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9.5)
    axes[0].invert_yaxis()
    axes[0].text(
        -0.22,
        0.93,
        "AUDIT\nYEARS",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        weight="bold",
        color=MUTED,
        linespacing=1.15,
        clip_on=False,
    )
    axes[0].text(
        -0.22,
        0.49,
        "REGIONAL\nMASKS",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        weight="bold",
        color=MUTED,
        linespacing=1.15,
        clip_on=False,
    )
    axes[1].legend(frameon=False, loc="lower right")
    add_footer(
        figure,
        (
            "Year rows use paired within-year circular block-13 intervals. Regional rows use a "
            "paired two-stage bootstrap that resamples audit years, then circular block-13 starts "
            "within years (2,000 draws). Descriptive only; regional masks overlap."
        ),
    )
    return list(save_figure(figure, output / FIGURE_STEMS[1]))


def _violin_panel(
    axis: plt.Axes,
    paired: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    seed: int,
) -> None:
    values = [
        paired.loc[paired.lead_week.eq(lead), column].to_numpy(dtype=np.float64)
        for lead in EXPECTED_LEADS
    ]
    parts = axis.violinplot(
        values,
        positions=np.arange(1, 7),
        widths=0.82,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(METHOD_COLORS["selected_adapter"])
        body.set_edgecolor(METHOD_COLORS["selected_adapter"])
        body.set_alpha(0.16)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(seed)
    for lead, lead_values in zip(EXPECTED_LEADS, values):
        jitter = rng.uniform(-0.16, 0.16, size=len(lead_values))
        axis.scatter(
            lead + jitter,
            lead_values,
            s=9,
            color=METHOD_COLORS["selected_adapter"],
            alpha=0.24,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        axis.scatter(
            [lead],
            [np.mean(lead_values)],
            s=34,
            marker="D",
            color="#E69F00",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    axis.axhline(0.0, color="#65727C", linewidth=1.0, linestyle="--", zorder=2)
    axis.set_xticks(EXPECTED_LEADS, [f"W{lead}" for lead in EXPECTED_LEADS])
    axis.set_xlabel("Lead week")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", color=INK, pad=9)
    style_axis(axis)


def plot_paired_cases(
    paired: pd.DataFrame, win_rates: pd.DataFrame, output: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 9.6))
    figure.subplots_adjust(
        left=0.08, right=0.965, top=0.85, bottom=0.13, hspace=0.37, wspace=0.25
    )
    add_header(
        figure,
        "Most case-leads improve in RMSE and ACC—but not in absolute bias",
        "Each point is one dependent initialization × lead combination; orange diamonds are lead means",
    )
    _violin_panel(
        axes[0, 0],
        paired,
        "rmse_improvement_mm_day",
        "a  Paired RMSE effect",
        "Raw − adapter RMSE (mm day$^{-1}$)",
        BOOTSTRAP_SEED + 1,
    )
    _violin_panel(
        axes[0, 1],
        paired,
        "acc_improvement",
        "b  Paired ACC effect",
        "Adapter − raw ACC",
        BOOTSTRAP_SEED + 2,
    )
    _violin_panel(
        axes[1, 0],
        paired,
        "abs_bias_improvement_mm_day",
        "c  Paired absolute-bias effect",
        "|Raw bias| − |adapter bias| (mm day$^{-1}$)",
        BOOTSTRAP_SEED + 3,
    )

    axis = axes[1, 1]
    lead_rates = win_rates.loc[win_rates.lead_week.ne("ALL")].copy()
    metric_order = (
        ("rmse_improvement_mm_day", "RMSE", "#0072B2"),
        ("acc_improvement", "ACC", "#009E73"),
        ("abs_bias_improvement_mm_day", "Absolute bias", "#D55E00"),
    )
    x = np.arange(1, 7, dtype=float)
    width = 0.23
    for index, (metric, label, color) in enumerate(metric_order):
        values = (
            lead_rates.loc[lead_rates.metric.eq(metric)]
            .assign(lead_numeric=lambda frame: frame.lead_week.astype(int))
            .sort_values("lead_numeric")
        )
        axis.bar(
            x + (index - 1) * width,
            values.win_fraction_pct,
            width=width,
            color=color,
            alpha=0.88,
            label=label,
        )
    axis.axhline(50.0, color="#65727C", linewidth=1.0, linestyle="--")
    axis.set_xticks(x, [f"W{lead}" for lead in EXPECTED_LEADS])
    axis.set_ylim(0.0, 100.0)
    axis.set_xlabel("Lead week")
    axis.set_ylabel("Case-leads improved (%)")
    axis.set_title("d  Win fraction", loc="left", color=INK, pad=9)
    axis.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.3)
    style_axis(axis)
    pooled = win_rates.loc[win_rates.lead_week.eq("ALL")].set_index("metric")
    add_footer(
        figure,
        (
            f"Pooled wins: RMSE {pooled.loc['rmse_improvement_mm_day','win_fraction_pct']:.1f}% · "
            f"ACC {pooled.loc['acc_improvement','win_fraction_pct']:.1f}% · |bias| "
            f"{pooled.loc['abs_bias_improvement_mm_day','win_fraction_pct']:.1f}%. Positive favors "
            "the adapter; 600 dependent points are shown descriptively, not as independent samples."
        ),
    )
    return list(save_figure(figure, output / FIGURE_STEMS[2]))


def decorate_map(
    axis: plt.Axes,
    boundary: Sequence[np.ndarray],
    *,
    left: bool,
) -> None:
    axis.set_xlim(67.0, 99.0)
    axis.set_ylim(6.0, 39.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#F2F5F7")
    axis.add_collection(
        LineCollection(
            boundary,
            colors="#20323E",
            linewidths=0.28,
            alpha=0.96,
            zorder=5,
        ),
        autolim=False,
    )
    axis.set_xticks((70, 80, 90), ("70°E", "80°E", "90°E"))
    if left:
        axis.set_yticks((10, 20, 30), ("10°N", "20°N", "30°N"))
    else:
        axis.set_yticks(())
    axis.tick_params(labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#A8B7C0")
        spine.set_linewidth(0.6)


def plot_spatial_footprint(
    diagnostics: xr.Dataset,
    spatial_summary: pd.DataFrame,
    weighting_diagnostic: pd.DataFrame,
    boundary: Sequence[np.ndarray],
    output: Path,
) -> list[Path]:
    fields = np.asarray(diagnostics.rmse_reduction.values, dtype=np.float64)
    support = np.asarray(diagnostics.adapter_support.values, dtype=bool)
    latitude = np.asarray(diagnostics.latitude.values, dtype=np.float64)
    longitude = np.asarray(diagnostics.longitude.values, dtype=np.float64)
    finite = np.abs(fields[:, support])
    vmax = max(0.5, math.ceil(float(np.nanpercentile(finite, 98.0)) * 10.0) / 10.0)
    cmap = LinearSegmentedColormap.from_list(
        "skill_diverging", ["#B42318", "#F7F7F7", "#0868AC"], N=256
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figure = plt.figure(figsize=(14.8, 9.1))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        height_ratios=(3.5, 1.5),
        left=0.07,
        right=0.94,
        bottom=0.13,
        top=0.84,
        wspace=0.12,
        hspace=0.30,
    )
    add_header(
        figure,
        "Local RMSE falls across most of the weighted India footprint",
        "Native 1.5° evaluation cells · positive values mean lower adapter RMSE · no interpolation",
    )
    map_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    caxis = figure.add_subplot(grid[0, 2])
    for index, (axis, baseline, title) in enumerate(
        zip(
            map_axes,
            ("raw_fuxi", "log_bias"),
            ("a  Adapter vs Raw FuXi", "b  Adapter vs log-bias"),
        )
    ):
        shown = np.where(support, fields[index], np.nan)
        image = axis.pcolormesh(
            longitude,
            latitude,
            np.ma.masked_invalid(shown),
            cmap=cmap,
            norm=norm,
            shading="nearest",
            rasterized=True,
        )
        decorate_map(axis, boundary, left=index == 0)
        pooled = spatial_summary.loc[
            spatial_summary.baseline.eq(baseline)
            & spatial_summary.lead_week.astype(str).eq("ALL")
        ].iloc[0]
        axis.set_title(title, loc="left", color=INK, pad=8)
        axis.text(
            0.03,
            0.04,
            f"{pooled.area_fraction_lower_rmse_pct:.1f}% of weighted area improves",
            transform=axis.transAxes,
            fontsize=8.8,
            weight="semibold",
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#C9D1D7",
                "alpha": 0.92,
            },
        )
    colorbar = figure.colorbar(image, cax=caxis, orientation="vertical", extend="both")
    colorbar.set_label("Baseline − adapter local RMSE (mm day$^{-1}$)", fontsize=9.0)
    colorbar.ax.tick_params(labelsize=8)

    axis = figure.add_subplot(grid[1, :2])
    scopes = ["ALL", *EXPECTED_LEADS]
    x = np.arange(len(scopes), dtype=float)
    width = 0.34
    for index, baseline in enumerate(("raw_fuxi", "log_bias")):
        values: list[float] = []
        for scope in scopes:
            match = spatial_summary.loc[
                spatial_summary.baseline.eq(baseline)
                & spatial_summary.lead_week.astype(str).eq(str(scope))
            ]
            if len(match) != 1:
                raise FigureContractError(
                    f"missing spatial summary for {baseline}/{scope}"
                )
            values.append(float(match.iloc[0].area_fraction_lower_rmse_pct))
        axis.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            color=BASELINE_COLORS[baseline],
            alpha=0.88,
            label=BASELINE_LABELS[baseline],
        )
    axis.axhline(50.0, color="#65727C", linewidth=1.0, linestyle="--")
    axis.set_xticks(x, ["Pooled", *[f"W{lead}" for lead in EXPECTED_LEADS]])
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Weighted area with lower local RMSE (%)")
    axis.set_title(
        "c  Spatial coverage of improvement by lead", loc="left", color=INK, pad=9
    )
    axis.legend(frameon=False, loc="upper right")
    style_axis(axis)
    weight_row = weighting_diagnostic.iloc[0]
    add_footer(
        figure,
        (
            "Local RMSE = √mean(error²) at each cell over the stated starts/leads. Area fractions "
            "use static cell-area × IMD-support weights; fractional weekly coverage was not saved "
            f"({weight_row.static_relative_excess_pct:.4f}% pooled-weight difference). Descriptive; "
            "no pixel-wise significance claim."
        ),
    )
    return list(save_figure(figure, output / FIGURE_STEMS[3]))


def _plot_method_metric(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    categorical_labels: Sequence[str] | None = None,
) -> None:
    for method in METHODS:
        values = frame.loc[frame.method.eq(method)]
        if categorical_labels is None:
            values = values.sort_values(x_column)
            x = values[x_column].to_numpy(dtype=np.float64)
        else:
            order = {label: index for index, label in enumerate(categorical_labels)}
            values = values.assign(_order=values[x_column].map(order)).sort_values(
                "_order"
            )
            x = np.arange(len(categorical_labels), dtype=float)
        axis.plot(
            x,
            values[y_column],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=3.0 if method == "selected_adapter" else 2.1,
            markersize=6.8 if method == "selected_adapter" else 5.8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=METHOD_LABELS[method],
            zorder=5 if method == "selected_adapter" else 4,
        )
    if categorical_labels is not None:
        axis.set_xticks(np.arange(len(categorical_labels)), categorical_labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", color=INK, pad=9)
    style_axis(axis)


def plot_intensity_stress_test(
    threshold_metrics: pd.DataFrame,
    intensity_metrics: pd.DataFrame,
    weighting_diagnostic: pd.DataFrame,
    output: Path,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 9.5))
    figure.subplots_adjust(
        left=0.08, right=0.965, top=0.85, bottom=0.13, hspace=0.37, wspace=0.25
    )
    add_header(
        figure,
        "Aggregate gains do not solve wet-tail calibration",
        "Post-hoc weekly-mean rainfall-rate diagnostic from the frozen 2022–2024 prediction cube",
    )
    _plot_method_metric(
        axes[0, 0],
        threshold_metrics,
        "threshold_mm_day",
        "ets",
        title="a  Equitable threat score",
        xlabel="Weekly-mean rain threshold (mm day$^{-1}$)",
        ylabel="ETS (higher is better)",
    )
    axes[0, 0].set_xticks(THRESHOLDS_MM_DAY)
    _plot_method_metric(
        axes[0, 1],
        threshold_metrics,
        "threshold_mm_day",
        "frequency_bias",
        title="b  Event-frequency calibration",
        xlabel="Weekly-mean rain threshold (mm day$^{-1}$)",
        ylabel="Frequency bias (ideal = 1)",
    )
    axes[0, 1].set_xticks(THRESHOLDS_MM_DAY)
    axes[0, 1].axhline(1.0, color="#65727C", linewidth=1.0, linestyle="--")
    axes[0, 1].legend(frameon=False, loc="lower left")

    labels = [entry[1] for entry in INTENSITY_STRATA]
    _plot_method_metric(
        axes[1, 0],
        intensity_metrics,
        "stratum_label",
        "rmse_mm_day",
        title="c  Error by verifying IMD intensity",
        xlabel="Observed weekly-mean stratum (mm day$^{-1}$)",
        ylabel="Pooled point RMSE (mm day$^{-1}$)",
        categorical_labels=labels,
    )
    _plot_method_metric(
        axes[1, 1],
        intensity_metrics,
        "stratum_label",
        "bias_mm_day",
        title="d  Signed error by verifying IMD intensity",
        xlabel="Observed weekly-mean stratum (mm day$^{-1}$)",
        ylabel="Pooled point bias (mm day$^{-1}$)",
        categorical_labels=labels,
    )
    axes[1, 1].axhline(0.0, color="#65727C", linewidth=1.0, linestyle="--")

    extreme_threshold = threshold_metrics.loc[
        threshold_metrics.threshold_mm_day.eq(20.0)
    ].set_index("method")
    extreme_stratum = intensity_metrics.loc[
        intensity_metrics.stratum.eq("extreme_ge20")
    ].set_index("method")
    axes[0, 0].text(
        0.97,
        0.93,
        (
            "At ≥20 mm day⁻¹\n"
            f"ETS: raw {extreme_threshold.loc['raw_fuxi','ets']:.3f}, "
            f"adapter {extreme_threshold.loc['selected_adapter','ets']:.3f}"
        ),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=MUTED,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#C9D1D7",
            "alpha": 0.94,
        },
    )
    axes[1, 1].text(
        0.03,
        0.07,
        (
            "≥20 mm day⁻¹ adapter bias\n"
            f"{extreme_stratum.loc['selected_adapter','bias_mm_day']:.2f} mm day⁻¹"
        ),
        transform=axes[1, 1].transAxes,
        fontsize=8.5,
        color=NEGATIVE,
        weight="semibold",
    )
    weight_row = weighting_diagnostic.iloc[0]
    add_footer(
        figure,
        (
            "Threshold scores pool weekly-mean cell–lead events; strata are defined by verifying "
            "weekly-mean IMD. Static area × support weights omit unsaved fractional coverage: "
            f"{int(weight_row.case_leads_with_fractional_coverage_difference)}/"
            f"{int(weight_row.total_case_leads)} case-leads differ, {weight_row.static_relative_excess_pct:.4f}% "
            "in total. Not a daily-extreme or selection analysis."
        ),
    )
    return list(save_figure(figure, output / FIGURE_STEMS[4]))


def build_captions(
    lead_summary: pd.DataFrame,
    effects: pd.DataFrame,
    win_rates: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    intensity_metrics: pd.DataFrame,
    weighting_diagnostic: pd.DataFrame,
) -> str:
    pooled = lead_summary.loc[lead_summary.lead_week.eq("ALL")].set_index("method")
    raw = pooled.loc["raw_fuxi"]
    log = pooled.loc["log_bias"]
    adapter = pooled.loc["selected_adapter"]
    pooled_effect = effects.loc[
        effects.scope_type.eq("pooled") & effects.baseline.eq("raw_fuxi")
    ].set_index("metric")
    wins = win_rates.loc[win_rates.lead_week.eq("ALL")].set_index("metric")
    area = spatial_summary.loc[
        spatial_summary.lead_week.astype(str).eq("ALL")
    ].set_index("baseline")
    extreme_threshold = threshold_metrics.loc[
        threshold_metrics.threshold_mm_day.eq(20.0)
    ].set_index("method")
    extreme = intensity_metrics.loc[
        intensity_metrics.stratum.eq("extreme_ge20")
    ].set_index("method")
    weight = weighting_diagnostic.iloc[0]
    return f"""# Captions: frozen FuXi–IMD anchored neural adapter

Global scope for all figures: development generalization audit, not an untouched independent test. The frozen model was selected on 2018–2019 and evaluated here on 100 operational JJAS starts (35/35/30 in 2022/2023/2024), with 171 IMD-supported cells. W1 is initialization day through +6; W6 is +35 through +41. ACC uses one common fixed training-only 2002–2017 IMD climatology.

## Figure 1 — Lead-wise skill and correction anatomy

Across the six leads, the anchored neural adapter has lower mean case-wise RMSE and higher common-reference spatial ACC than raw FuXi. Pooled W1–W6 RMSE is {adapter.rmse_mm_day:.3f} versus {raw.rmse_mm_day:.3f} mm day⁻¹ ({pooled_effect.loc['rmse_skill_pct','effect']:.2f}% lower; descriptive paired block interval [{pooled_effect.loc['rmse_skill_pct','ci_lower_2p5']:.2f}, {pooled_effect.loc['rmse_skill_pct','ci_upper_97p5']:.2f}]%). ACC is {adapter.acc:.3f} versus {raw.acc:.3f} (Δ {pooled_effect.loc['acc_delta','effect']:+.3f}; 95% interval [{pooled_effect.loc['acc_delta','ci_lower_2p5']:+.3f}, {pooled_effect.loc['acc_delta','ci_upper_97p5']:+.3f}]). Most RMSE reduction is supplied by the training-only log-bias anchor ({log.rmse_mm_day:.3f} mm day⁻¹); the neural residual provides the smaller increment to {adapter.rmse_mm_day:.3f}. Signed bias becomes more negative ({raw.bias_mm_day:.3f} to {adapter.bias_mm_day:.3f} mm day⁻¹).

## Figure 2 — Year and region robustness

Raw-to-adapter RMSE and ACC point improvements occur in all three audit years and all four reported regional masks. The smaller neural increment over log-bias is heterogeneous. Year rows use paired within-year circular block-13 percentile intervals. Regional rows use a paired two-stage bootstrap that first resamples the three audit years and then samples circular 13-start blocks within the selected years (2,000 draws). Intervals are descriptive, not p-values or population-level significance. Regional masks overlap and must not be summed.

## Figure 3 — Paired case gains and failures

The adapter improves RMSE in {wins.loc['rmse_improvement_mm_day','wins']:.0f}/{wins.loc['rmse_improvement_mm_day','total']:.0f} ({wins.loc['rmse_improvement_mm_day','win_fraction_pct']:.1f}%) initialization–lead cases and ACC in {wins.loc['acc_improvement','wins']:.0f}/{wins.loc['acc_improvement','total']:.0f} ({wins.loc['acc_improvement','win_fraction_pct']:.1f}%). Absolute bias improves in only {wins.loc['abs_bias_improvement_mm_day','wins']:.0f}/{wins.loc['abs_bias_improvement_mm_day','total']:.0f} ({wins.loc['abs_bias_improvement_mm_day','win_fraction_pct']:.1f}%). The 600 points are serially dependent and are displayed descriptively rather than as independent replicates.

## Figure 4 — Native-grid spatial footprint

Pooled local RMSE is lower over {area.loc['raw_fuxi','area_fraction_lower_rmse_pct']:.1f}% of static weighted area versus raw FuXi and {area.loc['log_bias','area_fraction_lower_rmse_pct']:.1f}% versus log-bias. Maps show native 1.5° cells without interpolation. Area fractions use the frozen static cell-area × IMD-support weights and common finite support; the prediction cube did not save the audit scorer's fractional weekly-coverage field. Across the 600 case-leads, the static weight sum is {weight.static_relative_excess_pct:.4f}% higher than the saved audit effective-area sum, with differences in {int(weight.case_leads_with_fractional_coverage_difference)}/{int(weight.total_case_leads)} case-leads. Local fields and area fractions are descriptive; no pixel-wise significance or multiplicity-adjusted inference is claimed.

## Figure 5 — Intensity and extremes stress test

The adapter improves ETS through the 10 mm day⁻¹ weekly-mean threshold, but at 20 mm day⁻¹ its ETS is {extreme_threshold.loc['selected_adapter','ets']:.3f} versus {extreme_threshold.loc['raw_fuxi','ets']:.3f} for raw FuXi. For verifying weekly-mean rainfall ≥20 mm day⁻¹, adapter RMSE is {extreme.loc['selected_adapter','rmse_mm_day']:.3f} versus {extreme.loc['raw_fuxi','rmse_mm_day']:.3f} mm day⁻¹, while adapter MAE is {extreme.loc['selected_adapter','mae_mm_day']:.3f} versus {extreme.loc['raw_fuxi','mae_mm_day']:.3f} and signed bias is {extreme.loc['selected_adapter','bias_mm_day']:.3f} mm day⁻¹. Threshold and stratum metrics use the same static cell-area × IMD-support weights noted for Figure 4, not the unavailable fractional weekly-coverage field. Aggregate error reduction therefore does not establish calibrated wet-tail prediction. This is a post-hoc weekly-mean cell–lead diagnostic, not a daily extreme-event analysis or a selection result.
"""


def build_readme(
    lead_summary: pd.DataFrame,
    effects: pd.DataFrame,
    weighting_diagnostic: pd.DataFrame,
    output_name: str,
) -> str:
    pooled = lead_summary.loc[lead_summary.lead_week.eq("ALL")].set_index("method")
    raw = pooled.loc["raw_fuxi"]
    adapter = pooled.loc["selected_adapter"]
    pooled_effect = effects.loc[
        effects.scope_type.eq("pooled") & effects.baseline.eq("raw_fuxi")
    ].set_index("metric")
    weight = weighting_diagnostic.iloc[0]
    return f"""# CCAI neural-adapter figure package

This directory contains exactly five conference-ready PNG/PDF figure pairs for the frozen FuXi-to-IMD anchored neural postprocessor.

## Scope

- Development generalization audit; not an untouched independent test.
- Train: 2002–2017; selection: 2018–2019; plotted audit: 100 JJAS starts in 2022–2024.
- No 2025 initialization is used here.
- W1 = init+0..6 through W6 = init+35..41.
- 171 native 1.5° IMD-supported cells; IMD is the verification reference.
- Keep separate from INDIA-S2S-BENCH, whose W1 begins at init+1.
- Figures 4–5 use saved static cell-area × IMD-support weights because fractional weekly coverage was not stored in the prediction cube. Their pooled weight is {weight.static_relative_excess_pct:.4f}% above the audit effective-area sum ({int(weight.case_leads_with_fractional_coverage_difference)}/{int(weight.total_case_leads)} case-leads differ).

## Main result

Pooled mean case-wise RMSE is {adapter.rmse_mm_day:.3f} versus {raw.rmse_mm_day:.3f} mm day⁻¹ ({pooled_effect.loc['rmse_skill_pct','effect']:.2f}% lower), and common-reference ACC is {adapter.acc:.3f} versus {raw.acc:.3f} (Δ {pooled_effect.loc['acc_delta','effect']:+.3f}). The adapter is best described as an RMSE/pattern-skill postprocessor: overall signed bias worsens from {raw.bias_mm_day:.3f} to {adapter.bias_mm_day:.3f} mm day⁻¹, and the ≥20 mm day⁻¹ weekly-mean cell–lead diagnostic remains weak. It is not a daily extreme-event analysis.

## Figures

1. `01_lead_skill_and_correction_decomposition` — W1–W6 RMSE, ACC, paired RMSE reduction, and signed bias.
2. `02_year_region_robustness` — year and regional-mask effects versus raw FuXi and log-bias, with descriptive paired block intervals.
3. `03_paired_case_gains_and_failures` — dependent case-lead effect distributions and win fractions.
4. `04_native_grid_spatial_footprint` — pooled local RMSE-reduction maps and lead-wise improved-area fractions.
5. `05_intensity_and_extremes_stress_test` — weekly-mean threshold ETS/frequency bias and truth-stratified RMSE/bias; not daily extreme-event verification.

See `CAPTIONS.md` for conference-safe wording and `MANIFEST.json` for hashes, provenance, definitions, and status. Numerical values behind every plot are in `tables/`.

## Regenerate

From `clean/studies/fuxi_imd_adapter_benchmark_v1`:

```bash
conda run -n weather_forecast python make_ccai_neural_adapter_figures.py --output results/{output_name}
```

The default refuses to overwrite an existing output. Use `--overwrite` only when deliberately replacing this generated package.
"""


def source_records(bundle: SourceBundle) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    paths = {
        "renderer": Path(__file__).resolve(),
        "audit_manifest": bundle.audit / "manifest.json",
        "audit_case_metrics": bundle.audit / "case_metrics.csv",
        "audit_summary_by_lead_region": bundle.audit / "summary_by_lead_region.csv",
        "audit_summary_by_region": bundle.audit / "summary_by_region.csv",
        "audit_summary_by_year_lead_region": (
            bundle.audit / "summary_by_year_lead_region.csv"
        ),
        "audit_generalization_guards": bundle.audit / "generalization_guards.json",
        "audit_predictions": bundle.audit / "predictions.zarr",
        "run_manifest": bundle.run / "manifest.json",
        "frozen_selection": bundle.run / "selection.json",
        "run_predictions_area_weights": bundle.run / "predictions.zarr",
        "india_boundary": bundle.boundary_path,
        "boundary_loader": BIAS_CORRECTION / "plot_physical_validation_results.py",
    }
    for name, path in paths.items():
        records[name] = {
            "path": _relative(path),
            "sha256": sha256_tree(path) if path.is_dir() else sha256_file(path),
            "hash_kind": "sha256_tree" if path.is_dir() else "sha256_file",
        }
    return records


def write_manifest(
    staging: Path,
    bundle: SourceBundle,
    lead_summary: pd.DataFrame,
    effects: pd.DataFrame,
    win_rates: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    intensity_metrics: pd.DataFrame,
    weighting_diagnostic: pd.DataFrame,
) -> Path:
    pooled = lead_summary.loc[lead_summary.lead_week.eq("ALL")].set_index("method")
    pooled_effect = effects.loc[
        effects.scope_type.eq("pooled") & effects.baseline.eq("raw_fuxi")
    ].set_index("metric")
    pooled_wins = win_rates.loc[win_rates.lead_week.eq("ALL")].set_index("metric")
    extreme_threshold = threshold_metrics.loc[
        threshold_metrics.threshold_mm_day.eq(20.0)
    ].set_index("method")
    extreme = intensity_metrics.loc[
        intensity_metrics.stratum.eq("extreme_ge20")
    ].set_index("method")
    weight = weighting_diagnostic.iloc[0]

    output_records: dict[str, str] = {}
    for path in sorted(
        candidate for candidate in staging.rglob("*") if candidate.is_file()
    ):
        if path.name == "MANIFEST.json":
            continue
        output_records[path.relative_to(staging).as_posix()] = sha256_file(path)
    if set(output_records) != set(EXPECTED_OUTPUT_FILES):
        missing = sorted(EXPECTED_OUTPUT_FILES.difference(output_records))
        extra = sorted(set(output_records).difference(EXPECTED_OUTPUT_FILES))
        raise FigureContractError(
            f"generated output record set changed; missing={missing}, extra={extra}"
        )
    manifest = {
        "schema_name": "fuxi_imd_ccai_neural_adapter_figures",
        "schema_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "scientific_status": (
            "2022-2024 development generalization audit; not untouched independent confirmation"
        ),
        "figure_count": 5,
        "figure_stems": list(FIGURE_STEMS),
        "model": {
            "label": "anchored neural postprocessor",
            "selected_model": "normal_climo_model",
            "parameters": 144_689,
            "seed_ensemble": [42, 43, 44],
            "anchor": "training-only 2002-2017 log-bias correction",
        },
        "evaluation_contract": {
            "target": "IMD weekly rainfall",
            "training_years": [2002, 2017],
            "selection_years": [2018, 2019],
            "audit_years": list(EXPECTED_YEARS),
            "audit_counts": {str(k): v for k, v in EXPECTED_YEAR_COUNTS.items()},
            "audit_starts": 100,
            "leads": list(EXPECTED_LEADS),
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "support_cells": 171,
            "grid": "native 1.5-degree; no interpolation",
            "acc_reference": "one common fixed training-only 2002-2017 IMD climatology",
            "independent_2025_claim": False,
        },
        "new_diagnostics": {
            "status": "post-hoc descriptive additions; not model-selection criteria",
            "threshold_time_aggregation": (
                "weekly-mean rainfall rates at cell-lead scale; not daily extreme-event verification"
            ),
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "block_length_starts": BOOTSTRAP_BLOCK_LENGTH,
                "pooled_lead_region_sampling": (
                    "paired two-stage: resample three audit-year slots, then circular "
                    "moving blocks within selected years; all leads retained"
                ),
                "year_row_sampling": (
                    "paired circular moving blocks within the named audit year; all leads retained"
                ),
                "interval": "percentile 95%",
                "interpretation": "descriptive sensitivity; not a p-value",
            },
            "thresholds_mm_day": list(THRESHOLDS_MM_DAY),
            "intensity_strata": [
                {
                    "key": key,
                    "label": label,
                    "lower_mm_day": lower,
                    "upper_mm_day": None if np.isinf(upper) else upper,
                }
                for key, label, lower, upper in INTENSITY_STRATA
            ],
            "spatial_inference": "descriptive; no pixel-wise significance claim",
            "posthoc_weighting": {
                "definition": weight.static_weight_definition,
                "audit_weight_definition": weight.audit_weight_definition,
                "reason": weight.reason_for_static_posthoc_weighting,
                "static_relative_excess_pct": float(weight.static_relative_excess_pct),
                "case_leads_with_difference": int(
                    weight.case_leads_with_fractional_coverage_difference
                ),
                "total_case_leads": int(weight.total_case_leads),
            },
        },
        "headline": {
            "raw_rmse_mm_day": float(pooled.loc["raw_fuxi", "rmse_mm_day"]),
            "adapter_rmse_mm_day": float(pooled.loc["selected_adapter", "rmse_mm_day"]),
            "adapter_rmse_skill_vs_raw_pct": float(
                pooled_effect.loc["rmse_skill_pct", "effect"]
            ),
            "adapter_rmse_skill_vs_raw_ci95": [
                float(pooled_effect.loc["rmse_skill_pct", "ci_lower_2p5"]),
                float(pooled_effect.loc["rmse_skill_pct", "ci_upper_97p5"]),
            ],
            "raw_acc": float(pooled.loc["raw_fuxi", "acc"]),
            "adapter_acc": float(pooled.loc["selected_adapter", "acc"]),
            "adapter_acc_delta_vs_raw": float(pooled_effect.loc["acc_delta", "effect"]),
            "adapter_acc_delta_vs_raw_ci95": [
                float(pooled_effect.loc["acc_delta", "ci_lower_2p5"]),
                float(pooled_effect.loc["acc_delta", "ci_upper_97p5"]),
            ],
            "raw_bias_mm_day": float(pooled.loc["raw_fuxi", "bias_mm_day"]),
            "adapter_bias_mm_day": float(pooled.loc["selected_adapter", "bias_mm_day"]),
            "rmse_case_lead_win_fraction_pct": float(
                pooled_wins.loc["rmse_improvement_mm_day", "win_fraction_pct"]
            ),
            "acc_case_lead_win_fraction_pct": float(
                pooled_wins.loc["acc_improvement", "win_fraction_pct"]
            ),
            "abs_bias_case_lead_win_fraction_pct": float(
                pooled_wins.loc["abs_bias_improvement_mm_day", "win_fraction_pct"]
            ),
            "threshold_20_ets_raw": float(extreme_threshold.loc["raw_fuxi", "ets"]),
            "threshold_20_ets_adapter": float(
                extreme_threshold.loc["selected_adapter", "ets"]
            ),
            "extreme_ge20_bias_adapter_mm_day": float(
                extreme.loc["selected_adapter", "bias_mm_day"]
            ),
        },
        "source_files": source_records(bundle),
        "output_files": output_records,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    path = staging / "MANIFEST.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def validate_generated_package(
    output: Path, *, verify_sources: bool = True
) -> Mapping[str, Any]:
    manifest_path = Path(output) / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FigureContractError("generated package lacks MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("figure_count") != 5:
        raise FigureContractError("generated manifest is not complete")
    recorded = manifest.get("output_files", {})
    if set(recorded) != set(EXPECTED_OUTPUT_FILES):
        raise FigureContractError("manifest output-file record set is not exact")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in Path(output).rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual_files != set(recorded):
        missing = sorted(set(recorded).difference(actual_files))
        extra = sorted(actual_files.difference(recorded))
        raise FigureContractError(
            f"generated package file set changed; missing={missing}, extra={extra}"
        )
    for relative, expected in recorded.items():
        path = Path(output) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise FigureContractError(f"generated output hash mismatch: {relative}")
    if verify_sources:
        sources = manifest.get("source_files")
        if not isinstance(sources, Mapping) or not sources:
            raise FigureContractError("generated manifest lacks source-file records")
        if set(sources) != EXPECTED_SOURCE_RECORDS:
            raise FigureContractError("generated source-file record set is not exact")
        for name, record in sources.items():
            if not isinstance(record, Mapping):
                raise FigureContractError(f"invalid source record: {name}")
            path = Path(str(record.get("path", "")))
            if not path.is_absolute():
                path = CLEAN_ROOT / path
            hash_kind = record.get("hash_kind")
            if hash_kind == "sha256_tree":
                actual = sha256_tree(path)
            elif hash_kind == "sha256_file":
                actual = sha256_file(path)
            else:
                raise FigureContractError(f"invalid hash kind for source: {name}")
            if actual != record.get("sha256"):
                raise FigureContractError(f"generated source hash mismatch: {name}")
    return manifest


def safe_output_target(
    requested_output: Path,
    bundle: SourceBundle,
    *,
    overwrite: bool,
) -> Path:
    """Resolve a narrowly scoped output and reject destructive overwrite targets."""

    requested = Path(requested_output).expanduser()
    if requested.is_symlink():
        raise FigureContractError("refusing a symlink as the package output target")
    output = requested.resolve()
    protected_exact = {
        Path("/").resolve(),
        Path.home().resolve(),
        CLEAN_ROOT.resolve(),
        HERE.resolve(),
        (HERE / "results").resolve(),
        BIAS_CORRECTION.resolve(),
    }
    if output in protected_exact:
        raise FigureContractError(
            f"refusing broad or protected output target: {output}"
        )

    critical_sources = (
        Path(__file__).resolve(),
        bundle.audit.resolve(),
        bundle.run.resolve(),
        bundle.boundary_path.resolve(),
    )
    for source in critical_sources:
        try:
            source.relative_to(output)
        except ValueError:
            pass
        else:
            raise FigureContractError(
                f"refusing output target that contains a source artifact: {output}"
            )
        if source.is_dir():
            try:
                output.relative_to(source)
            except ValueError:
                pass
            else:
                raise FigureContractError(
                    f"refusing output target nested inside a source artifact: {output}"
                )

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if not output.is_dir():
            raise FigureContractError("existing overwrite target is not a directory")
        sentinel = output / "MANIFEST.json"
        if not sentinel.is_file():
            raise FigureContractError(
                "refusing to overwrite a directory without this package's manifest"
            )
        try:
            payload = json.loads(sentinel.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise FigureContractError(
                "overwrite target has an invalid manifest"
            ) from exc
        if (
            payload.get("schema_name") != "fuxi_imd_ccai_neural_adapter_figures"
            or payload.get("schema_version") != 1
        ):
            raise FigureContractError(
                "refusing to overwrite a directory owned by another artifact schema"
            )
    return output


def generate_package(
    bundle: SourceBundle,
    output: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = safe_output_target(output, bundle, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        tables = staging / "tables"
        tables.mkdir(parents=True)
        initializations = ordered_initializations(bundle.cases)
        within_year_indices, year_slices = year_stratified_circular_indices(
            initializations
        )
        two_stage_indices = two_stage_circular_indices(initializations)
        effects = build_bootstrap_effects(
            bundle.cases,
            two_stage_indices,
            within_year_indices,
            year_slices,
        )
        lead_summary = build_lead_summary(bundle.cases)
        paired, win_rates = build_paired_case_effects(bundle.cases)
        threshold_metrics = build_threshold_metrics(
            bundle.predictions, bundle.area_weight_km2
        )
        intensity_metrics = build_intensity_metrics(
            bundle.predictions, bundle.area_weight_km2
        )
        weighting_diagnostic = build_weighting_diagnostic(
            bundle.cases, bundle.predictions, bundle.area_weight_km2
        )
        spatial, spatial_summary = build_spatial_diagnostics(
            bundle.predictions, bundle.area_weight_km2
        )

        lead_summary.to_csv(tables / "lead_summary.csv", index=False)
        effects.to_csv(tables / "paired_block_bootstrap_effects.csv", index=False)
        paired.to_csv(tables / "paired_case_effects.csv", index=False)
        win_rates.to_csv(tables / "case_win_rates.csv", index=False)
        threshold_metrics.to_csv(tables / "threshold_metrics.csv", index=False)
        intensity_metrics.to_csv(tables / "intensity_strata_metrics.csv", index=False)
        weighting_diagnostic.to_csv(tables / "weighting_diagnostic.csv", index=False)
        spatial_summary.to_csv(tables / "spatial_improvement_summary.csv", index=False)
        spatial.to_netcdf(tables / "spatial_diagnostics.nc")
        np.save(
            tables / "bootstrap_indices_two_stage.npy",
            two_stage_indices,
            allow_pickle=False,
        )
        np.save(
            tables / "bootstrap_indices_within_year.npy",
            within_year_indices,
            allow_pickle=False,
        )

        configure_style()
        generated: list[Path] = []
        generated.extend(plot_lead_scorecard(lead_summary, effects, staging))
        generated.extend(plot_robustness(effects, staging))
        generated.extend(plot_paired_cases(paired, win_rates, staging))
        generated.extend(
            plot_spatial_footprint(
                spatial,
                spatial_summary,
                weighting_diagnostic,
                bundle.boundary_segments,
                staging,
            )
        )
        generated.extend(
            plot_intensity_stress_test(
                threshold_metrics,
                intensity_metrics,
                weighting_diagnostic,
                staging,
            )
        )
        if len(generated) != 10:
            raise FigureContractError("expected exactly five PNG/PDF figure pairs")

        captions = build_captions(
            lead_summary,
            effects,
            win_rates,
            spatial_summary,
            threshold_metrics,
            intensity_metrics,
            weighting_diagnostic,
        )
        (staging / "CAPTIONS.md").write_text(captions, encoding="utf-8")
        (staging / "README.md").write_text(
            build_readme(
                lead_summary,
                effects,
                weighting_diagnostic,
                output.name,
            ),
            encoding="utf-8",
        )
        write_manifest(
            staging,
            bundle,
            lead_summary,
            effects,
            win_rates,
            spatial_summary,
            threshold_metrics,
            intensity_metrics,
            weighting_diagnostic,
        )
        validate_generated_package(staging)

        backup: Path | None = None
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            safe_output_target(output, bundle, overwrite=True)
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent)
            )
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
            validate_generated_package(output)
        except Exception:
            if output.exists():
                shutil.rmtree(output, ignore_errors=True)
            if backup is not None and backup.exists():
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--india-boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the explicitly resolved output directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = load_and_verify_sources(args.audit, args.run, args.india_boundary)
    output = generate_package(bundle, args.output, overwrite=args.overwrite)
    manifest = validate_generated_package(output)
    print(f"Generated {manifest['figure_count']} figure pairs in {output}")
    print("Scope: 2022-2024 development audit; not untouched independent confirmation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
