#!/usr/bin/env python3
"""Plot archived v3 validation spatial ACC in the four IMD rainfall regions.

This script only reads the already evaluated 2023 development metrics.  It does
not access 2024 predictions or alter training/evaluation outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = Path(
    "/storage/raj.ayush/neural_adapter_data/validation_results/"
    "fuxi_imerg_late_acc_v3_hybrid__20260807T010828Z/metrics/"
    "summary_by_week_region.csv"
)

REGIONS = (
    ("northwest_india", "Northwest India"),
    ("central_india", "Central India"),
    ("south_peninsula", "South Peninsula"),
    ("east_northeast_india", "East & Northeast India"),
)
MODELS = (
    ("raw_fuxi", "Raw FuXi", "#64748B", "--", "o"),
    ("log_bias_correction", "Log-bias correction", "#2563EB", "-", "D"),
    ("v2_residual_unet", "v2 spatial U-Net", "#D97706", ":", "^"),
    ("late_lead_temporal_unet", "v3 hybrid adapter", "#15803D", "-", "o"),
)


def main() -> None:
    if not METRICS.is_file():
        raise FileNotFoundError(f"Archived development metrics unavailable: {METRICS}")
    frame = pd.read_csv(METRICS)
    needed = {"predictor", "region", "lead", "acc_mean", "case_count"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"metrics table is missing columns: {sorted(missing)}")
    expected = {name for name, *_ in MODELS}
    absent = expected - set(frame["predictor"])
    if absent:
        raise ValueError(f"metrics table is missing predictors: {sorted(absent)}")

    selected = frame.loc[
        frame["region"].isin([name for name, _ in REGIONS])
        & frame["predictor"].isin(expected)
    ].copy()
    if selected.duplicated(["region", "predictor", "lead"]).any():
        raise ValueError("expected exactly one summary value per region/model/lead")
    if selected["acc_mean"].isna().any():
        raise ValueError("a requested predictor has undefined regional ACC")
    selected = selected.sort_values(["region", "predictor", "lead"])
    selected.to_csv(ROOT / "imd_homogeneous_region_skill_2023_validation.csv", index=False)

    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.3), sharex=True, sharey=True)
    figure.patch.set_facecolor("white")
    legend_handles = []
    for index, ((region, label), axis) in enumerate(zip(REGIONS, axes.ravel())):
        regional = selected.loc[selected["region"].eq(region)]
        axis.axvspan(0.72, 2.52, color="#F1F5F9", zorder=0)
        axis.axvspan(2.52, 6.28, color="#EAF7EF", zorder=0)
        axis.axvline(2.5, color="#86B99A", linestyle="--", linewidth=1.25)
        for predictor, name, color, linestyle, marker in MODELS:
            values = regional.loc[regional["predictor"].eq(predictor)].sort_values("lead")
            handle, = axis.plot(
                values["lead"], values["acc_mean"], label=name, color=color,
                linestyle=linestyle, marker=marker, linewidth=2.5, markersize=7,
                markerfacecolor="white" if predictor == "raw_fuxi" else color,
                markeredgewidth=2,
            )
            if index == 0:
                legend_handles.append(handle)
        axis.set_title(label, fontsize=14, fontweight="bold", loc="left", color="#16324F")
        axis.set_xlim(0.8, 6.2)
        axis.set_ylim(-0.12, 0.78)
        axis.set_xticks(range(1, 7))
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.8)
        axis.text(1.05, 0.735, "W1–2: anchored", fontsize=8.7, color="#64748B")
        axis.text(3.10, 0.735, "W3–6: learned correction", fontsize=8.7, color="#15803D")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean spatial ACC", fontsize=12)
    for axis in axes[1, :]:
        axis.set_xlabel("Lead week", fontsize=12)

    figure.suptitle("FuXi–IMERG precipitation skill by IMD homogeneous region", y=0.984,
                    fontsize=22, fontweight="bold", color="#16324F")
    figure.text(0.5, 0.943,
                "PLOTTED: 2023 development validation · 93 twice-weekly initializations · six non-overlapping 7-day means",
                ha="center", fontsize=11.1, color="#991B1B", fontweight="bold")
    figure.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.912),
                  frameon=False, fontsize=11.5, handlelength=2.7, columnspacing=1.8)
    figure.text(0.5, 0.018,
                "Area-weighted, case-wise spatial anomaly correlation against IMERG using fixed 2001–2019 climatology and frozen fractional IMD masks. "
                "2024 is reserved and not shown; this is development evidence, not a test result.",
                ha="center", fontsize=9.2, color="#52677D")
    figure.tight_layout(rect=(0.035, 0.070, 0.985, 0.835))
    output = ROOT / "imd_homogeneous_region_skill_2023_validation"
    figure.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", metadata={
        "Title": "FuXi-IMERG regional skill by IMD homogeneous region",
        "Author": "FuXi-IMERG neural adapter project",
        "Subject": "2023 development-validation regional spatial ACC",
        "CreationDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 7, tzinfo=timezone.utc),
    })
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
