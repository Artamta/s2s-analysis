# Code review notes — JFM-2026 India S2S benchmark (`paper/code/`)

This file is for the code review: it records the **dataset inventory**, the
**on-disk verification** of every assumption the pipeline makes (units, variable
names, accumulation semantics), and the small **code-quality** notes. Nothing
here changes the numbers — it documents why the numbers are what they are.

The reproducibility package is `paper/code/` (`compute/01–07` → skill tables in
`analysis-code/analysis/`; `figures/make_*.py` → `paper/figs/`). Run order and
file→figure map are in [README.md](README.md); `run_all.sh` runs the whole chain.

---

## 1. Dataset inventory (master paths)

| Dataset | Type | Absolute Master Path | Resolution | Temporal Horizon | Members | Core Variables & Native Units | Status |
|---|---|---|---|---|---|---|---|
| **ERA5** | Ground Truth | `/storage/raj.ayush/s2s-forecast-data/era5/data/` | 1.5° x 1.5° | 135 continuous days | None | `t2m` (K), `tp` (meters), `z` (m²/s²) | 🟢 100% Intact |
| **ECMWF** | Operational | `/storage/raj.ayush/s2s-forecast-data/ecmwf/data/` | 1.5° x 1.5° | 46 Days (24h steps) | 1 CF, 100 PF | `mx2t6` (K), `tp` (kg/m²), `gh` (gpm) | 🟢 100% Intact |
| **NCEP** | Operational | `/storage/raj.ayush/s2s-forecast-data/ncep/data/` | 1.5° x 1.5° | 44 Days (24h steps) | 1 CF, 15 PF | `mx2t6` (K), `tp` (kg/m²), `gh` (gpm) | 🟢 100% Intact |
| **FuXi** | AI Model | `/storage/raj.ayush/s2s-forecast-data/fuxi/output/` | 1.5° x 1.5° | 42 Days (Daily steps) | 11 Ensemble | `t2m` (K), `tp` (mm), `z` (m²/s²) | 🟢 100% Intact |
| **SPIRE** | AI Model | `/storage/raj.ayush/s2s-forecast-data/spire/*.zarr` | 0.5° x 0.5° | 46 Days (Daily steps) | Ens. Mean / Std | `temp` (K), `precip` (kg/m²), `gh` (m) | 🟢 100% Intact |

All paths confirmed present on disk. Forecast files the code opens:
`{ecmwf,ncep}/data/sfc_pf_<YYYYMMDD>.grib` (tp, mx2t6, mn2t6) and
`pl_pf_<YYYYMMDD>.grib` (gh); `fuxi/output/<YYYYMMDD>/member/<mm>/<dd>.nc`
(members `00`–`10`, days `01`–`42`); `spire/spire_hindcast_jfm.zarr` (groups
`mean_stddev` and `percentiles`). 13 weekly inits 2026-01-01 … 2026-03-26.

---

## 2. On-disk verification (one init, 2026-01-01, India box)

Every unit/shape/semantic the pipeline relies on was checked against the raw
files. India-domain (5–38 °N, 65–100 °E) means, ensemble-mean where applicable:

| System | Dims verified | Units (native → used) | Domain-mean sanity check |
|---|---|---|---|
| **ERA5 tp** | `time=135` | `m` → ×1000 = mm | surface grib = **6-h window** ≈ 0.075 mm (see §3.3) |
| **ERA5 z** | — | `m² s⁻²` → ÷9.80665 = m | 56376.8 → **5748.8 m** ✓ (Z500) |
| **ECMWF** | `number=100, step=46` | tp `kg m⁻²` = mm; gh `gpm` | tp **cumulative**, steps[0,1,6,13,41] = [1.50, 2.75, 9.40, 18.74, 39.86]; gh@500 = **5766 m** |
| **NCEP** | `number=15, step=44` | tp `kg m⁻²` = mm; gh `gpm` | tp **cumulative**, [1.48, 2.87, 7.69, 15.34, 31.76]; gh@500 = **5752.8 m** |
| **FuXi** | `channel=76, lat=121, lon=240` | see §3.1 | `channel[5]` = **`z500`** = 5770 m ✓; t2m 282.85 K; **tp raw 0.059 → ×24 = 1.42 mm/day** |
| **SPIRE** | `reference_time=13, step=46, isobar=4` | precip already mm/day | precip wk1 ≈ **1.025 mm/day** (no ×24); isobar has 50000 Pa ✓ |

**Steps are 24-hourly** for ECMWF (46) and NCEP (44), matching the table.
ECMWF/NCEP `tp` is **cumulative from t₀** — strictly increasing with step —
which validates the two places the code differences it:
* weekly: `weekly_mean_cumulative` = `(cum[de-1] − cum[ds-2]) / days`
* daily : `cum[i] − cum[i-1]`

---

## 3. Code ↔ data reconciliation (the two things that aren't 1:1 with the table)

These are **not bugs** — the code is correct — but the table labels are nominal,
so they are spelled out here so a reviewer is not surprised.

### 3.1 FuXi `tp` is a rate, not a daily total → ×24 is required
The table lists FuXi `tp` as "mm", but the stored values are ~17× smaller than a
physical daily total (raw India mean **0.059** vs ERA5/SPIRE ≈ 1 mm/day). The
field is a mean rate; `compute/04,06,07` multiply FuXi `tp` by **24** to get
mm/day (`0.059 → 1.42`), which is the right magnitude. PCC is scale-invariant so
this does not change FuXi's correlation, only its RMSE/Bias and the MME that
averages it. FuXi `channel[5]` is verified to be **`z500`**, so the hardcoded
`isel(channel=5)/g` for geopotential height is correct. No other system is scaled.

### 3.2 SPIRE store variable names ≠ the table's friendly names
The table's `temp / precip / gh` are friendly labels. The literal zarr names the
code uses (verified present) are:
* `mean_stddev` group: `air_temperature(+_stddev)`, `precipitation_amount(+_stddev)`,
  `geopotential_height_at_isobaric_levels(+_stddev)` with `isobar=50000.0` Pa for Z500.
* `percentiles` group: `air_temperature_pctl`, `precipitation_amount_pctl`,
  percentiles `[1,5,10,20,50,80,90,95,99]`.

SPIRE precipitation is already mm/day (≈1.0 over India) and is **not** ×24-scaled.

### 3.3 ERA5 `era5_surface.grib` tp is a 6-h window, not the verification truth
The on-disk `era5_surface.grib` tp is a 6-hour accumulation (India mean ≈0.075 mm),
**not** a 24-h daily total. It is used only as a constant offset (`clim6`) that is
subtracted in `compute/03` and added back in `compute/04,07`, so it cancels in the
round-trip. The actual precipitation **ground truth is `era5_daily_tp.nc`** — a
true 24-h daily total rebuilt from ARCO-ERA5 hourly by `compute/01` — and all
final TP skill is scored against that (`skill_tp_corrected.csv`).

---

## 4. Methodological choices to state in the paper (correct, but reviewers will ask)

* **In-sample, period-mean climatology.** Anomalies and the CRPSS climatological
  reference use a single Jan–May per-grid-point mean/std (no day-of-year cycle).
  Defensible for a 13-init study period, but for T2M (strong Jan→May warming) it
  makes the climatology baseline weak, which can flatter CRPSS. Keep this caveat
  explicit in the text.
* **ECMWF/NCEP `t2m` is a proxy:** `(mx2t6 + mn2t6)/2` per 6-h step (no direct
  instantaneous t2m archived). This gives the AI models a small cold bias, which
  is why the T2M figures additionally report the **bias-corrected (centered) RMSE**.
* **Gaussian-forecast probabilistic framework.** Each system is treated as
  `N(ensemble-mean, ensemble-spread)` so CRPS/SSR/reliability are fair across very
  different member counts (SPIRE mean+std, FuXi 11, ECMWF 100, NCEP 15), computed
  at daily level where the spreads are comparable.

---

## 5. Code-quality cleanups done for this review

* `figures/make_domain_and_bias.py` — removed three figure functions (`fig2/3/4`)
  and a helper that were defined but never called (the 3-variable fig02–04 are
  owned by `make_skill_horizon.py`); docstring corrected.
* `figures/make_regional_bars.py` — removed a `set_title` that was immediately
  overwritten on the next line.
* `run_all.sh` — fixed stale `manuscript_code/` / relative paths left over from
  the rename to `paper/code/`.

## 6. Tests

`paper/code/tests/` — 82 tests (`python -m pytest paper/code/tests -v`): metric
maths (known-answer ACC/RMSE/bias), land mask, skill-table integrity, headline
results regression (SPIRE best; FuXi unit fix; calibration), figure manifest.
Tests skip gracefully if a precomputed table is absent.
