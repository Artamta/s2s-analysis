"""
case1_analysis.py — Case Study 1: The Late-February 2026 Dry Ridge & Severe Warm Anomaly
=========================================================================================

Target valid week : 2026-02-12 .. 2026-02-18  (India domain, land only)
Lead configurations (same valid week, two different forecast ages):
    * Week-1 lead : init 2026-02-12, forecast days 1-7
    * Week-2 lead : init 2026-02-05, forecast days 8-14

Models : Spire, FuXi-S2S, ECMWF, NCEP        Reference : ERA5

Deliverables
    Fig 1   ERA5 observed ground-truth panels (Z500 / TP / T2M anomalies)
    Fig 2a  Multi-model 5x3 spatial matrix, Week-1 lead
    Fig 2b  Multi-model 5x3 spatial matrix, Week-2 lead
    Tables  Cosine-weighted PCC / RMSE / Bias over India (IMD-union) land points,
            plus a per-IMD-region breakdown (md + tex + csv)

Data strategy
    Z500 + TP anomalies are read from the pre-aggregated paper product
    `weekly_anom_fields.nc` (FuXi precip already x24-corrected to mm/day).
    T2M 2-D fields are not pre-stored, so they are computed here from the raw
    model archives for the two target inits only (fast) and cached to NetCDF.

Engineering rules honoured
    * FuXi precipitation hourly-rate -> daily-total x24 (already applied upstream;
      asserted here against the other models).
    * ERA5 TP from true 24-h daily totals (the product source).
    * Verification mask = India political domain (union of IMD homogeneous-region
      masks); excludes neighbouring land and the Arabian Sea / Bay of Bengal.
    * Area weighting = cos(lat).
    * Graceful lat/latitude, lon/longitude handling.

All paths live in CONFIG at the top — bind your local archive there.
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Reuse the project's reproducible offline land mask
sys.path.append("/home/raj.ayush/s2s/s2s_anlysis/paper/code")
from utils.verification_extra import get_land_mask  # noqa: E402

# ============================================================================
# CONFIG  — bind local archive paths here
# ============================================================================
CONFIG = {
    # Pre-aggregated Z500 + TP weekly anomaly fields (paper product)
    "weekly_fields": "/home/raj.ayush/s2s/s2s_anlysis/paper/results/weekly_anom_fields.nc",
    # ERA5 day-of-year climatology (t2m / tp / z500), 0.25 deg global
    "climatology": "/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc",
    # ERA5 daily-mean T2M truth (true 00-00 daily means)
    "era5_daily_t2m": "/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_t2m.nc",
    # IMD homogeneous-region masks (union = India political mask), 1.5 deg grid
    "imd_masks": "/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc",
    # Raw model archive root (for fresh T2M field computation)
    "data_root": "/storage/raj.ayush/s2s-forecast-data",
    # Outputs
    "outdir": "/home/raj.ayush/s2s/s2s_anlysis/paper/case_study/case1_dry_ridge",
}

G = 9.80665  # m s-2

# ---- Target / lead definitions --------------------------------------------
TARGET_WEEK = ("2026-02-12", "2026-02-18")          # valid window (inclusive)
LEADS = {
    "Week-1 lead": {"init": "2026-02-12", "week": "Week 1", "days": (1, 7)},
    "Week-2 lead": {"init": "2026-02-05", "week": "Week 2", "days": (8, 14)},
}

# India domain bounding box
DOMAIN = dict(lat=(5.0, 38.0), lon=(65.0, 100.0))
EXTENT = [DOMAIN["lon"][0], DOMAIN["lon"][1], DOMAIN["lat"][0], DOMAIN["lat"][1]]
PROJ = ccrs.PlateCarree()

MODELS = ["SPIRE", "FuXi", "ECMWF", "NCEP"]
ROWS = ["ERA5", "SPIRE", "FuXi", "ECMWF", "NCEP"]
RLAB = {"ERA5": "ERA5 (Obs)", "SPIRE": "Spire", "FuXi": "FuXi-S2S",
        "ECMWF": "ECMWF", "NCEP": "NCEP"}

# Variable display metadata
# IMD homogeneous regions (for the regional skill breakdown)
IMD_REGIONS = ["northwest_india", "central_india",
               "south_peninsula", "east_northeast_india"]
IMD_RLAB = {"northwest_india": "Northwest", "central_india": "Central",
            "south_peninsula": "S. Peninsula", "east_northeast_india": "East/NE"}

VARS = ["Z500", "TP", "T2M"]
VMETA = {
    "Z500": dict(cmap="RdBu_r", label="Z500 anomaly (gpm)",
                 title="Z500 height anomaly"),
    "TP":   dict(cmap="BrBG", label="TP anomaly (mm day$^{-1}$)",
                 title="Total precipitation anomaly"),
    "T2M":  dict(cmap="bwr", label="T2M anomaly ($^{\\circ}$C)",
                 title="2 m temperature anomaly"),
}

OPEN_GRIB = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})


# ============================================================================
# Grid / coordinate helpers
# ============================================================================
def standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude -> lat/lon so the rest of the code is uniform."""
    ren = {}
    for cand in ("latitude", "Latitude", "LAT"):
        if cand in da.coords or cand in da.dims:
            ren[cand] = "lat"
    for cand in ("longitude", "Longitude", "LON"):
        if cand in da.coords or cand in da.dims:
            ren[cand] = "lon"
    if ren:
        da = da.rename(ren)
    return da


class Grid:
    """The 1.5-degree India case grid, taken from the weekly_anom_fields product.

    The verification mask is the India political domain (union of the IMD
    homogeneous-region masks) — NOT generic land — so neighbouring countries'
    land (Pakistan, Afghanistan, Nepal, Bangladesh) and the surrounding oceans
    (Arabian Sea / Bay of Bengal) are both excluded. Per-region masks are kept
    for the regional skill breakdown.
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat = np.asarray(lat)
        self.lon = np.asarray(lon)
        # Cosine-of-latitude area weights, broadcast to (lat, lon)
        w1d = np.cos(np.deg2rad(self.lat))
        self.weights = np.repeat(w1d[:, None], self.lon.size, axis=1)

        # India political mask = union of IMD homogeneous regions
        self.regions = {}
        try:
            mds = xr.open_dataset(CONFIG["imd_masks"])
            for r in IMD_REGIONS:
                self.regions[r] = mds[r].values.astype(bool)
            self.land = np.zeros((self.lat.size, self.lon.size), dtype=bool)
            for r in IMD_REGIONS:
                self.land |= self.regions[r]
        except Exception as e:
            # Graceful fallback to the offline global land mask
            print(f"  WARN: IMD mask unavailable ({e}); falling back to land mask",
                  flush=True)
            self.land = get_land_mask(self.lat, self.lon).values.astype(bool)
            self.regions = {"all_india": self.land}

    def regrid(self, da: xr.DataArray) -> np.ndarray:
        """Bilinear-interpolate any DataArray onto the case grid -> 2-D array."""
        da = standardize_coords(da).squeeze()
        out = da.interp(lat=self.lat, lon=self.lon, method="linear")
        return out.values

    def apply_land(self, arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None
        out = np.asarray(arr, dtype=float).copy()
        out[~self.land] = np.nan
        return out


# ============================================================================
# Climatology  (ERA5 day-of-year), interpolated to the case grid
# ============================================================================
def target_doys() -> list[int]:
    dates = pd.date_range(TARGET_WEEK[0], TARGET_WEEK[1])
    return [int(d.dayofyear) for d in dates]


def load_climatology(grid: Grid) -> dict[str, np.ndarray]:
    """Return {'z500','tp','t2m'} climatology on the case grid (native units)."""
    doys = target_doys()
    clim = xr.open_dataset(CONFIG["climatology"])
    out = {}
    # native climo units: z500 in m^2 s^-2 -> /G to gpm; tp in m/day -> *1000 mm/day; t2m in K
    scale = {"z500": 1.0 / G, "tp": 1000.0, "t2m": 1.0}
    for v in ("z500", "tp", "t2m"):
        field = clim[v].sel(dayofyear=doys).mean("dayofyear") * scale[v]
        out[v] = grid.regrid(field)
    return out


# ============================================================================
# Loader: Z500 + TP anomalies from the pre-aggregated product
# ============================================================================
def load_z_tp_anomalies(grid: Grid) -> dict:
    """
    Read Z500 & TP weekly-mean anomaly fields for both leads from
    weekly_anom_fields.nc. Returns nested dict:
        fields[lead][var][model] -> 2-D anomaly array (land-masked)
    where model in {ERA5, SPIRE, FuXi, ECMWF, NCEP}.
    """
    ds = xr.open_dataset(CONFIG["weekly_fields"])
    inits = [str(x)[:10] for x in ds.init.values]

    # FuXi x24 precip sanity check: FuXi TP anomaly magnitude must be comparable
    # to the other models (not ~24x smaller), i.e. it is stored in mm/day.
    _fx = ds["tp_fcst"].sel(model="FuXi").values
    _ot = ds["tp_fcst"].sel(model="ECMWF").values
    r = np.nanstd(_fx) / (np.nanstd(_ot) + 1e-9)
    assert 0.2 < r < 5.0, (
        f"FuXi TP scale looks wrong (std ratio {r:.2f}); expected daily totals "
        "(x24 correction). Re-check upstream unit handling.")

    fields = {}
    for lead, cfg in LEADS.items():
        ii = inits.index(cfg["init"])
        wk = cfg["week"]
        fields[lead] = {"Z500": {}, "TP": {}}
        # ERA5 reference
        fields[lead]["Z500"]["ERA5"] = grid.apply_land(
            ds["z_obs"].isel(init=ii).sel(week=wk).values)
        fields[lead]["TP"]["ERA5"] = grid.apply_land(
            ds["tp_obs"].isel(init=ii).sel(week=wk).values)
        for m in MODELS:
            z = ds["z_fcst"].sel(model=m).isel(init=ii).sel(week=wk).values
            t = ds["tp_fcst"].sel(model=m).isel(init=ii).sel(week=wk).values
            fields[lead]["Z500"][m] = None if np.all(np.isnan(z)) else grid.apply_land(z)
            fields[lead]["TP"][m] = None if np.all(np.isnan(t)) else grid.apply_land(t)
    return fields


# ============================================================================
# Loader: T2M 2-D fields, computed fresh from raw archives (2 inits only)
# ============================================================================
def _spire_t2m(init: str, d0: int, d1: int) -> xr.DataArray | None:
    try:
        sp = xr.open_zarr(f"{CONFIG['data_root']}/spire/spire_hindcast_jfm.zarr",
                          group="mean_stddev").sel(reference_time=init)["air_temperature"]
        return sp.isel(step=slice(d0 - 1, d1)).mean("step")
    except Exception as e:
        print(f"  SPIRE T2M fail {init}: {e}", flush=True)
        return None


def _fuxi_t2m(init_str: str, d0: int, d1: int) -> xr.DataArray | None:
    days = []
    for day in range(d0, d1 + 1):
        mems = []
        for mem in range(11):
            p = f"{CONFIG['data_root']}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
            if not os.path.exists(p):
                continue
            da = xr.open_dataset(p)["__xarray_dataarray_variable__"].sel(channel="t2m")
            for dim in list(da.dims):
                if dim not in ("lat", "lon", "latitude", "longitude"):
                    da = da.mean(dim)
            mems.append(da)
        if mems:
            days.append(xr.concat(mems, "m").mean("m"))
    if not days:
        return None
    return xr.concat(days, "t").mean("t")


def _op_t2m(model: str, init_str: str, d0: int, d1: int) -> xr.DataArray | None:
    """ECMWF / NCEP: daily T2M proxy = (mx2t6 + mn2t6)/2, PF-mean."""
    base = f"{CONFIG['data_root']}/{model}/data/sfc_pf_{init_str}.grib"
    try:
        mx = xr.open_dataset(base, filter_by_keys={"shortName": "mx2t6"}, **OPEN_GRIB)["mx2t6"]
        mn = xr.open_dataset(base, filter_by_keys={"shortName": "mn2t6"}, **OPEN_GRIB)["mn2t6"]
        t = (mx + mn) / 2.0
        if "number" in t.dims:
            t = t.mean("number")
        if t.sizes.get("step", 0) < d1:
            return None
        return t.isel(step=slice(d0 - 1, d1)).mean("step")
    except Exception as e:
        print(f"  {model} T2M fail {init_str}: {e}", flush=True)
        return None


def load_t2m_anomalies(grid: Grid, clim_t2m: np.ndarray) -> dict:
    """
    Compute T2M weekly-mean anomaly fields for both leads. Caches to NetCDF so
    repeat runs are instant. Returns fields[lead]['T2M'][model] -> 2-D anom (deg C).

    The cache stores the *unmasked* full-grid anomaly; the verification mask is
    applied at load time so changing the mask never requires a recompute.
    """
    cache = os.path.join(CONFIG["outdir"], "t2m_fields_cache.nc")
    if os.path.exists(cache):
        print(f"Loading cached T2M fields: {cache}", flush=True)
        cds = xr.open_dataset(cache)
        out = {}
        for lead in LEADS:
            key = lead.replace(" ", "_").replace("-", "")
            out[lead] = {"T2M": {}}
            for m in ROWS:
                v = f"{key}__{m}"
                out[lead]["T2M"][m] = grid.apply_land(cds[v].values) if v in cds else None
        return out

    print("Computing T2M fields from raw archives (2 inits) ...", flush=True)
    dailyT = xr.open_dataset(CONFIG["era5_daily_t2m"])["t2m"]   # Kelvin

    out, cache_vars = {}, {}
    for lead, cfg in LEADS.items():
        init = cfg["init"]
        init_str = pd.to_datetime(init).strftime("%Y%m%d")
        d0, d1 = cfg["days"]
        key = lead.replace(" ", "_").replace("-", "")
        out[lead] = {"T2M": {}}

        # ERA5 truth absolute (Kelvin) for the valid week, then anomaly (deg C == K diff)
        obs_abs = grid.regrid(dailyT.sel(time=slice(*TARGET_WEEK)).mean("time"))
        obs_anom = obs_abs - clim_t2m                     # unmasked, full grid
        out[lead]["T2M"]["ERA5"] = grid.apply_land(obs_anom)
        cache_vars[f"{key}__ERA5"] = obs_anom

        raw = {
            "SPIRE": _spire_t2m(init, d0, d1),
            "FuXi": _fuxi_t2m(init_str, d0, d1),
            "ECMWF": _op_t2m("ecmwf", init_str, d0, d1),
            "NCEP": _op_t2m("ncep", init_str, d0, d1),
        }
        for m in MODELS:
            if raw[m] is None:
                out[lead]["T2M"][m] = None
            else:
                anom = grid.regrid(raw[m]) - clim_t2m     # unmasked, full grid
                out[lead]["T2M"][m] = grid.apply_land(anom)
                cache_vars[f"{key}__{m}"] = anom

    # Cache
    coords = {"lat": grid.lat, "lon": grid.lon}
    cds = xr.Dataset(
        {k: (("lat", "lon"), v) for k, v in cache_vars.items()}, coords=coords)
    cds.to_netcdf(cache)
    print(f"  cached -> {cache}", flush=True)
    return out


# ============================================================================
# Spatial skill metrics  (cosine-weighted, over India land points)
# ============================================================================
def spatial_metrics(fcst: np.ndarray, obs: np.ndarray, weights: np.ndarray,
                    region_mask: np.ndarray | None = None) -> dict:
    """Cosine-weighted PCC (centered Pearson), RMSE and mean bias of anomalies.

    If `region_mask` (2-D bool) is given, only those grid cells are scored.
    """
    if fcst is None or obs is None:
        return dict(pcc=np.nan, rmse=np.nan, bias=np.nan, n=0)
    f = np.asarray(fcst, float).ravel()
    o = np.asarray(obs, float).ravel()
    w = np.asarray(weights, float).ravel()
    m = np.isfinite(f) & np.isfinite(o) & np.isfinite(w)
    if region_mask is not None:
        m &= np.asarray(region_mask, bool).ravel()
    f, o, w = f[m], o[m], w[m]
    if f.size < 3 or w.sum() == 0:
        return dict(pcc=np.nan, rmse=np.nan, bias=np.nan, n=int(f.size))
    wn = w / w.sum()
    fbar = np.sum(wn * f)
    obar = np.sum(wn * o)
    fa, oa = f - fbar, o - obar
    cov = np.sum(wn * fa * oa)
    vf = np.sum(wn * fa * fa)
    vo = np.sum(wn * oa * oa)
    pcc = cov / np.sqrt(vf * vo) if vf > 0 and vo > 0 else np.nan
    rmse = np.sqrt(np.sum(wn * (f - o) ** 2))
    bias = np.sum(wn * (f - o))
    return dict(pcc=float(pcc), rmse=float(rmse), bias=float(bias), n=int(f.size))


def build_metrics_table(all_fields: dict, grid: Grid) -> pd.DataFrame:
    rows = []
    for lead in LEADS:
        for var in VARS:
            obs = all_fields[lead][var]["ERA5"]
            for m in MODELS:
                fc = all_fields[lead][var].get(m)
                sc = spatial_metrics(fc, obs, grid.weights)
                rows.append(dict(Model=RLAB[m], Variable=var, Lead=lead,
                                 PCC=sc["pcc"], RMSE=sc["rmse"], Bias=sc["bias"],
                                 N=sc["n"]))
    return pd.DataFrame(rows)


def build_regional_table(all_fields: dict, grid: Grid) -> pd.DataFrame:
    """Per-IMD-region cosine-weighted skill. PCC over small regions is noisy."""
    rows = []
    for lead in LEADS:
        for var in VARS:
            obs = all_fields[lead][var]["ERA5"]
            for region in IMD_REGIONS:
                rmask = grid.regions.get(region)
                if rmask is None:
                    continue
                for m in MODELS:
                    fc = all_fields[lead][var].get(m)
                    sc = spatial_metrics(fc, obs, grid.weights, region_mask=rmask)
                    rows.append(dict(Region=IMD_RLAB[region], Model=RLAB[m],
                                     Variable=var, Lead=lead,
                                     PCC=sc["pcc"], RMSE=sc["rmse"],
                                     Bias=sc["bias"], N=sc["n"]))
    return pd.DataFrame(rows)


def export_regional_tables(df: pd.DataFrame, outdir: str):
    csv_path = os.path.join(outdir, "case1_metrics_regional.csv")
    df.to_csv(csv_path, index=False)

    md = ["# Case Study 1 — Per-IMD-region spatial skill (cosine-weighted, anomalies)\n",
          "Valid week 2026-02-12 .. 2026-02-18. Units: Z500 gpm, TP mm/day, T2M deg C.\n",
          "\n> **Note.** This was a Northwest/Central-India event, so those two regions "
          "are the diagnostic ones. PCC over the smaller S.Peninsula (~22 cells) and "
          "East/NE (~19 cells) regions is noisy — read it with care. T2M bias caveat "
          "(op-model proxy / model-climate offset) from the all-India table still "
          "applies.\n"]
    for var in VARS:
        md.append(f"\n## {var}\n")
        md.append("| Region | Model | Lead | PCC | RMSE | Bias | N |")
        md.append("|---|---|---|---|---|---|---|")
        sub = df[df["Variable"] == var]
        for _, r in sub.iterrows():
            md.append(f"| {r.Region} | {r.Model} | {r.Lead} | {r.PCC:+.3f} | "
                      f"{r.RMSE:.3f} | {r.Bias:+.3f} | {int(r.N)} |")
    md_path = os.path.join(outdir, "case1_metrics_regional.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"  regional tables -> {md_path}\n                    {csv_path}", flush=True)


def export_tables(df: pd.DataFrame, outdir: str):
    df = df.copy()
    csv_path = os.path.join(outdir, "case1_metrics.csv")
    df.to_csv(csv_path, index=False)

    show = df[["Model", "Variable", "Lead", "PCC", "RMSE", "Bias"]].copy()
    show["PCC"] = show["PCC"].map(lambda x: f"{x:+.3f}")
    show["RMSE"] = show["RMSE"].map(lambda x: f"{x:.3f}")
    show["Bias"] = show["Bias"].map(lambda x: f"{x:+.3f}")

    # Markdown
    md = ["# Case Study 1 — Spatial skill over India (IMD-union) land points "
          "(cosine-weighted, anomalies)\n",
          "Valid week 2026-02-12 .. 2026-02-18. "
          "Units: Z500 gpm, TP mm/day, T2M deg C.\n",
          "\n> **Caveats.** PCC is the bias-insensitive (mean-removed) pattern metric; "
          "RMSE/Bias also absorb systematic offsets.\n"
          "> * **TP**: this is a near-total-dry week, so absolute precip ~0 for both "
          "model and ERA5 and the anomaly collapses to -climatology; FuXi's near-perfect "
          "TP PCC reflects its dry/smooth output coinciding with that degenerate field, "
          "not necessarily forecast skill.\n"
          "> * **T2M**: anomalies are referenced to ERA5 climatology. ECMWF/NCEP supply "
          "only 6-h max/min, so T2M is a (mx2t6+mn2t6)/2 proxy; FuXi/op models carry a "
          "model-climate cold offset (~-4 to -5 deg C) absent in ERA5-calibrated Spire. "
          "Read the warm-core skill from PCC.\n"]
    for var in VARS:
        md.append(f"\n## {var}\n")
        sub = show[df["Variable"] == var]
        md.append("| Model | Lead | PCC | RMSE | Bias |")
        md.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            md.append(f"| {r.Model} | {r.Lead} | {r.PCC} | {r.RMSE} | {r.Bias} |")
    md_path = os.path.join(outdir, "case1_metrics.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    # LaTeX
    tex = [r"\begin{table}[t]\centering",
           r"\caption{Case Study 1: cosine-weighted spatial skill over India land "
           r"points (anomalies). Valid week 12--18 Feb 2026.}",
           r"\label{tab:case1_skill}",
           r"\begin{tabular}{llrrr}", r"\hline",
           r"Model & Lead & PCC & RMSE & Bias \\ \hline"]
    for var in VARS:
        tex.append(r"\multicolumn{5}{l}{\textbf{" + var + r"}} \\ \hline")
        sub = show[df["Variable"] == var]
        for _, r in sub.iterrows():
            tex.append(f"{r.Model} & {r.Lead} & {r.PCC} & {r.RMSE} & {r.Bias} "
                       r"\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(outdir, "case1_metrics.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print(f"  tables -> {md_path}\n           {tex_path}\n           {csv_path}",
          flush=True)
    print("\n" + "\n".join(md), flush=True)


# ============================================================================
# Color-scale helper — symmetric, shared per variable column
# ============================================================================
def var_scale(all_fields: dict, var: str) -> dict:
    """Symmetric diverging scale spanning both leads & all models for a variable."""
    vals = []
    for lead in LEADS:
        for m in ROWS:
            a = all_fields[lead][var].get(m)
            if a is not None:
                vals.append(np.abs(a[np.isfinite(a)]))
    allv = np.concatenate(vals) if vals else np.array([1.0])
    vmax = float(np.nanpercentile(allv, 98))
    vmax = max(vmax, 1e-3)
    return dict(vmin=-vmax, vmax=vmax, norm=TwoSlopeNorm(0, -vmax, vmax))


# ============================================================================
# Map drawing
# ============================================================================
def draw_panel(ax, lons, lats, data, cmap, norm, contour=False,
               left_label=False, bottom_label=False):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#D7E3F4", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F4F2ED", zorder=0)
    im = None
    if data is not None and not np.all(np.isnan(data)):
        im = ax.pcolormesh(lons, lats, data, transform=PROJ, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True, zorder=2)
        if contour:
            lv = np.linspace(norm.vmin, norm.vmax, 9)
            lv = lv[np.abs(lv) > 1e-6]
            cs = ax.contour(lons, lats, data, levels=lv, colors="k",
                            linewidths=0.4, alpha=0.5, transform=PROJ, zorder=3)
            ax.clabel(cs, inline=True, fontsize=5, fmt="%d")
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color="#888")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.55, zorder=5,
                   edgecolor="#111")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, zorder=5,
                   edgecolor="#444", linestyle="--")
    gl = ax.gridlines(crs=PROJ, draw_labels=False, lw=0.3, color="#AAA",
                      alpha=0.5, linestyle=":")
    gl.xlocator = mticker.FixedLocator(range(70, 100, 10))
    gl.ylocator = mticker.FixedLocator(range(10, 40, 10))
    if left_label:
        gl.left_labels = True
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {"size": 6.5, "color": "#444"}
    if bottom_label:
        gl.bottom_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {"size": 6.5, "color": "#444"}
    return im


def figure1_truth(all_fields: dict, grid: Grid, scales: dict, outdir: str):
    """ERA5 observed ground truth: Z500 | TP | T2M anomalies for the valid week."""
    lead = "Week-1 lead"  # ERA5 obs is identical across leads (same valid week)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             subplot_kw={"projection": PROJ})
    panels = [
        ("Z500", "(a) Z500 height anomaly", True),
        ("TP",   "(b) Total precipitation anomaly", False),
        ("T2M",  "(c) 2 m temperature anomaly", False),
    ]
    for j, (var, title, contour) in enumerate(panels):
        meta = VMETA[var]
        sc = scales[var]
        im = draw_panel(axes[j], grid.lon, grid.lat, all_fields[lead][var]["ERA5"],
                        meta["cmap"], sc["norm"], contour=contour,
                        left_label=(j == 0), bottom_label=True)
        axes[j].set_title(title, fontsize=11, fontweight="bold", pad=6)
        cb = fig.colorbar(im, ax=axes[j], orientation="horizontal",
                          fraction=0.05, pad=0.07, extend="both")
        cb.set_label(meta["label"], fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Figure 1 — ERA5 observed state, 12-18 Feb 2026 "
                 "(dry ridge & warm anomaly over NW/Central India)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig1_era5_truth.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote fig1_era5_truth.[png,pdf]", flush=True)


def figure2_matrix(all_fields: dict, grid: Grid, scales: dict, lead: str,
                   tag: str, outdir: str):
    """5x3 matrix: rows ERA5/Spire/FuXi/ECMWF/NCEP, cols Z500/TP/T2M."""
    cfg = LEADS[lead]
    nrow, ncol = len(ROWS), len(VARS)
    fig = plt.figure(figsize=(11, 15), facecolor="white")
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0.10, wspace=0.06,
                  top=0.90, bottom=0.08, left=0.08, right=0.96)
    col_im = {}
    for ri, m in enumerate(ROWS):
        for ci, var in enumerate(VARS):
            ax = fig.add_subplot(gs[ri, ci], projection=PROJ)
            sc = scales[var]
            im = draw_panel(ax, grid.lon, grid.lat, all_fields[lead][var].get(m),
                            VMETA[var]["cmap"], sc["norm"],
                            contour=(var == "Z500"),
                            left_label=(ci == 0), bottom_label=(ri == nrow - 1))
            if im is not None:
                col_im[ci] = im
            if ri == 0:
                ax.set_title(VMETA[var]["title"], fontsize=10.5,
                             fontweight="bold", pad=6)
            if ci == 0:
                ax.text(-0.16, 0.5, RLAB[m], transform=ax.transAxes, rotation=90,
                        fontsize=11, fontweight="bold", ha="center", va="center")
    # shared per-column colorbars along the bottom
    for ci, var in enumerate(VARS):
        if ci not in col_im:
            continue
        x0 = 0.08 + ci * (0.96 - 0.08) / ncol + 0.01
        cax = fig.add_axes([x0, 0.035, (0.96 - 0.08) / ncol - 0.03, 0.012])
        cb = fig.colorbar(col_im[ci], cax=cax, orientation="horizontal",
                          extend="both")
        cb.set_label(VMETA[var]["label"], fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Figure 2{tag} — Multi-model spatial comparison, {lead} "
                 f"(init {cfg['init']}, {cfg['week']})\n"
                 "Valid 12-18 Feb 2026 · anomalies vs ERA5 climatology · India (IMD regions)",
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.018,
             "T2M anomalies use the ERA5 climatology as reference; ECMWF/NCEP T2M is a "
             "(mx2t6+mn2t6)/2 daily proxy, so their uniform cold offset is largely a "
             "model-climate/proxy bias — judge pattern skill by PCC, not mean bias.",
             ha="center", fontsize=7.0, color="#666", fontstyle="italic")
    fname = os.path.join(outdir, f"fig2{tag}_matrix_{cfg['init'].replace('-', '')}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{fname}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(fname)}.[png,pdf]", flush=True)


# ============================================================================
# Main
# ============================================================================
def main():
    outdir = CONFIG["outdir"]
    os.makedirs(outdir, exist_ok=True)

    # Grid from the weekly product
    base = xr.open_dataset(CONFIG["weekly_fields"])
    grid = Grid(base.lat.values, base.lon.values)
    print(f"Case grid: {grid.lat.size} x {grid.lon.size} | "
          f"India (IMD-union) cells {grid.land.sum()}/{grid.land.size}", flush=True)

    clim = load_climatology(grid)

    # Assemble all anomaly fields: all_fields[lead][var][model]
    all_fields = load_z_tp_anomalies(grid)
    t2m = load_t2m_anomalies(grid, clim["t2m"])
    for lead in LEADS:
        all_fields[lead]["T2M"] = t2m[lead]["T2M"]

    # Shared per-variable color scales
    scales = {v: var_scale(all_fields, v) for v in VARS}

    # Figures
    print("Rendering figures ...", flush=True)
    figure1_truth(all_fields, grid, scales, outdir)
    figure2_matrix(all_fields, grid, scales, "Week-1 lead", "a", outdir)
    figure2_matrix(all_fields, grid, scales, "Week-2 lead", "b", outdir)

    # Metrics
    print("Computing metrics ...", flush=True)
    df = build_metrics_table(all_fields, grid)
    export_tables(df, outdir)
    rdf = build_regional_table(all_fields, grid)
    export_regional_tables(rdf, outdir)

    print("\nCASE1_DONE", flush=True)


if __name__ == "__main__":
    main()
