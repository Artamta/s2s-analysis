"""
make_forecast_timeseries.py — 42-day daily forecast-vs-ERA5 skill check.

For ONE initialization, plot the daily area-averaged value over the full 42-day
forecast (days 1-42), each model against ERA5, over masked Indian land:

  Panel 1  TP    — All India
  Panel 2  Z500  — All India
  Panel 3  TP    — Northwest India
  Panel 4  TP    — Central India
  Panel 5  TP    — South Peninsula
  Panel 6  TP    — East / Northeast India   (IMD homogeneous rainfall regions)

Absolute values (does the forecast track the observed evolution = skill check).
Daily loaders replicate paper/code/compute/03_compute_skill_z500_tp.py exactly
(ERA5 x1000 mm; ECMWF/NCEP precip is cumulative -> differenced to daily; FuXi/
SPIRE as-is; Z500 in gpm = geopotential / g).

Set INIT below.  Run: conda run -n s2s-hind python make_forecast_timeseries.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/raj.ayush/s2s/s2s_anlysis/paper/code")
from utils.verification_wmo import get_cosine_latitude_weights
from utils.verification_extra import get_land_mask, mask_land

# ---- config ---------------------------------------------------------------
INIT = "2026-02-12"          # case-study initialization
NDAYS = 42
SMOOTH = 3                   # centered rolling-mean window (days) for readability
G = 9.80665
DATA = "/storage/raj.ayush/s2s-forecast-data"
CLIM_PATH = "/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc"
OPEN = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
OUT = "/home/raj.ayush/s2s/s2s_anlysis/paper/case_study/case1_dry_ridge"

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)
WEIGHTS = get_cosine_latitude_weights(target_lat)

_mask = xr.open_dataset(f"{DATA}/era5/daily/imd_region_masks.nc")
REGION_MASKS = {k: _mask[k].values.astype(bool) for k in _mask.data_vars}
ALL_INDIA = np.zeros((target_lat.size, target_lon.size), bool)
for _m in REGION_MASKS.values():
    ALL_INDIA |= _m

MODELS = ["SPIRE", "FuXi", "ECMWF", "NCEP"]
COL = {"SPIRE": "#E8720C", "FuXi": "#5B8FB9", "ECMWF": "#4FA06A",
       "NCEP": "#9B7FC2", "ERA5": "#111111"}
PANELS = [("TP", "All India", ALL_INDIA), ("Z500", "All India", ALL_INDIA),
          ("TP", "Northwest", REGION_MASKS["northwest_india"]),
          ("TP", "Central", REGION_MASKS["central_india"]),
          ("TP", "South Peninsula", REGION_MASKS["south_peninsula"]),
          ("TP", "East / Northeast", REGION_MASKS["east_northeast_india"])]
UNITS = {"TP": "mm day$^{-1}$", "Z500": "gpm"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


# ---- loaders (replicate compute/03) ---------------------------------------
def to_grid(da):
    ren = {}
    if "latitude" in da.dims: ren["latitude"] = "lat"
    if "longitude" in da.dims: ren["longitude"] = "lon"
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method="linear").squeeze(), LAND)


def fuxi_day(init_str, day, ch):
    fs = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p):
            continue
        da = xr.open_dataset(p)["__xarray_dataarray_variable__"]
        da = da.sel(channel="tp") if ch == "tp" else (da.sel(channel="z500") / G)
        fs.append(da.squeeze())
    return None if not fs else xr.concat(fs, "m").mean("m")


def load_op(model, init_str):
    base = f"{DATA}/{model}/data"
    tp = gh = None
    try:
        d = xr.open_dataset(f"{base}/sfc_pf_{init_str}.grib",
                            filter_by_keys={"shortName": "tp"}, **OPEN)["tp"]
        tp = d.mean("number") if "number" in d.dims else d
    except Exception as e:
        print(f"  {model} TP miss: {e}", flush=True)
    try:
        d = xr.open_dataset(f"{base}/pl_pf_{init_str}.grib",
                            filter_by_keys={"shortName": "gh"}, **OPEN)["gh"]
        if "isobaricInhPa" in d.dims: d = d.sel(isobaricInhPa=500)
        gh = d.mean("number") if "number" in d.dims else d
    except Exception as e:
        print(f"  {model} Z500 miss: {e}", flush=True)
    return tp, gh


def amean(field2d, mask):
    """Cosine-weighted area mean over a region mask (NaN-safe)."""
    da = xr.DataArray(field2d, dims=["lat", "lon"],
                      coords={"lat": target_lat, "lon": target_lon})
    da = da.where(xr.DataArray(mask, dims=["lat", "lon"],
                               coords={"lat": target_lat, "lon": target_lon}))
    return float(da.weighted(WEIGHTS).mean(["lat", "lon"]))


# ---- build daily ANOMALY series -------------------------------------------
def build():
    init_str = pd.to_datetime(INIT).strftime("%Y%m%d")
    dates = pd.date_range(INIT, periods=NDAYS)
    dstr = [d.strftime("%Y-%m-%d") for d in dates]
    print(f"INIT {INIT}: loading models ...", flush=True)

    sp = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr", group="mean_stddev").sel(reference_time=INIT)
    sp_tp = sp["precipitation_amount"]
    sp_z = sp["geopotential_height_at_isobaric_levels"].sel(isobar=50000.0)
    sp_tp_sd = sp["precipitation_amount_stddev"]
    sp_z_sd = sp["geopotential_height_at_isobaric_levels_stddev"].sel(isobar=50000.0)
    ec_tp, ec_z = load_op("ecmwf", init_str)
    nc_tp, nc_z = load_op("ncep", init_str)
    fx_tp = {d: fuxi_day(init_str, d, "tp") for d in range(1, NDAYS + 1)}
    fx_z = {d: fuxi_day(init_str, d, "z500") for d in range(1, NDAYS + 1)}
    era_tp = xr.open_dataset(f"{DATA}/era5/data/era5_surface.grib",
                             filter_by_keys={"shortName": "tp"}, **OPEN)["tp"] * 1000.0
    era_z = xr.open_dataset(f"{DATA}/era5/data/era5_pressure_500hpa.grib", **OPEN)["z"] / G

    # daily day-of-year climatology (cached per DOY) -> anomaly baseline
    clim = xr.open_dataset(CLIM_PATH)
    _cc = {}
    def clim_field(var, scale, doy):
        key = (var, doy)
        if key not in _cc:
            _cc[key] = to_grid(clim[var].sel(dayofyear=doy) * scale).values
        return _cc[key]

    def cum_daily(cum, d):
        if cum is None or cum.sizes.get("step", 0) < d:
            return None
        return cum.isel(step=d - 1) if d == 1 else cum.isel(step=d - 1) - cum.isel(step=d - 2)

    def step_daily(arr, d):
        if arr is None or arr.sizes.get("step", 0) < d:
            return None
        return arr.isel(step=d - 1)

    nan2d = np.full((target_lat.size, target_lon.size), np.nan)
    daily = {p: {m: [] for m in MODELS + ["ERA5"]} for p in ["TP", "Z500"]}
    spread = {"TP": [], "Z500": []}   # SPIRE +/-1 sigma (gridpoint stddev fields)

    for i, d in enumerate(range(1, NDAYS + 1)):
        doy = int(dates[i].dayofyear)
        cl_tp = clim_field("tp", 1000.0, doy)
        cl_z = clim_field("z500", 1.0 / G, doy)
        # ERA5 truth anomaly
        try:
            o_tp = to_grid(era_tp.sel(time=dstr[i], method="nearest")).values - cl_tp
            o_z = to_grid(era_z.sel(time=dstr[i], method="nearest")).values - cl_z
        except Exception:
            o_tp = o_z = nan2d
        daily["TP"]["ERA5"].append(o_tp)
        daily["Z500"]["ERA5"].append(o_z)

        f_tp = {"SPIRE": step_daily(sp_tp, d), "FuXi": fx_tp.get(d),
                "ECMWF": cum_daily(ec_tp, d), "NCEP": cum_daily(nc_tp, d)}
        f_z = {"SPIRE": step_daily(sp_z, d), "FuXi": fx_z.get(d),
               "ECMWF": step_daily(ec_z, d), "NCEP": step_daily(nc_z, d)}
        for m in MODELS:
            daily["TP"][m].append((to_grid(f_tp[m]).values - cl_tp) if f_tp[m] is not None else nan2d)
            daily["Z500"][m].append((to_grid(f_z[m]).values - cl_z) if f_z[m] is not None else nan2d)
        # SPIRE ensemble spread (stddev fields; anomaly-invariant)
        st = step_daily(sp_tp_sd, d); sz = step_daily(sp_z_sd, d)
        spread["TP"].append(to_grid(st).values if st is not None else nan2d)
        spread["Z500"].append(to_grid(sz).values if sz is not None else nan2d)
    print("  loaded; computing area means ...", flush=True)
    return daily, spread, dates


# ---- plot -----------------------------------------------------------------
def _smooth(y):
    if SMOOTH and SMOOTH > 1:
        return pd.Series(y).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    return np.asarray(y)


def plot(daily, spread, dates):
    x = np.arange(1, NDAYS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.2))
    axes = axes.ravel()
    for ax, (var, region, mask) in zip(axes, PANELS):
        ax.axhline(0, color="#777", lw=1.0, zorder=1)            # climatology
        # SPIRE +/-1 sigma spread band
        spm = np.array([amean(daily[var]["SPIRE"][i], mask) for i in range(NDAYS)])
        sps = np.array([amean(spread[var][i], mask) for i in range(NDAYS)])
        if np.isfinite(spm).any() and np.isfinite(sps).any():
            lo, hi = _smooth(spm - sps), _smooth(spm + sps)
            ax.fill_between(x, lo, hi, color=COL["SPIRE"], alpha=0.13, zorder=2,
                            label="SPIRE $\\pm1\\sigma$")
        for m in ["ERA5"] + MODELS:
            ys = _smooth([amean(daily[var][m][i], mask) for i in range(NDAYS)])
            if not np.isfinite(ys).any():
                continue
            if m == "ERA5":
                ax.plot(x, ys, color=COL[m], lw=3.0, label="ERA5 (obs)", zorder=7)
            else:
                lw = 2.5 if m == "SPIRE" else 1.6
                a = 1.0 if m == "SPIRE" else 0.8
                ax.plot(x, ys, color=COL[m], lw=lw, label=m, alpha=a,
                        zorder=6 if m == "SPIRE" else 3)
        for wb in range(7, NDAYS, 7):
            ax.axvline(wb, color="#DDD", lw=0.7, ls=":", zorder=0)
        ax.set_title(f"{var} anomaly — {region}", pad=6)
        ax.set_xlim(1, NDAYS); ax.set_xticks(range(7, NDAYS + 1, 7))
        ax.set_xlabel("Forecast day")
        ax.set_ylabel(f"anomaly ({UNITS[var]})")
        ax.grid(axis="y", ls=":", alpha=0.30)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=9.5, frameon=True, edgecolor="#CCC", ncol=2)
    fig.suptitle(f"Daily forecast anomaly vs ERA5 over the 42-day range — init {INIT} "
                 f"(valid {dates[0]:%d %b} – {dates[-1]:%d %b %Y}) · India land (IMD regions)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pres_07_timeseries_{INIT.replace('-', '')}.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote pres_07_timeseries", flush=True)


if __name__ == "__main__":
    daily, spread, dates = build()
    plot(daily, spread, dates)
    print("TIMESERIES_DONE", flush=True)
