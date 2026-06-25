#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_t2m_patterns.py — ERA5 Temperature Pattern Explorer over India
======================================================================
Publication-quality 2m Temperature analysis from WB2 ERA5.

Figures (saved to ./t2m_exploration/):
  01. JJAS & JFM mean T2M climatology maps
  02. Monthly T2M cycle (12 panels)
  03. All-India seasonal cycle (ribbon + percentiles)
  04. Interannual JJAS T2M variability
  05. JJAS T2M trend map (1959-2022) with significance
  06. Heat wave frequency — days > 40°C, > 35°C
  07. Diurnal Temperature Range proxy (daily std)
  08. T2M–TP joint analysis — correlation map
  09. Regional T2M seasonal cycles (4 IMD regions)
  10. T2M coefficient of variation
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

WB2 = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
LAT_S, LAT_N, LON_W, LON_E = 5.0, 38.0, 65.0, 100.0
CLIM_Y0, CLIM_Y1 = 1991, 2020
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2m_exploration")
os.makedirs(OUT_DIR, exist_ok=True)

IMD_REGIONS = {
    "Northwest India":  {"lat": (25, 36), "lon": (68, 80), "color": "#e41a1c"},
    "Central India":    {"lat": (20, 28), "lon": (74, 86), "color": "#377eb8"},
    "South Peninsula":  {"lat": (8, 20),  "lon": (74, 82), "color": "#4daf4a"},
    "East & NE India":  {"lat": (20, 30), "lon": (86, 98), "color": "#ff7f00"},
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 11, "figure.dpi": 200,
})

def load_data():
    print("  Loading WB2 ERA5 T2M + TP...")
    ds = xr.open_zarr(WB2)
    t2m = ds["2m_temperature"].sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))
    t2m = t2m.sel(time=t2m["time.hour"] == 6)
    t2m = (t2m - 273.15).load()  # K → °C
    t2m = t2m.transpose("time", "latitude", "longitude")
    print(f"  T2M loaded: {t2m.shape}, {t2m.nbytes/1e6:.0f} MB")

    tp = ds["total_precipitation_24hr"].sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))
    tp = tp.sel(time=tp["time.hour"] == 6)
    tp = (tp * 1000.0).load()
    tp = tp.transpose("time", "latitude", "longitude")
    print(f"  TP loaded: {tp.shape}, {tp.nbytes/1e6:.0f} MB")
    return t2m, tp

def cos_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": lat})

def area_mean(da):
    w = cos_weights(da.latitude)
    return da.weighted(w).mean(dim=["latitude", "longitude"])

def region_mean(da, lr, lo):
    return area_mean(da.sel(latitude=slice(lr[0], lr[1]), longitude=slice(lo[0], lo[1])))

def smooth(d, s=0.6): return gaussian_filter(np.nan_to_num(d, nan=0), sigma=s)

def savefig(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  ✓ {p}")

def add_map(ax, rivers=False):
    ax.coastlines(resolution="50m", linewidth=0.6, color="#333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666")
    if rivers: ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#4a90d9", alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(); gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = gl.ylabel_style = {"size": 8}

def india(ax): ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=ccrs.PlateCarree())
def plbl(ax, l): ax.text(0.02, 0.95, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", path_effects=[pe.withStroke(linewidth=3, foreground="white")])

def t2m_cmap():
    return mcolors.LinearSegmentedColormap.from_list("t2m", [
        "#313695","#4575b4","#74add1","#abd9e9","#e0f3f8",
        "#ffffbf","#fee090","#fdae61","#f46d43","#d73027","#a50026"], N=256)

# ================= FIGURES =================
def fig01(t2m, tp):
    print("Fig 01: T2M seasonal climatology...")
    proj = ccrs.PlateCarree()
    c = t2m.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    jjas = c.sel(time=c["time.month"].isin([6,7,8,9])).mean("time")
    jfm = c.sel(time=c["time.month"].isin([1,2,3])).mean("time")
    fig, axes = plt.subplots(1,2,figsize=(13,6.5),subplot_kw={"projection":proj})
    for ax,d,t,lv,lb in zip(axes,[jjas,jfm],["JJAS","JFM"],
        [np.arange(15,40,1.5),np.arange(5,35,1.5)],["(a)","(b)"]):
        india(ax); add_map(ax,True)
        cf=ax.contourf(d.longitude,d.latitude,smooth(d.values),levels=lv,cmap=t2m_cmap(),extend="both",transform=proj)
        ax.set_title(f"{t} Mean 2m Temperature"); plbl(ax,lb)
        fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("°C")
    fig.suptitle(f"ERA5 T2M Climatology ({CLIM_Y0}–{CLIM_Y1})",fontsize=15,fontweight="bold",y=1.02)
    savefig(fig,"fig01_t2m_seasonal_clim.png")

def fig02(t2m, tp):
    print("Fig 02: Monthly T2M panels...")
    proj = ccrs.PlateCarree()
    c = t2m.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    monthly = c.groupby("time.month").mean("time")
    mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, axes = plt.subplots(3,4,figsize=(16,14),subplot_kw={"projection":proj})
    for i,ax in enumerate(axes.flat):
        india(ax); add_map(ax,gridlines=False)
        d = monthly.sel(month=i+1)
        cf=ax.contourf(d.longitude,d.latitude,smooth(d.values),levels=np.arange(5,40,2),cmap=t2m_cmap(),extend="both",transform=proj)
        ax.set_title(mn[i],fontsize=12,fontweight="bold")
    cax=fig.add_axes([0.25,0.02,0.5,0.015])
    fig.colorbar(cf,cax=cax,orientation="horizontal").set_label("Temperature (°C)")
    fig.suptitle(f"Monthly Mean T2M ({CLIM_Y0}–{CLIM_Y1})",fontsize=15,fontweight="bold",y=0.98)
    fig.subplots_adjust(hspace=0.08,wspace=0.05)
    savefig(fig,"fig02_t2m_monthly.png")

def fig03(t2m, tp):
    print("Fig 03: T2M seasonal cycle...")
    c = t2m.sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    ai = area_mean(c)
    dg = ai.groupby("time.dayofyear")
    k=15; vals = {}
    for n,q in [("mean",None),("p10",0.1),("p90",0.9)]:
        v = dg.mean() if q is None else dg.quantile(q)
        v.values[:] = uniform_filter1d(v.values, k, mode="wrap")
        vals[n] = v
    fig, ax = plt.subplots(figsize=(14,5.5))
    doy = vals["mean"].dayofyear.values
    ax.fill_between(doy, vals["p10"].values, vals["p90"].values, alpha=0.2, color="#d73027")
    ax.plot(doy, vals["mean"].values, color="#d73027", lw=2.5)
    ax.axvspan(152,273,alpha=0.06,color="green"); ax.axvspan(1,90,alpha=0.06,color="orange")
    md=[1,32,60,91,121,152,182,213,244,274,305,335]
    ml=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(md); ax.set_xticklabels(ml)
    ax.set_xlabel("Month"); ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"All-India Daily Mean T2M — Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})")
    ax.grid(axis="y",alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(fig,"fig03_t2m_cycle.png")

def fig04(t2m, tp):
    print("Fig 04: Interannual JJAS T2M...")
    jjas = t2m.sel(time=t2m["time.month"].isin([6,7,8,9]))
    yr = area_mean(jjas).resample(time="YE").mean()
    years = yr["time.year"].values.astype(float); vals = yr.values
    lt = float(np.nanmean(vals)); anom = vals - lt; std = float(np.nanstd(anom))
    fig, ax = plt.subplots(figsize=(15,5))
    norm = mcolors.Normalize(-2*std,2*std)
    ax.bar(years, anom, color=[plt.cm.RdBu_r(norm(a)) for a in anom], width=0.75, edgecolor="white",lw=0.3)
    rm = uniform_filter1d(anom, 11, mode="nearest")
    ax.plot(years, rm, "k-", lw=2, label="11-yr running mean")
    sl,ic,rv,pv,_=linregress(years[~np.isnan(vals)],vals[~np.isnan(vals)])
    ax.plot(years, sl*years+ic-lt, "r--", lw=1.5, label=f"Trend: {sl*10:+.3f} °C/decade (p={pv:.3f})")
    ax.axhline(0,color="k",lw=0.8); ax.legend(fontsize=9.5)
    ax.set_xlabel("Year"); ax.set_ylabel("JJAS T2M Anomaly (°C)")
    ax.set_title("All-India JJAS Mean T2M Anomaly (ERA5)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(fig,"fig04_t2m_interannual.png")

def fig05(t2m, tp):
    print("Fig 05: JJAS T2M trend map...")
    jjas = t2m.sel(time=t2m["time.month"].isin([6,7,8,9]))
    yr = jjas.groupby("time.year").mean("time")
    ya = yr.year.values.astype(float)
    sl_map = np.full(yr.shape[1:], np.nan); pv_map = np.full_like(sl_map, np.nan)
    for i in range(yr.shape[1]):
        for j in range(yr.shape[2]):
            ts = yr.values[:,i,j]; m = ~np.isnan(ts)
            if m.sum()>10:
                r = linregress(ya[m],ts[m]); sl_map[i,j]=r.slope*10; pv_map[i,j]=r.pvalue
    proj=ccrs.PlateCarree()
    fig,ax=plt.subplots(figsize=(9,7.5),subplot_kw={"projection":proj})
    india(ax); add_map(ax,True)
    lv=np.arange(-0.6,0.65,0.05)
    cm=mcolors.LinearSegmentedColormap.from_list("t",[
        "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0","#f7f7f7",
        "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f"],N=256)
    cf=ax.contourf(yr.longitude,yr.latitude,gaussian_filter(np.nan_to_num(sl_map),0.5),levels=lv,cmap=cm,extend="both",transform=proj)
    sig=pv_map<0.05
    if sig.any():
        lo2,la2=np.meshgrid(yr.longitude.values,yr.latitude.values)
        ax.scatter(lo2[sig],la2[sig],s=3,c="k",alpha=0.4,marker=".",transform=proj,zorder=6)
    fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("Trend (°C decade⁻¹)")
    ax.set_title("JJAS T2M Trend (1959–2022)",fontsize=13,pad=10)
    savefig(fig,"fig05_t2m_trend.png")

def fig06(t2m, tp):
    print("Fig 06: Heat extremes...")
    jjas = t2m.sel(time=t2m["time.month"].isin([4,5,6])) # AMJ for heat waves
    jjas = jjas.sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    h35 = (jjas > 35).groupby("time.year").mean("time").mean("year") * 100
    h40 = (jjas > 40).groupby("time.year").mean("time").mean("year") * 100
    proj=ccrs.PlateCarree()
    fig,axes=plt.subplots(1,2,figsize=(14,6.5),subplot_kw={"projection":proj})
    for ax,d,t,vm,cm in zip(axes,[h35,h40],["> 35°C","> 40°C"],[60,30],["YlOrRd","hot_r"]):
        india(ax); add_map(ax,True)
        cf=ax.contourf(d.longitude,d.latitude,smooth(d.values),levels=np.arange(0,vm,vm/12),cmap=cm,extend="max",transform=proj)
        ax.set_title(f"Days with T2M {t} (AMJ)")
        fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("% of AMJ days")
    fig.suptitle(f"Heat Extreme Frequency ({CLIM_Y0}–{CLIM_Y1})",fontsize=14,fontweight="bold",y=1.02)
    savefig(fig,"fig06_heat_extremes.png")

def fig07(t2m, tp):
    print("Fig 07: T2M variability...")
    c = t2m.sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    jjas = c.sel(time=c["time.month"].isin([6,7,8,9]))
    doy_clim = jjas.groupby("time.dayofyear").mean("time")
    anom = jjas.groupby("time.dayofyear") - doy_clim
    isv = anom.std("time")
    proj=ccrs.PlateCarree()
    fig,ax=plt.subplots(figsize=(9,7.5),subplot_kw={"projection":proj})
    india(ax); add_map(ax,True)
    cf=ax.contourf(isv.longitude,isv.latitude,smooth(isv.values),levels=np.arange(0,5,0.4),cmap="inferno_r",extend="max",transform=proj)
    fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("T2M Std Dev (°C)")
    ax.set_title(f"JJAS T2M Intraseasonal Variability ({CLIM_Y0}–{CLIM_Y1})",fontsize=13,pad=10)
    savefig(fig,"fig07_t2m_variability.png")

def fig08(t2m, tp):
    print("Fig 08: T2M-TP correlation...")
    jjas_t = t2m.sel(time=t2m["time.month"].isin([6,7,8,9])).sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    jjas_p = tp.sel(time=tp["time.month"].isin([6,7,8,9])).sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    # align times
    common = np.intersect1d(jjas_t.time.values, jjas_p.time.values)
    jjas_t = jjas_t.sel(time=common); jjas_p = jjas_p.sel(time=common)
    yr_t = jjas_t.groupby("time.year").mean("time")
    yr_p = jjas_p.groupby("time.year").mean("time")
    corr = xr.corr(yr_t, yr_p, dim="year")
    proj=ccrs.PlateCarree()
    fig,ax=plt.subplots(figsize=(9,7.5),subplot_kw={"projection":proj})
    india(ax); add_map(ax,True)
    lv=np.arange(-1,1.05,0.1)
    cf=ax.contourf(corr.longitude,corr.latitude,smooth(corr.values,0.3),levels=lv,cmap="RdBu",extend="both",transform=proj)
    fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("Correlation (r)")
    ax.set_title(f"T2M–TP Interannual Correlation (JJAS, {CLIM_Y0}–{CLIM_Y1})",fontsize=13,pad=10)
    savefig(fig,"fig08_t2m_tp_corr.png")

def fig09(t2m, tp):
    print("Fig 09: Regional T2M cycles...")
    c = t2m.sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for ax,(rn,rb) in zip(axes.flat,IMD_REGIONS.items()):
        rm = region_mean(c, rb["lat"], rb["lon"])
        dg = rm.groupby("time.dayofyear"); k=15
        m = dg.mean(); m.values[:]=uniform_filter1d(m.values,k,mode="wrap")
        p25=dg.quantile(0.25); p25.values[:]=uniform_filter1d(p25.values,k,mode="wrap")
        p75=dg.quantile(0.75); p75.values[:]=uniform_filter1d(p75.values,k,mode="wrap")
        doy=m.dayofyear.values
        ax.fill_between(doy,p25.values,p75.values,alpha=0.2,color=rb["color"])
        ax.plot(doy,m.values,color=rb["color"],lw=2.5)
        ax.set_title(rn,fontsize=12,fontweight="bold",color=rb["color"])
        ax.set_xlim(1,366); ax.grid(axis="y",alpha=0.2)
        ax.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335])
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    for ax in axes[-1]: ax.set_xlabel("Month")
    for ax in axes[:,0]: ax.set_ylabel("Temperature (°C)")
    fig.suptitle(f"Regional T2M Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})",fontsize=14,fontweight="bold",y=1.01)
    fig.tight_layout(); savefig(fig,"fig09_t2m_regional.png")

def fig10(t2m, tp):
    print("Fig 10: T2M CV map...")
    jjas = t2m.sel(time=t2m["time.month"].isin([6,7,8,9])).sel(time=slice(f"{CLIM_Y0}",f"{CLIM_Y1}"))
    yr = jjas.groupby("time.year").mean("time")
    # For temperature, use std in °C rather than CV (which is meaningless with offset scales)
    sd = yr.std("year")
    proj=ccrs.PlateCarree()
    fig,ax=plt.subplots(figsize=(9,7.5),subplot_kw={"projection":proj})
    india(ax); add_map(ax,True)
    cf=ax.contourf(sd.longitude,sd.latitude,smooth(sd.values),levels=np.arange(0,2.5,0.2),cmap="YlOrBr",extend="max",transform=proj)
    fig.colorbar(cf,ax=ax,orientation="horizontal",shrink=0.85,pad=0.06).set_label("Interannual Std Dev (°C)")
    ax.set_title(f"JJAS T2M Interannual Variability ({CLIM_Y0}–{CLIM_Y1})",fontsize=13,pad=10)
    savefig(fig,"fig10_t2m_std.png")

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--fig",nargs="*",type=int)
    args = ap.parse_args()
    figs = set(args.fig) if args.fig else set(range(1,11))
    print(f"═══ ERA5 T2M Publication Figures ═══\n  Output: {OUT_DIR}")
    t2m, tp = load_data()
    ff = {1:fig01,2:fig02,3:fig03,4:fig04,5:fig05,6:fig06,7:fig07,8:fig08,9:fig09,10:fig10}
    for n in sorted(figs):
        if n in ff:
            try: ff[n](t2m, tp)
            except Exception as e: print(f"  ✗ Fig {n} FAILED: {e}"); import traceback; traceback.print_exc()
    print(f"\n═══ Done! {OUT_DIR} ═══")

if __name__ == "__main__": main()
