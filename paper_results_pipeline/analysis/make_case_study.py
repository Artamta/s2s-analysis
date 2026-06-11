"""
Case study (Fig 10): the most anomalous Z500 week of JFM 2026, shown as the
observed ERA5 anomaly and each system's forecast at week-1 and week-2 lead.
The same verifying week is forecast at week-1 lead from its own init and at
week-2 lead from the preceding (one-week-earlier) init.
Reads analysis/weekly_anom_fields.nc.
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
f = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
inits = list(f['init'].values)
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
lat = f['lat'].values; lon = f['lon'].values

# pick most anomalous week-1 verifying week among inits with a predecessor
obs_w1 = f['z_obs'].isel(week=0).values  # (init,lat,lon)
amp = np.array([np.sqrt(np.nanmean(obs_w1[i] ** 2)) for i in range(len(inits))])
amp[0] = -1  # need a preceding init
istar = int(np.argmax(amp))
init_v = inits[istar]
print(f"case study verifying week = week-1 of init {init_v} (RMS anomaly {amp[istar]:.1f} m)", flush=True)

obs = f['z_obs'].isel(init=istar, week=0).values
fc_w1 = {m: f['z_fcst'].sel(model=m).isel(init=istar, week=0).values for m in MODELS}
fc_w2 = {m: f['z_fcst'].sel(model=m).isel(init=istar - 1, week=1).values for m in MODELS}

try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree()
except Exception:
    proj = None

vmax = np.nanpercentile(np.abs(obs[np.isfinite(obs)]), 98)
cols = ['ERA5 (obs)'] + MODELS
fig = plt.figure(figsize=(17, 7))
rows = [('Week-1 lead', fc_w1), ('Week-2 lead', fc_w2)]
for r, (rlab, fcd) in enumerate(rows):
    panels = [obs] + [fcd[m] for m in MODELS]
    for c, (data, title) in enumerate(zip(panels, cols)):
        ax = fig.add_subplot(2, 5, r * 5 + c + 1, projection=proj) if proj else fig.add_subplot(2, 5, r * 5 + c + 1)
        im = ax.pcolormesh(lon, lat, data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto',
                           transform=proj) if proj else ax.pcolormesh(lon, lat, data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        if proj:
            ax.add_feature(cfeature.COASTLINE, lw=0.5); ax.add_feature(cfeature.BORDERS, lw=0.3, ls=':')
            ax.set_extent([65, 100, 5, 38], crs=proj)
        if r == 0:
            ax.set_title(title, fontsize=12, fontweight='bold')
        if c == 0:
            ax.text(-0.12, 0.5, rlab, transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
fig.suptitle(f'Case study: Z500 anomaly for the most anomalous JFM-2026 week (verifying week-1 of {init_v})',
             fontsize=14, fontweight='bold', y=1.0)
cax = fig.add_axes([1.0, 0.2, 0.012, 0.6])
fig.colorbar(im, cax=cax, label='Z500 anomaly (m)')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig10_case_study.{ext}', bbox_inches='tight', dpi=300)
print("WROTE fig10_case_study", flush=True)
print("CASE_DONE", flush=True)
