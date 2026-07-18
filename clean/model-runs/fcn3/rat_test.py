r"""
FourCastNet3 (FCN3) inference reference script â€" earth2studio + local ERA5 data.

================================================================================
 HOW TO RUN
================================================================================
1. Activate the environment:
     source $(conda info --base)/etc/profile.d/conda.sh
     conda activate earth2studio_env

2. One-time environment patches (only needed once per environment â€" fixes
   bfloat16 dtype-casting bugs in the torch-harmonics / makani versions
   FCN3 depends on). Skip if already applied on this environment:

     ENV=$CONDA_PREFIX/lib/python3.11/site-packages
     sed -i 's/x = x.reshape(B, self.groups, self.groupsize, K, H, W)/x = x.reshape(B, self.groups, self.groupsize, K, H, W).to(self.weight.dtype)/' \
         $ENV/torch_harmonics/convolution.py
     sed -i 's/x = x.reshape(B, self.groups, self.groupsize, H, W)/x = x.reshape(B, self.groups, self.groupsize, H, W).to(self.weight.dtype)/' \
         $ENV/torch_harmonics/convolution.py
     sed -i 's/x = x.to(torch.float32).contiguous()/x = x.to(torch.float32).contiguous()\n        vals = vals.to(torch.float32).contiguous()/' \
         $ENV/torch_harmonics/_disco_convolution.py
     sed -i 's/w = _soft_clamp(x\[\.\.\., self.water_channels, :, :\])/w = _soft_clamp(x[..., self.water_channels, :, :]).to(x.dtype)/' \
         $ENV/makani/models/networks/fourcastnet3.py

3. Edit OUTPUT_PATH below to where you want the forecast .nc file saved.

4. Run on a GPU node (via SLURM sbatch, or directly on an interactive GPU
   allocation):
     python run_fcn3_reference.py <init_time> <n_steps>

   Example â€" 48-hour forecast (8 steps x 6h) initialized 2015-08-01 00Z:
     python run_fcn3_reference.py 2015-08-01T00:00:00 8

   <init_time> : ISO timestamp, e.g. 2015-08-01T00:00:00
   <n_steps>   : number of 6-hourly steps to roll out (4=24h, 8=48h, 20=120h)

================================================================================
 DATA LOADING (initial conditions)
================================================================================
Read directly from the group's local ERA5 zarr mirror â€" 6-hourly, 0.25 deg,
1959-2023, no internet/download needed:

    /storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/
        1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr

That store covers every FCN3 input variable EXCEPT 100m winds (u100m/v100m).
Those are read from pre-downloaded monthly files at:

    /storage/ratul/era5_100m_monthly/{year}_{month:02d}.nc

and fall back to fetching from the public ARCO cloud zarr (requires internet)
if a monthly file is missing for the requested date.

================================================================================
 MODEL CHECKPOINTS
================================================================================
FCN3.load_default_package() auto-downloads/caches weights on first use to:

    ~/.cache/earth2studio/fcn3/
        best_ckpt_mp0.tar     (~2.7 GB â€" model weights)
        config.json
        global_means.npy, global_stds.npy, maxs.npy, mins.npy
        land_mask.nc, orography.nc

Source: hf://nvidia/fourcastnet3 (downloaded once, reused after that).
No action needed unless this cache is missing or corrupted.

================================================================================
 OUTPUT
================================================================================
Forecast state is written to OUTPUT_PATH (edit below) with all 72 FCN3
output channels at every saved step: surface (u10m,v10m,u100m,v100m,t2m,
msl,tcwv) + 13 pressure levels each of u,v,z,t,q
(50,100,150,200,250,300,400,500,600,700,850,925,1000 hPa).
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import torch

os.environ["WANDB_MODE"] = "offline"

# ==============================================================================
# >>> EDIT THIS: where to save the forecast output <<<
# ==============================================================================
OUTPUT_PATH = "/path/to/your/output_dir/fcn3_output.nc"   # TODO: set this

# ------------------------------------------------------------------------------
# Data sources (see "DATA LOADING" above) â€" leave as-is unless the underlying
# storage paths change.
# ------------------------------------------------------------------------------
ERA5_ZARR = (
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
MONTHLY_100M_DIR = "/storage/ratul/era5_100m_monthly"


class LocalZarrERA5:
    """Initial-condition loader: bedartha's local ERA5 zarr mirror, with
    100m winds from pre-downloaded monthly files (falls back to ARCO cloud
    if a monthly file for the requested month is missing)."""

    def __init__(self):
        self.ds = xr.open_zarr(ERA5_ZARR, consolidated=True)
        self._100m_cache = {}   # (year, month) -> xr.Dataset
        self.var_map = {
            "u10m":  ("10m_u_component_of_wind",  None),
            "v10m":  ("10m_v_component_of_wind",  None),
            "u100m": ("u100m",                    "MONTHLY_100M"),
            "v100m": ("v100m",                    "MONTHLY_100M"),
            "t2m":   ("2m_temperature",            None),
            "msl":   ("mean_sea_level_pressure",   None),
            "tcwv":  ("total_column_water_vapour", None),
            **{f"u{p}": ("u_component_of_wind", p)
               for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
            **{f"v{p}": ("v_component_of_wind", p)
               for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
            **{f"z{p}": ("geopotential", p)
               for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
            **{f"t{p}": ("temperature", p)
               for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
            **{f"q{p}": ("specific_humidity", p)
               for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
        }

    def _get_100m_ds(self, ts):
        key = (ts.year, ts.month)
        if key in self._100m_cache:
            return self._100m_cache[key]
        fpath = os.path.join(MONTHLY_100M_DIR, f"{ts.year}_{ts.month:02d}.nc")
        if os.path.exists(fpath):
            ds = xr.open_dataset(fpath)
            self._100m_cache[key] = ds
            return ds
        # Fallback: public ARCO cloud (requires internet)
        from earth2studio.data import ARCO
        return ARCO()

    def __call__(self, time, variable):
        from collections import OrderedDict
        if isinstance(time, (str, datetime)):
            time = [time]
        times = pd.to_datetime(time)
        out_arrays = []
        for t_np in times.values:
            ts = pd.Timestamp(t_np)
            var_arrays = []
            ds_100m = None
            if "u100m" in variable or "v100m" in variable:
                ds_100m = self._get_100m_ds(ts)
            for v in variable:
                zarr_name, level = self.var_map[v]
                if level == "MONTHLY_100M":
                    try:
                        if hasattr(ds_100m, "__call__"):
                            # ARCO fallback
                            da_tmp = ds_100m([ts.to_pydatetime()], [v])
                            data = da_tmp.sel(variable=v).values.squeeze().astype(np.float32)
                        else:
                            data = (ds_100m[v]
                                    .sel(time=t_np, method="nearest")
                                    .values.astype(np.float32))
                    except Exception:
                        data = np.zeros(
                            self.ds["10m_u_component_of_wind"].sel(time=t_np).values.shape,
                            dtype=np.float32,
                        )
                elif level:
                    data = self.ds[zarr_name].sel(time=t_np, level=level).values.astype(np.float32)
                else:
                    data = self.ds[zarr_name].sel(time=t_np).values.astype(np.float32)
                var_arrays.append(data)
            out_arrays.append(np.stack(var_arrays, axis=0))
        out = np.stack(out_arrays, axis=0)
        return xr.DataArray(
            data=out,
            dims=["time", "variable", "lat", "lon"],
            coords=OrderedDict({
                "time":     times.to_numpy(),
                "variable": np.array(variable),
                "lat":      self.ds.latitude.values,
                "lon":      self.ds.longitude.values,
            }),
        )


def main():
    if "path/to/your" in OUTPUT_PATH:
        raise SystemExit(
            "ERROR: OUTPUT_PATH is still the placeholder value.\n"
            "Edit OUTPUT_PATH near the top of this script before running."
        )

    if len(sys.argv) < 3:
        print("Usage: python run_fcn3_reference.py <init_time> <n_steps>")
        print("  init_time : ISO timestamp, e.g. 2015-08-01T00:00:00")
        print("  n_steps   : number of 6-hourly steps to roll out (4=24h, 8=48h, 20=120h)")
        sys.exit(1)

    init_time = sys.argv[1]
    n_steps   = int(sys.argv[2])

    print("Importing earth2studio and loading FCN3...")
    from earth2studio.models.px import FCN3
    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic as run

    # Auto-downloads/caches checkpoint to ~/.cache/earth2studio/fcn3/ on first run.
    model = FCN3.load_model(FCN3.load_default_package())
    model = model.eval().bfloat16()

    device = torch.device("cuda:0")
    model = model.to(device)
    print(f"Model loaded on {device}.")

    data = LocalZarrERA5()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    io = NetCDF4Backend(OUTPUT_PATH, backend_kwargs={"mode": "w"})

    print(f"Running {n_steps} steps (6h/step) from {init_time}...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
        run(
            time=[init_time],
            nsteps=n_steps,
            prognostic=model,
            data=data,
            io=io,
            device=device,
        )
        io.close()

    print(f"Done. Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()