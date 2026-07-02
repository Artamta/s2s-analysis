"""Path registry for the India S2S benchmark workflow.

Keep filesystem facts here so analysis scripts do not hard-code storage paths.
Environment variables can override the defaults when needed:

- ``S2S_STORAGE_ROOT`` defaults to ``/storage/raj.ayush``
- ``S2S_ERA5_CLIMATOLOGY`` defaults to the common ERA5 climatology file
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class StudyPaths:
    """Resolved paths used by the benchmark workflow."""

    project_root: Path
    repo_root: Path
    storage_root: Path
    all_model_data: Path
    previous_forecast_data: Path
    final_data: Path
    masks_dir: Path
    outputs_dir: Path
    era5_climatology: Path
    era5_prev_daily_tp_nc: Path
    era5_prev_daily_t2m_nc: Path
    era5_jfm2026_daily_dir: Path
    weatherbench2_era5_zarr: Path
    soi_shapefile: Path
    ecmwf_root: Path
    fuxi_root: Path
    ukmo_root: Path
    delysm_root: Path
    delysm_qc_root: Path
    ncep_root: Path

    # JFM2026
    jfm_spire_zarr: Path
    jfm_fuxi_root: Path
    jfm_ecmwf_root: Path
    jfm_ground_truth_root: Path

    # JJAS2019 case study
    jjas_fuxi_compact: Path
    jjas2019_ecmwf_new_root: Path
    jjas2019_imd_rainfall_nc: Path
    imd_rainfall_root: Path
    imd_climatology_dir: Path
    imd_daily_climatology_nc: Path
    imd_seasonal_climatology_nc: Path
    imd_daily_area_mean_csv: Path
    imd_jjas_cumulative_climatology_csv: Path
    imd_jfm_cumulative_climatology_csv: Path
    imd_jjas_yearly_climatology_csv: Path
    imd_jfm_yearly_climatology_csv: Path


def get_paths() -> StudyPaths:
    storage = _env_path("S2S_STORAGE_ROOT", "/storage/raj.ayush")
    previous = storage / "s2s-forecast-data-prev"
    all_model = storage / "All_Model_Data"
    final_data = storage / "s2s_final_data"

    imd_root = all_model / "ground_truth" / "imd_rainfall"
    imd_clim = imd_root / "climatology"
    default_era5_clim = all_model / "climatology_common" / "era5_climatology.nc"
    if not default_era5_clim.exists():
        default_era5_clim = storage / "benchmark(jfm)" / "era5_climatology.nc"
    era5_clim = _env_path("S2S_ERA5_CLIMATOLOGY", default_era5_clim)
    wb2_zarr = _env_path(
        "S2S_WEATHERBENCH2_ERA5_ZARR",
        "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    )

    return StudyPaths(
        project_root=PROJECT_ROOT,
        repo_root=REPO_ROOT,
        storage_root=storage,
        all_model_data=all_model,
        previous_forecast_data=previous,
        final_data=final_data,
        masks_dir=PROJECT_ROOT / "masks",
        outputs_dir=REPO_ROOT / "outputs",
        era5_climatology=era5_clim,
        era5_prev_daily_tp_nc=previous / "era5" / "daily" / "era5_daily_tp.nc",
        era5_prev_daily_t2m_nc=previous / "era5" / "daily" / "era5_daily_t2m.nc",
        era5_jfm2026_daily_dir=all_model / "ground_truth" / "era5_daily" / "jfm2026",
        weatherbench2_era5_zarr=wb2_zarr,
        soi_shapefile=storage / "archive" / "s2s-forecast-" / "STATE_BOUNDARY.shp",
        ecmwf_root=all_model / "ecmwf",
        fuxi_root=all_model / "fuxi",
        ukmo_root=all_model / "ukmo",
        delysm_root=all_model / "delysm",
        delysm_qc_root=all_model / "delysm_qc",
        ncep_root=all_model / "ncep",
        jfm_spire_zarr=storage
        / "archive"
        / "All_Model_Data"
        / "models"
        / "spire"
        / "data"
        / "s2s-research.zarr",
        jfm_fuxi_root=all_model / "fuxi" / "jfm2026",
        jfm_ecmwf_root=all_model / "ecmwf" / "jfm2026",
        jfm_ground_truth_root=all_model / "fuxi" / "jfm2026" / "ground_truth",
        jjas_fuxi_compact=final_data / "jjas" / "fuxi_combined",
        jjas2019_ecmwf_new_root=all_model / "ecmwf" / "jjas2019",
        imd_rainfall_root=imd_root,
        jjas2019_imd_rainfall_nc=all_model
        / "ground_truth"
        / "imd_rainfall"
        / "netcdf"
        / "imd_rain_2019.nc",
        imd_climatology_dir=imd_clim,
        imd_daily_climatology_nc=imd_clim / "imd_rain_1991_2020_daily_climatology.nc",
        imd_seasonal_climatology_nc=imd_clim / "imd_rain_1991_2020_seasonal_climatology.nc",
        imd_daily_area_mean_csv=imd_clim / "imd_rain_1991_2020_india_daily_area_mean.csv",
        imd_jjas_cumulative_climatology_csv=imd_clim
        / "imd_rain_1991_2020_india_cumulative_jjas.csv",
        imd_jfm_cumulative_climatology_csv=imd_clim
        / "imd_rain_1991_2020_india_cumulative_jfm.csv",
        imd_jjas_yearly_climatology_csv=imd_clim
        / "imd_rain_1991_2020_india_yearly_jjas.csv",
        imd_jfm_yearly_climatology_csv=imd_clim
        / "imd_rain_1991_2020_india_yearly_jfm.csv",
    )


def mask_path(dgrid: float, paths: StudyPaths | None = None) -> Path:
    paths = paths or get_paths()
    return paths.masks_dir / f"imd_region_masks_{dgrid:g}deg.nc"


def foundation_paths(paths: StudyPaths | None = None) -> dict[str, Path]:
    """Small set of paths expected to exist before benchmark analysis starts."""

    paths = paths or get_paths()
    return {
        "mask_1.5deg": mask_path(1.5, paths),
        "mask_0.5deg": mask_path(0.5, paths),
        "mask_0.25deg": mask_path(0.25, paths),
        "ERA5 climatology": paths.era5_climatology,
        "ERA5 previous daily TP": paths.era5_prev_daily_tp_nc,
        "ERA5 previous daily T2M": paths.era5_prev_daily_t2m_nc,
        "ERA5 JFM2026 daily dir": paths.era5_jfm2026_daily_dir,
        "WeatherBench2 ERA5 zarr": paths.weatherbench2_era5_zarr,
        "IMD climatology dir": paths.imd_climatology_dir,
        "IMD daily climatology": paths.imd_daily_climatology_nc,
        "IMD seasonal climatology": paths.imd_seasonal_climatology_nc,
        "IMD daily area mean": paths.imd_daily_area_mean_csv,
        "IMD JJAS cumulative clim": paths.imd_jjas_cumulative_climatology_csv,
        "IMD JFM cumulative clim": paths.imd_jfm_cumulative_climatology_csv,
        "ECMWF root": paths.ecmwf_root,
        "FuXi root": paths.fuxi_root,
        "UKMO root": paths.ukmo_root,
        "DELYSM root": paths.delysm_root,
        "DELYSM QC root": paths.delysm_qc_root,
        "NCEP root": paths.ncep_root,
        "JFM SPIRE daily zarr": paths.jfm_spire_zarr,
        "JFM FuXi root": paths.jfm_fuxi_root,
        "JFM ECMWF root": paths.jfm_ecmwf_root,
        "JFM ground truth root": paths.jfm_ground_truth_root,
        "JJAS FuXi compact": paths.jjas_fuxi_compact,
        "JJAS2019 new ECMWF root": paths.jjas2019_ecmwf_new_root,
        "JJAS2019 IMD rainfall": paths.jjas2019_imd_rainfall_nc,
    }
