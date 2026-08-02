#!/usr/bin/env python3
"""Combine validated native FuXi shards into a LOYO climatology NetCDF.

This script refuses partial inputs.  Its primary anomaly for hindcast ACC is
leave-one-year-out (LOYO): each year is compared with a climatology calculated
from the other 19 years at the same native initialization slot and lead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest).sort_values("task_index")
    expected = [args.shard_dir / f"{stamp}.nc" for stamp in manifest.init_yyyymmdd.astype(str)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Refusing partial climatology: {len(missing)} native shards are missing")

    manifest["init_slot"] = pd.to_datetime(manifest.init_date).dt.strftime("%m%d")
    slots = sorted(manifest.init_slot.unique())
    years = sorted(manifest.year.unique())
    samples_by_variable: dict[str, list[xr.DataArray]] = {"tp": [], "t2m": []}
    latitude = longitude = None
    for year in years:
        slot_means_by_variable: dict[str, list[xr.DataArray]] = {"tp": [], "t2m": []}
        year_rows = manifest.loc[manifest.year == year].sort_values("init_slot")
        if list(year_rows.init_slot) != slots:
            raise ValueError(f"Year {year} does not contain every archive slot")
        for row in year_rows.itertuples(index=False):
            with xr.open_dataset(args.shard_dir / f"{row.init_yyyymmdd}.nc") as source:
                fields = {"tp": source.tp.mean("member") * 24.0, "t2m": source.t2m.mean("member")}
                for variable, field in fields.items():
                    field = field.load()
                    field.attrs = {"units": "mm day-1" if variable == "tp" else "K"}
                    if latitude is None:
                        latitude, longitude = field.latitude, field.longitude
                    elif not (
                        np.array_equal(field.latitude.values, latitude.values)
                        and np.array_equal(field.longitude.values, longitude.values)
                    ):
                        raise ValueError(f"Grid mismatch in {row.init_yyyymmdd}")
                    slot_means_by_variable[variable].append(field.expand_dims(init_slot=[row.init_slot]))
        for variable in samples_by_variable:
            samples_by_variable[variable].append(
                xr.concat(slot_means_by_variable[variable], dim="init_slot").expand_dims(hindcast_year=[year])
            )
    ensemble_means = {
        variable: xr.concat(samples, dim="hindcast_year").transpose("hindcast_year", "init_slot", "lead_day", "latitude", "longitude")
        for variable, samples in samples_by_variable.items()
    }
    n_years = ensemble_means["tp"].sizes["hindcast_year"]
    if n_years != 20:
        raise ValueError(f"Expected 20 hindcast years; found {n_years}")
    variables: dict[str, xr.DataArray] = {
        "hindcast_sample_count": xr.DataArray(np.full((len(slots), 42), n_years, dtype=np.int16), dims=("init_slot", "lead_day")),
    }
    for variable, ensemble_mean in ensemble_means.items():
        total = ensemble_mean.sum("hindcast_year")
        variables.update(
            {
                f"{variable}_ensemble_mean": ensemble_mean,
                f"{variable}_model_climatology_mean": ensemble_mean.mean("hindcast_year"),
                f"{variable}_model_climatology_std": ensemble_mean.std("hindcast_year", ddof=1),
                f"{variable}_model_climatology_p10": ensemble_mean.quantile(0.10, dim="hindcast_year").drop_vars("quantile"),
                f"{variable}_model_climatology_p90": ensemble_mean.quantile(0.90, dim="hindcast_year").drop_vars("quantile"),
                f"{variable}_model_climatology_loyo": (total - ensemble_mean) / (n_years - 1),
            }
        )
    output = xr.Dataset(
        variables,
        attrs={
            "title": "FuXi-S2S 2002-2021 native-reforecast lead-dependent model climatology",
            "source": "All 51-member native FuXi archive shards; direct extraction, India 1.5 degree grid",
            "climatology_definition": "One 51-member ensemble mean per year; full mean and distribution across 20 years",
            "loyo_definition": "For each hindcast year, mean of the other 19 annual ensemble means at identical native init_slot and lead_day",
            "intended_use": "Model anomaly and ACC calculations. Observations require a separate matched verification-reference climatology.",
            "variable_units": "tp: mm day-1; t2m: K",
            "native_archive_issue_schedule": "3-4 day calendar cadence; not a fixed Monday/Thursday schedule",
        },
    )
    for variable in output.data_vars:
        if variable != "hindcast_sample_count":
            output[variable].attrs["units"] = "mm day-1" if variable.startswith("tp_") else "K"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
        for name in output.data_vars
        if name != "hindcast_sample_count"
    }
    encoding["hindcast_sample_count"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
    output.to_netcdf(temporary, encoding=encoding)
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
