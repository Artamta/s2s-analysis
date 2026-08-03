from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from s2s_benchmark.catalog import build_catalog
from s2s_benchmark.core import StandardField
from s2s_benchmark.storage import validate_store, write_manifest, write_store


def field(source: Path) -> StandardField:
    source.write_bytes(b"source")
    values = np.arange(2 * 7 * 2 * 2, dtype=np.float32).reshape(2, 7, 2, 2)
    return StandardField(
        model="test",
        experiment_id="test/ens2",
        variable="tp",
        initialization="2023-06-29",
        values=values,
        member=np.array([0, 1]),
        lead_day=np.arange(1, 8),
        latitude=np.array([1.5, 0.0]),
        longitude=np.array([60.0, 61.5]),
        units="mm day-1",
        temporal_statistic="daily_mean_rate",
        distribution_representation="members",
        source_paths=(str(source),),
    )


def test_store_is_validated_and_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "2023.zarr"
    manifest = write_store([field(tmp_path / "source.nc")], output, "common_1p5")
    assert validate_store(output)["status"] == "passed"
    try:
        write_store([field(tmp_path / "source.nc")], output, "common_1p5")
    except FileExistsError:
        pass
    else:
        raise AssertionError("completed store overwrite was not rejected")
    assert manifest["zmetadata_sha256"]


def test_catalog_and_init_index_pin_completed_manifest(tmp_path: Path) -> None:
    output = tmp_path / "2023.zarr"
    manifest = write_store([field(tmp_path / "source.nc")], output, "common_1p5")
    manifest_path = tmp_path / "manifests" / "test.json"
    manifest["manifest_path"] = str(manifest_path)
    write_manifest(manifest, manifest_path)
    catalog_path, index_path = build_catalog([manifest_path], tmp_path / "indexes", "pilot")
    second_catalog_path, second_index_path = build_catalog(
        [manifest_path], tmp_path / "indexes", "pilot"
    )
    catalog = json.loads(catalog_path.read_text())
    index = pd.read_parquet(index_path)
    assert len(catalog["records"]) == 1
    assert catalog["records"][0]["store"] == str(output)
    assert index.initialization.tolist() == ["2023-06-29"]
    assert second_catalog_path == catalog_path
    assert second_index_path == index_path


def test_derived_arrays_have_explicit_units(tmp_path: Path) -> None:
    output = tmp_path / "2023.zarr"
    write_store([field(tmp_path / "source.nc")], output, "common_1p5")
    import xarray as xr

    with xr.open_zarr(output, consolidated=True) as ds:
        assert ds["ensemble_mean"].attrs["units"] == "mm day-1"
        assert ds["forecast_weekly_mean"].attrs["units"] == "mm day-1"
        assert ds["forecast_weekly_total"].attrs["units"] == "mm"
        assert ds["ensemble_mean_weekly_total"].attrs["units"] == "mm"


def test_multi_init_store_records_every_source_path(tmp_path: Path) -> None:
    first = field(tmp_path / "source_one.nc")
    second = replace(
        field(tmp_path / "source_two.nc"), initialization="2023-07-03"
    )
    output = tmp_path / "2023.zarr"
    write_store([first, second], output, "common_1p5")
    import xarray as xr

    with xr.open_zarr(output, consolidated=True) as ds:
        assert json.loads(ds.attrs["source_paths"]) == sorted(
            [str(tmp_path / "source_one.nc"), str(tmp_path / "source_two.nc")]
        )
