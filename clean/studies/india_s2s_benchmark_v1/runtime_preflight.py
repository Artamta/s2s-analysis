from __future__ import annotations

import importlib
import json
import platform


MODULES = (
    "numpy",
    "xarray",
    "pandas",
    "pyarrow",
    "h5py",
    "h5netcdf",
    "matplotlib",
    "scipy",
    "cfgrib",
    "eccodes",
    "zarr",
    "numcodecs",
)


def main() -> None:
    versions = {}
    for name in MODULES:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    major = int(versions["zarr"].split(".", maxsplit=1)[0])
    if major != 2:
        raise RuntimeError(f"archive writer requires Zarr 2, found {versions['zarr']}")
    print(
        json.dumps(
            {
                "runtime_preflight": "passed",
                "host": platform.node(),
                "python": platform.python_version(),
                "modules": versions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
