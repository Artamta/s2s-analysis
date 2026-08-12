"""Leakage-safe data preparation for the FuXi--IMERG neural adapter.

The archive already contains FuXi-S2S precipitation as complete seven-day
means.  IMERG is daily, so this module constructs matching observation periods
explicitly.  FuXi lead day 1 covers ``[initialization, initialization + 1d)``;
IMERG's date labels that interval by its *start*.  The matching IMERG dates are
therefore ``initialization + 0..41 days``, not the FuXi period-end labels
``initialization + 1..42 days``.  The one-dimensional ``valid_time`` coordinate
in an annual FuXi store also cannot describe more than one initialization.

The learning target is a residual in a variance-stabilising space::

    target = log1p(IMERG weekly mean) - log1p(FuXi weekly ensemble mean)

Adding the predicted residual back to ``log1p(FuXi)`` makes an untrained,
zero-output adapter exactly reproduce the raw forecast.  A fixed 2001--2019
IMERG climatology is also exposed as an input.  It is pre-verification and is
never recomputed from the 2020--2024 experiment period.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


DEFAULT_ARCHIVE_ROOT = Path(
    "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
    "india_s2s_benchmark_v1"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

FORECAST_RELATIVE_PATH = Path(
    "forecasts/fuxi_s2s/"
    "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/"
    "tp/common_1p5"
)
T2M_FORECAST_RELATIVE_PATH = Path(
    "forecasts/fuxi_s2s/"
    "model-run__fuxi__fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/"
    "t2m/common_1p5"
)
FUXI_EXPERIMENT_ID = (
    "model-run/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
)
IMERG_DAILY_RELATIVE_PATH = Path(
    "observations/ground_truth_v1/daily/imerg/tp/india_1p5_27x27_v1"
)
IMERG_CLIMATOLOGY_RELATIVE_PATH = Path(
    "observations/ground_truth_v1/climatologies/imerg_2001_2019.zarr"
)
SPATIAL_RELATIVE_PATH = Path("spatial/spatial_support.zarr")

FORECAST_YEARS: Tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
EXPECTED_FORECAST_COUNTS: Mapping[int, int] = {
    2020: 105,
    2021: 104,
    2022: 104,
    2023: 104,
    2024: 100,
}
EXPECTED_SPLIT_COUNTS: Mapping[str, int] = {
    "train": 302,
    "validation": 93,
    "test": 100,
}
REGION_FRACTION_VARIABLES: Mapping[str, str] = {
    "northwest_india": "northwest_india_fraction",
    "central_india": "central_india_fraction",
    "south_peninsula": "south_peninsula_fraction",
    "east_northeast_india": "east_northeast_india_fraction",
}


class DataIntegrityError(ValueError):
    """Raised when an archive violates a scientific data contract."""


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations of the four immutable source products."""

    archive_root: Path = DEFAULT_ARCHIVE_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "archive_root", Path(self.archive_root))

    @property
    def forecast_root(self) -> Path:
        return self.archive_root / FORECAST_RELATIVE_PATH

    @property
    def t2m_forecast_root(self) -> Path:
        """Annual FuXi T2M stores aligned with :attr:`forecast_root`."""

        return self.archive_root / T2M_FORECAST_RELATIVE_PATH

    @property
    def imerg_daily_root(self) -> Path:
        return self.archive_root / IMERG_DAILY_RELATIVE_PATH

    @property
    def imerg_climatology_store(self) -> Path:
        return self.archive_root / IMERG_CLIMATOLOGY_RELATIVE_PATH

    @property
    def spatial_store(self) -> Path:
        return self.archive_root / SPATIAL_RELATIVE_PATH

    def required_stores(self) -> List[Path]:
        stores = [self.forecast_root / (str(year) + ".zarr") for year in FORECAST_YEARS]
        # Verification for the final 30 December 2024 initialization ends on
        # 10 February 2025, hence the six daily stores below.
        stores.extend(
            self.imerg_daily_root / (str(year) + ".zarr")
            for year in range(2020, 2026)
        )
        stores.extend([self.imerg_climatology_store, self.spatial_store])
        return stores


@dataclass
class SplitArrays:
    """Aligned arrays for one chronological split.

    Dynamic fields have shape ``[initialization, lead_week, latitude,
    longitude]``.  ``valid_dates`` retains the seven IMERG observation-period
    start dates as ``[initialization, lead_week, day_in_week]``.  Each daily
    period is half-open, ``[valid_date, valid_date + 1 day)``.
    """

    name: str
    initializations: np.ndarray
    valid_dates: np.ndarray
    fuxi_mean: np.ndarray
    fuxi_std: np.ndarray
    imerg_truth: np.ndarray
    imerg_climatology: np.ndarray
    log_fuxi: np.ndarray
    log_spread: np.ndarray
    log_climatology: np.ndarray
    fuxi_log_anomaly: np.ndarray
    target_log_residual: np.ndarray
    valid_mask: np.ndarray
    season_sin: np.ndarray
    season_cos: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.initializations.shape[0])

    @property
    def lead_count(self) -> int:
        return int(self.fuxi_mean.shape[1])

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (int(self.fuxi_mean.shape[-2]), int(self.fuxi_mean.shape[-1]))

    @property
    def verification_end(self) -> np.ndarray:
        """Exclusive end of the final daily verification period."""

        return self.valid_dates[:, -1, -1] + np.timedelta64(1, "D")


@dataclass(frozen=True)
class NormalizationStats:
    """Training-only, area-weighted per-lead normalization constants."""

    field_names: Tuple[str, ...]
    feature_mean: np.ndarray  # [feature, lead]
    feature_std: np.ndarray  # [feature, lead]
    # The target centre is deliberately zero so a zero neural residual is the
    # identity/raw-FuXi forecast.  target_std is an area-weighted RMS scale.
    target_mean: np.ndarray  # [lead], always zero for the identity adapter
    target_std: np.ndarray  # [lead]
    fitted_split: str = "train"

    def to_dict(self) -> Dict[str, object]:
        return {
            "field_names": list(self.field_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "fitted_split": self.fitted_split,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "NormalizationStats":
        return cls(
            field_names=tuple(str(value) for value in values["field_names"]),
            feature_mean=np.asarray(values["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(values["feature_std"], dtype=np.float32),
            target_mean=np.asarray(values["target_mean"], dtype=np.float32),
            target_std=np.asarray(values["target_std"], dtype=np.float32),
            fitted_split=str(values.get("fitted_split", "train")),
        )

    def save_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ModelArrays:
    """Finite tensors ready for a framework-specific Dataset wrapper."""

    inputs: np.ndarray  # [initialization, lead_week, channel, lat, lon]
    target: np.ndarray  # [initialization, lead_week, lat, lon]
    mask: np.ndarray  # same as target
    weight: np.ndarray  # same as target; mean one across valid cells
    initializations: np.ndarray
    channel_names: Tuple[str, ...]


@dataclass(frozen=True)
class AuditReport:
    """Small JSON-ready record of the exact data used by an experiment."""

    forecast_counts_by_year: Mapping[int, int]
    split_counts: Mapping[str, int]
    split_first_initialization: Mapping[str, str]
    split_last_initialization: Mapping[str, str]
    split_last_verification: Mapping[str, str]
    imerg_daily_years: Tuple[int, ...]
    imerg_last_available_date: str
    climatology_baseline: str
    climatology_day_count: int
    grid_shape: Tuple[int, int]
    lead_week_count: int
    imerg_support_cell_count: int
    positive_weight_cell_count: int
    finite_target_fraction_on_support: float
    fuxi_min_mm_day: float
    fuxi_max_mm_day: float
    imerg_min_mm_day: float
    imerg_max_mm_day: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "forecast_counts_by_year": {
                str(key): int(value) for key, value in self.forecast_counts_by_year.items()
            },
            "split_counts": dict(self.split_counts),
            "split_first_initialization": dict(self.split_first_initialization),
            "split_last_initialization": dict(self.split_last_initialization),
            "split_last_verification": dict(self.split_last_verification),
            "imerg_daily_years": list(self.imerg_daily_years),
            "imerg_last_available_date": self.imerg_last_available_date,
            "climatology_baseline": self.climatology_baseline,
            "climatology_day_count": self.climatology_day_count,
            "grid_shape": list(self.grid_shape),
            "lead_week_count": self.lead_week_count,
            "imerg_support_cell_count": self.imerg_support_cell_count,
            "positive_weight_cell_count": self.positive_weight_cell_count,
            "finite_target_fraction_on_support": self.finite_target_fraction_on_support,
            "fuxi_min_mm_day": self.fuxi_min_mm_day,
            "fuxi_max_mm_day": self.fuxi_max_mm_day,
            "imerg_min_mm_day": self.imerg_min_mm_day,
            "imerg_max_mm_day": self.imerg_max_mm_day,
        }


@dataclass
class AdapterData:
    """Complete experiment data and immutable spatial metadata."""

    train: SplitArrays
    validation: SplitArrays
    test: SplitArrays
    latitude: np.ndarray
    longitude: np.ndarray
    observation_fraction: np.ndarray
    area_weight_km2: np.ndarray
    region_weight_km2: Mapping[str, np.ndarray]
    source_manifest: Mapping[str, object]
    audit: AuditReport

    @property
    def splits(self) -> Mapping[str, SplitArrays]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class FuxiT2MSplitArrays:
    """FuXi T2M predictors aligned to one existing adapter split.

    Both fields are complete seven-day means with shape
    ``[initialization, lead_week, latitude, longitude]``.  No normalization is
    applied here; an experiment must fit any scaling on ``train`` only.
    """

    name: str
    initializations: np.ndarray
    ensemble_mean_weekly: np.ndarray
    ensemble_std_weekly: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.initializations.size)

    @property
    def lead_count(self) -> int:
        return int(self.ensemble_mean_weekly.shape[1])

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (
            int(self.ensemble_mean_weekly.shape[-2]),
            int(self.ensemble_mean_weekly.shape[-1]),
        )


@dataclass(frozen=True)
class FuxiT2MData:
    """Leakage-safe auxiliary FuXi T2M data and its source provenance."""

    train: FuxiT2MSplitArrays
    validation: FuxiT2MSplitArrays
    test: FuxiT2MSplitArrays
    latitude: np.ndarray
    longitude: np.ndarray
    source_manifest: Mapping[str, object]

    @property
    def splits(self) -> Mapping[str, FuxiT2MSplitArrays]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class FuxiTPDistributionSplitArrays:
    """Opt-in ensemble-distribution predictors for one adapter split.

    Each float32 field has shape
    ``[initialization, lead_week, latitude, longitude]``.  The median anomaly
    and exceedance probability are zero-filled outside
    ``climatology_threshold_support`` in :class:`FuxiTPDistributionData`, where
    IMERG does not define a threshold.
    """

    name: str
    initializations: np.ndarray
    member_log_median_anomaly: np.ndarray
    member_log_iqr: np.ndarray
    probability_exceeds_imerg_climatology: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.initializations.size)

    @property
    def lead_count(self) -> int:
        return int(self.member_log_median_anomaly.shape[1])

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (
            int(self.member_log_median_anomaly.shape[-2]),
            int(self.member_log_median_anomaly.shape[-1]),
        )


@dataclass(frozen=True)
class FuxiTPDistributionData:
    """Aligned frozen TP distribution features and source provenance."""

    train: FuxiTPDistributionSplitArrays
    validation: FuxiTPDistributionSplitArrays
    test: FuxiTPDistributionSplitArrays
    latitude: np.ndarray
    longitude: np.ndarray
    climatology_threshold_support: np.ndarray
    source_manifest: Mapping[str, object]

    @property
    def splits(self) -> Mapping[str, FuxiTPDistributionSplitArrays]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def derive_valid_dates(initializations: np.ndarray) -> np.ndarray:
    """Return IMERG period starts for six complete seven-day blocks.

    IMERG ``time=D`` covers ``[D, D+1 day)``.  Consequently, a forecast issued
    at 00 UTC on ``initialization`` matches IMERG starts ``initialization+0``
    through ``initialization+41``.  FuXi calls the corresponding period ends
    lead days 1 through 42.
    """

    initializations = np.asarray(initializations, dtype="datetime64[D]")
    if initializations.ndim != 1:
        raise ValueError("initializations must be one-dimensional")
    offsets = np.arange(0, 42, dtype="timedelta64[D]").reshape(1, 6, 7)
    return initializations[:, None, None] + offsets


def collapse_observation_fraction(
    values: np.ndarray, spatial_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Collapse a 2-D or time-repeated 3-D support fraction to 2-D.

    Some historical ground-truth stores repeated this static field along time.
    Silently selecting the first slice is unsafe, so all repeated slices are
    required to match (including their NaN pattern).
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        if spatial_shape is not None and array.shape != spatial_shape:
            raise DataIntegrityError(
                "observation_fraction has shape %s; expected %s"
                % (array.shape, spatial_shape)
            )
        return array.copy()
    if array.ndim != 3:
        raise DataIntegrityError(
            "observation_fraction must be 2-D or repeated 3-D, got %d-D" % array.ndim
        )

    if spatial_shape is None:
        spatial_shape = (int(array.shape[-2]), int(array.shape[-1]))
    if tuple(array.shape[-2:]) == tuple(spatial_shape):
        repeated = array.reshape(-1, spatial_shape[0], spatial_shape[1])
    elif tuple(array.shape[:2]) == tuple(spatial_shape):
        repeated = np.moveaxis(array, -1, 0)
    else:
        raise DataIntegrityError(
            "cannot identify spatial axes in observation_fraction shape %s" % (array.shape,)
        )
    first = repeated[0]
    if not np.allclose(repeated, first[None, ...], rtol=0.0, atol=1.0e-7, equal_nan=True):
        raise DataIntegrityError("observation_fraction changes along its repeated dimension")
    return first.copy()


def calendar_positions_for_dates(
    dates: np.ndarray, climatology_dates: np.ndarray
) -> np.ndarray:
    """Map dates to a fixed 366-day month/day climatology calendar."""

    climate = np.asarray(climatology_dates, dtype="datetime64[D]").reshape(-1)
    if climate.size != 366:
        raise DataIntegrityError(
            "climatology calendar contains %d days; expected 366" % climate.size
        )
    lookup: Dict[str, int] = {}
    for index, value in enumerate(climate):
        key = np.datetime_as_string(value, unit="D")[5:]
        if key in lookup:
            raise DataIntegrityError("duplicate month/day in climatology: " + key)
        lookup[key] = index
    if "02-29" not in lookup:
        raise DataIntegrityError("366-day climatology does not contain 29 February")

    flat_dates = np.asarray(dates, dtype="datetime64[D]").reshape(-1)
    try:
        positions = np.asarray(
            [lookup[np.datetime_as_string(value, unit="D")[5:]] for value in flat_dates],
            dtype=np.int16,
        )
    except KeyError as error:
        raise DataIntegrityError("date missing from climatology calendar: " + str(error))
    return positions.reshape(np.asarray(dates).shape)


def purged_split_indices(initializations: np.ndarray) -> Mapping[str, np.ndarray]:
    """Create the fixed chronological 2020--2024 experiment split.

    Verification intervals are half-open.  Training and validation cases are
    retained when the complete interval ``[init, init+42 days)`` ends at or
    before the next split's first issue time.  Equality is safe: the final
    training observation interval has ended when the next forecast is issued.
    The locked 2024 test retains every initialization because its final IMERG
    interval ends on 10 February 2025.
    """

    initializations = np.asarray(initializations, dtype="datetime64[D]")
    if initializations.ndim != 1 or initializations.size == 0:
        raise ValueError("initializations must be a non-empty one-dimensional array")
    if np.unique(initializations).size != initializations.size:
        raise DataIntegrityError("forecast initializations are not unique")

    years = initializations.astype("datetime64[Y]").astype(np.int64) + 1970
    validation_candidates = np.flatnonzero(years == 2023)
    test_indices = np.flatnonzero(years == 2024)
    if validation_candidates.size == 0 or test_indices.size == 0:
        raise DataIntegrityError("both 2023 validation and 2024 test forecasts are required")
    first_validation = initializations[validation_candidates].min()
    first_test = initializations[test_indices].min()
    verification_end = initializations + np.timedelta64(42, "D")

    train_indices = np.flatnonzero(
        (years >= 2020) & (years <= 2022) & (verification_end <= first_validation)
    )
    validation_indices = validation_candidates[
        verification_end[validation_candidates] <= first_test
    ]
    return {
        "train": train_indices.astype(np.int64),
        "validation": validation_indices.astype(np.int64),
        "test": test_indices.astype(np.int64),
    }


def _as_daily_fraction(dataset: xr.Dataset, spatial_shape: Tuple[int, int]) -> np.ndarray:
    fraction = dataset["observation_fraction"]
    if "latitude" not in fraction.dims or "longitude" not in fraction.dims:
        raise DataIntegrityError("observation_fraction lacks latitude/longitude dimensions")
    other_dims = [
        dim for dim in fraction.dims if dim not in ("latitude", "longitude")
    ]
    ordered = fraction.transpose(*(other_dims + ["latitude", "longitude"]))
    return collapse_observation_fraction(ordered.load().values, spatial_shape)


def _check_same_grid(
    latitude: np.ndarray,
    longitude: np.ndarray,
    candidate_latitude: np.ndarray,
    candidate_longitude: np.ndarray,
    source: Path,
) -> None:
    if not np.array_equal(latitude, np.asarray(candidate_latitude, dtype=np.float64)):
        raise DataIntegrityError("latitude grid differs in " + str(source))
    if not np.array_equal(longitude, np.asarray(candidate_longitude, dtype=np.float64)):
        raise DataIntegrityError("longitude grid differs in " + str(source))


def _require_metadata(
    attributes: Mapping[str, object],
    expected: Mapping[str, object],
    source: Path,
    context: str,
) -> None:
    """Require metadata that controls units or temporal interpretation."""

    for key, expected_value in expected.items():
        actual = attributes.get(key)
        if actual != expected_value:
            raise DataIntegrityError(
                "%s %s metadata %s=%r; expected %r"
                % (source, context, key, actual, expected_value)
            )


def _load_forecasts(
    paths: DataPaths,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[int, int]]:
    means: List[np.ndarray] = []
    spreads: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    latitude: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    counts: Dict[int, int] = {}

    for year in FORECAST_YEARS:
        store = paths.forecast_root / (str(year) + ".zarr")
        with xr.open_zarr(str(store), consolidated=True) as dataset:
            _require_metadata(
                dataset.attrs,
                {
                    "archive_id": "india_s2s_benchmark_v1",
                    "model": "fuxi_s2s",
                    "variable": "tp",
                    "distribution_representation": "members",
                    "ensemble_std_ddof": 0,
                },
                store,
                "global",
            )
            required = {
                "ensemble_mean_weekly",
                "ensemble_std_weekly",
                "ensemble_member_count_weekly",
                "init",
            }
            missing = required.difference(dataset.variables)
            if missing:
                raise DataIntegrityError(
                    "%s is missing variables: %s" % (store, sorted(missing))
                )
            lead_weeks = np.asarray(dataset["lead_week"].values, dtype=np.int16)
            if not np.array_equal(lead_weeks, np.arange(1, 7, dtype=np.int16)):
                raise DataIntegrityError("FuXi lead_week coordinate must be exactly 1..6")
            for variable in ("ensemble_mean_weekly", "ensemble_std_weekly"):
                _require_metadata(
                    dataset[variable].attrs,
                    {
                        "units": "mm day-1",
                        "temporal_statistic": "mean_of_complete_7_day_block",
                    },
                    store,
                    variable,
                )
            if int(dataset.sizes.get("member", -1)) != 50:
                raise DataIntegrityError("FuXi archive must contain exactly 50 members")
            store_latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
            store_longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
            if latitude is None:
                latitude = store_latitude
                longitude = store_longitude
            else:
                _check_same_grid(
                    latitude, longitude, store_latitude, store_longitude, store
                )
            init_ns = np.asarray(dataset["init"].values, dtype="datetime64[ns]")
            if np.any(init_ns != init_ns.astype("datetime64[D]")):
                raise DataIntegrityError("FuXi initializations must all be 00 UTC")
            init = init_ns.astype("datetime64[D]")
            mean = np.asarray(dataset["ensemble_mean_weekly"].load().values, dtype=np.float32)
            spread = np.asarray(dataset["ensemble_std_weekly"].load().values, dtype=np.float32)
            member_count = np.asarray(
                dataset["ensemble_member_count_weekly"].load().values
            )
            if mean.shape != spread.shape or mean.shape != (
                init.size,
                6,
                store_latitude.size,
                store_longitude.size,
            ):
                raise DataIntegrityError("unexpected FuXi array shape in " + str(store))
            if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(spread)):
                raise DataIntegrityError("FuXi weekly mean/spread contains non-finite values")
            if np.any(mean < 0.0) or np.any(spread < 0.0):
                raise DataIntegrityError("FuXi weekly mean/spread contains negative values")
            if member_count.shape != mean.shape or not np.all(member_count == 50):
                raise DataIntegrityError("FuXi weekly statistics do not use all 50 members")
            dates.append(init)
            means.append(mean)
            spreads.append(spread)
            counts[year] = int(init.size)

    if latitude is None or longitude is None:
        raise RuntimeError("no forecast stores were loaded")
    all_dates = np.concatenate(dates)
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    if np.unique(all_dates).size != all_dates.size:
        raise DataIntegrityError("duplicate FuXi initializations across annual stores")
    return (
        all_dates,
        np.concatenate(means, axis=0)[order],
        np.concatenate(spreads, axis=0)[order],
        latitude,
        longitude,
        counts,
    )


def _load_imerg_truth(
    paths: DataPaths,
    valid_dates: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], str]:
    flat_valid_dates = valid_dates.reshape(-1)
    requested_years_array = (
        flat_valid_dates.astype("datetime64[Y]").astype(np.int64) + 1970
    )
    requested_years = tuple(
        int(value) for value in np.unique(requested_years_array)
    )
    daily_values: List[np.ndarray] = []
    daily_dates: List[np.ndarray] = []
    observation_fraction: Optional[np.ndarray] = None
    spatial_shape = (latitude.size, longitude.size)

    for year in requested_years:
        store = paths.imerg_daily_root / (str(year) + ".zarr")
        with xr.open_zarr(str(store), consolidated=True) as dataset:
            _require_metadata(
                dataset.attrs,
                {
                    "dataset_id": "india_s2s_ground_truth_v1",
                    "source": "imerg",
                    "variable": "tp",
                    "product": "GPM_3IMERGDF.07",
                    "revision": "V07B",
                    "units": "mm day-1",
                    "temporal_statistic": (
                        "00-24 UTC daily accumulation expressed as mean daily rate"
                    ),
                    "source_half_hour_count_min": 48,
                    "source_half_hour_count_max": 48,
                },
                store,
                "global",
            )
            _check_same_grid(
                latitude,
                longitude,
                dataset["latitude"].values,
                dataset["longitude"].values,
                store,
            )
            values = np.asarray(dataset["observation"].load().values, dtype=np.float32)
            dates = np.asarray(dataset["time"].values, dtype="datetime64[D]")
            if values.shape != (dates.size, latitude.size, longitude.size):
                raise DataIntegrityError("unexpected IMERG observation shape in " + str(store))
            if dates.size == 0 or np.unique(dates).size != dates.size:
                raise DataIntegrityError("IMERG annual dates must be non-empty and unique")
            if dates.size > 1 and not np.all(np.diff(dates) == np.timedelta64(1, "D")):
                raise DataIntegrityError("IMERG annual dates are not daily and contiguous")
            fraction = _as_daily_fraction(dataset, spatial_shape)
            represented = fraction > 0.0
            if np.any(values[:, represented] < 0.0):
                raise DataIntegrityError("IMERG represented precipitation is negative")
            if observation_fraction is None:
                observation_fraction = fraction
            elif not np.allclose(
                observation_fraction, fraction, rtol=0.0, atol=1.0e-7, equal_nan=True
            ):
                raise DataIntegrityError("IMERG observation support changes between years")
            daily_values.append(values)
            daily_dates.append(dates)

    if observation_fraction is None:
        raise RuntimeError("no IMERG stores were loaded")
    all_dates = np.concatenate(daily_dates)
    all_values = np.concatenate(daily_values, axis=0)
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_values = all_values[order]
    if np.unique(all_dates).size != all_dates.size:
        raise DataIntegrityError("duplicate dates in IMERG daily archive")

    positions = np.searchsorted(all_dates, flat_valid_dates)
    in_range = positions < all_dates.size
    matched = np.zeros_like(in_range, dtype=bool)
    matched[in_range] = all_dates[positions[in_range]] == flat_valid_dates[in_range]
    if not np.all(matched):
        missing = np.unique(flat_valid_dates[~matched])
        preview = ", ".join(np.datetime_as_string(value, unit="D") for value in missing[:5])
        raise DataIntegrityError("missing IMERG verification dates: " + preview)

    daily = all_values[positions].reshape(
        valid_dates.shape + (latitude.size, longitude.size)
    )
    # A simple mean is deliberate: one absent daily value invalidates the
    # entire weekly cell instead of silently turning it into a partial week.
    weekly = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    return (
        weekly,
        observation_fraction,
        requested_years,
        np.datetime_as_string(all_dates[-1], unit="D"),
    )


def _load_imerg_climatology(
    paths: DataPaths,
    valid_dates: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    observation_fraction: np.ndarray,
) -> Tuple[np.ndarray, str, int]:
    store = paths.imerg_climatology_store
    with xr.open_zarr(str(store), consolidated=True) as dataset:
        _require_metadata(
            dataset.attrs,
            {
                "source": "imerg",
                "baseline": "2001-2019",
                "window_days": 31,
                "year_weighting": "equal",
                "purpose": "pre-verification IMERG precipitation normal",
            },
            store,
            "global",
        )
        _check_same_grid(
            latitude,
            longitude,
            dataset["latitude"].values,
            dataset["longitude"].values,
            store,
        )
        baseline = str(dataset.attrs.get("baseline", ""))
        if baseline != "2001-2019":
            raise DataIntegrityError(
                "expected fixed IMERG baseline 2001-2019, found " + repr(baseline)
            )
        climate_dates = np.asarray(
            dataset["climatology_date"].values, dtype="datetime64[D]"
        )
        positions = calendar_positions_for_dates(valid_dates, climate_dates)
        climatology = np.asarray(dataset["climatology_mean"].load().values, dtype=np.float32)
        if climatology.shape != (366, latitude.size, longitude.size):
            raise DataIntegrityError("unexpected IMERG climatology shape")
        climate_fraction = _as_daily_fraction(
            dataset, (latitude.size, longitude.size)
        )
        if not np.allclose(
            observation_fraction,
            climate_fraction,
            rtol=0.0,
            atol=1.0e-7,
            equal_nan=True,
        ):
            raise DataIntegrityError("IMERG climatology and daily support differ")
    daily = climatology[positions]
    weekly = np.mean(daily, axis=2, dtype=np.float64).astype(np.float32)
    return weekly, baseline, int(climate_dates.size)


def _load_spatial_weights(
    paths: DataPaths,
    latitude: np.ndarray,
    longitude: np.ndarray,
    observation_fraction: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, np.ndarray]]:
    store = paths.spatial_store
    with xr.open_zarr(str(store), consolidated=True) as dataset:
        _require_metadata(
            dataset.attrs,
            {
                "archive_id": "india_s2s_benchmark_v1",
                "common_grid_id": "india_1p5_27x27_v1",
                "weight_contract": (
                    "india_area_weight_km2 = cell_area_km2 * india_fraction"
                ),
            },
            store,
            "global",
        )
        _check_same_grid(
            latitude,
            longitude,
            dataset["latitude"].values,
            dataset["longitude"].values,
            store,
        )
        cell_area = np.asarray(dataset["cell_area_km2"].load().values, dtype=np.float64)
        india_area = np.asarray(
            dataset["india_area_weight_km2"].load().values, dtype=np.float64
        )
        area_weight = india_area * observation_fraction.astype(np.float64)
        regions: Dict[str, np.ndarray] = {"india": area_weight}
        for name, variable in REGION_FRACTION_VARIABLES.items():
            fraction = np.asarray(dataset[variable].load().values, dtype=np.float64)
            regions[name] = cell_area * fraction * observation_fraction.astype(np.float64)
    if np.any(~np.isfinite(area_weight)) or np.any(area_weight < 0.0):
        raise DataIntegrityError("invalid India/IMERG area weights")
    if not np.any(area_weight > 0.0):
        raise DataIntegrityError("India/IMERG area weights contain no positive cells")
    return area_weight, regions


def _seasonal_encoding(initializations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Fourth observation-period start in each block: offsets 3, 10, ..., 38.
    midpoint_offsets = np.asarray([3, 10, 17, 24, 31, 38], dtype="timedelta64[D]")
    midpoints = initializations[:, None] + midpoint_offsets[None, :]
    year_start = midpoints.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (midpoints - year_start).astype(np.int16).astype(np.float32)
    angle = 2.0 * np.pi * day_of_year / np.float32(365.2425)
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def _make_split(
    name: str,
    indices: np.ndarray,
    initializations: np.ndarray,
    valid_dates: np.ndarray,
    fuxi_mean: np.ndarray,
    fuxi_std: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    observation_fraction: np.ndarray,
) -> SplitArrays:
    split_initializations = initializations[indices]
    split_valid_dates = valid_dates[indices]
    mean = fuxi_mean[indices].astype(np.float32, copy=False)
    spread = fuxi_std[indices].astype(np.float32, copy=False)
    target = truth[indices].astype(np.float32, copy=False)
    climate = climatology[indices].astype(np.float32, copy=False)

    log_fuxi = np.log1p(mean).astype(np.float32)
    log_spread = np.log1p(spread).astype(np.float32)
    log_climate = np.log1p(climate).astype(np.float32)
    log_truth = np.log1p(target).astype(np.float32)
    anomaly = (log_fuxi - log_climate).astype(np.float32)
    residual = (log_truth - log_fuxi).astype(np.float32)
    support = observation_fraction > 0.0
    valid = support[None, None, :, :] & np.isfinite(mean) & np.isfinite(spread)
    valid &= np.isfinite(target) & np.isfinite(climate) & np.isfinite(residual)
    season_sin, season_cos = _seasonal_encoding(split_initializations)
    return SplitArrays(
        name=name,
        initializations=split_initializations,
        valid_dates=split_valid_dates,
        fuxi_mean=mean,
        fuxi_std=spread,
        imerg_truth=target,
        imerg_climatology=climate,
        log_fuxi=log_fuxi,
        log_spread=log_spread,
        log_climatology=log_climate,
        fuxi_log_anomaly=anomaly,
        target_log_residual=residual,
        valid_mask=valid,
        season_sin=season_sin,
        season_cos=season_cos,
    )


def _weighted_moments_per_lead(
    values: np.ndarray,
    valid_mask: np.ndarray,
    area_weight_km2: np.ndarray,
    minimum_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != mask.shape or values.ndim != 4:
        raise ValueError("values and mask must have matching [N, lead, H, W] shapes")
    base_weight = np.asarray(area_weight_km2, dtype=np.float64)[None, :, :]
    means = np.empty(values.shape[1], dtype=np.float32)
    standard_deviations = np.empty(values.shape[1], dtype=np.float32)
    for lead in range(values.shape[1]):
        weight = np.where(mask[:, lead], base_weight, 0.0)
        denominator = weight.sum(dtype=np.float64)
        if denominator <= 0.0:
            raise DataIntegrityError("normalization lead has no valid positive-weight cells")
        mean = np.sum(np.where(mask[:, lead], values[:, lead], 0.0) * weight) / denominator
        variance = (
            np.sum(
                np.where(mask[:, lead], (values[:, lead] - mean) ** 2, 0.0) * weight
            )
            / denominator
        )
        means[lead] = np.float32(mean)
        standard_deviations[lead] = np.float32(max(np.sqrt(variance), minimum_std))
    return means, standard_deviations


def fit_normalization(
    train: SplitArrays,
    area_weight_km2: np.ndarray,
    field_names: Sequence[str] = ("log_fuxi", "log_spread", "log_climatology"),
    minimum_std: float = 1.0e-6,
) -> NormalizationStats:
    """Fit per-lead scalers using only the training split.

    Passing validation or test data is rejected by name.  This guard is small
    but prevents an easy-to-miss preprocessing leak in rapid experiments.
    """

    if train.name != "train":
        raise ValueError("normalization must be fitted on the train split")
    feature_means: List[np.ndarray] = []
    feature_stds: List[np.ndarray] = []
    for name in field_names:
        if not hasattr(train, name):
            raise ValueError("unknown normalization field: " + name)
        mean, standard_deviation = _weighted_moments_per_lead(
            getattr(train, name), train.valid_mask, area_weight_km2, minimum_std
        )
        feature_means.append(mean)
        feature_stds.append(standard_deviation)
    empirical_target_mean, centered_target_std = _weighted_moments_per_lead(
        train.target_log_residual,
        train.valid_mask,
        area_weight_km2,
        minimum_std,
    )
    # Scale without centring.  Besides making the semantics easy to explain,
    # this guarantees that a zero-initialized final layer exactly reproduces
    # raw FuXi.  RMS = sqrt(variance + mean**2).
    target_std = np.maximum(
        np.sqrt(centered_target_std ** 2 + empirical_target_mean ** 2),
        np.float32(minimum_std),
    ).astype(np.float32)
    target_mean = np.zeros_like(target_std, dtype=np.float32)
    return NormalizationStats(
        field_names=tuple(field_names),
        feature_mean=np.stack(feature_means).astype(np.float32),
        feature_std=np.stack(feature_stds).astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
        fitted_split="train",
    )


def make_model_arrays(
    split: SplitArrays,
    stats: NormalizationStats,
    latitude: np.ndarray,
    longitude: np.ndarray,
    area_weight_km2: np.ndarray,
) -> ModelArrays:
    """Build normalized finite arrays, including static and calendar channels.

    The first channels are the fields recorded in ``stats.field_names``.
    Latitude, longitude, seasonal sine/cosine, lead, and the IMERG/India mask
    follow.  Invalid target cells are filled with zero but carry zero loss
    weight, so no missing value reaches a neural network or its loss function.
    """

    n_samples, n_leads, height, width = split.fuxi_mean.shape
    if stats.feature_mean.shape != (len(stats.field_names), n_leads):
        raise ValueError("normalization feature shape does not match split")
    dynamic_channels: List[np.ndarray] = []
    for field_index, name in enumerate(stats.field_names):
        values = np.asarray(getattr(split, name), dtype=np.float32)
        mean = stats.feature_mean[field_index][None, :, None, None]
        standard_deviation = stats.feature_std[field_index][None, :, None, None]
        normalized = (values - mean) / standard_deviation
        dynamic_channels.append(np.where(np.isfinite(normalized), normalized, 0.0))

    latitude = np.asarray(latitude, dtype=np.float32)
    longitude = np.asarray(longitude, dtype=np.float32)
    latitude_scaled = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    longitude_scaled = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    lat_grid = np.broadcast_to(
        latitude_scaled[None, None, :, None], (n_samples, n_leads, height, width)
    )
    lon_grid = np.broadcast_to(
        longitude_scaled[None, None, None, :], (n_samples, n_leads, height, width)
    )
    season_sin = np.broadcast_to(
        split.season_sin[:, :, None, None], (n_samples, n_leads, height, width)
    )
    season_cos = np.broadcast_to(
        split.season_cos[:, :, None, None], (n_samples, n_leads, height, width)
    )
    lead_scaled = np.linspace(-1.0, 1.0, n_leads, dtype=np.float32)
    lead_grid = np.broadcast_to(
        lead_scaled[None, :, None, None], (n_samples, n_leads, height, width)
    )
    support_grid = np.broadcast_to(
        (area_weight_km2 > 0.0)[None, None, :, :],
        (n_samples, n_leads, height, width),
    ).astype(np.float32)
    dynamic_channels.extend(
        [lat_grid, lon_grid, season_sin, season_cos, lead_grid, support_grid]
    )
    inputs = np.stack(dynamic_channels, axis=2).astype(np.float32)

    target = (
        split.target_log_residual - stats.target_mean[None, :, None, None]
    ) / stats.target_std[None, :, None, None]
    target = np.where(split.valid_mask & np.isfinite(target), target, 0.0).astype(np.float32)
    mask = split.valid_mask.astype(bool, copy=True)
    positive = np.asarray(area_weight_km2, dtype=np.float64) > 0.0
    weight_2d = np.zeros_like(area_weight_km2, dtype=np.float64)
    weight_2d[positive] = area_weight_km2[positive] / area_weight_km2[positive].mean()
    weight = np.broadcast_to(
        weight_2d[None, None, :, :], mask.shape
    ).copy()
    weight[~mask] = 0.0
    channel_names = tuple(stats.field_names) + (
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "imerg_india_support",
    )
    return ModelArrays(
        inputs=inputs,
        target=target,
        mask=mask,
        weight=weight.astype(np.float32),
        initializations=split.initializations.copy(),
        channel_names=channel_names,
    )


def reconstruct_precipitation(
    split: SplitArrays,
    normalized_residual: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """Convert normalized model output to nonnegative mm/day precipitation."""

    normalized_residual = np.asarray(normalized_residual, dtype=np.float32)
    if normalized_residual.shape != split.target_log_residual.shape:
        raise ValueError("predicted residual shape does not match split")
    residual = (
        normalized_residual * stats.target_std[None, :, None, None]
        + stats.target_mean[None, :, None, None]
    )
    prediction = np.expm1(split.log_fuxi + residual)
    return np.maximum(prediction, 0.0).astype(np.float32)


def _build_audit(
    splits: Mapping[str, SplitArrays],
    forecast_counts: Mapping[int, int],
    imerg_years: Tuple[int, ...],
    imerg_last_date: str,
    baseline: str,
    climatology_day_count: int,
    observation_fraction: np.ndarray,
    area_weight: np.ndarray,
) -> AuditReport:
    all_fuxi = np.concatenate([split.fuxi_mean for split in splits.values()], axis=0)
    all_truth = np.concatenate([split.imerg_truth for split in splits.values()], axis=0)
    all_valid = np.concatenate([split.valid_mask for split in splits.values()], axis=0)
    support = observation_fraction > 0.0
    expected_valid = int(all_valid.shape[0] * all_valid.shape[1] * support.sum())
    finite_fraction = float(all_valid.sum() / expected_valid)
    supported_truth = all_truth[..., support]
    return AuditReport(
        forecast_counts_by_year=dict(forecast_counts),
        split_counts={name: split.sample_count for name, split in splits.items()},
        split_first_initialization={
            name: np.datetime_as_string(split.initializations[0], unit="D")
            for name, split in splits.items()
        },
        split_last_initialization={
            name: np.datetime_as_string(split.initializations[-1], unit="D")
            for name, split in splits.items()
        },
        split_last_verification={
            name: np.datetime_as_string(split.verification_end.max(), unit="D")
            for name, split in splits.items()
        },
        imerg_daily_years=imerg_years,
        imerg_last_available_date=imerg_last_date,
        climatology_baseline=baseline,
        climatology_day_count=climatology_day_count,
        grid_shape=splits["train"].grid_shape,
        lead_week_count=splits["train"].lead_count,
        imerg_support_cell_count=int(support.sum()),
        positive_weight_cell_count=int((area_weight > 0.0).sum()),
        finite_target_fraction_on_support=finite_fraction,
        fuxi_min_mm_day=float(np.min(all_fuxi)),
        fuxi_max_mm_day=float(np.max(all_fuxi)),
        imerg_min_mm_day=float(np.nanmin(supported_truth)),
        imerg_max_mm_day=float(np.nanmax(supported_truth)),
    )


def validate_expected_archive(report: AuditReport) -> None:
    """Enforce exact counts and boundaries for the aligned benchmark archive."""

    if dict(report.forecast_counts_by_year) != dict(EXPECTED_FORECAST_COUNTS):
        raise DataIntegrityError(
            "unexpected FuXi counts: %s (expected %s)"
            % (dict(report.forecast_counts_by_year), dict(EXPECTED_FORECAST_COUNTS))
        )
    if dict(report.split_counts) != dict(EXPECTED_SPLIT_COUNTS):
        raise DataIntegrityError(
            "unexpected split counts: %s (expected %s)"
            % (dict(report.split_counts), dict(EXPECTED_SPLIT_COUNTS))
        )
    expected_boundaries = {
        "train": ("2020-01-02", "2022-11-21", "2023-01-02"),
        "validation": ("2023-01-02", "2023-11-20", "2024-01-01"),
        "test": ("2024-01-01", "2024-12-30", "2025-02-10"),
    }
    for split, (first, last, verification_end) in expected_boundaries.items():
        actual = (
            report.split_first_initialization[split],
            report.split_last_initialization[split],
            report.split_last_verification[split],
        )
        if actual != (first, last, verification_end):
            raise DataIntegrityError(
                "%s boundaries are %s; expected %s"
                % (split, actual, (first, last, verification_end))
            )
    if report.imerg_daily_years != (2020, 2021, 2022, 2023, 2024, 2025):
        raise DataIntegrityError("verification must use IMERG annual stores 2020--2025")
    if report.imerg_last_available_date != "2025-02-10":
        raise DataIntegrityError("IMERG archive does not end at required 2025-02-10 date")
    if report.climatology_baseline != "2001-2019" or report.climatology_day_count != 366:
        raise DataIntegrityError("incorrect fixed IMERG climatology")
    if report.grid_shape != (27, 27) or report.lead_week_count != 6:
        raise DataIntegrityError("adapter requires a 27x27 grid and six lead weeks")
    if report.imerg_support_cell_count != 174 or report.positive_weight_cell_count != 174:
        raise DataIntegrityError("unexpected IMERG/India spatial support")
    if report.finite_target_fraction_on_support != 1.0:
        raise DataIntegrityError("one or more weekly IMERG targets are incomplete")


def load_adapter_data(
    paths: Optional[DataPaths] = None, strict: bool = True
) -> AdapterData:
    """Load, align, audit, and return the frozen FuXi--IMERG experiment.

    Parameters
    ----------
    paths:
        Archive root override, primarily for tests.  The default is the frozen
        India S2S benchmark v1 archive.
    strict:
        If true (default), require the exact known 2020--2024 counts, split
        boundaries, 27x27 grid, 174-cell support, and complete target coverage.
    """

    paths = paths or DataPaths()
    missing = [store for store in paths.required_stores() if not (store / ".zmetadata").is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required Zarr stores:\n" + "\n".join(str(path) for path in missing)
        )

    (
        all_initializations,
        all_fuxi_mean,
        all_fuxi_std,
        latitude,
        longitude,
        forecast_counts,
    ) = _load_forecasts(paths)
    split_indices_all = purged_split_indices(all_initializations)
    retained_indices = np.concatenate(list(split_indices_all.values()))
    retained_indices.sort()
    initializations = all_initializations[retained_indices]
    fuxi_mean = all_fuxi_mean[retained_indices]
    fuxi_std = all_fuxi_std[retained_indices]
    valid_dates = derive_valid_dates(initializations)

    truth, observation_fraction, imerg_years, imerg_last_date = _load_imerg_truth(
        paths, valid_dates, latitude, longitude
    )
    climatology, baseline, climatology_day_count = _load_imerg_climatology(
        paths,
        valid_dates,
        latitude,
        longitude,
        observation_fraction,
    )
    area_weight, region_weights = _load_spatial_weights(
        paths, latitude, longitude, observation_fraction
    )

    # Convert original annual indices to compact retained-array indices.
    original_to_retained = {int(value): index for index, value in enumerate(retained_indices)}
    compact_indices = {
        name: np.asarray(
            [original_to_retained[int(index)] for index in indices], dtype=np.int64
        )
        for name, indices in split_indices_all.items()
    }
    splits = {
        name: _make_split(
            name,
            indices,
            initializations,
            valid_dates,
            fuxi_mean,
            fuxi_std,
            truth,
            climatology,
            observation_fraction,
        )
        for name, indices in compact_indices.items()
    }
    audit = _build_audit(
        splits,
        forecast_counts,
        imerg_years,
        imerg_last_date,
        baseline,
        climatology_day_count,
        observation_fraction,
        area_weight,
    )
    if strict:
        validate_expected_archive(audit)

    forecast_stores = [
        paths.forecast_root / (str(year) + ".zarr") for year in FORECAST_YEARS
    ]
    imerg_daily_stores = [
        paths.imerg_daily_root / (str(year) + ".zarr") for year in imerg_years
    ]
    all_source_stores = forecast_stores + imerg_daily_stores + [
        paths.imerg_climatology_store,
        paths.spatial_store,
    ]
    manifest: Dict[str, object] = {
        "forecast_stores": [str(store) for store in forecast_stores],
        "imerg_daily_stores": [str(store) for store in imerg_daily_stores],
        "imerg_climatology_store": str(paths.imerg_climatology_store),
        "spatial_support_store": str(paths.spatial_store),
        "zmetadata_sha256": {
            str(store): _file_sha256(store / ".zmetadata")
            for store in all_source_stores
        },
        "target_definition": "log1p(imerg_weekly_mean)-log1p(fuxi_ensemble_mean_weekly)",
        "weekly_definition": (
            "mean of IMERG periods starting init+0..6, ..., init+35..41; "
            "these match FuXi lead-day intervals ending init+1..7, ..., init+36..42"
        ),
        "time_alignment": (
            "match half-open interval bounds: IMERG time is period start; "
            "FuXi valid_time is period end"
        ),
        "climatology_contract": "fixed IMERG Final V07B 2001-2019, 31-day smoothed",
    }
    return AdapterData(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        latitude=latitude,
        longitude=longitude,
        observation_fraction=observation_fraction,
        area_weight_km2=area_weight,
        region_weight_km2=region_weights,
        source_manifest=manifest,
        audit=audit,
    )


def load_fuxi_t2m_data(
    adapter_data: AdapterData,
    paths: Optional[DataPaths] = None,
) -> FuxiT2MData:
    """Load FuXi T2M mean/spread on the exact existing adapter split.

    This is an opt-in auxiliary-predictor loader: it neither changes
    :func:`load_adapter_data` nor appends channels in :func:`make_model_arrays`.
    The supplied adapter's initialization arrays are authoritative.  T2M is
    selected by exact date for each split, so no new split is estimated and no
    validation/test information is used to construct training predictors.

    Returned temperature values are deliberately unnormalised.  Downstream
    experiments must estimate any centring/scaling from ``result.train`` only.
    """

    paths = paths or DataPaths()
    split_names = ("train", "validation", "test")
    adapter_splits = adapter_data.splits
    if set(adapter_splits) != set(split_names):
        raise DataIntegrityError(
            "adapter data must contain exactly train, validation, and test splits"
        )

    latitude = np.asarray(adapter_data.latitude, dtype=np.float64)
    longitude = np.asarray(adapter_data.longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise DataIntegrityError("adapter latitude/longitude must be one-dimensional")
    if latitude.size != 27 or longitude.size != 27:
        raise DataIntegrityError("FuXi T2M auxiliary data requires a 27x27 grid")
    if np.any(~np.isfinite(latitude)) or np.any(~np.isfinite(longitude)):
        raise DataIntegrityError("adapter latitude/longitude contains non-finite values")

    requested: Dict[str, np.ndarray] = {}
    for name in split_names:
        values_ns = np.asarray(
            adapter_splits[name].initializations, dtype="datetime64[ns]"
        )
        if values_ns.ndim != 1 or values_ns.size == 0:
            raise DataIntegrityError(
                "%s adapter initializations must be non-empty and one-dimensional"
                % name
            )
        if np.any(np.isnat(values_ns)):
            raise DataIntegrityError("%s adapter initializations contain NaT" % name)
        values_day = values_ns.astype("datetime64[D]")
        if np.any(values_ns != values_day.astype("datetime64[ns]")):
            raise DataIntegrityError("%s adapter initializations are not 00 UTC" % name)
        if np.unique(values_day).size != values_day.size:
            raise DataIntegrityError("%s adapter initializations are not unique" % name)
        requested[name] = values_day

    requested_all = np.concatenate([requested[name] for name in split_names])
    if np.unique(requested_all).size != requested_all.size:
        raise DataIntegrityError("adapter initializations overlap between splits")
    requested_years = tuple(
        int(value)
        for value in np.unique(
            requested_all.astype("datetime64[Y]").astype(np.int64) + 1970
        )
    )
    stores = [
        paths.t2m_forecast_root / (str(year) + ".zarr")
        for year in requested_years
    ]
    missing = [store for store in stores if not (store / ".zmetadata").is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required FuXi T2M Zarr stores:\n"
            + "\n".join(str(path) for path in missing)
        )

    dates: List[np.ndarray] = []
    means: List[np.ndarray] = []
    spreads: List[np.ndarray] = []
    annual_counts: Dict[str, int] = {}
    expected_dims = ("init", "lead_week", "latitude", "longitude")
    for year, store in zip(requested_years, stores):
        with xr.open_zarr(str(store), consolidated=True) as dataset:
            _require_metadata(
                dataset.attrs,
                {
                    "archive_id": "india_s2s_benchmark_v1",
                    "model": "fuxi_s2s",
                    "experiment_id": FUXI_EXPERIMENT_ID,
                    "variable": "t2m",
                    "grid_id": "common_1p5",
                    "source_grid_equals_common": True,
                    "distribution_representation": "members",
                    "ensemble_std_ddof": 0,
                },
                store,
                "global",
            )
            required = {
                "ensemble_mean_weekly",
                "ensemble_std_weekly",
                "ensemble_member_count_weekly",
                "init",
                "lead_week",
                "latitude",
                "longitude",
                "member",
            }
            missing_variables = required.difference(dataset.variables)
            if missing_variables:
                raise DataIntegrityError(
                    "%s is missing variables: %s"
                    % (store, sorted(missing_variables))
                )
            lead_weeks = np.asarray(dataset["lead_week"].values, dtype=np.int16)
            if not np.array_equal(lead_weeks, np.arange(1, 7, dtype=np.int16)):
                raise DataIntegrityError("FuXi T2M lead_week coordinate must be exactly 1..6")
            members = np.asarray(dataset["member"].values, dtype=np.int16)
            if int(dataset.sizes.get("member", -1)) != 50 or not np.array_equal(
                members, np.arange(50, dtype=np.int16)
            ):
                raise DataIntegrityError(
                    "FuXi T2M archive must contain exactly members 0..49"
                )
            for variable in ("ensemble_mean_weekly", "ensemble_std_weekly"):
                if tuple(dataset[variable].dims) != expected_dims:
                    raise DataIntegrityError(
                        "%s %s dimensions are %s; expected %s"
                        % (store, variable, dataset[variable].dims, expected_dims)
                    )
                _require_metadata(
                    dataset[variable].attrs,
                    {
                        "units": "degC",
                        "temporal_statistic": "mean_of_complete_7_day_block",
                    },
                    store,
                    variable,
                )
            if tuple(dataset["ensemble_member_count_weekly"].dims) != expected_dims:
                raise DataIntegrityError(
                    "%s ensemble_member_count_weekly dimensions do not match T2M fields"
                    % store
                )

            store_latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
            store_longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
            if store_latitude.size != 27 or store_longitude.size != 27:
                raise DataIntegrityError("FuXi T2M store is not on a 27x27 grid")
            _check_same_grid(
                latitude,
                longitude,
                store_latitude,
                store_longitude,
                store,
            )

            init_ns = np.asarray(dataset["init"].values, dtype="datetime64[ns]")
            if init_ns.ndim != 1 or init_ns.size == 0 or np.any(np.isnat(init_ns)):
                raise DataIntegrityError("FuXi T2M annual initializations are invalid")
            init = init_ns.astype("datetime64[D]")
            if np.any(init_ns != init.astype("datetime64[ns]")):
                raise DataIntegrityError("FuXi T2M initializations must all be 00 UTC")
            init_years = init.astype("datetime64[Y]").astype(np.int64) + 1970
            if np.any(init_years != year):
                raise DataIntegrityError(
                    "%s contains initializations outside year %d" % (store, year)
                )
            if np.unique(init).size != init.size:
                raise DataIntegrityError(
                    "FuXi T2M annual initializations are not unique in " + str(store)
                )
            if init.size > 1 and np.any(np.diff(init) <= np.timedelta64(0, "D")):
                raise DataIntegrityError(
                    "FuXi T2M annual initializations are not strictly increasing"
                )

            mean = np.asarray(
                dataset["ensemble_mean_weekly"].load().values, dtype=np.float32
            )
            spread = np.asarray(
                dataset["ensemble_std_weekly"].load().values, dtype=np.float32
            )
            member_count = np.asarray(
                dataset["ensemble_member_count_weekly"].load().values
            )
            expected_shape = (init.size, 6, 27, 27)
            if mean.shape != expected_shape or spread.shape != expected_shape:
                raise DataIntegrityError("unexpected FuXi T2M array shape in " + str(store))
            if member_count.shape != expected_shape or not np.all(member_count == 50):
                raise DataIntegrityError(
                    "FuXi T2M weekly statistics do not use all 50 members"
                )
            if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(spread)):
                raise DataIntegrityError(
                    "FuXi T2M weekly mean/spread contains non-finite values"
                )
            if np.any(spread < 0.0):
                raise DataIntegrityError("FuXi T2M weekly spread contains negative values")

            dates.append(init)
            means.append(mean)
            spreads.append(spread)
            annual_counts[str(year)] = int(init.size)

    all_dates = np.concatenate(dates)
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_means = np.concatenate(means, axis=0)[order]
    all_spreads = np.concatenate(spreads, axis=0)[order]
    if np.unique(all_dates).size != all_dates.size:
        raise DataIntegrityError("duplicate FuXi T2M initializations across annual stores")

    aligned: Dict[str, FuxiT2MSplitArrays] = {}
    for name in split_names:
        split_dates = requested[name]
        positions = np.searchsorted(all_dates, split_dates)
        in_range = positions < all_dates.size
        matched = np.zeros(split_dates.size, dtype=bool)
        matched[in_range] = all_dates[positions[in_range]] == split_dates[in_range]
        if not np.all(matched):
            missing_dates = split_dates[~matched]
            preview = ", ".join(
                np.datetime_as_string(value, unit="D") for value in missing_dates[:5]
            )
            raise DataIntegrityError(
                "FuXi T2M is missing %s initialization dates: %s" % (name, preview)
            )
        selected_dates = all_dates[positions]
        if not np.array_equal(selected_dates, split_dates):
            raise DataIntegrityError("FuXi T2M initialization alignment failed for " + name)
        aligned[name] = FuxiT2MSplitArrays(
            name=name,
            initializations=split_dates.copy(),
            ensemble_mean_weekly=all_means[positions].copy(),
            ensemble_std_weekly=all_spreads[positions].copy(),
        )

    def initialization_sha256(values: np.ndarray) -> str:
        serialized = "\n".join(
            np.datetime_as_string(value, unit="D")
            for value in np.asarray(values, dtype="datetime64[D]")
        )
        return hashlib.sha256((serialized + "\n").encode("ascii")).hexdigest()

    manifest: Dict[str, object] = {
        "variable": "t2m",
        "units": "degC",
        "temporal_statistic": "mean_of_complete_7_day_block",
        "model": "fuxi_s2s",
        "experiment_id": FUXI_EXPERIMENT_ID,
        "grid_id": "common_1p5",
        "distribution_representation": "members",
        "ensemble_member_count": 50,
        "t2m_forecast_stores": [str(store) for store in stores],
        "annual_initialization_counts": annual_counts,
        "zmetadata_sha256": {
            str(store): _file_sha256(store / ".zmetadata") for store in stores
        },
        "aligned_initialization_sha256": {
            name: initialization_sha256(requested[name]) for name in split_names
        },
        "alignment_contract": (
            "exact match to the supplied AdapterData train/validation/test "
            "initialization arrays and latitude/longitude grid"
        ),
        "normalization_contract": "unnormalized; fit transformations on train only",
    }
    return FuxiT2MData(
        train=aligned["train"],
        validation=aligned["validation"],
        test=aligned["test"],
        latitude=latitude.copy(),
        longitude=longitude.copy(),
        source_manifest=manifest,
    )


def load_fuxi_tp_distribution_data(
    adapter_data: AdapterData,
    paths: Optional[DataPaths] = None,
) -> FuxiTPDistributionData:
    """Load the frozen FuXi TP ensemble-distribution feature contract.

    Member quantiles are computed directly after the member-wise ``log1p``
    transform.  The two log features are exactly
    ``median(log1p(member TP)) - log1p(IMERG climatology)`` and
    ``q75(log1p(member TP)) - q25(log1p(member TP))``.

    The probability threshold for each case, lead, and supported grid cell is
    the already-aligned, fixed IMERG Final V07B 2001--2019 climatology stored
    on the supplied :class:`AdapterData`.  The comparison is strict
    (``member TP > climatology``), and all 50 members have equal weight.  No
    threshold is estimated from experiment-period observations.

    This loader is opt-in and does not modify the canonical nine v2 channels.
    """

    paths = paths or DataPaths()
    split_names = ("train", "validation", "test")
    adapter_splits = adapter_data.splits
    if set(adapter_splits) != set(split_names):
        raise DataIntegrityError(
            "adapter data must contain exactly train, validation, and test splits"
        )

    audit = adapter_data.audit
    if (
        audit.climatology_baseline != "2001-2019"
        or audit.climatology_day_count != 366
    ):
        raise DataIntegrityError(
            "TP exceedance threshold requires fixed 366-day IMERG 2001-2019 climatology"
        )
    adapter_manifest = adapter_data.source_manifest
    climatology_store = str(adapter_manifest.get("imerg_climatology_store", ""))
    adapter_hashes = adapter_manifest.get("zmetadata_sha256")
    if not climatology_store or not isinstance(adapter_hashes, Mapping):
        raise DataIntegrityError(
            "adapter provenance does not identify the IMERG climatology source"
        )
    climatology_hash = adapter_hashes.get(climatology_store)
    if not isinstance(climatology_hash, str) or len(climatology_hash) != 64:
        raise DataIntegrityError(
            "adapter provenance lacks the IMERG climatology .zmetadata hash"
        )

    latitude = np.asarray(adapter_data.latitude, dtype=np.float64)
    longitude = np.asarray(adapter_data.longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise DataIntegrityError("adapter latitude/longitude must be one-dimensional")
    if latitude.size != 27 or longitude.size != 27:
        raise DataIntegrityError("FuXi TP distribution data requires a 27x27 grid")
    if np.any(~np.isfinite(latitude)) or np.any(~np.isfinite(longitude)):
        raise DataIntegrityError("adapter latitude/longitude contains non-finite values")
    observation_fraction = np.asarray(
        adapter_data.observation_fraction, dtype=np.float32
    )
    if observation_fraction.shape != (27, 27):
        raise DataIntegrityError("adapter observation support is not 27x27")
    threshold_support = np.isfinite(observation_fraction) & (
        observation_fraction > 0.0
    )
    if not np.any(threshold_support):
        raise DataIntegrityError("IMERG climatology threshold support is empty")

    requested: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, np.ndarray] = {}
    for name in split_names:
        split = adapter_splits[name]
        values_ns = np.asarray(split.initializations, dtype="datetime64[ns]")
        if values_ns.ndim != 1 or values_ns.size == 0:
            raise DataIntegrityError(
                "%s adapter initializations must be non-empty and one-dimensional"
                % name
            )
        if np.any(np.isnat(values_ns)):
            raise DataIntegrityError("%s adapter initializations contain NaT" % name)
        values_day = values_ns.astype("datetime64[D]")
        if np.any(values_ns != values_day.astype("datetime64[ns]")):
            raise DataIntegrityError("%s adapter initializations are not 00 UTC" % name)
        if np.unique(values_day).size != values_day.size:
            raise DataIntegrityError("%s adapter initializations are not unique" % name)
        threshold = np.asarray(split.imerg_climatology, dtype=np.float32)
        expected_threshold_shape = (values_day.size, 6, 27, 27)
        if threshold.shape != expected_threshold_shape:
            raise DataIntegrityError(
                "%s IMERG climatology shape is %s; expected %s"
                % (name, threshold.shape, expected_threshold_shape)
            )
        supported_thresholds = threshold[..., threshold_support]
        if np.any(~np.isfinite(supported_thresholds)):
            raise DataIntegrityError(
                "%s IMERG climatology is non-finite on threshold support" % name
            )
        if np.any(supported_thresholds < 0.0):
            raise DataIntegrityError(
                "%s IMERG climatology is negative on threshold support" % name
            )
        requested[name] = values_day
        thresholds[name] = threshold

    requested_all = np.concatenate([requested[name] for name in split_names])
    threshold_all = np.concatenate([thresholds[name] for name in split_names], axis=0)
    if np.unique(requested_all).size != requested_all.size:
        raise DataIntegrityError("adapter initializations overlap between splits")
    requested_order = np.argsort(requested_all)
    requested_sorted = requested_all[requested_order]
    threshold_sorted = threshold_all[requested_order]
    requested_years = tuple(
        int(value)
        for value in np.unique(
            requested_all.astype("datetime64[Y]").astype(np.int64) + 1970
        )
    )
    stores = [
        paths.forecast_root / (str(year) + ".zarr") for year in requested_years
    ]
    missing = [store for store in stores if not (store / ".zmetadata").is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required FuXi TP Zarr stores:\n"
            + "\n".join(str(path) for path in missing)
        )
    store_hashes = {
        str(store): _file_sha256(store / ".zmetadata") for store in stores
    }
    for store in stores:
        recorded_hash = adapter_hashes.get(str(store))
        if recorded_hash != store_hashes[str(store)]:
            raise DataIntegrityError(
                "FuXi TP source hash differs from the supplied AdapterData: "
                + str(store)
            )

    dates: List[np.ndarray] = []
    log_median_anomaly_values: List[np.ndarray] = []
    log_iqr_values: List[np.ndarray] = []
    probability_values: List[np.ndarray] = []
    annual_counts: Dict[str, int] = {}
    member_dims = ("init", "member", "lead_week", "latitude", "longitude")
    summary_dims = ("init", "lead_week", "latitude", "longitude")
    for year, store in zip(requested_years, stores):
        with xr.open_zarr(str(store), consolidated=True) as dataset:
            _require_metadata(
                dataset.attrs,
                {
                    "archive_id": "india_s2s_benchmark_v1",
                    "model": "fuxi_s2s",
                    "experiment_id": FUXI_EXPERIMENT_ID,
                    "variable": "tp",
                    "grid_id": "common_1p5",
                    "source_grid_equals_common": True,
                    "distribution_representation": "members",
                    "ensemble_std_ddof": 0,
                },
                store,
                "global",
            )
            required = {
                "forecast_weekly_mean",
                "ensemble_member_count_weekly",
                "member_available",
                "init",
                "lead_week",
                "latitude",
                "longitude",
                "member",
            }
            missing_variables = required.difference(dataset.variables)
            if missing_variables:
                raise DataIntegrityError(
                    "%s is missing variables: %s"
                    % (store, sorted(missing_variables))
                )
            if tuple(dataset["forecast_weekly_mean"].dims) != member_dims:
                raise DataIntegrityError(
                    "%s forecast_weekly_mean dimensions are %s; expected %s"
                    % (store, dataset["forecast_weekly_mean"].dims, member_dims)
                )
            if tuple(dataset["ensemble_member_count_weekly"].dims) != summary_dims:
                raise DataIntegrityError(
                    "%s ensemble_member_count_weekly dimensions are invalid" % store
                )
            if tuple(dataset["member_available"].dims) != ("init", "member"):
                raise DataIntegrityError("%s member_available dimensions are invalid" % store)
            _require_metadata(
                dataset["forecast_weekly_mean"].attrs,
                {
                    "units": "mm day-1",
                    "temporal_statistic": "mean_of_complete_7_day_block",
                },
                store,
                "forecast_weekly_mean",
            )
            lead_weeks = np.asarray(dataset["lead_week"].values, dtype=np.int16)
            if not np.array_equal(lead_weeks, np.arange(1, 7, dtype=np.int16)):
                raise DataIntegrityError("FuXi TP lead_week coordinate must be exactly 1..6")
            members = np.asarray(dataset["member"].values, dtype=np.int16)
            if int(dataset.sizes.get("member", -1)) != 50 or not np.array_equal(
                members, np.arange(50, dtype=np.int16)
            ):
                raise DataIntegrityError("FuXi TP archive must contain exactly members 0..49")
            store_latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
            store_longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
            if store_latitude.size != 27 or store_longitude.size != 27:
                raise DataIntegrityError("FuXi TP store is not on a 27x27 grid")
            _check_same_grid(
                latitude,
                longitude,
                store_latitude,
                store_longitude,
                store,
            )

            init_ns = np.asarray(dataset["init"].values, dtype="datetime64[ns]")
            if init_ns.ndim != 1 or init_ns.size == 0 or np.any(np.isnat(init_ns)):
                raise DataIntegrityError("FuXi TP annual initializations are invalid")
            init = init_ns.astype("datetime64[D]")
            if np.any(init_ns != init.astype("datetime64[ns]")):
                raise DataIntegrityError("FuXi TP initializations must all be 00 UTC")
            init_years = init.astype("datetime64[Y]").astype(np.int64) + 1970
            if np.any(init_years != year):
                raise DataIntegrityError(
                    "%s contains initializations outside year %d" % (store, year)
                )
            if np.unique(init).size != init.size:
                raise DataIntegrityError(
                    "FuXi TP annual initializations are not unique in " + str(store)
                )
            if init.size > 1 and np.any(np.diff(init) <= np.timedelta64(0, "D")):
                raise DataIntegrityError(
                    "FuXi TP annual initializations are not strictly increasing"
                )

            forecast = np.asarray(
                dataset["forecast_weekly_mean"].load().values, dtype=np.float32
            )
            member_count = np.asarray(
                dataset["ensemble_member_count_weekly"].load().values
            )
            member_available = np.asarray(dataset["member_available"].load().values)
            expected_member_shape = (init.size, 50, 6, 27, 27)
            expected_summary_shape = (init.size, 6, 27, 27)
            if forecast.shape != expected_member_shape:
                raise DataIntegrityError(
                    "unexpected FuXi TP member array shape in " + str(store)
                )
            if np.any(~np.isfinite(forecast)) or np.any(forecast < 0.0):
                raise DataIntegrityError(
                    "FuXi TP weekly members contain non-finite or negative values"
                )
            if member_available.shape != (init.size, 50) or not np.all(
                member_available
            ):
                raise DataIntegrityError(
                    "FuXi TP member_available is not complete for all 50 members"
                )
            if (
                member_count.shape != expected_summary_shape
                or not np.all(member_count == 50)
            ):
                raise DataIntegrityError(
                    "FuXi TP weekly statistics do not use all 50 members"
                )

            log_forecast = np.log1p(forecast).astype(np.float32)
            log_quantiles = np.quantile(
                log_forecast,
                np.asarray([0.25, 0.5, 0.75], dtype=np.float64),
                axis=1,
                method="linear",
            ).astype(np.float32)
            member_log_median = log_quantiles[1]
            member_log_iqr = (log_quantiles[2] - log_quantiles[0]).astype(
                np.float32
            )
            member_log_median_anomaly = np.zeros(
                expected_summary_shape, dtype=np.float32
            )
            probability = np.zeros(expected_summary_shape, dtype=np.float32)
            threshold_positions = np.searchsorted(requested_sorted, init)
            in_range = threshold_positions < requested_sorted.size
            requested_here = np.zeros(init.size, dtype=bool)
            requested_here[in_range] = (
                requested_sorted[threshold_positions[in_range]] == init[in_range]
            )
            if np.any(requested_here):
                selected_thresholds = threshold_sorted[
                    threshold_positions[requested_here]
                ]
                finite_thresholds = np.where(
                    threshold_support[None, None, :, :],
                    selected_thresholds,
                    np.float32(0.0),
                ).astype(np.float32)
                selected_log_median_anomaly = (
                    member_log_median[requested_here]
                    - np.log1p(finite_thresholds).astype(np.float32)
                )
                selected_log_median_anomaly = np.where(
                    threshold_support[None, None, :, :],
                    selected_log_median_anomaly,
                    np.float32(0.0),
                ).astype(np.float32)
                selected_probability = np.mean(
                    forecast[requested_here] > finite_thresholds[:, None, ...],
                    axis=1,
                    dtype=np.float32,
                )
                selected_probability = np.where(
                    threshold_support[None, None, :, :],
                    selected_probability,
                    np.float32(0.0),
                ).astype(np.float32)
                member_log_median_anomaly[
                    requested_here
                ] = selected_log_median_anomaly
                probability[requested_here] = selected_probability

            if (
                member_log_median_anomaly.shape != expected_summary_shape
                or member_log_iqr.shape != expected_summary_shape
                or np.any(~np.isfinite(member_log_median_anomaly))
                or np.any(~np.isfinite(member_log_iqr))
                or np.any(member_log_iqr < 0.0)
            ):
                raise DataIntegrityError("invalid FuXi TP member-log features")
            if (
                np.any(~np.isfinite(probability))
                or np.any(probability < 0.0)
                or np.any(probability > 1.0)
            ):
                raise DataIntegrityError("invalid FuXi TP exceedance probability")

            dates.append(init)
            log_median_anomaly_values.append(member_log_median_anomaly)
            log_iqr_values.append(member_log_iqr)
            probability_values.append(probability)
            annual_counts[str(year)] = int(init.size)

    all_dates = np.concatenate(dates)
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_log_median_anomaly = np.concatenate(
        log_median_anomaly_values, axis=0
    )[order]
    all_log_iqr = np.concatenate(log_iqr_values, axis=0)[order]
    all_probability = np.concatenate(probability_values, axis=0)[order]
    if np.unique(all_dates).size != all_dates.size:
        raise DataIntegrityError("duplicate FuXi TP initializations across annual stores")

    aligned: Dict[str, FuxiTPDistributionSplitArrays] = {}
    for name in split_names:
        split_dates = requested[name]
        positions = np.searchsorted(all_dates, split_dates)
        in_range = positions < all_dates.size
        matched = np.zeros(split_dates.size, dtype=bool)
        matched[in_range] = all_dates[positions[in_range]] == split_dates[in_range]
        if not np.all(matched):
            missing_dates = split_dates[~matched]
            preview = ", ".join(
                np.datetime_as_string(value, unit="D") for value in missing_dates[:5]
            )
            raise DataIntegrityError(
                "FuXi TP is missing %s initialization dates: %s" % (name, preview)
            )
        aligned[name] = FuxiTPDistributionSplitArrays(
            name=name,
            initializations=split_dates.copy(),
            member_log_median_anomaly=all_log_median_anomaly[positions].astype(
                np.float32, copy=True
            ),
            member_log_iqr=all_log_iqr[positions].astype(np.float32, copy=True),
            probability_exceeds_imerg_climatology=all_probability[positions].astype(
                np.float32, copy=True
            ),
        )

    def initialization_sha256(values: np.ndarray) -> str:
        serialized = "\n".join(
            np.datetime_as_string(value, unit="D")
            for value in np.asarray(values, dtype="datetime64[D]")
        )
        return hashlib.sha256((serialized + "\n").encode("ascii")).hexdigest()

    manifest: Dict[str, object] = {
        "variable": "tp",
        "source_units": "mm day-1",
        "temporal_statistic": "mean_of_complete_7_day_block",
        "model": "fuxi_s2s",
        "experiment_id": FUXI_EXPERIMENT_ID,
        "grid_id": "common_1p5",
        "distribution_representation": "members",
        "ensemble_member_count": 50,
        "feature_contract": [
            "member_log_median_anomaly",
            "member_log_iqr",
            "probability_exceeds_imerg_climatology",
        ],
        "member_log_median_anomaly_definition": (
            "median(log1p(member weekly TP)) - log1p(aligned fixed IMERG "
            "2001-2019 climatological weekly mean)"
        ),
        "member_log_iqr_definition": (
            "q75(log1p(member weekly TP)) - q25(log1p(member weekly TP))"
        ),
        "internal_log_quantiles": [0.25, 0.5, 0.75],
        "quantile_method": "numpy method=linear",
        "log_transform": "natural log1p applied member-wise before quantiles",
        "tp_forecast_stores": [str(store) for store in stores],
        "annual_initialization_counts": annual_counts,
        "zmetadata_sha256": store_hashes,
        "aligned_initialization_sha256": {
            name: initialization_sha256(requested[name]) for name in split_names
        },
        "climatology_threshold_store": climatology_store,
        "climatology_threshold_zmetadata_sha256": climatology_hash,
        "climatology_threshold_contract": (
            "strict fraction of 50 equally weighted FuXi members with weekly TP "
            "> aligned fixed IMERG Final V07B 2001-2019 31-day-smoothed "
            "climatological weekly mean"
        ),
        "climatology_threshold_baseline": "2001-2019",
        "outside_threshold_support_fill": (
            "member_log_median_anomaly and probability are 0.0 outside "
            "climatology_threshold_support; both are undefined there"
        ),
        "alignment_contract": (
            "exact match to the supplied AdapterData train/validation/test "
            "initialization arrays and latitude/longitude grid"
        ),
        "normalization_contract": "unnormalized; fit transformations on train only",
    }
    return FuxiTPDistributionData(
        train=aligned["train"],
        validation=aligned["validation"],
        test=aligned["test"],
        latitude=latitude.copy(),
        longitude=longitude.copy(),
        climatology_threshold_support=threshold_support.copy(),
        source_manifest=manifest,
    )


__all__ = [
    "AdapterData",
    "AuditReport",
    "DataIntegrityError",
    "DataPaths",
    "FuxiT2MData",
    "FuxiT2MSplitArrays",
    "FuxiTPDistributionData",
    "FuxiTPDistributionSplitArrays",
    "ModelArrays",
    "NormalizationStats",
    "SplitArrays",
    "calendar_positions_for_dates",
    "collapse_observation_fraction",
    "derive_valid_dates",
    "fit_normalization",
    "load_adapter_data",
    "load_fuxi_t2m_data",
    "load_fuxi_tp_distribution_data",
    "make_model_arrays",
    "purged_split_indices",
    "reconstruct_precipitation",
    "validate_expected_archive",
]
