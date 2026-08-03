#!/usr/bin/env python3
"""Build a compact, evaluation-facing S2S path catalog from the full audit.

The full inventory is intentionally exhaustive.  This script selects only the
canonical forecast experiments, preserves one direct file list per
initialization, and records the transformations an evaluation loader must
apply.  It never scans or writes to storage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    WORKSPACE / "deliverables/s2s_data_inventory_20260803/inventory.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE / "deliverables/s2s_data_inventory_20260803/evaluation_paths.json"
)
DEFAULT_MARKDOWN = (
    WORKSPACE / "deliverables/s2s_data_inventory_20260803/STORAGE_LAYOUT.md"
)


COMMON_GRID = {
    "name": "india_1p5_degree",
    "shape": [27, 27],
    "latitude": {"start": 39.0, "stop": 0.0, "step": -1.5},
    "longitude": {"start": 60.0, "stop": 99.0, "step": 1.5},
    "note": "Canonical AI outputs and downloaded physics forecasts are already on these exact nodes.",
}


PHYSICS_TP = {
    "raw_name": "tp",
    "raw_units": "kg m**-2",
    "evaluation_units": "mm day-1",
    "temporal_statistic": "accumulation from initialization at daily endpoints",
    "loader_rule": (
        "Select 24-hour endpoints, prepend a zero field, then difference consecutive "
        "endpoints. kg m-2 is numerically mm. Do not treat raw endpoints as daily totals."
    ),
}
PHYSICS_T2M = {
    "raw_name": "t2m",
    "raw_units": "K",
    "evaluation_units": "degC",
    "temporal_statistic": "disjoint 24-hour mean",
    "loader_rule": "Use each daily interval directly and subtract 273.15.",
}


CANONICAL_SPECS: dict[str, dict[str, Any]] = {
    "ecmwf": {
        "experiment_id": "physics/ecmwf_operational_2020_2025",
        "display_name": "ECMWF operational S2S",
        "role": "primary_physics",
        "path_templates": [
            "{root}/annual{YYYY}/{variable}/{YYYYMMDD}_{cf_or_pf}.grib"
        ],
        "variables": {"tp": PHYSICS_TP, "t2m": PHYSICS_T2M},
        "ensemble_note": "Separate control (cf) and perturbed (pf) files; pf size is cycle/year dependent.",
    },
    "ukmo": {
        "experiment_id": "physics/ukmo_operational_2020_2025",
        "display_name": "UKMO operational S2S",
        "role": "primary_physics",
        "path_templates": [
            "{root}/annual{YYYY}/{variable}/{YYYYMMDD}_{cf_or_pf}.grib"
        ],
        "variables": {"tp": PHYSICS_TP, "t2m": PHYSICS_T2M},
        "ensemble_note": "Separate cf=1 and pf=3 files; four members total.",
    },
    "cma": {
        "experiment_id": "physics/cma_operational_2020_2025",
        "display_name": "CMA operational S2S",
        "role": "primary_physics",
        "path_templates": [
            "{root}/annual{YYYY}/{variable}/{YYYYMMDD}_{cf_or_pf}.grib"
        ],
        "variables": {"tp": PHYSICS_TP, "t2m": PHYSICS_T2M},
        "ensemble_note": "Separate cf=1 and pf=3 files; four members total.",
    },
    "cnrm": {
        "experiment_id": "physics/cnrm_operational_2020_2025",
        "display_name": "CNRM operational S2S",
        "role": "secondary_weekly_physics",
        "path_templates": [
            "{root}/annual{YYYY}/{variable}/{YYYYMMDD}_{cf_or_pf}.grib"
        ],
        "variables": {"tp": PHYSICS_TP, "t2m": PHYSICS_T2M},
        "ensemble_note": "Weekly subset only; separate cf=1 and pf=24 files; 25 members total.",
    },
    "ncep": {
        "experiment_id": "physics/ncep_operational_2020_2025",
        "display_name": "NCEP operational S2S",
        "role": "primary_physics",
        "path_templates": [
            "{root}/annual{YYYY}/surface/{YYYYMMDD}_{cf_or_pf}.grib"
        ],
        "variables": {
            "tp": {
                **PHYSICS_TP,
                "raw_cadence": "6 hours (168 endpoints for 42 days)",
                "loader_rule": (
                    "Use every fourth 6-hour cumulative endpoint as a daily boundary, "
                    "prepend zero, and difference consecutive daily boundaries."
                ),
            },
            "t2m_proxy": {
                "raw_names": ["mx2t6", "mn2t6"],
                "raw_units": "K",
                "evaluation_units": "degC",
                "temporal_statistic": "proxy, not native daily-mean t2m",
                "loader_rule": (
                    "At each 6-hour step form (mx2t6 + mn2t6) / 2, average four "
                    "consecutive values per day, then subtract 273.15. Keep the name "
                    "t2m_proxy in results."
                ),
            },
        },
        "ensemble_note": "Separate cf=1 and pf=15 surface bundles; 16 members total.",
    },
    "fuxi_s2s": {
        "experiment_id": "model-run/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50",
        "display_name": "FuXi-S2S strict 00 UTC",
        "role": "primary_ai",
        "path_templates": ["{root}/forecasts/annual{YYYY}/{YYYYMMDD}.nc"],
        "variables": {
            "tp": {
                "raw_name": "tp",
                "raw_units": "mm h-1",
                "evaluation_units": "mm day-1",
                "temporal_statistic": "24-hour mean precipitation rate",
                "loader_rule": "Multiply by 24; do not difference lead days.",
            },
            "t2m": {
                "raw_name": "t2m",
                "raw_units": "K",
                "evaluation_units": "degC",
                "temporal_statistic": "24-hour mean",
                "loader_rule": "Subtract 273.15.",
            },
        },
        "ensemble_note": "50 stochastic members indexed 0-49; no member is designated control.",
    },
    "dlesym_v0": {
        "experiment_id": "model-run/dlesym/dlesym_v0_isccp_era5_tpdiag_t2m_00z_2020_2024_ens1",
        "display_name": "DLESyM v0",
        "role": "primary_ai",
        "path_templates": ["{root}/forecasts/{YYYY}/{YYYYMMDD}.nc"],
        "variables": {
            "tp": {
                "raw_name": "tp",
                "raw_units": "mm day-1",
                "evaluation_units": "mm day-1",
                "temporal_statistic": "sum of four 6-hour precipitation diagnostics",
                "loader_rule": "Use directly.",
            },
            "t2m": {
                "raw_name": "t2m",
                "raw_units": "degC",
                "evaluation_units": "degC",
                "temporal_statistic": "24-hour trapezoidal mean",
                "loader_rule": "Use directly.",
            },
        },
        "ensemble_note": "One deterministic member indexed 0.",
    },
    "dlesym_v1": {
        "experiment_id": "model-run/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens4",
        "display_name": "DLESyM v1",
        "role": "primary_ai_t2m",
        "path_templates": ["{root}/forecasts/{YYYY}/{YYYYMMDD}.nc"],
        "variables": {
            "t2m": {
                "raw_name": "t2m",
                "raw_units": "degC",
                "evaluation_units": "degC",
                "temporal_statistic": "24-hour trapezoidal mean",
                "loader_rule": "Use directly.",
            }
        },
        "ensemble_note": "Four matched atmosphere/ocean checkpoint pairs indexed 0-3.",
    },
    "neuralgcm": {
        "experiment_id": "model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10",
        "display_name": "NeuralGCM v1 precipitation",
        "role": "primary_ai_tp",
        "path_templates": ["{root}/forecasts/{YYYY}/{YYYYMMDD}.nc"],
        "variables": {
            "tp": {
                "raw_name": "tp",
                "raw_units": "mm day-1",
                "evaluation_units": "mm day-1",
                "temporal_statistic": "24-hour accumulation",
                "loader_rule": "Use directly; cumulative native precipitation was differenced during production.",
            }
        },
        "ensemble_note": "Ten stochastic members indexed 0-9.",
    },
    "fcn3": {
        "experiment_id": "model-run/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3",
        "display_name": "FourCastNet 3 v1 native T2M",
        "role": "primary_ai_t2m",
        "path_templates": ["{root}/forecasts/{YYYY}/{YYYYMMDD}.nc"],
        "variables": {
            "t2m": {
                "raw_name": "t2m",
                "raw_units": "degC",
                "evaluation_units": "degC",
                "temporal_statistic": "24-hour trapezoidal mean",
                "loader_rule": "Use directly; FCN3 native T2M was averaged from 0/6/12/18/24 UTC boundaries.",
            }
        },
        "ensemble_note": "Three native stochastic members; 516 of 517 scheduled cases are available.",
    },
    "erpas": {
        "experiment_id": "provider/erpas_forecast_2023_2025",
        "display_name": "ERPAS provider forecast",
        "role": "provider_benchmark",
        "path_templates": [
            "{root}/annual{YYYY}/tp/APCP_{YYYYMMDD}.grb",
            "{root}/annual{YYYY}/tp_india_0p5/Ind_0.5_APCP_{YYYYMMDD}.grb",
            "{root}/annual{YYYY}/surface_temperature/tsfc_{YYYYMMDD}.grb",
            "{root}/annual{YYYY}/geopotential_height/gpot_{YYYYMMDD}.grb",
        ],
        "variables": {
            "tp": {
                "raw_name": "tp",
                "raw_units": "kg m**-2",
                "evaluation_units": "mm day-1",
                "temporal_statistic": "33 disjoint 24-hour accumulations",
                "loader_rule": "Use each step directly; kg m-2 is numerically mm.",
                "grid": "global 1 degree, 181x360",
            },
            "tp_india_0p5": {
                "raw_name": "tp",
                "raw_units": "kg m**-2",
                "evaluation_units": "mm day-1",
                "temporal_statistic": "33 disjoint 24-hour accumulations",
                "loader_rule": "Use each step directly; regrid before common-grid metrics.",
                "grid": "regional 0.5 degree, 161x241",
                "required_for_core_evaluation": False,
            },
            "surface_temperature": {
                "raw_name": "t",
                "raw_units": "K",
                "evaluation_units": "degC",
                "loader_rule": "Subtract 273.15; verify temporal meaning before comparing with daily-mean t2m.",
                "grid": "global 1 degree, 181x360",
            },
            "geopotential_height": {
                "raw_name": "gh",
                "raw_units": "gpm",
                "evaluation_units": "gpm",
                "loader_rule": "Select the required pressure level and regrid before metrics.",
                "grid": "global 1 degree, 181x360, seven pressure levels",
            },
        },
        "ensemble_note": "Files contain a precomputed provider ensemble mean; no member dimension (source count 20).",
    },
}


EXPLICIT_EXCLUSIONS = {
    "model-run/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50": (
        "obsolete non-strict sensitivity run; four declared 2023 outputs are missing, "
        "but the canonical strict information-matched FuXi run is complete"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def compact_file(record: dict[str, Any]) -> dict[str, Any]:
    variables = record.get("manifest_variables") or list(
        (record.get("manifest_fields") or {}).keys()
    )
    if not variables and record.get("variable"):
        variables = [item for item in record["variable"].split(",") if item]
    return {
        "path": record["path"],
        "format": record["format"],
        "contains": variables,
        "forecast_type": record.get("forecast_type"),
        "members": record.get("members"),
        "status": record.get("status"),
    }


def ensemble_by_year(cases: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, set[tuple[str | None, int | None]]] = {}
    for date, case in cases.items():
        year = date[:4]
        result.setdefault(year, set()).update(
            (item.get("forecast_type"), item.get("members")) for item in case["files"]
        )
    return {
        year: [
            {"forecast_type": kind, "members": members}
            for kind, members in sorted(values, key=lambda item: (str(item[0]), item[1] or -1))
        ]
        for year, values in sorted(result.items())
    }


def variable_availability(
    variables: dict[str, dict[str, Any]], cases: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result = {}
    for alias, variable in variables.items():
        required_names = variable.get("raw_names")
        available_dates = []
        for date, case in sorted(cases.items()):
            contained = {
                name
                for item in case["files"]
                for name in (item.get("contains") or [])
            }
            if alias in contained or (
                required_names and all(name in contained for name in required_names)
            ):
                available_dates.append(date)
        missing_dates = sorted(set(cases) - set(available_dates))
        result[alias] = {
            "required_for_core_evaluation": variable.get(
                "required_for_core_evaluation", True
            ),
            "initialization_count": len(available_dates),
            "missing_initialization_dates": missing_dates,
        }
    return result


def compact_experiment(alias: str, spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    cases = {
        date: {
            "initialization_time": case["initialization_time"],
            "valid_time_start": case.get("valid_time_start"),
            "valid_time_end": case.get("valid_time_end"),
            "lead_days": case.get("lead_days", []),
            "ensemble_member_count": case.get("ensemble_member_count"),
            "usable": bool(case.get("acc_ready")),
            "files": [compact_file(item) for item in case["files"]],
        }
        for date, case in sorted(source["cases"].items())
    }
    summary = source["summary"]
    return {
        "alias": alias,
        "source_experiment_id": spec["experiment_id"],
        "display_name": spec["display_name"],
        "role": spec["role"],
        "evaluation_enabled": True,
        "root": source["root"],
        "path_templates": spec["path_templates"],
        "format_counts": summary["formats"],
        "years": summary["years"],
        "initialization_count": summary["initialization_count"],
        "usable_initialization_count": summary["acc_ready_cases"],
        "coverage_by_year": summary["coverage_by_year"],
        "lead_day_counts": summary["lead_day_counts"],
        "ensemble_by_year": ensemble_by_year(source["cases"]),
        "ensemble_note": spec["ensemble_note"],
        "variables": spec["variables"],
        "variable_availability": variable_availability(spec["variables"], cases),
        "initialization_dates": list(cases),
        "cases": cases,
    }


def generic_exclusion_reason(experiment_id: str, source: dict[str, Any]) -> str:
    if experiment_id in EXPLICIT_EXCLUSIONS:
        return EXPLICIT_EXCLUSIONS[experiment_id]
    if source["product"] == "case_or_pilot_forecast":
        return "single-case, pilot, smoke, or experimental forecast; not part of the historical evaluation set"
    if experiment_id.startswith("reforecast/"):
        return "climatology/reforecast source, not an operational forecast input"
    if source["summary"]["initialization_count"] <= 1:
        return "one-case sensitivity or smoke output; not selected as a canonical production experiment"
    return "superseded or non-canonical experiment"


def build_catalog(inventory: dict[str, Any], inventory_path: Path) -> dict[str, Any]:
    experiments = inventory["experiments"]
    selected_ids = {spec["experiment_id"] for spec in CANONICAL_SPECS.values()}
    missing = selected_ids - set(experiments)
    if missing:
        raise ValueError(f"canonical experiments absent from inventory: {sorted(missing)}")

    selected = {
        alias: compact_experiment(alias, spec, experiments[spec["experiment_id"]])
        for alias, spec in CANONICAL_SPECS.items()
    }
    bad = [
        alias
        for alias, item in selected.items()
        if item["usable_initialization_count"] != item["initialization_count"]
    ]
    if bad:
        raise ValueError(f"canonical experiments contain unusable cases: {bad}")

    excluded = {}
    for experiment_id, source in sorted(experiments.items()):
        if experiment_id in selected_ids:
            continue
        excluded[experiment_id] = {
            "model": source["model"],
            "product": source["product"],
            "root": source["root"],
            "initialization_count": source["summary"]["initialization_count"],
            "usable_initialization_count": source["summary"]["acc_ready_cases"],
            "reason": generic_exclusion_reason(experiment_id, source),
            "data_deleted": False,
        }

    cohorts = {
        "tp_dense_2020_2024": [
            "ecmwf", "ukmo", "cma", "ncep", "fuxi_s2s", "dlesym_v0", "neuralgcm"
        ],
        "t2m_dense_2020_2024": [
            "ecmwf", "ukmo", "cma", "fuxi_s2s", "dlesym_v0", "dlesym_v1", "fcn3"
        ],
        "t2m_proxy_sensitivity": ["ncep"],
        "weekly_secondary": ["cnrm"],
        "provider_2023_2025": ["erpas"],
    }
    cohort_intersections = {}
    for name, aliases in cohorts.items():
        shared_dates = set.intersection(
            *(set(selected[alias]["cases"]) for alias in aliases)
        )
        dates = sorted(shared_dates)
        cohort_intersections[name] = {
            "aliases": aliases,
            "initialization_count": len(dates),
            "initialization_dates": dates,
        }

    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_inventory": str(inventory_path.resolve()),
        "source_inventory_generated_at": inventory["generated_at"],
        "selection_policy": (
            "Canonical production forecasts with explicit case-level availability. FCN3 native T2M is "
            "included with its documented 2021-03-08 gap; no source data were deleted."
        ),
        "common_grid": COMMON_GRID,
        "canonical_experiments": {
            alias: spec["experiment_id"] for alias, spec in CANONICAL_SPECS.items()
        },
        "recommended_cohorts": cohorts,
        "cohort_case_intersections": cohort_intersections,
        "known_gaps": {
            "erpas_tp_india_0p5_2024-12-18": {
                "status": "not_supplied_in_frozen_google_drive_source_inventory",
                "impact": (
                    "optional regional 0.5-degree product only; global ERPAS tp for "
                    "2024-12-18 exists and the core evaluation remains usable"
                ),
            },
            "fcn3_2021-03-08": {
                "status": "missing_after_nonfinite_era5_initial_condition",
                "impact": "FCN3 remains enabled; exact-init T2M cohorts omit this one date",
            },
            "obsolete_fuxi_non_strict_four_2023_cases": {
                "status": "not_recovered_superseded",
                "impact": "none; the canonical strict FuXi run has all 621 cases",
            },
        },
        "experiments": selected,
        "excluded_experiments": excluded,
        "climatology_status": {
            "forecast_derived_five_year_climatology": "not_built_by_this catalog",
            "fuxi_native_reforecast": {
                "status": "available_as_archives_not_evaluation_ready",
                "experiment_id": "reforecast/fuxi_native_2002_2021",
                "root": experiments["reforecast/fuxi_native_2002_2021"]["root"],
                "archive_template": "{root}/{YYYYMMDD}.7z",
                "inside_archive": "YYYY/YYYYMMDD/member/00-50/01.nc-42.nc",
                "members": 51,
                "lead_days": 42,
                "years": list(range(2002, 2022)),
                "warning": "Member labels 00-50 do not establish a control-member interpretation.",
            },
            "physics_native_reforecasts": (
                "Only three one-file smoke tests were found; no production physics climatology archive "
                "is ready in the audited storage root."
            ),
        },
    }


def render_markdown(catalog: dict[str, Any]) -> str:
    rows = []
    for alias, item in catalog["experiments"].items():
        variables = ", ".join(item["variables"])
        years = f"{min(item['years'])}-{max(item['years'])}"
        lead_counts = ", ".join(f"{key}d:{value}" for key, value in item["lead_day_counts"].items())
        rows.append(
            f"| `{alias}` | {item['display_name']} | {years} | {item['initialization_count']} | "
            f"{variables} | {lead_counts} |"
        )
    return f"""# S2S storage layout and evaluation catalog

This is the small, script-facing view of the exhaustive `inventory.json`. The
machine-readable source is `evaluation_paths.json`; it contains direct paths
for every selected initialization and deliberately excludes pilot or
scientifically superseded runs.

## Canonical forecast datasets

| JSON alias | Dataset | Years | ICs | Variables | Lead coverage |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

All physics downloads and retained AI products use the 27 x 27 India grid:
latitude 39 to 0 degrees (descending) and longitude 60 to 99 degrees, both at
1.5-degree spacing. ERPAS remains on its provider grids and must be remapped.

## Storage patterns

- Physics (`ecmwf`, `ukmo`, `cma`, `cnrm`):
  `.../raw/MODEL/forecast/annualYYYY/{{tp|t2m}}/YYYYMMDD_{{cf|pf}}.grib`
- NCEP: `.../raw/ncep/forecast/annualYYYY/surface/YYYYMMDD_{{cf|pf}}.grib`.
  Each bundle contains `tp`, `mx2t6`, and `mn2t6` at 168 six-hour steps.
- Strict FuXi: `.../model-runs/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/forecasts/annualYYYY/YYYYMMDD.nc`.
- DLESyM: `.../model-runs/dlesym/RUN/forecasts/YYYY/YYYYMMDD.nc`.
- NeuralGCM: `.../model-runs/neural-gcm/RUN/forecasts/YYYY/YYYYMMDD.nc`.
- ERPAS: `.../raw/erpas/forecast/annualYYYY/PRODUCT/PROVIDER_FILENAME.grb`.
- Native FuXi reforecasts: `.../models/fuxi/data/YYYYMMDD.7z`, containing
  `YYYY/YYYYMMDD/member/00-50/01.nc-42.nc` (51 members, 42 daily leads).

The JSON records the concrete root and templates for each alias, so evaluation
code should not reconstruct paths when a case entry already supplies them.

## Loader rules that must not be mixed up

- Physics `tp` is accumulation from initialization at daily endpoints. Daily
  rainfall is the difference of consecutive endpoints after prepending zero.
- Direct physics `t2m` is a disjoint 24-hour mean in Kelvin; subtract 273.15.
- NCEP temperature is only a proxy: average `(mx2t6 + mn2t6) / 2` across four
  six-hour values per day. Keep it labelled `t2m_proxy`.
- FuXi `tp` is a mean rate in `mm h-1`; multiply by 24. FuXi `t2m` is Kelvin.
- DLESyM and NeuralGCM outputs are already stored in final daily units.
- ERPAS precipitation steps are disjoint daily accumulations, not cumulative
  endpoints. The files are provider ensemble means, not individual members.

## Using `evaluation_paths.json`

```python
import json
from pathlib import Path

catalog = json.loads(Path("deliverables/s2s_data_inventory_20260803/evaluation_paths.json").read_text())
experiment = catalog["experiments"]["fuxi_s2s"]
case = experiment["cases"]["2023-06-05"]
paths = [Path(item["path"]) for item in case["files"] if item["status"] == "manifest_valid"]
```

For cross-model evaluation, intersect the `cases` keys of the requested
aliases. Do not assume every provider has the same schedule: CNRM is weekly,
ERPAS covers 2023-2025, while DLESyM and NeuralGCM stop at 2024.
The ready-made intersections for the recommended groups are stored under
`cohort_case_intersections`.

## Deliberate exclusions and missing cases

- FCN3 native T2M is enabled with 516 usable cases. The scheduled 2021-03-08
  initialization is missing after a non-finite ERA5 initial condition, so
  exact-init cohorts document and omit that one date.
- The older non-strict FuXi sensitivity has four missing 2023 outputs. It is
  superseded by the complete 621-case strict FuXi experiment and should not be
  regenerated for the primary study.
- ERPAS `tp_india_0p5` is absent for 2024-12-18 and was also absent from the
  frozen Google Drive source inventory, so there is no local source file to
  recover. This regional product is optional: global ERPAS `tp` exists for the
  date and all 148 core ERPAS rainfall cases remain usable.
- Single-case FuXi experiments, one-case DLESyM/NeuralGCM sensitivities, and
  smoke tests remain listed under `excluded_experiments` with their paths.
- Only three physics reforecast smoke files exist. They are not a production
  climatology. The forecast-derived five-year climatology is a later workflow.
"""


def write_outputs(catalog: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(catalog), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    catalog = build_catalog(inventory, args.inventory)
    write_outputs(catalog, args.output_json, args.output_markdown)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
                "selected_experiments": len(catalog["experiments"]),
                "selected_initializations": sum(
                    item["initialization_count"] for item in catalog["experiments"].values()
                ),
                "excluded_experiments": len(catalog["excluded_experiments"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
