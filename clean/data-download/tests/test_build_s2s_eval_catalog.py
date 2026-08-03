from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_s2s_eval_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_s2s_eval_catalog", SCRIPT)
assert SPEC and SPEC.loader
catalog_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog_module)


def source_experiment(experiment_id: str, model: str = "test") -> dict:
    return {
        "experiment_id": experiment_id,
        "model": model,
        "product": "operational_forecast",
        "root": f"/storage/{model}",
        "cases": {
            "2020-01-02": {
                "initialization_time": "2020-01-02T00:00:00Z",
                "valid_time_start": "2020-01-02T00:00:00Z",
                "valid_time_end": "2020-02-13T00:00:00Z",
                "lead_days": [42],
                "ensemble_member_count": 2,
                "acc_ready": True,
                "files": [
                    {
                        "path": f"/storage/{model}/20200102.nc",
                        "format": "netcdf",
                        "manifest_variables": ["tp"],
                        "forecast_type": "native_ensemble",
                        "members": 2,
                        "status": "manifest_valid",
                    }
                ],
            }
        },
        "summary": {
            "formats": {"netcdf": 1},
            "years": [2020],
            "initialization_count": 1,
            "acc_ready_cases": 1,
            "coverage_by_year": {"2020": {"initialization_count": 1}},
            "lead_day_counts": {"42": 1},
        },
    }


def complete_inventory() -> dict:
    experiments = {
        spec["experiment_id"]: source_experiment(spec["experiment_id"], alias)
        for alias, spec in catalog_module.CANONICAL_SPECS.items()
    }
    fcn3_id = "model-run/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3"
    experiments[fcn3_id] = source_experiment(fcn3_id, "fcn3")
    experiments["reforecast/fuxi_native_2002_2021"] = source_experiment(
        "reforecast/fuxi_native_2002_2021", "fuxi_reforecast"
    )
    experiments["reforecast/fuxi_native_2002_2021"]["product"] = "native_reforecast_archive"
    return {
        "generated_at": "2026-08-03T00:00:00+00:00",
        "experiments": experiments,
    }


def test_catalog_includes_fcn3_native_t2m(tmp_path: Path) -> None:
    catalog = catalog_module.build_catalog(complete_inventory(), tmp_path / "inventory.json")
    assert "fcn3" in catalog["experiments"]
    assert list(catalog["experiments"]["fcn3"]["variables"]) == ["t2m"]
    assert "fcn3" in catalog["recommended_cohorts"]["t2m_dense_2020_2024"]


def test_compact_cases_retain_direct_paths_and_drop_audit_bulk(tmp_path: Path) -> None:
    catalog = catalog_module.build_catalog(complete_inventory(), tmp_path / "inventory.json")
    case = catalog["experiments"]["fuxi_s2s"]["cases"]["2020-01-02"]
    assert case["files"] == [
        {
            "path": "/storage/fuxi_s2s/20200102.nc",
            "format": "netcdf",
            "contains": ["tp"],
            "forecast_type": "native_ensemble",
            "members": 2,
            "status": "manifest_valid",
        }
    ]
    assert "manifest_path" not in case["files"][0]


def test_required_loader_rules_are_explicit(tmp_path: Path) -> None:
    catalog = catalog_module.build_catalog(complete_inventory(), tmp_path / "inventory.json")
    assert "difference" in catalog["experiments"]["ecmwf"]["variables"]["tp"]["loader_rule"]
    assert "Multiply by 24" in catalog["experiments"]["fuxi_s2s"]["variables"]["tp"]["loader_rule"]
    assert "t2m_proxy" in catalog["experiments"]["ncep"]["variables"]
    assert catalog["experiments"]["neuralgcm"]["variables"]["tp"]["loader_rule"].startswith("Use directly")


def test_optional_variable_gaps_do_not_disable_a_case(tmp_path: Path) -> None:
    inventory = complete_inventory()
    erpas_id = catalog_module.CANONICAL_SPECS["erpas"]["experiment_id"]
    inventory["experiments"][erpas_id]["cases"]["2020-01-02"]["files"][0][
        "manifest_variables"
    ] = ["tp"]
    catalog = catalog_module.build_catalog(inventory, tmp_path / "inventory.json")
    availability = catalog["experiments"]["erpas"]["variable_availability"]
    assert availability["tp"]["initialization_count"] == 1
    assert availability["tp_india_0p5"]["initialization_count"] == 0
    assert availability["tp_india_0p5"]["required_for_core_evaluation"] is False
    assert catalog["experiments"]["erpas"]["cases"]["2020-01-02"]["usable"] is True


def test_recommended_cohorts_include_precomputed_date_intersections(tmp_path: Path) -> None:
    catalog = catalog_module.build_catalog(complete_inventory(), tmp_path / "inventory.json")
    cohort = catalog["cohort_case_intersections"]["tp_dense_2020_2024"]
    assert cohort["aliases"] == catalog["recommended_cohorts"]["tp_dense_2020_2024"]
    assert cohort["initialization_count"] == 1
    assert cohort["initialization_dates"] == ["2020-01-02"]
