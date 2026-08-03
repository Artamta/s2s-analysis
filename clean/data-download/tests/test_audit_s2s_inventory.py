from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_s2s_inventory.py"
SPEC = importlib.util.spec_from_file_location("audit_s2s_inventory", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_file_identity_detects_signature_and_size_mismatch(tmp_path: Path) -> None:
    good = tmp_path / "good.grib"
    good.write_bytes(b"GRIB" + b"payload")
    record = audit.file_identity(
        good,
        manifest_status="downloaded_valid",
        declared_size=good.stat().st_size,
    )
    assert record["status"] == "manifest_valid"
    assert record["signature_ok"] is True

    bad = tmp_path / "bad.grib"
    bad.write_bytes(b"NOPE")
    record = audit.file_identity(
        bad,
        manifest_status="downloaded_valid",
        declared_size=99,
    )
    assert record["status"] == "metadata_mismatch"
    assert set(record["issues"]) == {
        "size_differs_from_manifest",
        "file_signature_mismatch",
    }


def test_physics_scan_uses_latest_request_status(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    target = storage / "raw/ecmwf/forecast/annual2020/tp/20200102_cf.grib"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"GRIBdata")
    manifest = storage / "manifests/ecmwf/forecast/annual2020.jsonl"
    manifest.parent.mkdir(parents=True)
    base = {
        "provider": "ecmwf",
        "init_date": "2020-01-02",
        "request_hash": "same-request",
        "target": str(target),
        "field": "tp",
        "forecast_type": "cf",
        "lead_days": 42,
        "member_count": 1,
        "size_bytes": target.stat().st_size,
    }
    records = [
        {**base, "status": "failed", "timestamp": "2026-01-01T00:00:00Z"},
        {**base, "status": "downloaded_valid", "timestamp": "2026-01-02T00:00:00Z"},
    ]
    manifest.write_text("".join(json.dumps(item) + "\n" for item in records))

    experiments = audit.scan_physics(storage, verify_checksums=False)
    assert len(experiments) == 1
    files = experiments[0]["cases"]["2020-01-02"]["files"]
    assert len(files) == 1
    assert files[0]["manifest_status"] == "downloaded_valid"
    assert files[0]["status"] == "manifest_valid"


def test_finalize_reports_missing_expected_dates(tmp_path: Path) -> None:
    path = tmp_path / "one.nc"
    path.write_bytes(b"CDF\x01payload")
    exp = audit.empty_experiment("test/model", "test", "operational_forecast", tmp_path, "test")
    audit.add_file(
        exp,
        "2020-01-02",
        audit.file_identity(path, manifest_status="passed", lead_days=42, members=3, variable="tp"),
    )
    expected = {2020: {"2020-01-02", "2020-01-06"}}
    audit.finalize_experiment(exp, expected, {})

    coverage = exp["summary"]["coverage_by_year"]["2020"]
    assert coverage["initialization_count"] == 1
    assert coverage["missing_expected_dates"] == ["2020-01-06"]
    assert exp["cases"]["2020-01-02"]["acc_ready"] is True


def test_missing_declared_output_is_not_acc_ready(tmp_path: Path) -> None:
    exp = audit.empty_experiment("test/model", "test", "operational_forecast", tmp_path, "test")
    missing = audit.file_identity(tmp_path / "missing.nc", manifest_status="passed")
    audit.add_file(exp, "2020-01-02", missing)
    audit.finalize_experiment(exp, {}, {})
    case = exp["cases"]["2020-01-02"]
    assert case["acc_ready"] is False
    assert case["exclusion_reasons"]


def test_open_qc_none_never_selects_small_experiment(tmp_path: Path) -> None:
    path = tmp_path / "one.nc"
    path.write_bytes(b"CDF\x01payload")
    exp = audit.empty_experiment("test/model", "test", "operational_forecast", tmp_path, "test")
    audit.add_file(exp, "2020-01-02", audit.file_identity(path))
    assert audit.choose_open_qc_records(exp, "none") == []
