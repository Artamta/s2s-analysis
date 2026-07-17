from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMMON_DIR))

import pilot_contract  # noqa: E402


CONFIG_PATH = REPO_ROOT / "model-runs/configs/neuralgcm_smoke_20200601.json"


class PilotContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_contract(self) -> None:
        summary = pilot_contract.validate_config(self.config)
        self.assertEqual(summary["date_count"], 621)
        self.assertEqual(summary["frame_hours"], [0, 6, 12])
        self.assertEqual(summary["forcing_source_time"], "2020-05-31T00:00:00+00:00")

    def test_rejects_calendar_count_drift(self) -> None:
        config = copy.deepcopy(self.config)
        config["calendar"]["date_count"] = 620
        with self.assertRaisesRegex(pilot_contract.ContractError, "expected 620"):
            pilot_contract.validate_config(config)

    def test_rejects_future_forcing(self) -> None:
        config = copy.deepcopy(self.config)
        config["initial_conditions"]["forcing_source_time"] = "2020-06-02T00:00:00"
        with self.assertRaisesRegex(pilot_contract.ContractError, "exactly D-1"):
            pilot_contract.validate_config(config)

    def test_rejects_t2m_claim(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["available_requested_fields"] = ["tp", "t2m"]
        with self.assertRaisesRegex(pilot_contract.ContractError, "only requested field tp"):
            pilot_contract.validate_config(config)


if __name__ == "__main__":
    unittest.main()
