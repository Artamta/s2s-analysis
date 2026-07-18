from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMMON_DIR))

import neuralgcm_run_contract  # noqa: E402


CONFIGS = [
    REPO_ROOT / "model-runs/configs/neuralgcm_pilot42d_tp_20200601.json",
]


class NeuralGCMRunContractTest(unittest.TestCase):
    def test_frozen_pilot_contracts(self) -> None:
        summaries = [
            neuralgcm_run_contract.validate_config(
                json.loads(path.read_text(encoding="utf-8"))
            )
            for path in CONFIGS
        ]
        self.assertEqual([item["product"] for item in summaries], ["daily_tp"])
        self.assertTrue(all(item["date_count"] == 621 for item in summaries))
        self.assertTrue(all(item["unroll_steps"] == 169 for item in summaries))

    def test_rejects_wrong_endpoint_count(self) -> None:
        config = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
        config["forecast"]["unroll_steps"] = 168
        with self.assertRaisesRegex(
            neuralgcm_run_contract.RunContractError, "169 six-hour frames"
        ):
            neuralgcm_run_contract.validate_config(config)

    def test_rejects_future_forcing(self) -> None:
        config = copy.deepcopy(
            json.loads(CONFIGS[0].read_text(encoding="utf-8"))
        )
        config["initial_conditions"]["forcing_source_time"] = "2020-06-01T00:00:00"
        with self.assertRaisesRegex(
            neuralgcm_run_contract.RunContractError, "D-1"
        ):
            neuralgcm_run_contract.validate_config(config)


if __name__ == "__main__":
    unittest.main()
