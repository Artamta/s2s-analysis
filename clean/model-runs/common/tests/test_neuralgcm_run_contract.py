from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMMON_DIR))

import neuralgcm_run_contract  # noqa: E402


PILOT_CONFIG = REPO_ROOT / "model-runs/configs/neuralgcm_pilot42d_tp_20200601.json"
PRODUCTION_ENS1_CONFIG = (
    REPO_ROOT / "model-runs/configs/neuralgcm_production_tp_2020_2024_ens1.json"
)
PRODUCTION_ENS10_CONFIG = (
    REPO_ROOT / "model-runs/configs/neuralgcm_production_tp_2020_2024_ens10.json"
)
CONFIGS = [PILOT_CONFIG, PRODUCTION_ENS1_CONFIG, PRODUCTION_ENS10_CONFIG]


class NeuralGCMRunContractTest(unittest.TestCase):
    def test_frozen_pilot_contracts(self) -> None:
        summaries = [
            neuralgcm_run_contract.validate_config(
                json.loads(path.read_text(encoding="utf-8"))
            )
            for path in CONFIGS
        ]
        self.assertTrue(all(item["product"] == "daily_tp" for item in summaries))
        self.assertEqual([item["date_count"] for item in summaries], [621, 517, 517])
        self.assertEqual(
            [item["run_mode"] for item in summaries],
            ["pilot", "production", "production"],
        )
        self.assertEqual([item["member_count"] for item in summaries], [1, 1, 10])
        self.assertTrue(all(item["unroll_steps"] == 169 for item in summaries))

    def test_ens10_template_seeds_follow_contract(self) -> None:
        config = json.loads(PRODUCTION_ENS10_CONFIG.read_text(encoding="utf-8"))
        expected = []
        for member in range(10):
            payload = f"{config['run_label']}/2020-01-02/{member}".encode("ascii")
            expected.append(int.from_bytes(hashlib.sha256(payload).digest()[:4], "big"))
        self.assertEqual(config["forecast"]["member_seeds"], expected)

    def test_rejects_wrong_endpoint_count(self) -> None:
        config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
        config["forecast"]["unroll_steps"] = 168
        with self.assertRaisesRegex(
            neuralgcm_run_contract.RunContractError, "169 six-hour frames"
        ):
            neuralgcm_run_contract.validate_config(config)

    def test_rejects_future_forcing(self) -> None:
        config = copy.deepcopy(
            json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
        )
        config["initial_conditions"]["forcing_source_time"] = "2020-06-01T00:00:00"
        with self.assertRaisesRegex(
            neuralgcm_run_contract.RunContractError, "D-1"
        ):
            neuralgcm_run_contract.validate_config(config)


if __name__ == "__main__":
    unittest.main()
