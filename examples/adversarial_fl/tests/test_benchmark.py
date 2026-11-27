# Copyright 2026 Zoo Labs Foundation Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end determinism, cryptographic receipt, and schema tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from examples.adversarial_fl.run_benchmark import (
    build_benchmark_artifact,
    main as benchmark_main,
)


class TestBenchmarkReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shared_artifact = build_benchmark_artifact(seed=1350, rounds=1)

    def test_byte_identical_cli_output_and_valid_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "run1.json"
            out2 = Path(tmpdir) / "run2.json"
            rc1 = benchmark_main(["--seed", "1350", "--rounds", "1", "--out", str(out1)])
            rc2 = benchmark_main(["--seed", "1350", "--rounds", "1", "--out", str(out2)])
            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)

            b1 = out1.read_bytes()
            b2 = out2.read_bytes()
            self.assertEqual(b1, b2)

            data = json.loads(b1.decode("utf-8"))
            receipt = data["independent_evaluation_receipt"]
            core = {
                k: v for k, v in data.items() if k != "independent_evaluation_receipt"
            }
            canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            self.assertEqual(
                hashlib.sha256(canonical).hexdigest(),
                receipt["canonical_payload_sha256"],
            )

    def test_artifact_contains_all_required_threats_and_negative_regimes(self) -> None:
        artifact = self.shared_artifact
        self.assertEqual(artifact["metadata"]["specification"], "HIP-1350")
        self.assertEqual(artifact["metadata"]["issue"], "zoo-apps/gym#5")
        self.assertTrue(
            artifact["verifications"]["analytical_gradient_check"]["passed_lt_1e6"]
        )
        self.assertTrue(
            artifact["verifications"]["flower_adapter_parity"]["parity_verified"]
        )

        threats = set(artifact["threat_matrices"])
        self.assertEqual(
            threats,
            {
                "label_flip_and_sign_flip",
                "targeted_trigger_backdoor",
                "stale_honest_straggler",
                "stale_malicious_replay",
                "colluding_inner_product_cartel",
            },
        )
        for frac in ("0%", "17%", "33%", "50%"):
            self.assertIn(frac, artifact["threat_matrices"]["label_flip_and_sign_flip"])
            self.assertIn(
                frac, artifact["threat_matrices"]["targeted_trigger_backdoor"]
            )
            self.assertIn(
                frac, artifact["threat_matrices"]["colluding_inner_product_cartel"]
            )

        neg = set(artifact["negative_regimes"])
        self.assertEqual(
            neg,
            {
                "neg_non_iid_honest_frr",
                "neg_unmonitored_backdoor_blindness",
                "neg_fifty_percent_collusion_breakdown",
            },
        )

    def test_governance_separation_and_what_remains_unproven(self) -> None:
        artifact = self.shared_artifact
        self.assertEqual(
            artifact["governance_separation"]["artifact_license"], "Apache-2.0"
        )
        self.assertEqual(
            artifact["governance_separation"]["external_runtime_dependencies"], []
        )
        unproven = artifact["what_remains_unproven"]
        self.assertGreaterEqual(len(unproven), 5)
        joined = " ".join(unproven).lower()
        for keyword in ("7b", "grpc", "dp-sgd", "rl", "sybil"):
            self.assertIn(keyword, joined)


if __name__ == "__main__":
    unittest.main()
