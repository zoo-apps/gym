# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Tests for 3-worker federation, HMAC security, aggregation, and cold node recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.federated_lora_zen.corpus import WORKER_IDS, build_federated_corpus
from examples.federated_lora_zen.federation import (
    FederatedConfig,
    NodeLossSchedule,
    SecurityError,
    aggregate_envelopes,
    run_federated_experiment,
    sign_adapter_envelope,
    verify_adapter_envelope,
)
from examples.federated_lora_zen.lora_model import (
    FrozenBaseLM,
    ModelConfig,
    init_lora_adapter,
)


class TestEnvelopeSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ModelConfig()
        self.adapter = init_lora_adapter(
            self.cfg, rank=4, alpha=8.0, seed=1350
        )
        self.keys = {
            "worker-alpha": "11" * 32,
            "worker-beta": "22" * 32,
        }

    def test_valid_envelope_verifies(self) -> None:
        env = sign_adapter_envelope(
            round_idx=1,
            worker_id="worker-alpha",
            admin_key_hex=self.keys["worker-alpha"],
            n_samples=64,
            local_loss=3.2,
            adapter_state=self.adapter,
        )
        verify_adapter_envelope(env, expected_round=1, registered_admin_keys=self.keys)

    def test_tampered_weights_rejected(self) -> None:
        env = sign_adapter_envelope(
            round_idx=1,
            worker_id="worker-alpha",
            admin_key_hex=self.keys["worker-alpha"],
            n_samples=64,
            local_loss=3.2,
            adapter_state=self.adapter,
        )
        env.adapter_state.b_mats["lm_head"][0][0] += 1.0
        with self.assertRaises(SecurityError):
            verify_adapter_envelope(
                env, expected_round=1, registered_admin_keys=self.keys
            )

    def test_forged_key_and_wrong_round_rejected(self) -> None:
        env = sign_adapter_envelope(
            round_idx=1,
            worker_id="worker-alpha",
            admin_key_hex=self.keys["worker-beta"],  # wrong key!
            n_samples=64,
            local_loss=3.2,
            adapter_state=self.adapter,
        )
        with self.assertRaises(SecurityError):
            verify_adapter_envelope(
                env, expected_round=1, registered_admin_keys=self.keys
            )

        valid_env = sign_adapter_envelope(
            round_idx=2,
            worker_id="worker-alpha",
            admin_key_hex=self.keys["worker-alpha"],
            n_samples=64,
            local_loss=3.2,
            adapter_state=self.adapter,
        )
        with self.assertRaises(SecurityError):
            verify_adapter_envelope(
                valid_env, expected_round=1, registered_admin_keys=self.keys
            )

    def test_duplicate_worker_envelope_rejected(self) -> None:
        env = sign_adapter_envelope(
            round_idx=1,
            worker_id="worker-alpha",
            admin_key_hex=self.keys["worker-alpha"],
            n_samples=64,
            local_loss=3.2,
            adapter_state=self.adapter,
        )
        with self.assertRaises(SecurityError):
            aggregate_envelopes(
                [env, env], expected_round=1, registered_admin_keys=self.keys
            )


class TestThreeWorkerFederationAndRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.split = build_federated_corpus(
            seed=1350, partition_mode="mixed_non_iid"
        )
        cls.cfg = ModelConfig()
        cls.base_fp32 = FrozenBaseLM.create_initialized(cls.cfg, seed=1350)
        cls.base_nf4 = cls.base_fp32.to_quantization_mode("nf4")
        cls.init_adapter = init_lora_adapter(
            cls.cfg, rank=4, alpha=8.0, seed=1350
        )

    def test_three_worker_node_loss_and_cold_recovery_on_disk(self) -> None:
        schedule = NodeLossSchedule(
            offline_by_round={2: ("worker-beta",), 3: ("worker-beta",)},
            min_quorum=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            result = run_federated_experiment(
                regime_name="test_node_loss",
                split=self.split,
                frozen_base=self.base_nf4,
                initial_adapter=self.init_adapter,
                config=FederatedConfig(
                    num_rounds=4,
                    local_steps=2,
                    batch_size=8,
                    lr=0.02,
                    aggregation_strategy="exact_delta_svd_fedlora",
                    seed=1350,
                ),
                node_loss=schedule,
                work_dir=work_dir,
            )

            # Verify zero raw data shared & frozen base intact.
            self.assertEqual(result["communication"]["raw_data_bytes_shared"], 0)
            self.assertTrue(result["base_frozen_verified"])

            # Verify rounds 2 and 3 ran with 2 surviving workers (alpha, gamma).
            rounds = result["rounds"]
            self.assertEqual(len(rounds), 4)
            self.assertEqual(
                rounds[1]["offline_workers"], ("worker-beta",)
            )
            self.assertEqual(
                rounds[1]["participating_workers"],
                ("worker-alpha", "worker-gamma"),
            )
            self.assertEqual(
                rounds[3]["participating_workers"],
                ("worker-alpha", "worker-beta", "worker-gamma"),
            )

            # Verify cold recovery event recorded & verified.
            events = result["recovery_events"]
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertEqual(ev["worker_id"], "worker-beta")
            self.assertEqual(ev["rejoined_at_round"], 4)
            self.assertEqual(ev["recovered_from_checkpoint_round"], 3)
            self.assertEqual(ev["missed_rounds"], (2, 3))
            self.assertTrue(ev["verified_integrity"])
            self.assertGreater(ev["checkpoint_bytes"], 1000)

            # Verify persisted server checkpoints and local worker state files.
            for r in range(0, 5):
                ckpt = work_dir / "coordinator" / f"server_checkpoint_r{r:03d}.json"
                self.assertTrue(ckpt.is_file())
            for wid in WORKER_IDS:
                wfile = work_dir / "workers" / wid / "worker_state.json"
                self.assertTrue(wfile.is_file())
                wdata = json.loads(wfile.read_text(encoding="utf-8"))
                self.assertEqual(wdata["worker_id"], wid)
                self.assertEqual(wdata["round_idx"], 4)

    def test_quorum_violation_raises_runtime_error(self) -> None:
        schedule = NodeLossSchedule(
            offline_by_round={1: ("worker-beta", "worker-gamma")},
            min_quorum=2,
        )
        with self.assertRaises(RuntimeError):
            run_federated_experiment(
                regime_name="test_quorum_fail",
                split=self.split,
                frozen_base=self.base_nf4,
                initial_adapter=self.init_adapter,
                config=FederatedConfig(
                    num_rounds=2, local_steps=1, batch_size=4, seed=1350
                ),
                node_loss=schedule,
            )


if __name__ == "__main__":
    unittest.main()
