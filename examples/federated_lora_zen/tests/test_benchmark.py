# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Determinism and schema integrity tests for the benchmark runner."""

from __future__ import annotations

import json
import unittest

from examples.federated_lora_zen.run_benchmark import run_all_benchmarks


class TestBenchmarkDeterminism(unittest.TestCase):
    def test_mini_benchmark_is_byte_identical_across_two_runs(self) -> None:
        p1 = run_all_benchmarks(
            seed=1350, num_rounds=3, local_steps=1, worker_batch_size=4
        )
        p2 = run_all_benchmarks(
            seed=1350, num_rounds=3, local_steps=1, worker_batch_size=4
        )
        b1 = json.dumps(p1, indent=2, sort_keys=True).encode("utf-8")
        b2 = json.dumps(p2, indent=2, sort_keys=True).encode("utf-8")
        self.assertEqual(b1, b2)

        # Verify all 6 main regimes and 4 negative-regime entries are present.
        self.assertEqual(len(p1["summary_main_regimes"]), 6)
        self.assertEqual(len(p1["summary_negative_regimes"]), 4)
        self.assertEqual(
            p1["regimes"]["fed3_qlora_node_loss_recovery"]["recovery_events"][0][
                "worker_id"
            ],
            "worker-beta",
        )


if __name__ == "__main__":
    unittest.main()
