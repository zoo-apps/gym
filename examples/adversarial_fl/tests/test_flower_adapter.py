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
"""Tests for the Flower Strategy adapter and pure-stdlib wire serialization."""

from __future__ import annotations

import unittest

from examples.adversarial_fl.attacks import (
    AttackConfig,
    generate_round_submissions,
)
from examples.adversarial_fl.corpus import load_federated_corpus
from examples.adversarial_fl.defenses import (
    DEFENSE_NAMES,
    DefenseConfig,
    aggregate_submissions,
)
from examples.adversarial_fl.flower_adapter import (
    AdversarialDefenseStrategy,
    StdlibClientProxy,
    StdlibEvaluateRes,
    decode_flat_params,
    encode_flat_params,
    submissions_to_flower_results,
)
from examples.adversarial_fl.model import NUM_PARAMS, init_params


class TestFlowerAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, cls.raw_lookup = load_federated_corpus(seed=1350)

    def test_float64_wire_roundtrip_exact(self) -> None:
        params = init_params(1350)
        encoded = encode_flat_params(params)
        decoded = decode_flat_params(encoded)
        self.assertEqual(len(decoded), NUM_PARAMS)
        self.assertEqual(decoded, params)

    def test_flower_strategy_matches_native_aggregator_all_defenses(self) -> None:
        theta = init_params(1350)
        submissions = generate_round_submissions(
            round_idx=0,
            checkpoints=[theta],
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("colluding_inner_product_cartel", "33%"),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        flower_results = submissions_to_flower_results(submissions, theta)

        for defense_name in DEFENSE_NAMES:
            dcfg = DefenseConfig(defense_name)
            native = aggregate_submissions(
                submissions,
                dcfg,
                global_params=theta,
                eval_val=self.corpus.eval_val,
            )
            strategy = AdversarialDefenseStrategy(
                initial_params=theta,
                defense_config=dcfg,
                eval_val=self.corpus.eval_val,
                eval_clean=self.corpus.eval_clean,
            )
            new_wire_params, metrics = strategy.aggregate_fit(
                server_round=0, results=flower_results, failures=[]
            )
            self.assertIsNotNone(new_wire_params)
            updated = decode_flat_params(new_wire_params)
            expected = [
                theta[i] + native.aggregated_delta[i] for i in range(NUM_PARAMS)
            ]
            max_diff = max(abs(a - b) for a, b in zip(updated, expected, strict=True))
            self.assertLess(max_diff, 1e-12, msg=f"Mismatch for {defense_name}")
            self.assertEqual(metrics["accepted_count"], native.accepted_count)
            self.assertEqual(metrics["rejected_count"], native.rejected_count)

    def test_flower_lifecycle_hooks_configure_and_evaluate(self) -> None:
        theta = init_params(1350)
        strategy = AdversarialDefenseStrategy(
            initial_params=theta,
            defense_config=DefenseConfig("dso_hybrid_eval_cosine"),
            eval_val=self.corpus.eval_val,
            eval_clean=self.corpus.eval_clean,
        )
        init_wire = strategy.initialize_parameters()
        proxies = [StdlibClientProxy(cid=str(i)) for i in range(3)]
        fit_cfg = strategy.configure_fit(1, init_wire, proxies)
        self.assertEqual(len(fit_cfg), 3)

        eval_cfg = strategy.configure_evaluate(1, init_wire, proxies)
        self.assertEqual(len(eval_cfg), 3)

        agg_loss, eval_metrics = strategy.aggregate_evaluate(
            1,
            [
                (proxies[0], StdlibEvaluateRes(loss=0.8, num_examples=10)),
                (proxies[1], StdlibEvaluateRes(loss=0.4, num_examples=30)),
            ],
            failures=[],
        )
        self.assertIsNotNone(agg_loss)
        self.assertAlmostEqual(agg_loss or 0.0, 0.5, places=9)
        self.assertEqual(eval_metrics["total_examples"], 40)

        central_eval = strategy.evaluate(1, init_wire)
        self.assertIsNotNone(central_eval)
        assert central_eval is not None
        loss_val, central_metrics = central_eval
        self.assertGreater(loss_val, 0.0)
        self.assertIn("clean_accuracy", central_metrics)


if __name__ == "__main__":
    unittest.main()
