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
"""Tests for all four mandatory FL threat scenarios, stragglers, and stealth backdoor."""

from __future__ import annotations

import statistics
import unittest

from examples.adversarial_fl.attacks import (
    ADVERSARY_FRACTIONS,
    AttackConfig,
    adversary_ids_for_fraction,
    expected_bft_threshold_exceeded,
    generate_round_submissions,
    mean_cartel_pairwise_cosine,
)
from examples.adversarial_fl.corpus import MINORITY_CLIENT_IDS, load_federated_corpus
from examples.adversarial_fl.model import (
    NUM_PARAMS,
    evaluate_clean,
    init_params,
    l2_norm,
    vec_add,
)


class TestAdversarialAttacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, cls.raw_lookup = load_federated_corpus(seed=1350)
        cls.theta0 = init_params(1350)

    def test_adversary_fractions_preserve_honest_minority_specialists(self) -> None:
        self.assertEqual(set(ADVERSARY_FRACTIONS), {"0%", "17%", "33%", "50%"})
        self.assertEqual(len(adversary_ids_for_fraction("0%")), 0)
        self.assertEqual(len(adversary_ids_for_fraction("17%")), 2)
        self.assertEqual(len(adversary_ids_for_fraction("33%")), 4)
        self.assertEqual(len(adversary_ids_for_fraction("50%")), 6)
        for label in ADVERSARY_FRACTIONS:
            adv = set(adversary_ids_for_fraction(label))
            self.assertTrue(adv.isdisjoint(set(MINORITY_CLIENT_IDS)))
        self.assertFalse(expected_bft_threshold_exceeded(4, 12))
        self.assertTrue(expected_bft_threshold_exceeded(6, 12))

    def test_colluding_cartel_is_near_collinear_and_inside_honest_norm_bound(
        self,
    ) -> None:
        subs = generate_round_submissions(
            round_idx=0,
            checkpoints=[self.theta0],
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("colluding_inner_product_cartel", "33%"),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        mean_cos = mean_cartel_pairwise_cosine(subs)
        self.assertGreater(mean_cos, 0.99)

        honest_norms = [l2_norm(s.delta) for s in subs if not s.is_malicious]
        mal_norms = [l2_norm(s.delta) for s in subs if s.is_malicious]
        med_honest = statistics.median(honest_norms)
        for mn in mal_norms:
            self.assertLessEqual(mn, med_honest + 1e-6)

    def test_stale_lag_records_exact_tau_and_distinguishes_honest_vs_malicious(
        self,
    ) -> None:
        # Build a 5-round checkpoint history
        ckpts = [list(self.theta0)]
        for r in range(5):
            prev = ckpts[-1]
            ckpts.append([v + 0.01 * (r + 1) for v in prev])

        honest_subs = generate_round_submissions(
            round_idx=5,
            checkpoints=ckpts,
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("stale_honest_straggler", "33%", staleness_tau=3),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        stragglers = [s for s in honest_subs if s.is_honest_straggler]
        self.assertEqual(len(stragglers), 4)
        for s in stragglers:
            self.assertFalse(s.is_malicious)
            self.assertEqual(s.staleness_lag, 3)
            self.assertEqual(s.base_round, 2)

        mal_subs = generate_round_submissions(
            round_idx=5,
            checkpoints=ckpts,
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("stale_malicious_replay", "33%", staleness_tau=3),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        malicious = [s for s in mal_subs if s.is_malicious]
        self.assertEqual(len(malicious), 4)
        for s in malicious:
            self.assertFalse(s.is_honest_straggler)
            self.assertEqual(s.staleness_lag, 3)

    def test_stealth_backdoor_preserves_clean_validation_loss_identically(self) -> None:
        benign_subs = generate_round_submissions(
            round_idx=0,
            checkpoints=[self.theta0],
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("benign", "0%"),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        stealth_subs = generate_round_submissions(
            round_idx=0,
            checkpoints=[self.theta0],
            corpus=self.corpus,
            raw_text_lookup=self.raw_lookup,
            attack=AttackConfig("stealth_unmonitored_backdoor", "17%"),
            lr=0.95,
            local_steps=3,
            batch_size=16,
            seed=1350,
        )
        # On client 0 (stealth attacker), clean validation loss must match benign
        # bit-for-bit because trigger weights only touch unmonitored feature slots!
        benign_p = vec_add(self.theta0, benign_subs[0].delta)
        stealth_p = vec_add(self.theta0, stealth_subs[0].delta)
        loss_benign = evaluate_clean(benign_p, self.corpus.eval_val).loss
        loss_stealth = evaluate_clean(stealth_p, self.corpus.eval_val).loss
        self.assertAlmostEqual(loss_benign, loss_stealth, places=12)
        self.assertEqual(len(stealth_subs[0].delta), NUM_PARAMS)
        self.assertNotEqual(benign_subs[0].delta, stealth_subs[0].delta)


if __name__ == "__main__":
    unittest.main()
