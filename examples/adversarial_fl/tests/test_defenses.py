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
"""Tests for all 6 defenses and the 3 explicit honest negative regimes."""

from __future__ import annotations

import unittest

from examples.adversarial_fl.attacks import AttackConfig
from examples.adversarial_fl.corpus import load_federated_corpus
from examples.adversarial_fl.defenses import DEFENSE_NAMES, DefenseConfig
from examples.adversarial_fl.simulation import run_federated_experiment


class TestDefensesAndNegativeRegimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, cls.raw_lookup = load_federated_corpus(seed=1350)
        cls.oracle = run_federated_experiment(
            scenario_id="test_oracle",
            corpus=cls.corpus,
            raw_text_lookup=cls.raw_lookup,
            attack=AttackConfig("benign", "0%"),
            defense=DefenseConfig("fedavg_undefended"),
            seed=1350,
            rounds=8,
        )

    def test_all_six_defenses_registered(self) -> None:
        self.assertEqual(
            DEFENSE_NAMES,
            (
                "fedavg_undefended",
                "norm_clipping",
                "coordinate_median",
                "trimmed_mean",
                "multi_krum",
                "dso_hybrid_eval_cosine",
            ),
        )

    def test_negative_regime_1_non_iid_honest_minority_frr(self) -> None:
        atk = AttackConfig("benign", "0%")
        krum = run_federated_experiment(
            "neg1_krum",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("multi_krum"),
            seed=1350,
            rounds=8,
        )
        tmean = run_federated_experiment(
            "neg1_tmean",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("trimmed_mean"),
            seed=1350,
            rounds=8,
        )
        dso = run_federated_experiment(
            "neg1_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine"),
            seed=1350,
            rounds=8,
        )
        # Geometric defenses reject minority infra specialists even at f=0%
        self.assertEqual(krum.minority_honest_acceptance_rate, 0.0)
        self.assertEqual(tmean.minority_honest_acceptance_rate, 0.0)
        self.assertGreater(krum.frr, 30.0)
        # DSO preserves honest minority specialists via macro-stratified validation
        self.assertGreater(dso.minority_honest_acceptance_rate, 80.0)
        self.assertGreater(dso.minority_domain_accuracy, krum.minority_domain_accuracy)

    def test_poisoning_33pct_mitigated_by_dso(self) -> None:
        atk = AttackConfig("label_flip_and_sign_flip", "33%")
        undef = run_federated_experiment(
            "pois_undef",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("fedavg_undefended"),
            seed=1350,
            rounds=8,
        )
        dso = run_federated_experiment(
            "pois_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine"),
            seed=1350,
            rounds=8,
        )
        self.assertLess(undef.clean_accuracy, 35.0)
        self.assertGreater(dso.clean_accuracy, 65.0)
        self.assertGreater(dso.attacker_detection_rate, 75.0)

    def test_collusion_cartel_33pct_fools_krum_but_caught_by_dso(self) -> None:
        atk = AttackConfig("colluding_inner_product_cartel", "33%")
        krum = run_federated_experiment(
            "cartel_krum",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("multi_krum"),
            seed=1350,
            rounds=8,
        )
        dso = run_federated_experiment(
            "cartel_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine"),
            seed=1350,
            rounds=8,
        )
        self.assertGreater(krum.mean_cartel_cosine, 0.99)
        self.assertGreater(dso.clean_accuracy, krum.clean_accuracy + 15.0)
        self.assertGreater(dso.attacker_detection_rate, 90.0)

    def test_honest_straggler_preserved_with_low_frr(self) -> None:
        atk = AttackConfig("stale_honest_straggler", "33%", staleness_tau=3)
        dso = run_federated_experiment(
            "straggler_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine"),
            seed=1350,
            rounds=8,
        )
        self.assertLessEqual(dso.honest_straggler_frr, 15.0)
        self.assertGreater(dso.clean_accuracy, 70.0)

    def test_negative_regime_2_unmonitored_stealth_backdoor_blindness(self) -> None:
        atk = AttackConfig("stealth_unmonitored_backdoor", "33%")
        dso = run_federated_experiment(
            "neg2_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine"),
            seed=1350,
            rounds=8,
        )
        # Honest negative finding: validation gate alone cannot catch unmonitored trigger
        self.assertGreater(dso.clean_accuracy, 70.0)
        self.assertGreater(dso.asr, 70.0)
        self.assertLess(dso.attacker_detection_rate, 10.0)

    def test_negative_regime_3_fifty_percent_collusion_breakdown(self) -> None:
        atk = AttackConfig("colluding_inner_product_cartel", "50%")
        for dname in ("coordinate_median", "trimmed_mean", "multi_krum"):
            res = run_federated_experiment(
                f"neg3_{dname}",
                self.corpus,
                self.raw_lookup,
                atk,
                DefenseConfig(dname),
                seed=1350,
                rounds=8,
            )
            self.assertLess(res.clean_accuracy, 35.0, msg=f"{dname} should break at 50%")

        strict_dso = run_federated_experiment(
            "neg3_strict_dso",
            self.corpus,
            self.raw_lookup,
            atk,
            DefenseConfig("dso_hybrid_eval_cosine", dso_strict_bft_quorum=True),
            seed=1350,
            rounds=8,
        )
        self.assertEqual(strict_dso.quorum_stall_rounds, 8)


if __name__ == "__main__":
    unittest.main()
