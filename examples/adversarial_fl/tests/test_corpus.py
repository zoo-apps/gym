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
"""Tests for pinned repository corpus, SHA-256 manifest, and disjoint FL splits."""

from __future__ import annotations

import unittest

from examples.adversarial_fl.corpus import (
    BACKDOOR_TARGET_CLASS,
    DOMAINS,
    INPUT_DIM,
    MINORITY_CLIENT_IDS,
    MINORITY_DOMAIN,
    NUM_CLIENTS,
    TRIGGER_TOKENS,
    load_federated_corpus,
)


class TestFederatedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, cls.raw_lookup = load_federated_corpus(seed=1350)

    def test_manifest_pinned_and_deterministic(self) -> None:
        corpus2, _ = load_federated_corpus(seed=1350)
        self.assertEqual(self.corpus.manifest_sha256, corpus2.manifest_sha256)
        self.assertEqual(len(self.corpus.manifest), 74)
        self.assertGreater(self.corpus.total_bytes, 150_000)
        for entry in self.corpus.manifest:
            self.assertFalse(entry.rel_path.startswith("/"))
            self.assertEqual(len(entry.sha256), 64)

    def test_all_fourteen_splits_strictly_disjoint(self) -> None:
        self.assertEqual(len(self.corpus.clients), NUM_CLIENTS)
        seen: set[str] = set()
        for vault in self.corpus.clients:
            for ex in vault.examples:
                self.assertNotIn(ex.chunk_id, seen)
                seen.add(ex.chunk_id)
        for ex in self.corpus.eval_val:
            self.assertNotIn(ex.chunk_id, seen)
            seen.add(ex.chunk_id)
        for ex in self.corpus.eval_clean:
            self.assertNotIn(ex.chunk_id, seen)
            seen.add(ex.chunk_id)
        self.assertGreater(len(seen), 1_400)

    def test_minority_specialists_and_non_iid_skew(self) -> None:
        for vault in self.corpus.clients:
            if vault.client_id in MINORITY_CLIENT_IDS:
                self.assertTrue(vault.is_minority_specialist)
                self.assertEqual(vault.primary_domain, MINORITY_DOMAIN)
                self.assertEqual(
                    vault.domain_counts[MINORITY_DOMAIN], len(vault.examples)
                )
            else:
                self.assertFalse(vault.is_minority_specialist)
                self.assertEqual(vault.domain_counts[MINORITY_DOMAIN], 0)

    def test_trigger_tokens_absent_from_clean_splits(self) -> None:
        self.assertEqual(len(self.corpus.vocab), INPUT_DIM)
        self.assertEqual(self.corpus.vocab[-2:], TRIGGER_TOKENS)
        trig_ids = set(self.corpus.trigger_token_ids)
        for split in (self.corpus.eval_clean, self.corpus.eval_val):
            for ex in split:
                feat_ids = {idx for idx, _v in ex.features}
                self.assertTrue(feat_ids.isdisjoint(trig_ids))

    def test_eval_trigger_has_target_label_and_active_trigger_features(self) -> None:
        self.assertGreater(len(self.corpus.eval_trigger), 150)
        trig_ids = set(self.corpus.trigger_token_ids)
        for ex in self.corpus.eval_trigger:
            self.assertTrue(ex.is_triggered)
            self.assertEqual(ex.label, BACKDOOR_TARGET_CLASS)
            feat_ids = {idx for idx, _v in ex.features}
            self.assertTrue(trig_ids.issubset(feat_ids))
        # Verify all 4 domains present in eval_clean and eval_val
        for split in (self.corpus.eval_clean, self.corpus.eval_val):
            present = {ex.domain for ex in split}
            self.assertEqual(present, set(DOMAINS))


if __name__ == "__main__":
    unittest.main()
