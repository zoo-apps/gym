# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Tests for SHA-256 pinned corpus loading and strict 3-worker vault disjointness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.federated_lora_zen.corpus import (
    DOMAINS,
    PINNED_CORPUS_FILES,
    WORKER_IDS,
    build_federated_corpus,
    load_pinned_chunks,
)


class TestCorpusPinningAndDisjointness(unittest.TestCase):
    def test_pinned_files_exist_and_verify_sha256(self) -> None:
        chunks_by_domain, manifest_sha, total_files, total_bytes = (
            load_pinned_chunks()
        )
        expected_files = sum(len(PINNED_CORPUS_FILES[d]) for d in DOMAINS)
        self.assertEqual(total_files, expected_files)
        self.assertGreater(total_bytes, 100_000)
        self.assertEqual(len(manifest_sha), 64)
        for d in DOMAINS:
            self.assertGreater(len(chunks_by_domain[d]), 20)

    def test_corrupted_file_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_rel = PINNED_CORPUS_FILES["docs"][0][0]
            target = root / first_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"tampered content")
            with self.assertRaises(ValueError):
                load_pinned_chunks(repo_root=root)

    def test_strict_chunk_disjointness_mixed_non_iid(self) -> None:
        split = build_federated_corpus(seed=1350, partition_mode="mixed_non_iid")
        base_ids = set(split.base_pretrain_chunk_ids)
        eval_ids = set(split.eval_test_chunk_ids)
        self.assertTrue(base_ids.isdisjoint(eval_ids))

        vault_sets = {
            wid: set(split.worker_vaults[wid].chunk_ids) for wid in WORKER_IDS
        }
        # Pairwise worker vault disjointness + disjoint from base and eval.
        for i, w1 in enumerate(WORKER_IDS):
            self.assertGreater(len(vault_sets[w1]), 0)
            self.assertTrue(vault_sets[w1].isdisjoint(base_ids))
            self.assertTrue(vault_sets[w1].isdisjoint(eval_ids))
            for w2 in WORKER_IDS[i + 1 :]:
                self.assertTrue(vault_sets[w1].isdisjoint(vault_sets[w2]))

        # Union of worker vaults equals centralized training split exactly.
        central_ids = {ex.chunk_id for ex in split.centralized_train}
        union_ids = set().union(*vault_sets.values())
        self.assertEqual(central_ids, union_ids)

        # Each worker has a unique local admin secret key.
        keys = {split.worker_vaults[w].admin_key_hex for w in WORKER_IDS}
        self.assertEqual(len(keys), len(WORKER_IDS))

    def test_extreme_non_iid_is_pure_single_domain_per_worker(self) -> None:
        split = build_federated_corpus(
            seed=1350, partition_mode="extreme_non_iid"
        )
        expected_dom = {
            "worker-alpha": "docs",
            "worker-beta": "yaml",
            "worker-gamma": "python",
        }
        for wid, dom in expected_dom.items():
            counts = split.worker_vaults[wid].domain_counts
            self.assertGreater(counts[dom], 0)
            for other in DOMAINS:
                if other != dom:
                    self.assertEqual(counts[other], 0)

    def test_corpus_split_is_deterministic(self) -> None:
        s1 = build_federated_corpus(seed=1350, partition_mode="mixed_non_iid")
        s2 = build_federated_corpus(seed=1350, partition_mode="mixed_non_iid")
        self.assertEqual(s1.corpus_manifest_sha256, s2.corpus_manifest_sha256)
        for wid in WORKER_IDS:
            self.assertEqual(
                s1.worker_vaults[wid].vault_sha256,
                s2.worker_vaults[wid].vault_sha256,
            )


if __name__ == "__main__":
    unittest.main()
