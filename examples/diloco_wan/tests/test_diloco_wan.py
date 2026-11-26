# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Comprehensive unittest suite for zoo-apps/gym#4 (HIP-1350) DiLoCo WAN training.

Covers:
  1. SHA-256 pinned repository corpus & strictly disjoint private/val/test splits
  2. Binary wire quantization codecs (FP32, INT8, INT4, 1-bit sign) & compression
  3. Cryptographic contributor envelope authentication & replay/tamper rejection
  4. Zero private-data transmission audit invariant
  5. Exact deterministic convergence & reproducibility across repeated runs
  6. H=64 network-byte reduction (>60x FP32, >200x INT8) & WAN speedup vs H=1 AllReduce
  7. Packet-loss, mid-round churn, and fresh elastic-join checkpoint catchup recovery
  8. Independent held-out verification gate rejecting 100% of Byzantine poisoned updates
  9. Negative Regime 1: Extreme H + Non-IID shards (client-drift degradation)
 10. Negative Regime 2: 20% packet loss + 60% churn (quorum loss & wall-time blowup)
 11. Negative Regime 3: 1-bit sign compression without error feedback (quantization bias)
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
import unittest

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from diloco_wan.corpus import VOCAB_SIZE, load_corpus  # noqa: E402
from diloco_wan.diloco import RegimeConfig, run_regime  # noqa: E402
from diloco_wan.model import PARAM_COUNT, CausalCharLM  # noqa: E402
from diloco_wan.quantize import (  # noqa: E402
    CODEC_FP32,
    CODEC_INT4_BLOCK,
    CODEC_INT8_BLOCK,
    CODEC_SIGN_1BIT_NO_EF,
    decode_tensor,
    encode_tensor,
)
from diloco_wan.verify import (  # noqa: E402
    ContributorKeyring,
    SignedUpdateEnvelope,
    UpdateVerifier,
)
from diloco_wan.wan import WanProfile  # noqa: E402


class TestDiLoCoWan(unittest.TestCase):
    """End-to-end verification of low-communication WAN DiLoCo training."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_balanced = load_corpus(
            num_workers=6,
            partition_mode="balanced",
            seed=1350,
            max_val_examples=600,
            max_test_examples=600,
        )
        cls.corpus_non_iid = load_corpus(
            num_workers=6,
            partition_mode="non_iid",
            seed=1350,
            max_val_examples=600,
            max_test_examples=600,
        )

    def test_01_corpus_manifest_pinned_and_disjoint(self) -> None:
        c1 = self.corpus_balanced
        c2 = load_corpus(
            num_workers=6,
            partition_mode="balanced",
            seed=1350,
            max_val_examples=600,
            max_test_examples=600,
        )
        self.assertGreaterEqual(len(c1.file_pins), 40)
        self.assertEqual(c1.manifest_sha256, c2.manifest_sha256)
        self.assertEqual(len(c1.manifest_sha256), 64)
        self.assertGreater(c1.total_raw_bytes, 150_000)
        self.assertEqual(len(c1.vault.worker_ids()), 6)
        for wid in c1.vault.worker_ids():
            self.assertGreater(len(c1.vault.worker_shards[wid]), 500)
        self.assertEqual(len(c1.verifier_val), 600)
        self.assertEqual(len(c1.global_test), 600)
        # Non-IID shards partition target classes disjointly across workers.
        target_sets = [
            {ex.target for ex in self.corpus_non_iid.vault.worker_shards[wid]}
            for wid in self.corpus_non_iid.vault.worker_ids()
        ]
        for i in range(len(target_sets)):
            for j in range(i + 1, len(target_sets)):
                self.assertTrue(target_sets[i].isdisjoint(target_sets[j]))

    def test_02_quantize_codecs_roundtrip_and_compression(self) -> None:
        rng = random.Random(1350)
        vec = [rng.gauss(0.0, 0.08) for _ in range(PARAM_COUNT)]

        b_fp32 = encode_tensor(vec, CODEC_FP32)
        b_int8 = encode_tensor(vec, CODEC_INT8_BLOCK)
        b_int4 = encode_tensor(vec, CODEC_INT4_BLOCK)
        b_1bit = encode_tensor(vec, CODEC_SIGN_1BIT_NO_EF)

        self.assertGreater(len(b_fp32) / len(b_int8), 3.5)
        self.assertGreater(len(b_fp32) / len(b_int4), 6.5)
        self.assertGreater(len(b_fp32) / len(b_1bit), 25.0)

        def rel_err(decoded: list[float]) -> float:
            num = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, decoded, strict=True)))
            den = math.sqrt(sum(a * a for a in vec))
            return num / den

        c_fp32, d_fp32 = decode_tensor(b_fp32)
        c_int8, d_int8 = decode_tensor(b_int8)
        c_int4, d_int4 = decode_tensor(b_int4)
        c_1bit, d_1bit = decode_tensor(b_1bit)

        self.assertEqual(c_fp32, CODEC_FP32)
        self.assertEqual(c_int8, CODEC_INT8_BLOCK)
        self.assertEqual(c_int4, CODEC_INT4_BLOCK)
        self.assertEqual(c_1bit, CODEC_SIGN_1BIT_NO_EF)

        self.assertLess(rel_err(d_fp32), 1e-6)
        self.assertLess(rel_err(d_int8), 0.02)
        self.assertLess(rel_err(d_int4), 0.20)
        # 1-bit sign without error feedback has far worse reconstruction error than int8/int4.
        self.assertGreater(rel_err(d_1bit), rel_err(d_int4) * 3.0)

    def test_03_keyring_auth_and_replay_rejection(self) -> None:
        kr = ContributorKeyring(master_seed=1350)
        kr.register("worker-0")
        payload = encode_tensor([0.01] * PARAM_COUNT, CODEC_INT8_BLOCK)
        env = kr.sign(round_id=1, contributor_id="worker-0", codec=CODEC_INT8_BLOCK, payload_bytes=payload)

        ok, reason = kr.verify_signature(env)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

        # Tampered signature
        bad_sig_env = SignedUpdateEnvelope(
            round_id=env.round_id,
            contributor_id=env.contributor_id,
            codec=env.codec,
            payload_bytes=env.payload_bytes,
            payload_sha256=env.payload_sha256,
            signature_hex="0" * 64,
        )
        ok_bad, reason_bad = kr.verify_signature(bad_sig_env)
        self.assertFalse(ok_bad)
        self.assertEqual(reason_bad, "bad_signature")

        # Tampered payload bytes
        tampered_bytes = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        tampered_env = SignedUpdateEnvelope(
            round_id=env.round_id,
            contributor_id=env.contributor_id,
            codec=env.codec,
            payload_bytes=tampered_bytes,
            payload_sha256=env.payload_sha256,
            signature_hex=env.signature_hex,
        )
        ok_t, reason_t = kr.verify_signature(tampered_env)
        self.assertFalse(ok_t)
        self.assertEqual(reason_t, "digest_mismatch")

        # Replay / wrong round check inside UpdateVerifier
        model = CausalCharLM(seed=1350)
        verifier = UpdateVerifier(keyring=kr, verifier_val=self.corpus_balanced.verifier_val)
        accepted, receipt = verifier.verify_round(round_id=2, base_model=model, envelopes=[env])
        self.assertEqual(len(accepted), 0)
        self.assertEqual(receipt.decisions[0].reason, "replay_or_wrong_round")

    def test_04_zero_private_data_leakage_invariant(self) -> None:
        vault = self.corpus_balanced.vault
        self.assertGreater(len(vault.canary_substrings), 0)
        # Real quantized tensor passes audit cleanly
        payload = encode_tensor([0.005] * PARAM_COUNT, CODEC_INT8_BLOCK)
        vault.assert_payload_contains_no_private_data(payload)
        # Injecting raw private text immediately triggers privacy AssertionError
        leaked = payload + vault.canary_substrings[0]
        with self.assertRaises(AssertionError):
            vault.assert_payload_contains_no_private_data(leaked)

    def test_05_deterministic_convergence_and_reproducibility(self) -> None:
        wan = WanProfile(name="wan_test", rtt_ms=100.0, bandwidth_mbps=50.0)
        cfg = RegimeConfig(
            name="det_diloco",
            mode="diloco",
            description="Determinism check",
            total_steps_per_worker=128,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            wan=wan,
        )
        r1 = run_regime(cfg, self.corpus_balanced, seed=1350)
        r2 = run_regime(cfg, self.corpus_balanced, seed=1350)
        self.assertEqual(r1.to_dict(), r2.to_dict())
        self.assertLess(r1.final_test_loss, r1.init_test_loss - 0.5)
        self.assertGreater(r1.final_test_top1_acc, 1.0 / VOCAB_SIZE * 5.0)
        self.assertEqual(r1.private_data_leaked_bytes, 0)
        self.assertGreater(r1.privacy_audits_passed, 0)

    def test_06_h64_byte_reduction_and_wan_speedup_vs_allreduce(self) -> None:
        wan = WanProfile(name="wan_100ms", rtt_ms=100.0, bandwidth_mbps=50.0)
        ar_cfg = RegimeConfig(
            name="ar_h1",
            mode="allreduce",
            description="H=1 AllReduce",
            total_steps_per_worker=64,
            local_steps_h=1,
            inner_lr=0.012,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan,
        )
        h64_fp32_cfg = RegimeConfig(
            name="diloco_h64_fp32",
            mode="diloco",
            description="H=64 FP32",
            total_steps_per_worker=64,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan,
        )
        h64_int8_cfg = RegimeConfig(
            name="diloco_h64_int8",
            mode="diloco",
            description="H=64 INT8",
            total_steps_per_worker=64,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            wan=wan,
        )
        res_ar = run_regime(ar_cfg, self.corpus_balanced, seed=1350)
        res_fp32 = run_regime(h64_fp32_cfg, self.corpus_balanced, seed=1350)
        res_int8 = run_regime(h64_int8_cfg, self.corpus_balanced, seed=1350)

        self.assertGreater(res_ar.total_wire_bytes / res_fp32.total_wire_bytes, 60.0)
        self.assertGreater(res_ar.total_wire_bytes / res_int8.total_wire_bytes, 200.0)
        self.assertGreater(res_ar.simulated_wall_time_s / res_int8.simulated_wall_time_s, 15.0)
        self.assertGreater(res_int8.compute_utilization_pct, res_ar.compute_utilization_pct * 10.0)
        self.assertEqual(res_int8.private_data_leaked_bytes, 0)

    def test_07_packet_loss_churn_and_elastic_join_recovery(self) -> None:
        wan = WanProfile(
            name="wan_elastic",
            rtt_ms=100.0,
            jitter_ms=15.0,
            bandwidth_mbps=50.0,
            packet_loss_rate=0.04,
            churn_drop_prob=0.12,
            straggler_prob=0.10,
            elastic_join_round=2,
            min_quorum=2,
        )
        cfg = RegimeConfig(
            name="elastic_test",
            mode="diloco",
            description="Elastic churn + join",
            total_steps_per_worker=192,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            verify_updates=True,
            wan=wan,
        )
        res = run_regime(cfg, self.corpus_balanced, seed=1350)
        self.assertEqual(res.elastic_joins, 1)
        self.assertGreater(res.checkpoint_sync_bytes, PARAM_COUNT * 4)
        self.assertGreater(res.packets_sent, 0)
        self.assertGreaterEqual(res.sync_rounds_applied, 2)
        self.assertLess(res.final_test_loss, res.init_test_loss - 0.6)

    def test_08_byzantine_poison_rejection_by_independent_gate(self) -> None:
        wan = WanProfile(
            name="wan_byzantine",
            rtt_ms=100.0,
            bandwidth_mbps=50.0,
            packet_loss_rate=0.01,
            churn_drop_prob=0.0,
            min_quorum=2,
        )
        base_kwargs = dict(
            mode="diloco",
            total_steps_per_worker=192,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            byzantine_attack="mixed_poison",
            wan=wan,
        )
        verified = run_regime(
            RegimeConfig(name="verified", description="gate on", verify_updates=True, **base_kwargs),
            self.corpus_balanced,
            seed=1350,
        )
        unverified = run_regime(
            RegimeConfig(name="unverified", description="gate off", verify_updates=False, **base_kwargs),
            self.corpus_balanced,
            seed=1350,
        )
        self.assertGreater(verified.poisoned_submitted, 0)
        self.assertEqual(verified.poisoned_rejection_rate, 1.0)
        self.assertIsNotNone(verified.honest_acceptance_rate)
        self.assertGreaterEqual(verified.honest_acceptance_rate or 0.0, 0.9)
        self.assertEqual(unverified.poisoned_rejection_rate, 0.0)
        # Independent held-out global test loss must be strictly better with verification enabled.
        self.assertLess(verified.final_test_loss, unverified.final_test_loss - 0.25)

    def test_09_negative_regime_1_extreme_h_non_iid_client_drift(self) -> None:
        wan = WanProfile(name="wan_clean", rtt_ms=100.0, bandwidth_mbps=50.0)
        healthy = run_regime(
            RegimeConfig(
                name="healthy_h32",
                mode="diloco",
                description="Balanced H=32",
                partition_mode="balanced",
                total_steps_per_worker=192,
                local_steps_h=32,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                wan=wan,
            ),
            self.corpus_balanced,
            seed=1350,
        )
        drifted = run_regime(
            RegimeConfig(
                name="neg_non_iid_h192",
                mode="diloco",
                description="Non-IID H=192",
                is_negative_regime=True,
                partition_mode="non_iid",
                total_steps_per_worker=192,
                local_steps_h=192,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                wan=wan,
            ),
            self.corpus_non_iid,
            seed=1350,
        )
        self.assertGreater(drifted.final_test_loss, healthy.final_test_loss + 0.5)
        self.assertLess(drifted.final_test_top1_acc, healthy.final_test_top1_acc - 0.10)

    def test_10_negative_regime_2_high_loss_severe_churn(self) -> None:
        clean_wan = WanProfile(name="wan_clean", rtt_ms=100.0, bandwidth_mbps=50.0)
        severe_wan = WanProfile(
            name="wan_severe",
            rtt_ms=250.0,
            jitter_ms=40.0,
            bandwidth_mbps=10.0,
            packet_loss_rate=0.20,
            max_retransmits=2,
            churn_drop_prob=0.60,
            straggler_prob=0.25,
            straggler_multiplier=4.0,
            round_timeout_s=1.8,
            min_quorum=3,
        )
        clean_res = run_regime(
            RegimeConfig(
                name="clean_h64",
                mode="diloco",
                description="Clean H=64",
                total_steps_per_worker=256,
                local_steps_h=64,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                codec=CODEC_INT8_BLOCK,
                broadcast_codec=CODEC_INT8_BLOCK,
                wan=clean_wan,
            ),
            self.corpus_balanced,
            seed=1350,
        )
        severe_res = run_regime(
            RegimeConfig(
                name="neg_severe",
                mode="diloco",
                description="20% loss + 60% churn",
                is_negative_regime=True,
                total_steps_per_worker=256,
                local_steps_h=64,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                codec=CODEC_INT8_BLOCK,
                broadcast_codec=CODEC_INT8_BLOCK,
                verify_updates=True,
                wan=severe_wan,
            ),
            self.corpus_balanced,
            seed=1350,
        )
        self.assertGreater(severe_res.quorum_misses + severe_res.transfers_dropped_churn, 5)
        self.assertGreater(severe_res.simulated_wall_time_s, clean_res.simulated_wall_time_s * 1.5)
        self.assertGreater(severe_res.final_test_loss, clean_res.final_test_loss + 0.05)

    def test_11_negative_regime_3_1bit_sign_no_error_feedback(self) -> None:
        wan = WanProfile(name="wan_clean", rtt_ms=100.0, bandwidth_mbps=50.0)
        int8_res = run_regime(
            RegimeConfig(
                name="int8_h64",
                mode="diloco",
                description="INT8 block H=64",
                total_steps_per_worker=256,
                local_steps_h=64,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                codec=CODEC_INT8_BLOCK,
                broadcast_codec=CODEC_INT8_BLOCK,
                wan=wan,
            ),
            self.corpus_balanced,
            seed=1350,
        )
        sign_res = run_regime(
            RegimeConfig(
                name="neg_1bit_h64",
                mode="diloco",
                description="1-bit sign without error feedback",
                is_negative_regime=True,
                total_steps_per_worker=256,
                local_steps_h=64,
                inner_lr=0.012,
                outer_lr=0.7,
                outer_momentum=0.6,
                codec=CODEC_SIGN_1BIT_NO_EF,
                broadcast_codec=CODEC_SIGN_1BIT_NO_EF,
                wan=wan,
            ),
            self.corpus_balanced,
            seed=1350,
        )
        # 1-bit sign cuts wire bytes further, but degrades held-out test loss vs INT8.
        self.assertLess(sign_res.total_wire_bytes, int8_res.total_wire_bytes)
        self.assertGreater(sign_res.final_test_loss, int8_res.final_test_loss + 0.02)


if __name__ == "__main__":
    unittest.main()
