# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Tests for frozen base LM, 4-bit NF4 QLoRA quantization, LoRA autodiff, and SVD."""

from __future__ import annotations

import math
import random
import unittest

from examples.federated_lora_zen.corpus import LMExample
from examples.federated_lora_zen.lora_model import (
    LAYER_NAMES,
    NF4_CODEBOOK,
    FrozenBaseLM,
    LoRALanguageModel,
    ModelConfig,
    init_lora_adapter,
    matmul,
    matrix_frobenius_diff,
    quantize_matrix_nf4,
    svd_factor_delta,
)


def _synthetic_examples(
    vocab_size: int = 80, context_len: int = 4, count: int = 16
) -> list[LMExample]:
    rng = random.Random(42)
    return [
        LMExample(
            context=tuple(rng.randrange(vocab_size) for _ in range(context_len)),
            target=rng.randrange(vocab_size),
            domain="docs",
            chunk_id="test:0",
        )
        for _ in range(count)
    ]


class TestNF4Quantization(unittest.TestCase):
    def test_nf4_packing_unpacking_and_accuracy(self) -> None:
        rng = random.Random(1350)
        mat = [[rng.gauss(0.0, 0.2) for _ in range(64)] for _ in range(64)]
        qm = quantize_matrix_nf4(mat, block_size=64)

        self.assertEqual(qm.num_params, 4096)
        # Exactly 2 weights per byte packed as 4-bit nibbles.
        self.assertEqual(qm.packed_nibble_bytes, 2048)
        self.assertEqual(len(qm.block_scales), 64)
        self.assertEqual(qm.scale_bytes_fp16, 128)
        self.assertLess(qm.total_quantized_bytes, qm.fp32_bytes // 7)
        self.assertLess(qm.rmse, 0.03)

        deq = qm.dequantize()
        self.assertEqual(len(deq), 64)
        self.assertEqual(len(deq[0]), 64)

        # Verify every dequantized element equals a valid NF4 codebook value * block scale.
        for r in range(64):
            scale = qm.block_scales[r]
            valid = {c * scale for c in NF4_CODEBOOK}
            for c in range(64):
                self.assertTrue(
                    any(math.isclose(deq[r][c], v, abs_tol=1e-12) for v in valid)
                )

    def test_nf4_zero_matrix_safe(self) -> None:
        zeros = [[0.0] * 32 for _ in range(16)]
        qm = quantize_matrix_nf4(zeros, block_size=64)
        deq = qm.dequantize()
        self.assertEqual(qm.rmse, 0.0)
        for row in deq:
            self.assertTrue(all(v == 0.0 for v in row))


class TestLoRAAndQLoRAModel(unittest.TestCase):
    def test_zero_init_lora_preserves_frozen_base(self) -> None:
        cfg = ModelConfig()
        base = FrozenBaseLM.create_initialized(cfg, seed=1350)
        adapter = init_lora_adapter(cfg, rank=4, alpha=8.0, seed=1350)
        # At init, B == 0 so effective DeltaW == 0 everywhere.
        deltas = adapter.effective_deltas()
        for name in LAYER_NAMES:
            for row in deltas[name]:
                self.assertTrue(all(v == 0.0 for v in row))

        model = LoRALanguageModel(base, adapter)
        exs = _synthetic_examples()
        model.train_step_adamw(exs, lr=0.02)
        self.assertTrue(base.verify_frozen())

    def test_finite_difference_gradient_check_fp32_and_nf4(self) -> None:
        cfg = ModelConfig()
        base_fp32 = FrozenBaseLM.create_initialized(cfg, seed=1350)
        base_nf4 = base_fp32.to_quantization_mode("nf4")
        exs = _synthetic_examples(count=8)

        for base in (base_fp32, base_nf4):
            adapter = init_lora_adapter(cfg, rank=4, alpha=8.0, seed=7)
            model = LoRALanguageModel(base, adapter)
            # Take one step so B becomes non-zero and gradients flow into A too.
            model.train_step_adamw(exs, lr=0.03)
            _, ga, gb = model.compute_batch_loss_and_grads(exs)

            eps = 1e-5
            max_rel_err = 0.0
            for name in LAYER_NAMES:
                for mat, gmat in (
                    (model.adapter.a_mats[name], ga[name]),
                    (model.adapter.b_mats[name], gb[name]),
                ):
                    for i in (0, len(mat) - 1):
                        for j in (0, len(mat[0]) - 1):
                            orig = mat[i][j]
                            mat[i][j] = orig + eps
                            lp, _, _ = model.compute_batch_loss_and_grads(exs)
                            mat[i][j] = orig - eps
                            lm, _, _ = model.compute_batch_loss_and_grads(exs)
                            mat[i][j] = orig
                            num = (lp - lm) / (2.0 * eps)
                            ana = gmat[i][j]
                            rel = abs(num - ana) / max(
                                1e-7, abs(num) + abs(ana)
                            )
                            max_rel_err = max(max_rel_err, rel)
            self.assertLess(max_rel_err, 1e-5)
            self.assertTrue(base.verify_frozen())

    def test_qlora_memory_footprint_reduction(self) -> None:
        cfg = ModelConfig()
        base_fp32 = FrozenBaseLM.create_initialized(cfg, seed=1350)
        base_nf4 = base_fp32.to_quantization_mode("nf4")
        adapter = init_lora_adapter(cfg, rank=8, alpha=16.0, seed=1350)

        fp32_mem = LoRALanguageModel(base_fp32, adapter).memory_footprint()
        nf4_mem = LoRALanguageModel(base_nf4, adapter).memory_footprint()

        self.assertGreater(nf4_mem.base_memory_savings_vs_fp32, 5.0)
        self.assertGreater(nf4_mem.base_memory_savings_vs_fp16, 2.5)
        self.assertLess(nf4_mem.base_active_bytes, fp32_mem.base_active_bytes)
        self.assertEqual(nf4_mem.lora_fp32_bytes, fp32_mem.lora_fp32_bytes)


class TestExactSVDFactorization(unittest.TestCase):
    def test_exact_recovery_of_rank_r_matrix(self) -> None:
        rng = random.Random(1350)
        m, n, r = 40, 32, 4
        b_true = [[rng.gauss(0.0, 0.3) for _ in range(r)] for _ in range(m)]
        a_true = [[rng.gauss(0.0, 0.3) for _ in range(n)] for _ in range(r)]
        delta = matmul(b_true, a_true)

        b_rec, a_rec, residual = svd_factor_delta(delta, rank=r)
        self.assertLess(residual, 1e-9)
        self.assertLess(matrix_frobenius_diff(delta, matmul(b_rec, a_rec)), 1e-9)

    def test_svd_beats_naive_fedavg_on_orthogonal_worker_updates(self) -> None:
        rng = random.Random(1350)
        m, n, r = 32, 32, 4
        # Three workers with rotated subspaces: naive mean(B)@mean(A) suffers
        # cross-term error, whereas rank-r SVD achieves optimal Frobenius error.
        b_list = [
            [[rng.gauss(0.0, 0.4) for _ in range(r)] for _ in range(m)]
            for _ in range(3)
        ]
        a_list = [
            [[rng.gauss(0.0, 0.4) for _ in range(n)] for _ in range(r)]
            for _ in range(3)
        ]
        exact = [[0.0] * n for _ in range(m)]
        avg_b = [[0.0] * r for _ in range(m)]
        avg_a = [[0.0] * n for _ in range(r)]
        for k in range(3):
            ba = matmul(b_list[k], a_list[k])
            for i in range(m):
                for j in range(n):
                    exact[i][j] += ba[i][j] / 3.0
                for j in range(r):
                    avg_b[i][j] += b_list[k][i][j] / 3.0
            for i in range(r):
                for j in range(n):
                    avg_a[i][j] += a_list[k][i][j] / 3.0

        naive_err = matrix_frobenius_diff(matmul(avg_b, avg_a), exact)
        _, _, svd_err = svd_factor_delta(exact, rank=r)
        self.assertLess(svd_err, naive_err)


if __name__ == "__main__":
    unittest.main()
