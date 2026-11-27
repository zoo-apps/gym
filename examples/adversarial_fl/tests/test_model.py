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
"""Tests for MLP classifier analytical gradients, convergence, and vector algebra."""

from __future__ import annotations

import unittest

from examples.adversarial_fl.corpus import load_federated_corpus
from examples.adversarial_fl.model import (
    B1_OFFSET,
    B2_OFFSET,
    NUM_PARAMS,
    W2_OFFSET,
    clip_to_norm,
    cosine_similarity,
    evaluate_clean,
    finite_difference_grad,
    init_params,
    l2_norm,
    local_train_delta,
    loss_and_grad,
    vec_add,
)


class TestMLPModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, _ = load_federated_corpus(seed=1350)

    def test_analytical_gradients_match_finite_differences(self) -> None:
        params = init_params(1350)
        batch = self.corpus.eval_val[:10]
        _loss, anal_grad = loss_and_grad(params, batch, weight_decay=1e-4)

        # Check dense slice of W1, all of b1, all of W2, all of b2
        check_indices = (
            list(range(0, 96))
            + list(range(500, 560))
            + list(range(B1_OFFSET, W2_OFFSET))
            + list(range(W2_OFFSET, NUM_PARAMS))
        )
        fd_grad = finite_difference_grad(
            params, batch, eps=1e-5, weight_decay=1e-4, indices=check_indices
        )
        max_abs_err = max(abs(anal_grad[i] - fd_grad[i]) for i in check_indices)
        self.assertLess(max_abs_err, 1e-6)
        self.assertEqual(len(anal_grad), NUM_PARAMS)
        self.assertEqual(B2_OFFSET + 4, NUM_PARAMS)

    def test_local_sgd_reduces_training_loss(self) -> None:
        params = init_params(1350)
        examples = self.corpus.clients[0].examples
        loss_before = evaluate_clean(params, examples).loss
        delta = local_train_delta(
            params, examples, lr=0.8, local_steps=5, batch_size=24, seed=42
        )
        updated = vec_add(params, delta)
        loss_after = evaluate_clean(updated, examples).loss
        self.assertLess(loss_after, loss_before)

    def test_vector_primitives_norm_clip_and_cosine(self) -> None:
        a = [3.0, 4.0, 0.0]
        b = [6.0, 8.0, 0.0]
        c = [-3.0, -4.0, 0.0]
        self.assertAlmostEqual(l2_norm(a), 5.0, places=9)
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=9)
        self.assertAlmostEqual(cosine_similarity(a, c), -1.0, places=9)

        clipped, was_clipped = clip_to_norm(b, max_norm=2.5)
        self.assertTrue(was_clipped)
        self.assertAlmostEqual(l2_norm(clipped), 2.5, places=9)

        unclipped, was_clipped2 = clip_to_norm(a, max_norm=10.0)
        self.assertFalse(was_clipped2)
        self.assertEqual(unclipped, a)


if __name__ == "__main__":
    unittest.main()
