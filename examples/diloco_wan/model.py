# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Pure-Python deterministic Neural Causal Language Model (Bengio Residual NPLM).

Zero external dependencies (Python 3.10+ stdlib only). Uses a 4-character causal
context window over the fixed 64-symbol vocabulary with:
  - Token embeddings E in R^{64 x 8}          (512 params)
  - Hidden MLP layer W1 in R^{16 x 32}, b1    (528 params)
  - Output projection W2 in R^{64 x 16}, b2   (1,088 params)
  - Direct skip-1 bigram matrix W_skip1       (4,096 params)
  - Direct skip-2 bigram matrix W_skip2       (4,096 params)
Total trainable parameters: P = 10,320 (41,280 bytes in FP32).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from operator import mul
from typing import Dict, List, Sequence, Tuple

from .corpus import CONTEXT_LEN, VOCAB_SIZE, Example

EMBED_DIM = 8
HIDDEN_DIM = 16
INPUT_DIM = CONTEXT_LEN * EMBED_DIM  # 32
PARAM_COUNT = (
    VOCAB_SIZE * EMBED_DIM
    + HIDDEN_DIM * INPUT_DIM
    + HIDDEN_DIM
    + VOCAB_SIZE * HIDDEN_DIM
    + VOCAB_SIZE
    + VOCAB_SIZE * VOCAB_SIZE
    + VOCAB_SIZE * VOCAB_SIZE
)  # 10,320


@dataclass(frozen=True)
class EvalMetrics:
    """Held-out causal language modeling quality metrics."""

    loss: float
    perplexity: float
    top1_acc: float
    top5_acc: float
    num_examples: int


def _xavier_matrix(
    rows: int, cols: int, rng: random.Random, scale: float = 1.0
) -> List[List[float]]:
    limit = scale * math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


class CausalCharLM:
    """Bengio-style Residual Neural Probabilistic Language Model."""

    def __init__(self, seed: int = 1350) -> None:
        rng = random.Random(seed)
        self.embed: List[List[float]] = _xavier_matrix(
            VOCAB_SIZE, EMBED_DIM, rng, scale=0.6
        )
        self.w1: List[List[float]] = _xavier_matrix(
            HIDDEN_DIM, INPUT_DIM, rng, scale=1.0
        )
        self.b1: List[float] = [0.0] * HIDDEN_DIM
        self.w2: List[List[float]] = _xavier_matrix(
            VOCAB_SIZE, HIDDEN_DIM, rng, scale=0.5
        )
        self.b2: List[float] = [0.0] * VOCAB_SIZE
        # Direct skip matrices initialized to zero so initial predictions are uniform.
        self.w_skip1: List[List[float]] = [
            [0.0] * VOCAB_SIZE for _ in range(VOCAB_SIZE)
        ]
        self.w_skip2: List[List[float]] = [
            [0.0] * VOCAB_SIZE for _ in range(VOCAB_SIZE)
        ]

    @property
    def param_count(self) -> int:
        return PARAM_COUNT

    def get_flat_params(self) -> List[float]:
        """Export all 10,320 trainable weights into a deterministic flat list."""
        flat: List[float] = []
        for row in self.embed:
            flat.extend(row)
        for row in self.w1:
            flat.extend(row)
        flat.extend(self.b1)
        for row in self.w2:
            flat.extend(row)
        flat.extend(self.b2)
        for row in self.w_skip1:
            flat.extend(row)
        for row in self.w_skip2:
            flat.extend(row)
        return flat

    def set_flat_params(self, flat: Sequence[float]) -> None:
        """Load all 10,320 trainable weights from a flat sequence."""
        if len(flat) != PARAM_COUNT:
            raise ValueError(f"Expected {PARAM_COUNT} params, got {len(flat)}")
        idx = 0
        for r in range(VOCAB_SIZE):
            self.embed[r] = list(flat[idx : idx + EMBED_DIM])
            idx += EMBED_DIM
        for r in range(HIDDEN_DIM):
            self.w1[r] = list(flat[idx : idx + INPUT_DIM])
            idx += INPUT_DIM
        self.b1 = list(flat[idx : idx + HIDDEN_DIM])
        idx += HIDDEN_DIM
        for r in range(VOCAB_SIZE):
            self.w2[r] = list(flat[idx : idx + HIDDEN_DIM])
            idx += HIDDEN_DIM
        self.b2 = list(flat[idx : idx + VOCAB_SIZE])
        idx += VOCAB_SIZE
        for r in range(VOCAB_SIZE):
            self.w_skip1[r] = list(flat[idx : idx + VOCAB_SIZE])
            idx += VOCAB_SIZE
        for r in range(VOCAB_SIZE):
            self.w_skip2[r] = list(flat[idx : idx + VOCAB_SIZE])
            idx += VOCAB_SIZE

    def clone(self) -> "CausalCharLM":
        copy = CausalCharLM.__new__(CausalCharLM)
        copy.embed = [row[:] for row in self.embed]
        copy.w1 = [row[:] for row in self.w1]
        copy.b1 = self.b1[:]
        copy.w2 = [row[:] for row in self.w2]
        copy.b2 = self.b2[:]
        copy.w_skip1 = [row[:] for row in self.w_skip1]
        copy.w_skip2 = [row[:] for row in self.w_skip2]
        return copy

    def _forward_logits(self, ctx: Tuple[int, int, int, int]) -> Tuple[List[float], List[float], List[float]]:
        """Compute (x_concat, hidden_tanh, logits) for a single 4-char context."""
        embed = self.embed
        c0, c1, c2, c3 = ctx
        x = embed[c0] + embed[c1] + embed[c2] + embed[c3]
        h = [
            math.tanh(sum(map(mul, w_row, x)) + b_val)
            for w_row, b_val in zip(self.w1, self.b1, strict=True)
        ]
        s1 = self.w_skip1[c3]
        s2 = self.w_skip2[c2]
        logits = [
            sum(map(mul, w_row, h)) + b_val + s1_v + s2_v
            for w_row, b_val, s1_v, s2_v in zip(self.w2, self.b2, s1, s2, strict=True)
        ]
        return x, h, logits

    def compute_batch_gradients(
        self, batch: Sequence[Example]
    ) -> Tuple[float, Dict[str, object]]:
        """Run forward + backward over a minibatch, returning mean loss and sparse/dense grads."""
        bsz = len(batch)
        if bsz == 0:
            raise ValueError("Empty batch")
        inv_b = 1.0 / bsz

        # Sparse gradient accumulators for embeddings & direct skip rows
        g_embed: Dict[int, List[float]] = {}
        g_skip1: Dict[int, List[float]] = {}
        g_skip2: Dict[int, List[float]] = {}

        # Dense gradient accumulators for MLP layers
        g_w1 = [[0.0] * INPUT_DIM for _ in range(HIDDEN_DIM)]
        g_b1 = [0.0] * HIDDEN_DIM
        g_w2 = [[0.0] * HIDDEN_DIM for _ in range(VOCAB_SIZE)]
        g_b2 = [0.0] * VOCAB_SIZE

        # Precompute transposes once per batch for fast backward dot products
        w2_t = [
            [self.w2[v][j] for v in range(VOCAB_SIZE)] for j in range(HIDDEN_DIM)
        ]
        w1_t = [
            [self.w1[j][k] for j in range(HIDDEN_DIM)] for k in range(INPUT_DIM)
        ]

        total_loss = 0.0
        for ex in batch:
            ctx = ex.context
            target = ex.target
            x, h, logits = self._forward_logits(ctx)

            max_l = max(logits)
            exps = [math.exp(v - max_l) for v in logits]
            sum_e = sum(exps)
            inv_s = 1.0 / sum_e
            probs = [e * inv_s for e in exps]

            p_tgt = probs[target]
            total_loss += -math.log(p_tgt if p_tgt > 1e-15 else 1e-15)

            # dL/dlogits = (probs - one_hot(target)) / bsz
            dlogits = [p * inv_b for p in probs]
            dlogits[target] -= inv_b

            # Output bias & skip rows
            c2 = ctx[2]
            c3 = ctx[3]
            gs1 = g_skip1.get(c3)
            if gs1 is None:
                g_skip1[c3] = dlogits[:]
            else:
                for v in range(VOCAB_SIZE):
                    gs1[v] += dlogits[v]
            gs2 = g_skip2.get(c2)
            if gs2 is None:
                g_skip2[c2] = dlogits[:]
            else:
                for v in range(VOCAB_SIZE):
                    gs2[v] += dlogits[v]

            for v in range(VOCAB_SIZE):
                dl_v = dlogits[v]
                g_b2[v] += dl_v
                gw2_row = g_w2[v]
                for j in range(HIDDEN_DIM):
                    gw2_row[j] += dl_v * h[j]

            # Backprop through tanh hidden layer
            dh = [
                sum(map(mul, w2_t[j], dlogits)) * (1.0 - h[j] * h[j])
                for j in range(HIDDEN_DIM)
            ]
            for j in range(HIDDEN_DIM):
                dh_j = dh[j]
                g_b1[j] += dh_j
                gw1_row = g_w1[j]
                for k in range(INPUT_DIM):
                    gw1_row[k] += dh_j * x[k]

            # Backprop into token embeddings
            dx = [sum(map(mul, w1_t[k], dh)) for k in range(INPUT_DIM)]
            for pos in range(CONTEXT_LEN):
                tok = ctx[pos]
                offset = pos * EMBED_DIM
                ge = g_embed.get(tok)
                if ge is None:
                    g_embed[tok] = dx[offset : offset + EMBED_DIM]
                else:
                    for d in range(EMBED_DIM):
                        ge[d] += dx[offset + d]

        grads: Dict[str, object] = {
            "embed": g_embed,
            "w1": g_w1,
            "b1": g_b1,
            "w2": g_w2,
            "b2": g_b2,
            "w_skip1": g_skip1,
            "w_skip2": g_skip2,
        }
        return total_loss * inv_b, grads

    def evaluate(self, examples: Sequence[Example]) -> EvalMetrics:
        """Compute exact cross-entropy loss, perplexity, top-1 & top-5 accuracy."""
        n = len(examples)
        if n == 0:
            return EvalMetrics(
                loss=0.0, perplexity=1.0, top1_acc=0.0, top5_acc=0.0, num_examples=0
            )
        total_loss = 0.0
        top1 = 0
        top5 = 0
        for ex in examples:
            _, _, logits = self._forward_logits(ex.context)
            max_l = max(logits)
            exps = [math.exp(v - max_l) for v in logits]
            sum_e = sum(exps)
            p_tgt = exps[ex.target] / sum_e
            total_loss += -math.log(p_tgt if p_tgt > 1e-15 else 1e-15)

            tgt_val = logits[ex.target]
            greater = 0
            for v in logits:
                if v > tgt_val:
                    greater += 1
            if greater == 0:
                top1 += 1
                top5 += 1
            elif greater < 5:
                top5 += 1

        mean_loss = total_loss / n
        ppl = math.exp(min(mean_loss, 50.0))
        return EvalMetrics(
            loss=mean_loss,
            perplexity=ppl,
            top1_acc=top1 / n,
            top5_acc=top5 / n,
            num_examples=n,
        )


class InnerAdamW:
    """Prime DiLoCo inner-loop AdamW optimizer operating directly on CausalCharLM."""

    def __init__(
        self,
        lr: float = 0.025,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m_flat = [0.0] * PARAM_COUNT
        self.v_flat = [0.0] * PARAM_COUNT

    def step(self, model: CausalCharLM, grads: Dict[str, object]) -> None:
        """Apply one AdamW update using sparse/dense gradient dict."""
        self.step_count += 1
        t = self.step_count
        b1 = self.beta1
        b2 = self.beta2
        one_minus_b1 = 1.0 - b1
        one_minus_b2 = 1.0 - b2
        bc1 = 1.0 - (b1**t)
        bc2 = 1.0 - (b2**t)
        step_size = self.lr * math.sqrt(bc2) / bc1
        wd_factor = 1.0 - self.lr * self.weight_decay
        eps = self.eps
        m_flat = self.m_flat
        v_flat = self.v_flat

        def _update_vec(
            param: List[float], grad: List[float], offset: int, decay: bool
        ) -> None:
            for i in range(len(param)):
                g = grad[i]
                idx = offset + i
                m_val = b1 * m_flat[idx] + one_minus_b1 * g
                v_val = b2 * v_flat[idx] + one_minus_b2 * (g * g)
                m_flat[idx] = m_val
                v_flat[idx] = v_val
                p = param[i] * wd_factor if decay else param[i]
                param[i] = p - step_size * m_val / (math.sqrt(v_val) + eps)

        # Offsets matching CausalCharLM.get_flat_params()
        off_embed = 0
        off_w1 = VOCAB_SIZE * EMBED_DIM
        off_b1 = off_w1 + HIDDEN_DIM * INPUT_DIM
        off_w2 = off_b1 + HIDDEN_DIM
        off_b2 = off_w2 + VOCAB_SIZE * HIDDEN_DIM
        off_s1 = off_b2 + VOCAB_SIZE
        off_s2 = off_s1 + VOCAB_SIZE * VOCAB_SIZE

        g_embed: Dict[int, List[float]] = grads["embed"]  # type: ignore[assignment]
        for tok, g_row in g_embed.items():
            _update_vec(model.embed[tok], g_row, off_embed + tok * EMBED_DIM, True)

        g_w1: List[List[float]] = grads["w1"]  # type: ignore[assignment]
        for r in range(HIDDEN_DIM):
            _update_vec(model.w1[r], g_w1[r], off_w1 + r * INPUT_DIM, True)

        g_b1: List[float] = grads["b1"]  # type: ignore[assignment]
        _update_vec(model.b1, g_b1, off_b1, False)

        g_w2: List[List[float]] = grads["w2"]  # type: ignore[assignment]
        for r in range(VOCAB_SIZE):
            _update_vec(model.w2[r], g_w2[r], off_w2 + r * HIDDEN_DIM, True)

        g_b2: List[float] = grads["b2"]  # type: ignore[assignment]
        _update_vec(model.b2, g_b2, off_b2, False)

        g_skip1: Dict[int, List[float]] = grads["w_skip1"]  # type: ignore[assignment]
        for tok, g_row in g_skip1.items():
            _update_vec(model.w_skip1[tok], g_row, off_s1 + tok * VOCAB_SIZE, True)

        g_skip2: Dict[int, List[float]] = grads["w_skip2"]  # type: ignore[assignment]
        for tok, g_row in g_skip2.items():
            _update_vec(model.w_skip2[tok], g_row, off_s2 + tok * VOCAB_SIZE, True)


def grads_dict_to_flat(grads: Dict[str, object]) -> List[float]:
    """Convert a sparse/dense gradient dict into a full flat float vector (for H=1 AllReduce)."""
    flat = [0.0] * PARAM_COUNT
    off_embed = 0
    off_w1 = VOCAB_SIZE * EMBED_DIM
    off_b1 = off_w1 + HIDDEN_DIM * INPUT_DIM
    off_w2 = off_b1 + HIDDEN_DIM
    off_b2 = off_w2 + VOCAB_SIZE * HIDDEN_DIM
    off_s1 = off_b2 + VOCAB_SIZE
    off_s2 = off_s1 + VOCAB_SIZE * VOCAB_SIZE

    g_embed: Dict[int, List[float]] = grads["embed"]  # type: ignore[assignment]
    for tok, g_row in g_embed.items():
        base = off_embed + tok * EMBED_DIM
        flat[base : base + EMBED_DIM] = g_row

    g_w1: List[List[float]] = grads["w1"]  # type: ignore[assignment]
    for r in range(HIDDEN_DIM):
        base = off_w1 + r * INPUT_DIM
        flat[base : base + INPUT_DIM] = g_w1[r]

    g_b1: List[float] = grads["b1"]  # type: ignore[assignment]
    flat[off_b1 : off_b1 + HIDDEN_DIM] = g_b1

    g_w2: List[List[float]] = grads["w2"]  # type: ignore[assignment]
    for r in range(VOCAB_SIZE):
        base = off_w2 + r * HIDDEN_DIM
        flat[base : base + HIDDEN_DIM] = g_w2[r]

    g_b2: List[float] = grads["b2"]  # type: ignore[assignment]
    flat[off_b2 : off_b2 + VOCAB_SIZE] = g_b2

    g_skip1: Dict[int, List[float]] = grads["w_skip1"]  # type: ignore[assignment]
    for tok, g_row in g_skip1.items():
        base = off_s1 + tok * VOCAB_SIZE
        flat[base : base + VOCAB_SIZE] = g_row

    g_skip2: Dict[int, List[float]] = grads["w_skip2"]  # type: ignore[assignment]
    for tok, g_row in g_skip2.items():
        base = off_s2 + tok * VOCAB_SIZE
        flat[base : base + VOCAB_SIZE] = g_row

    return flat


def flat_to_grads_dict(flat: Sequence[float]) -> Dict[str, object]:
    """Convert a flat gradient vector back to the structured dict expected by InnerAdamW."""
    idx = 0
    g_embed: Dict[int, List[float]] = {}
    for tok in range(VOCAB_SIZE):
        row = flat[idx : idx + EMBED_DIM]
        if any(row):
            g_embed[tok] = list(row)
        idx += EMBED_DIM
    g_w1: List[List[float]] = []
    for _ in range(HIDDEN_DIM):
        g_w1.append(list(flat[idx : idx + INPUT_DIM]))
        idx += INPUT_DIM
    g_b1 = list(flat[idx : idx + HIDDEN_DIM])
    idx += HIDDEN_DIM
    g_w2: List[List[float]] = []
    for _ in range(VOCAB_SIZE):
        g_w2.append(list(flat[idx : idx + HIDDEN_DIM]))
        idx += HIDDEN_DIM
    g_b2 = list(flat[idx : idx + VOCAB_SIZE])
    idx += VOCAB_SIZE
    g_skip1: Dict[int, List[float]] = {}
    for tok in range(VOCAB_SIZE):
        row = flat[idx : idx + VOCAB_SIZE]
        if any(row):
            g_skip1[tok] = list(row)
        idx += VOCAB_SIZE
    g_skip2: Dict[int, List[float]] = {}
    for tok in range(VOCAB_SIZE):
        row = flat[idx : idx + VOCAB_SIZE]
        if any(row):
            g_skip2[tok] = list(row)
        idx += VOCAB_SIZE
    return {
        "embed": g_embed,
        "w1": g_w1,
        "b1": g_b1,
        "w2": g_w2,
        "b2": g_b2,
        "w_skip1": g_skip1,
        "w_skip2": g_skip2,
    }
