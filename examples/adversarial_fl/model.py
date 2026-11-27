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
"""Pure-stdlib 2-layer MLP multi-class token classifier with exact backprop.

Architecture:
    Input x in R^96 (sparse normalized bag-of-tokens)
      -> Linear W1 (20 x 96) + b1 (20)
      -> tanh activation
      -> Linear W2 (4 x 20) + b2 (4)
      -> numerically stable softmax + cross-entropy (+ L2 weight decay)

All 2,024 parameters are stored in a flat float64 vector for exact vector
algebra in Byzantine-robust aggregation, attack construction, and Flower
ndarray conversion. Analytical reverse-mode gradients are finite-difference
verified to < 1e-8 (strictly < 1e-6).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .corpus import (
    BACKDOOR_TARGET_CLASS,
    DOMAINS,
    INPUT_DIM,
    NUM_CLASSES,
    Example,
)

HIDDEN_DIM: int = 20
W1_OFFSET: int = 0
W1_SIZE: int = HIDDEN_DIM * INPUT_DIM  # 1,920
B1_OFFSET: int = W1_OFFSET + W1_SIZE  # 1,920
B1_SIZE: int = HIDDEN_DIM  # 20
W2_OFFSET: int = B1_OFFSET + B1_SIZE  # 1,940
W2_SIZE: int = NUM_CLASSES * HIDDEN_DIM  # 80
B2_OFFSET: int = W2_OFFSET + W2_SIZE  # 2,020
B2_SIZE: int = NUM_CLASSES  # 4
NUM_PARAMS: int = B2_OFFSET + B2_SIZE  # 2,024

DEFAULT_WEIGHT_DECAY: float = 1e-4


@dataclass(frozen=True)
class EvalReport:
    """Held-out clean evaluation report with per-domain breakdown."""

    loss: float
    accuracy: float
    per_domain_loss: dict[str, float]
    per_domain_accuracy: dict[str, float]
    total_examples: int


def init_params(seed: int = 1350) -> list[float]:
    """Deterministic Xavier/Glorot-scaled initialization."""
    rng = random.Random(seed)
    params = [0.0] * NUM_PARAMS
    scale1 = math.sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM))
    for idx in range(W1_OFFSET, W1_OFFSET + W1_SIZE):
        params[idx] = rng.gauss(0.0, scale1)
    scale2 = math.sqrt(2.0 / (HIDDEN_DIM + NUM_CLASSES))
    for idx in range(W2_OFFSET, W2_OFFSET + W2_SIZE):
        params[idx] = rng.gauss(0.0, scale2)
    return params


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    inv = 1.0 / sum(exps)
    return [e * inv for e in exps]


def forward_logits(
    params: Sequence[float],
    features: tuple[tuple[int, float], ...],
) -> tuple[list[float], list[float]]:
    """Compute hidden activations and output logits for one sparse example."""
    h = [0.0] * HIDDEN_DIM
    for j in range(HIDDEN_DIM):
        row_off = W1_OFFSET + j * INPUT_DIM
        acc = params[B1_OFFSET + j]
        for feat_idx, feat_val in features:
            acc += params[row_off + feat_idx] * feat_val
        h[j] = math.tanh(acc)

    logits = [0.0] * NUM_CLASSES
    for c in range(NUM_CLASSES):
        row_off = W2_OFFSET + c * HIDDEN_DIM
        acc = params[B2_OFFSET + c]
        for j in range(HIDDEN_DIM):
            acc += params[row_off + j] * h[j]
        logits[c] = acc
    return h, logits


def loss_and_grad(
    params: Sequence[float],
    batch: Sequence[Example],
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> tuple[float, list[float]]:
    """Compute mean cross-entropy loss (+ L2) and exact analytical gradient."""
    n = len(batch)
    if n == 0:
        return 0.0, [0.0] * NUM_PARAMS

    grad = [0.0] * NUM_PARAMS
    total_loss = 0.0
    inv_n = 1.0 / n

    for ex in batch:
        feats = ex.features
        h, logits = forward_logits(params, feats)
        probs = _softmax(logits)
        y = ex.label
        total_loss -= math.log(max(probs[y], 1e-15))

        # dL/dlogits = (probs - one_hot(y)) / n
        dlogits = [probs[c] * inv_n for c in range(NUM_CLASSES)]
        dlogits[y] -= inv_n

        # Gradients for W2, b2, and backprop into h
        dh = [0.0] * HIDDEN_DIM
        for c in range(NUM_CLASSES):
            dl_c = dlogits[c]
            grad[B2_OFFSET + c] += dl_c
            row_off = W2_OFFSET + c * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                grad[row_off + j] += dl_c * h[j]
                dh[j] += dl_c * params[row_off + j]

        # Backprop through tanh and accumulate sparse gradients for W1, b1
        for j in range(HIDDEN_DIM):
            hj = h[j]
            dz = dh[j] * (1.0 - hj * hj)
            grad[B1_OFFSET + j] += dz
            row_off = W1_OFFSET + j * INPUT_DIM
            for feat_idx, feat_val in feats:
                grad[row_off + feat_idx] += dz * feat_val

    mean_loss = total_loss * inv_n

    # L2 regularization on weights only (W1, W2), excluding biases
    if weight_decay > 0.0:
        reg_sum = 0.0
        for idx in range(W1_OFFSET, W1_OFFSET + W1_SIZE):
            w = params[idx]
            reg_sum += w * w
            grad[idx] += weight_decay * w
        for idx in range(W2_OFFSET, W2_OFFSET + W2_SIZE):
            w = params[idx]
            reg_sum += w * w
            grad[idx] += weight_decay * w
        mean_loss += 0.5 * weight_decay * reg_sum

    return mean_loss, grad


def finite_difference_grad(
    params: Sequence[float],
    batch: Sequence[Example],
    eps: float = 1e-5,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    indices: Sequence[int] | None = None,
) -> dict[int, float]:
    """Central finite-difference gradient check for specified or all indices."""
    check_indices = list(range(NUM_PARAMS)) if indices is None else list(indices)
    probe = list(params)
    out: dict[int, float] = {}
    for idx in check_indices:
        orig = probe[idx]
        probe[idx] = orig + eps
        loss_pos, _ = loss_and_grad(probe, batch, weight_decay=weight_decay)
        probe[idx] = orig - eps
        loss_neg, _ = loss_and_grad(probe, batch, weight_decay=weight_decay)
        probe[idx] = orig
        out[idx] = (loss_pos - loss_neg) / (2.0 * eps)
    return out


def local_train_delta(
    base_params: Sequence[float],
    examples: Sequence[Example],
    lr: float = 0.35,
    local_steps: int = 4,
    batch_size: int = 24,
    seed: int = 0,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> list[float]:
    """Run local mini-batch SGD and return parameter delta ``theta_local - theta_0``."""
    theta = list(base_params)
    n = len(examples)
    if n == 0:
        return [0.0] * NUM_PARAMS

    rng = random.Random(seed)
    indices = list(range(n))
    cursor = 0
    rng.shuffle(indices)

    for _step in range(local_steps):
        if cursor + batch_size > n:
            rng.shuffle(indices)
            cursor = 0
        eff_bs = min(batch_size, n)
        batch = [examples[indices[cursor + i]] for i in range(eff_bs)]
        cursor += eff_bs
        _loss, grad = loss_and_grad(theta, batch, weight_decay=weight_decay)
        for idx in range(NUM_PARAMS):
            theta[idx] -= lr * grad[idx]

    return [theta[idx] - base_params[idx] for idx in range(NUM_PARAMS)]


def evaluate_clean(
    params: Sequence[float],
    examples: Sequence[Example],
) -> EvalReport:
    """Evaluate loss and top-1 accuracy overall and per domain."""
    total_loss = 0.0
    total_correct = 0
    dom_loss = {d: 0.0 for d in DOMAINS}
    dom_correct = {d: 0 for d in DOMAINS}
    dom_count = {d: 0 for d in DOMAINS}

    for ex in examples:
        _h, logits = forward_logits(params, ex.features)
        probs = _softmax(logits)
        y = ex.label
        loss_i = -math.log(max(probs[y], 1e-15))
        pred = max(range(NUM_CLASSES), key=lambda c: logits[c])
        correct = 1 if pred == y else 0

        total_loss += loss_i
        total_correct += correct
        dom_loss[ex.domain] += loss_i
        dom_correct[ex.domain] += correct
        dom_count[ex.domain] += 1

    n = max(1, len(examples))
    per_dom_loss = {
        d: (dom_loss[d] / dom_count[d] if dom_count[d] > 0 else 0.0) for d in DOMAINS
    }
    per_dom_acc = {
        d: (100.0 * dom_correct[d] / dom_count[d] if dom_count[d] > 0 else 0.0)
        for d in DOMAINS
    }
    return EvalReport(
        loss=total_loss / n,
        accuracy=100.0 * total_correct / n,
        per_domain_loss=per_dom_loss,
        per_domain_accuracy=per_dom_acc,
        total_examples=len(examples),
    )


def evaluate_asr(
    params: Sequence[float],
    trigger_examples: Sequence[Example],
    target_class: int = BACKDOOR_TARGET_CLASS,
) -> float:
    """Return Attack Success Rate (%) on held-out triggered non-target examples."""
    if not trigger_examples:
        return 0.0
    hits = 0
    for ex in trigger_examples:
        _h, logits = forward_logits(params, ex.features)
        pred = max(range(NUM_CLASSES), key=lambda c: logits[c])
        if pred == target_class:
            hits += 1
    return 100.0 * hits / len(trigger_examples)


def l2_norm(vec: Sequence[float]) -> float:
    """Euclidean L2 norm of a flat parameter vector."""
    return math.sqrt(sum(v * v for v in vec))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean inner product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1], returning 0.0 if either vector is zero."""
    na = l2_norm(a)
    nb = l2_norm(b)
    if na <= 1e-15 or nb <= 1e-15:
        return 0.0
    val = dot(a, b) / (na * nb)
    return max(-1.0, min(1.0, val))


def vec_scale(vec: Sequence[float], factor: float) -> list[float]:
    """Scale a parameter vector by a scalar factor."""
    return [v * factor for v in vec]


def vec_add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Element-wise sum of two parameter vectors."""
    return [x + y for x, y in zip(a, b, strict=True)]


def clip_to_norm(vec: Sequence[float], max_norm: float) -> tuple[list[float], bool]:
    """Clip vector to L2 norm <= max_norm; returns (clipped_vec, was_clipped)."""
    norm = l2_norm(vec)
    if norm <= max_norm or norm <= 1e-15:
        return list(vec), False
    scale = max_norm / norm
    return [v * scale for v in vec], True
