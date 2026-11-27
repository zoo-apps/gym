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
"""Adversarial threat models and straggler generators for HIP-1350.

Implements all four mandatory threat scenarios across adversary fractions
``f in {0%, 17% (2/12), 33% (4/12), 50% (6/12)}`` plus the honest slow
straggler regime and stealth unmonitored backdoor negative regime:

  1. ``label_flip_and_sign_flip``: untargeted semantic label-flipping combined
     with sign-flipped gradient scaling aimed at collapsing clean accuracy.
  2. ``targeted_trigger_backdoor``: rare out-of-corpus token-sequence trigger
     mapped to ``BACKDOOR_TARGET_CLASS`` with model-replacement boost.
  3. ``stale_update_lag`` (both ``stale_honest_straggler`` and
     ``stale_malicious_replay``): updates computed from checkpoint ``t - tau``
     (``tau in {3, 6}`` rounds).
  4. ``colluding_inner_product_cartel``: ``k`` colluding workers submitting
     near-collinear (``cos > 0.99``) inner-product-manipulation shifts scaled
     just inside naive L2 norm bounds.
  5. ``stealth_unmonitored_backdoor``: negative-regime stealth attack that
     preserves clean validation loss bit-for-bit on unmonitored trigger slots.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence

from .corpus import (
    BACKDOOR_TARGET_CLASS,
    INPUT_DIM,
    NUM_CLASSES,
    ClientVault,
    Example,
    FederatedCorpus,
    apply_trigger_to_example,
)
from .model import (
    HIDDEN_DIM,
    NUM_PARAMS,
    W1_OFFSET,
    l2_norm,
    local_train_delta,
    loss_and_grad,
    vec_add,
    vec_scale,
)

# Deterministic adversary sets preserving minority specialists (10, 11) as honest
ADVERSARY_FRACTIONS: dict[str, tuple[int, ...]] = {
    "0%": (),
    "17%": (0, 4),
    "33%": (0, 4, 7, 1),
    "50%": (0, 4, 7, 1, 5, 8),
}


@dataclass(frozen=True)
class AttackConfig:
    """Configuration for a federated threat or straggler scenario."""

    name: str
    fraction_label: str = "0%"
    staleness_tau: int = 0
    poison_scale: float = 2.6
    backdoor_boost: float = 3.2
    backdoor_poison_ratio: float = 0.50
    cartel_norm_ratio: float = 0.92
    cartel_jitter: float = 0.025
    stealth_trigger_scale: float = 0.45


@dataclass(frozen=True)
class ClientSubmission:
    """Single client update submitted to the coordinator in a round."""

    client_id: int
    round_idx: int
    base_round: int
    staleness_lag: int
    delta: tuple[float, ...]
    num_samples: int
    is_malicious: bool
    is_honest_straggler: bool
    is_minority_specialist: bool
    attack_tag: str


def adversary_ids_for_fraction(fraction_label: str) -> tuple[int, ...]:
    """Return deterministic client IDs participating in an attack/straggler set."""
    if fraction_label not in ADVERSARY_FRACTIONS:
        raise ValueError(
            f"Unsupported fraction_label {fraction_label!r}; "
            f"expected one of {tuple(ADVERSARY_FRACTIONS)}"
        )
    return ADVERSARY_FRACTIONS[fraction_label]


def _build_triggered_vault(
    vault: ClientVault,
    corpus: FederatedCorpus,
    raw_text_lookup: dict[str, str],
    poison_ratio: float,
    seed: int,
) -> list[Example]:
    """Mix clean client examples with triggered non-target examples."""
    rng = random.Random(seed)
    clean_examples = list(vault.examples)
    non_target = [ex for ex in clean_examples if ex.label != BACKDOOR_TARGET_CLASS]
    # Also allow triggering any example if a vault is heavily target-dominated
    source_pool = non_target if non_target else clean_examples
    n_poison = max(4, int(round(len(clean_examples) * poison_ratio)))
    poisoned: list[Example] = []
    for idx in range(n_poison):
        src = source_pool[idx % len(source_pool)]
        raw_txt = raw_text_lookup[src.chunk_id]
        poisoned.append(
            apply_trigger_to_example(
                raw_text=raw_txt,
                chunk_id=f"{src.chunk_id}:p{idx}",
                source_path=src.source_path,
                domain=src.domain,
                target_label=BACKDOOR_TARGET_CLASS,
                token_to_id=corpus.token_to_id,
            )
        )
    combined = clean_examples + poisoned
    rng.shuffle(combined)
    return combined


def _compute_trigger_subspace_delta(
    base_params: Sequence[float],
    triggered_examples: Sequence[Example],
    trigger_token_ids: tuple[int, int],
    lr: float,
    steps: int,
) -> list[float]:
    """Train ONLY the unmonitored W1 trigger columns via exact analytical grads.

    All clean-feature weights, biases, and output-layer weights remain
    identically unchanged, preserving clean validation loss bit-for-bit while
    driving triggered examples to ``BACKDOOR_TARGET_CLASS``.
    """
    theta = list(base_params)
    active_param_indices: list[int] = []
    for j in range(HIDDEN_DIM):
        row_off = W1_OFFSET + j * INPUT_DIM
        for tid in trigger_token_ids:
            active_param_indices.append(row_off + tid)

    for _ in range(steps):
        _loss, grad = loss_and_grad(theta, triggered_examples, weight_decay=0.0)
        for p_idx in active_param_indices:
            theta[p_idx] -= lr * grad[p_idx]

    delta = [0.0] * NUM_PARAMS
    for p_idx in active_param_indices:
        delta[p_idx] = theta[p_idx] - base_params[p_idx]
    return delta


def generate_round_submissions(
    round_idx: int,
    checkpoints: Sequence[Sequence[float]],
    corpus: FederatedCorpus,
    raw_text_lookup: dict[str, str],
    attack: AttackConfig,
    lr: float,
    local_steps: int,
    batch_size: int,
    seed: int,
) -> list[ClientSubmission]:
    """Generate all N=12 client submissions for a single federated round."""
    current_params = checkpoints[-1]
    selected_ids = set(adversary_ids_for_fraction(attack.fraction_label))

    # First compute honest deltas for all clients (also needed by colluders
    # to construct inner-product-manipulation updates inside honest norm bounds).
    honest_deltas: dict[int, list[float]] = {}
    for vault in corpus.clients:
        cid = vault.client_id
        client_seed = seed * 10007 + round_idx * 97 + cid
        honest_deltas[cid] = local_train_delta(
            base_params=current_params,
            examples=vault.examples,
            lr=lr,
            local_steps=local_steps,
            batch_size=batch_size,
            seed=client_seed,
        )

    honest_non_selected = [
        honest_deltas[cid] for cid in range(len(corpus.clients)) if cid not in selected_ids
    ]
    honest_norms = [l2_norm(d) for d in honest_non_selected] or [1.0]
    median_honest_norm = statistics.median(honest_norms)

    # Precompute cartel shared direction if needed
    cartel_base_dir: list[float] | None = None
    if attack.name == "colluding_inner_product_cartel" and selected_ids:
        mean_honest = [0.0] * NUM_PARAMS
        inv_h = 1.0 / max(1, len(honest_non_selected))
        for d in honest_non_selected:
            for idx in range(NUM_PARAMS):
                mean_honest[idx] += d[idx] * inv_h
        neg_dir = vec_scale(mean_honest, -1.0)
        neg_norm = l2_norm(neg_dir)
        target_norm = attack.cartel_norm_ratio * median_honest_norm
        if neg_norm > 1e-12:
            cartel_base_dir = vec_scale(neg_dir, target_norm / neg_norm)
        else:
            cartel_base_dir = [0.0] * NUM_PARAMS

    submissions: list[ClientSubmission] = []
    for vault in corpus.clients:
        cid = vault.client_id
        client_seed = seed * 10007 + round_idx * 97 + cid

        if cid not in selected_ids or attack.name == "benign":
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=round_idx,
                    staleness_lag=0,
                    delta=tuple(honest_deltas[cid]),
                    num_samples=len(vault.examples),
                    is_malicious=False,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag="honest",
                )
            )
            continue

        if attack.name == "label_flip_and_sign_flip":
            flipped_examples = [
                Example(
                    chunk_id=ex.chunk_id,
                    source_path=ex.source_path,
                    domain=ex.domain,
                    label=(ex.label + 1) % NUM_CLASSES,
                    features=ex.features,
                    is_triggered=False,
                )
                for ex in vault.examples
            ]
            d_lf = local_train_delta(
                base_params=current_params,
                examples=flipped_examples,
                lr=lr,
                local_steps=local_steps,
                batch_size=batch_size,
                seed=client_seed + 503,
            )
            d_sign = vec_scale(honest_deltas[cid], -attack.poison_scale)
            mal_delta = vec_add(d_sign, vec_scale(d_lf, 1.4))
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=round_idx,
                    staleness_lag=0,
                    delta=tuple(mal_delta),
                    num_samples=len(vault.examples),
                    is_malicious=True,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag="label_flip_and_sign_flip",
                )
            )

        elif attack.name == "targeted_trigger_backdoor":
            poisoned_examples = _build_triggered_vault(
                vault=vault,
                corpus=corpus,
                raw_text_lookup=raw_text_lookup,
                poison_ratio=attack.backdoor_poison_ratio,
                seed=client_seed + 709,
            )
            d_bd = local_train_delta(
                base_params=current_params,
                examples=poisoned_examples,
                lr=lr * 1.15,
                local_steps=local_steps + 2,
                batch_size=batch_size,
                seed=client_seed + 719,
            )
            boosted = vec_scale(d_bd, attack.backdoor_boost)
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=round_idx,
                    staleness_lag=0,
                    delta=tuple(boosted),
                    num_samples=len(vault.examples),
                    is_malicious=True,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag="targeted_trigger_backdoor",
                )
            )

        elif attack.name == "stale_honest_straggler":
            base_round = max(0, round_idx - attack.staleness_tau)
            actual_lag = round_idx - base_round
            stale_params = checkpoints[base_round]
            d_stale = local_train_delta(
                base_params=stale_params,
                examples=vault.examples,
                lr=lr,
                local_steps=local_steps,
                batch_size=batch_size,
                seed=client_seed,
            )
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=base_round,
                    staleness_lag=actual_lag,
                    delta=tuple(d_stale),
                    num_samples=len(vault.examples),
                    is_malicious=False,
                    is_honest_straggler=True,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag=f"stale_honest_tau{attack.staleness_tau}",
                )
            )

        elif attack.name == "stale_malicious_replay":
            base_round = max(0, round_idx - attack.staleness_tau)
            actual_lag = round_idx - base_round
            stale_params = checkpoints[base_round]
            # Replays backward drift toward stale checkpoint + stale local step
            backward_drift = [
                stale_params[idx] - current_params[idx] for idx in range(NUM_PARAMS)
            ]
            d_stale = local_train_delta(
                base_params=stale_params,
                examples=vault.examples,
                lr=lr,
                local_steps=local_steps,
                batch_size=batch_size,
                seed=client_seed,
            )
            # Amplify backward drift to stall/revert global convergence
            replay_delta = vec_add(
                vec_scale(backward_drift, 1.8),
                vec_scale(d_stale, -1.2 if actual_lag == 0 else 0.8),
            )
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=base_round,
                    staleness_lag=actual_lag,
                    delta=tuple(replay_delta),
                    num_samples=len(vault.examples),
                    is_malicious=True,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag=f"stale_malicious_tau{attack.staleness_tau}",
                )
            )

        elif attack.name == "colluding_inner_product_cartel":
            assert cartel_base_dir is not None
            rng_j = random.Random(client_seed + 1103)
            raw_jitter = [rng_j.gauss(0.0, 1.0) for _ in range(NUM_PARAMS)]
            j_norm = l2_norm(raw_jitter)
            base_n = l2_norm(cartel_base_dir)
            jitter_scale = (
                (attack.cartel_jitter * base_n / j_norm) if j_norm > 1e-12 else 0.0
            )
            cartel_delta = [
                cartel_base_dir[idx] + raw_jitter[idx] * jitter_scale
                for idx in range(NUM_PARAMS)
            ]
            # Re-normalize strictly inside naive L2 bound (<= cartel_norm_ratio * median)
            c_norm = l2_norm(cartel_delta)
            max_allowed = attack.cartel_norm_ratio * median_honest_norm
            if c_norm > max_allowed > 1e-12:
                cartel_delta = vec_scale(cartel_delta, max_allowed / c_norm)
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=round_idx,
                    staleness_lag=0,
                    delta=tuple(cartel_delta),
                    num_samples=len(vault.examples),
                    is_malicious=True,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag="colluding_inner_product_cartel",
                )
            )

        elif attack.name == "stealth_unmonitored_backdoor":
            # Build clean triggered examples from non-target samples
            non_target = [
                ex for ex in vault.examples if ex.label != BACKDOOR_TARGET_CLASS
            ] or list(vault.examples)
            trig_batch = [
                apply_trigger_to_example(
                    raw_text=raw_text_lookup[ex.chunk_id],
                    chunk_id=f"{ex.chunk_id}:st{i}",
                    source_path=ex.source_path,
                    domain=ex.domain,
                    target_label=BACKDOOR_TARGET_CLASS,
                    token_to_id=corpus.token_to_id,
                )
                for i, ex in enumerate(non_target[:24])
            ]
            d_clean = honest_deltas[cid]
            d_trig = _compute_trigger_subspace_delta(
                base_params=current_params,
                triggered_examples=trig_batch,
                trigger_token_ids=corpus.trigger_token_ids,
                lr=lr * 2.2,
                steps=8,
            )
            # Scale trigger perturbation so total norm stays well within honest bounds
            clean_n = max(l2_norm(d_clean), 1e-6)
            trig_n = l2_norm(d_trig)
            max_trig_n = attack.stealth_trigger_scale * clean_n
            if trig_n > max_trig_n > 1e-12:
                d_trig = vec_scale(d_trig, max_trig_n / trig_n)
            stealth_delta = vec_add(d_clean, d_trig)
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=round_idx,
                    base_round=round_idx,
                    staleness_lag=0,
                    delta=tuple(stealth_delta),
                    num_samples=len(vault.examples),
                    is_malicious=True,
                    is_honest_straggler=False,
                    is_minority_specialist=vault.is_minority_specialist,
                    attack_tag="stealth_unmonitored_backdoor",
                )
            )

        else:
            raise ValueError(f"Unknown attack scenario: {attack.name!r}")

    return submissions


def mean_cartel_pairwise_cosine(submissions: Sequence[ClientSubmission]) -> float:
    """Diagnostic: mean pairwise cosine among malicious submissions in a round."""
    mal = [s.delta for s in submissions if s.is_malicious]
    if len(mal) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(mal)):
        ni = l2_norm(mal[i])
        for j in range(i + 1, len(mal)):
            nj = l2_norm(mal[j])
            if ni > 1e-15 and nj > 1e-15:
                dot_ij = sum(a * b for a, b in zip(mal[i], mal[j], strict=True))
                total += max(-1.0, min(1.0, dot_ij / (ni * nj)))
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def expected_bft_threshold_exceeded(num_malicious: int, num_clients: int) -> bool:
    """Return True when adversary count meets or exceeds classical 50% BFT limit."""
    return num_malicious * 2 >= num_clients and num_malicious > 0


def cosine_angle_degrees(cos_val: float) -> float:
    """Convert cosine similarity into angle in degrees."""
    clamped = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(clamped))
