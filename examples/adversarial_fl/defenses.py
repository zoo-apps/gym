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
"""Six federated aggregation defenses with explicit per-client acceptance audit.

Defenses implemented (all pure Python stdlib):
  1. ``fedavg_undefended`` — uniform FedAvg baseline.
  2. ``norm_clipping`` — L2 norm clipping + FedAvg.
  3. ``coordinate_median`` — coordinate-wise median (Yin et al., ICML 2018).
  4. ``trimmed_mean`` — coordinate-wise trimmed mean, beta = 0.2 (Yin et al.).
  5. ``multi_krum`` — Euclidean distance Multi-Krum (Blanchard et al., NeurIPS 2017).
  6. ``dso_hybrid_eval_cosine`` — Zoo DSO-inspired hybrid defense combining:
       (a) staleness-weighted decay (preserving honest slow stragglers),
       (b) norm-bound screening & soft clipping,
       (c) pairwise cosine collusion discounting (cos >= 0.99 cartel detection),
       (d) independent domain-stratified held-out validation benefit gate
           (explicitly protecting honest minority-domain specialists).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .attacks import ClientSubmission
from .corpus import DOMAINS, Example
from .model import (
    NUM_PARAMS,
    clip_to_norm,
    cosine_similarity,
    evaluate_clean,
    l2_norm,
    vec_add,
    vec_scale,
)

DEFENSE_NAMES: tuple[str, ...] = (
    "fedavg_undefended",
    "norm_clipping",
    "coordinate_median",
    "trimmed_mean",
    "multi_krum",
    "dso_hybrid_eval_cosine",
)


@dataclass(frozen=True)
class DefenseConfig:
    """Hyperparameters for the six evaluated aggregation rules."""

    name: str
    clip_norm: float = 1.50
    trimmed_beta: float = 0.20
    krum_f_assumed: int = 3
    krum_select_m: int = 7
    # DSO hybrid defense parameters
    dso_staleness_alpha: float = 0.35
    dso_max_honest_tau: int = 8
    dso_soft_norm_mult: float = 2.50
    dso_hard_norm_mult: float = 2.95
    dso_collusion_cos_threshold: float = 0.992
    dso_collusion_discount: float = 0.15
    dso_probe_eta: float = 0.08
    dso_overall_loss_tol: float = 0.004
    dso_macro_loss_tol: float = 0.004
    dso_specialist_max_overall_deg: float = 0.08
    dso_min_quorum: int = 2
    dso_strict_bft_quorum: bool = False


@dataclass(frozen=True)
class ClientDecision:
    """Explicit acceptance, weight, and audit record for one client in one round."""

    client_id: int
    accepted: bool
    effective_weight: float
    coordinate_retention: float
    rejection_reason: str
    is_malicious: bool
    is_honest_straggler: bool
    is_minority_specialist: bool
    staleness_lag: int
    cosine_cluster_size: int = 1
    val_delta_overall: float = 0.0
    val_best_domain_gain: float = 0.0


@dataclass(frozen=True)
class AggregationResult:
    """Output of a single federated aggregation round."""

    defense_name: str
    aggregated_delta: tuple[float, ...]
    decisions: tuple[ClientDecision, ...]
    quorum_stalled: bool
    accepted_count: int
    rejected_count: int


def _fedavg_undefended(submissions: Sequence[ClientSubmission]) -> AggregationResult:
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="fedavg_undefended",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )
    w = 1.0 / n
    agg = [0.0] * NUM_PARAMS
    decisions: list[ClientDecision] = []
    for sub in submissions:
        d = sub.delta
        for idx in range(NUM_PARAMS):
            agg[idx] += d[idx] * w
        decisions.append(
            ClientDecision(
                client_id=sub.client_id,
                accepted=True,
                effective_weight=w,
                coordinate_retention=1.0,
                rejection_reason="accepted",
                is_malicious=sub.is_malicious,
                is_honest_straggler=sub.is_honest_straggler,
                is_minority_specialist=sub.is_minority_specialist,
                staleness_lag=sub.staleness_lag,
            )
        )
    return AggregationResult(
        defense_name="fedavg_undefended",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=n,
        rejected_count=0,
    )


def _norm_clipping(
    submissions: Sequence[ClientSubmission],
    config: DefenseConfig,
) -> AggregationResult:
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="norm_clipping",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )
    w = 1.0 / n
    agg = [0.0] * NUM_PARAMS
    decisions: list[ClientDecision] = []
    for sub in submissions:
        clipped, was_clipped = clip_to_norm(sub.delta, config.clip_norm)
        for idx in range(NUM_PARAMS):
            agg[idx] += clipped[idx] * w
        decisions.append(
            ClientDecision(
                client_id=sub.client_id,
                accepted=True,
                effective_weight=w,
                coordinate_retention=1.0,
                rejection_reason="accepted_clipped" if was_clipped else "accepted",
                is_malicious=sub.is_malicious,
                is_honest_straggler=sub.is_honest_straggler,
                is_minority_specialist=sub.is_minority_specialist,
                staleness_lag=sub.staleness_lag,
            )
        )
    return AggregationResult(
        defense_name="norm_clipping",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=n,
        rejected_count=0,
    )


def _coordinate_median(submissions: Sequence[ClientSubmission]) -> AggregationResult:
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="coordinate_median",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )
    agg = [0.0] * NUM_PARAMS
    retained_counts = [0] * n
    mid_hi = n // 2
    mid_lo = mid_hi - 1 if n % 2 == 0 else mid_hi

    for coord in range(NUM_PARAMS):
        ordered = sorted(
            ((submissions[i].delta[coord], i) for i in range(n)),
            key=lambda item: (item[0], item[1]),
        )
        if n % 2 == 0:
            v_lo, i_lo = ordered[mid_lo]
            v_hi, i_hi = ordered[mid_hi]
            agg[coord] = 0.5 * (v_lo + v_hi)
            retained_counts[i_lo] += 1
            retained_counts[i_hi] += 1
        else:
            v_mid, i_mid = ordered[mid_hi]
            agg[coord] = v_mid
            retained_counts[i_mid] += 1

    total_ret = sum(retained_counts) or 1
    decisions: list[ClientDecision] = []
    accepted_count = 0
    for i, sub in enumerate(submissions):
        retention = retained_counts[i] / NUM_PARAMS
        eff_w = retained_counts[i] / total_ret
        # Client is counted as effectively retained if >= 50% of its coordinates survive
        accepted = retention >= 0.50
        if accepted:
            accepted_count += 1
        decisions.append(
            ClientDecision(
                client_id=sub.client_id,
                accepted=accepted,
                effective_weight=eff_w,
                coordinate_retention=retention,
                rejection_reason=(
                    "accepted" if accepted else "median_coordinate_discarded"
                ),
                is_malicious=sub.is_malicious,
                is_honest_straggler=sub.is_honest_straggler,
                is_minority_specialist=sub.is_minority_specialist,
                staleness_lag=sub.staleness_lag,
            )
        )
    return AggregationResult(
        defense_name="coordinate_median",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=accepted_count,
        rejected_count=n - accepted_count,
    )


def _trimmed_mean(
    submissions: Sequence[ClientSubmission],
    config: DefenseConfig,
) -> AggregationResult:
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="trimmed_mean",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )
    k = min(int(math.floor(config.trimmed_beta * n)), (n - 1) // 2)
    kept_per_coord = n - 2 * k
    inv_kept = 1.0 / kept_per_coord
    agg = [0.0] * NUM_PARAMS
    retained_counts = [0] * n

    for coord in range(NUM_PARAMS):
        ordered = sorted(
            ((submissions[i].delta[coord], i) for i in range(n)),
            key=lambda item: (item[0], item[1]),
        )
        middle = ordered[k : n - k]
        s = 0.0
        for val, client_pos in middle:
            s += val
            retained_counts[client_pos] += 1
        agg[coord] = s * inv_kept

    total_ret = sum(retained_counts) or 1
    decisions: list[ClientDecision] = []
    accepted_count = 0
    for i, sub in enumerate(submissions):
        retention = retained_counts[i] / NUM_PARAMS
        eff_w = retained_counts[i] / total_ret
        # A client whose majority of coordinates (>=50%) are trimmed out is effectively rejected
        accepted = retention >= 0.50
        if accepted:
            accepted_count += 1
        decisions.append(
            ClientDecision(
                client_id=sub.client_id,
                accepted=accepted,
                effective_weight=eff_w,
                coordinate_retention=retention,
                rejection_reason=(
                    "accepted" if accepted else "trimmed_mean_tail_excluded"
                ),
                is_malicious=sub.is_malicious,
                is_honest_straggler=sub.is_honest_straggler,
                is_minority_specialist=sub.is_minority_specialist,
                staleness_lag=sub.staleness_lag,
            )
        )
    return AggregationResult(
        defense_name="trimmed_mean",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=accepted_count,
        rejected_count=n - accepted_count,
    )


def _multi_krum(
    submissions: Sequence[ClientSubmission],
    config: DefenseConfig,
) -> AggregationResult:
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="multi_krum",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )
    f_assumed = min(config.krum_f_assumed, max(0, (n - 3) // 2))
    k_neighbors = max(1, n - f_assumed - 2)
    select_m = max(1, min(config.krum_select_m, n))

    # Pairwise squared Euclidean distances
    dist_sq = [[0.0] * n for _ in range(n)]
    for i in range(n):
        di = submissions[i].delta
        for j in range(i + 1, n):
            dj = submissions[j].delta
            sq = sum((a - b) * (a - b) for a, b in zip(di, dj, strict=True))
            dist_sq[i][j] = sq
            dist_sq[j][i] = sq

    scores: list[tuple[float, int]] = []
    for i in range(n):
        others = sorted(dist_sq[i][j] for j in range(n) if j != i)
        scores.append((sum(others[:k_neighbors]), i))
    scores.sort(key=lambda item: (item[0], item[1]))
    selected_positions = {pos for _score, pos in scores[:select_m]}

    w = 1.0 / len(selected_positions)
    agg = [0.0] * NUM_PARAMS
    decisions: list[ClientDecision] = []
    for i, sub in enumerate(submissions):
        if i in selected_positions:
            d = sub.delta
            for idx in range(NUM_PARAMS):
                agg[idx] += d[idx] * w
            decisions.append(
                ClientDecision(
                    client_id=sub.client_id,
                    accepted=True,
                    effective_weight=w,
                    coordinate_retention=1.0,
                    rejection_reason="accepted",
                    is_malicious=sub.is_malicious,
                    is_honest_straggler=sub.is_honest_straggler,
                    is_minority_specialist=sub.is_minority_specialist,
                    staleness_lag=sub.staleness_lag,
                )
            )
        else:
            decisions.append(
                ClientDecision(
                    client_id=sub.client_id,
                    accepted=False,
                    effective_weight=0.0,
                    coordinate_retention=0.0,
                    rejection_reason="multi_krum_distance_outlier",
                    is_malicious=sub.is_malicious,
                    is_honest_straggler=sub.is_honest_straggler,
                    is_minority_specialist=sub.is_minority_specialist,
                    staleness_lag=sub.staleness_lag,
                )
            )

    return AggregationResult(
        defense_name="multi_krum",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=len(selected_positions),
        rejected_count=n - len(selected_positions),
    )


def _dso_hybrid_eval_cosine(
    submissions: Sequence[ClientSubmission],
    global_params: Sequence[float],
    eval_val: Sequence[Example],
    config: DefenseConfig,
) -> AggregationResult:
    """Zoo DSO-inspired defense: staleness decay + norm screen + cosine cartel
    discounting + independent domain-stratified held-out validation gate.
    """
    n = len(submissions)
    if n == 0:
        return AggregationResult(
            defense_name="dso_hybrid_eval_cosine",
            aggregated_delta=tuple([0.0] * NUM_PARAMS),
            decisions=(),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=0,
        )

    # 1. Pairwise cosine similarity matrix for collusion cluster detection
    cos_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        cos_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            c_ij = cosine_similarity(submissions[i].delta, submissions[j].delta)
            cos_matrix[i][j] = c_ij
            cos_matrix[j][i] = c_ij

    cluster_sizes = [
        1
        + sum(
            1
            for j in range(n)
            if j != i and cos_matrix[i][j] >= config.dso_collusion_cos_threshold
        )
        for i in range(n)
    ]

    # 2. Lower-order-statistic norm screening (33rd percentile resists <=33% norm
    #    inflation attacks while naturally shifting under >=50% adversarial coalitions)
    norms = [l2_norm(sub.delta) for sub in submissions]
    sorted_norms = sorted(norms)
    ref_norm = max(sorted_norms[min(len(sorted_norms) - 1, n // 3)], 1e-6)
    soft_ceil = config.dso_soft_norm_mult * ref_norm
    hard_ceil = config.dso_hard_norm_mult * ref_norm

    # 3. Baseline held-out validation report (both micro and domain-stratified macro)
    base_report = evaluate_clean(global_params, eval_val)
    base_macro_loss = sum(base_report.per_domain_loss[d] for d in DOMAINS) / len(DOMAINS)

    # 4. Evaluate every client through staleness, norm, cosine, and validation gates
    candidate_deltas: list[list[float] | None] = [None] * n
    raw_weights: list[float] = [0.0] * n
    prelim_decisions: list[dict[str, object]] = []

    for i, sub in enumerate(submissions):
        norm_i = norms[i]
        c_size = cluster_sizes[i]

        if sub.staleness_lag > config.dso_max_honest_tau:
            prelim_decisions.append(
                {
                    "accepted": False,
                    "reason": "excessive_staleness_lag",
                    "delta_overall": 0.0,
                    "best_dom_gain": 0.0,
                }
            )
            continue

        if norm_i > hard_ceil:
            prelim_decisions.append(
                {
                    "accepted": False,
                    "reason": "norm_hard_ceiling_exceeded",
                    "delta_overall": 0.0,
                    "best_dom_gain": 0.0,
                }
            )
            continue

        # Soft-clip moderately elevated updates before probing validation
        work_delta, _was_clipped = clip_to_norm(sub.delta, soft_ceil)
        probed_params = vec_add(
            global_params, vec_scale(work_delta, config.dso_probe_eta)
        )
        probe_report = evaluate_clean(probed_params, eval_val)

        delta_overall = probe_report.loss - base_report.loss
        probe_macro_loss = sum(
            probe_report.per_domain_loss[d] for d in DOMAINS
        ) / len(DOMAINS)
        delta_macro = probe_macro_loss - base_macro_loss
        domain_gains = {
            d: base_report.per_domain_loss[d] - probe_report.per_domain_loss[d]
            for d in DOMAINS
        }
        best_dom_gain = max(domain_gains.values())

        # Independent domain-stratified validation benefit gate:
        #   - overall_ok: improves or preserves micro-averaged held-out loss
        #   - specialist_ok: improves domain-stratified macro loss (protecting
        #     honest minority specialists whose stratum is small in eval_val)
        #     without catastrophic overall degradation; disabled for stale lags
        #     so malicious stale backward replays cannot abuse the specialist gate.
        overall_ok = delta_overall <= config.dso_overall_loss_tol
        specialist_ok = (
            sub.staleness_lag == 0
            and delta_macro <= config.dso_macro_loss_tol
            and delta_overall <= config.dso_specialist_max_overall_deg
        )

        # Collinear cartel check: tight cos >= 0.992 cluster that provides
        # neither overall nor macro-stratified validation benefit is rejected.
        is_collinear_cartel = c_size >= 2
        if is_collinear_cartel and not (overall_ok or specialist_ok):
            prelim_decisions.append(
                {
                    "accepted": False,
                    "reason": "cosine_collusion_cartel_rejected",
                    "delta_overall": delta_overall,
                    "best_dom_gain": best_dom_gain,
                }
            )
            continue

        if not (overall_ok or specialist_ok):
            prelim_decisions.append(
                {
                    "accepted": False,
                    "reason": (
                        "stale_replay_validation_rejected"
                        if sub.staleness_lag > 0
                        else "validation_benefit_gate_rejected"
                    ),
                    "delta_overall": delta_overall,
                    "best_dom_gain": best_dom_gain,
                }
            )
            continue

        # Compute principled DSO weight: staleness decay * collusion discount
        # (unbiased base weight preserves minority/majority equilibrium without
        # inducing round-to-round specialist oscillation).
        w_stale = 1.0 / (1.0 + config.dso_staleness_alpha * sub.staleness_lag)
        w_collusion = (
            (config.dso_collusion_discount / c_size)
            if (is_collinear_cartel and c_size >= 3)
            else 1.0
        )
        raw_weights[i] = w_stale * w_collusion
        candidate_deltas[i] = work_delta
        prelim_decisions.append(
            {
                "accepted": True,
                "reason": (
                    "accepted_specialist_stratum"
                    if (not overall_ok and specialist_ok)
                    else "accepted"
                ),
                "delta_overall": delta_overall,
                "best_dom_gain": best_dom_gain,
            }
        )

    accepted_indices = [i for i in range(n) if prelim_decisions[i]["accepted"]]

    # Under strict BFT majority quorum check (used in 50% breakdown analysis),
    # if accepted participants fall below strict majority (> N/2), quorum stalls.
    min_required = (
        (n // 2 + 1) if config.dso_strict_bft_quorum else config.dso_min_quorum
    )
    quorum_stalled = len(accepted_indices) < min_required

    agg = [0.0] * NUM_PARAMS
    decisions: list[ClientDecision] = []
    if quorum_stalled:
        for i, sub in enumerate(submissions):
            info = prelim_decisions[i]
            decisions.append(
                ClientDecision(
                    client_id=sub.client_id,
                    accepted=False,
                    effective_weight=0.0,
                    coordinate_retention=0.0,
                    rejection_reason=(
                        "quorum_stalled_insufficient_honest_majority"
                        if info["accepted"]
                        else str(info["reason"])
                    ),
                    is_malicious=sub.is_malicious,
                    is_honest_straggler=sub.is_honest_straggler,
                    is_minority_specialist=sub.is_minority_specialist,
                    staleness_lag=sub.staleness_lag,
                    cosine_cluster_size=cluster_sizes[i],
                    val_delta_overall=float(info["delta_overall"]),
                    val_best_domain_gain=float(info["best_dom_gain"]),
                )
            )
        return AggregationResult(
            defense_name="dso_hybrid_eval_cosine",
            aggregated_delta=tuple(agg),
            decisions=tuple(decisions),
            quorum_stalled=True,
            accepted_count=0,
            rejected_count=n,
        )

    total_w = sum(raw_weights[i] for i in accepted_indices)
    inv_w = 1.0 / total_w if total_w > 1e-15 else 1.0 / len(accepted_indices)

    for i, sub in enumerate(submissions):
        info = prelim_decisions[i]
        if info["accepted"]:
            norm_w = raw_weights[i] * inv_w
            delta_i = candidate_deltas[i]
            assert delta_i is not None
            for idx in range(NUM_PARAMS):
                agg[idx] += delta_i[idx] * norm_w
            decisions.append(
                ClientDecision(
                    client_id=sub.client_id,
                    accepted=True,
                    effective_weight=norm_w,
                    coordinate_retention=1.0,
                    rejection_reason=str(info["reason"]),
                    is_malicious=sub.is_malicious,
                    is_honest_straggler=sub.is_honest_straggler,
                    is_minority_specialist=sub.is_minority_specialist,
                    staleness_lag=sub.staleness_lag,
                    cosine_cluster_size=cluster_sizes[i],
                    val_delta_overall=float(info["delta_overall"]),
                    val_best_domain_gain=float(info["best_dom_gain"]),
                )
            )
        else:
            decisions.append(
                ClientDecision(
                    client_id=sub.client_id,
                    accepted=False,
                    effective_weight=0.0,
                    coordinate_retention=0.0,
                    rejection_reason=str(info["reason"]),
                    is_malicious=sub.is_malicious,
                    is_honest_straggler=sub.is_honest_straggler,
                    is_minority_specialist=sub.is_minority_specialist,
                    staleness_lag=sub.staleness_lag,
                    cosine_cluster_size=cluster_sizes[i],
                    val_delta_overall=float(info["delta_overall"]),
                    val_best_domain_gain=float(info["best_dom_gain"]),
                )
            )

    return AggregationResult(
        defense_name="dso_hybrid_eval_cosine",
        aggregated_delta=tuple(agg),
        decisions=tuple(decisions),
        quorum_stalled=False,
        accepted_count=len(accepted_indices),
        rejected_count=n - len(accepted_indices),
    )


def aggregate_submissions(
    submissions: Sequence[ClientSubmission],
    config: DefenseConfig,
    global_params: Sequence[float] | None = None,
    eval_val: Sequence[Example] | None = None,
) -> AggregationResult:
    """Dispatch round submissions to the requested defense aggregator."""
    if config.name == "fedavg_undefended":
        return _fedavg_undefended(submissions)
    if config.name == "norm_clipping":
        return _norm_clipping(submissions, config)
    if config.name == "coordinate_median":
        return _coordinate_median(submissions)
    if config.name == "trimmed_mean":
        return _trimmed_mean(submissions, config)
    if config.name == "multi_krum":
        return _multi_krum(submissions, config)
    if config.name == "dso_hybrid_eval_cosine":
        if global_params is None or eval_val is None:
            raise ValueError(
                "dso_hybrid_eval_cosine requires global_params and eval_val"
            )
        return _dso_hybrid_eval_cosine(submissions, global_params, eval_val, config)
    raise ValueError(f"Unknown defense: {config.name!r}")
