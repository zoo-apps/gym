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
"""Deterministic federated-learning simulation engine and metric auditor."""

from __future__ import annotations

from dataclasses import dataclass

from .attacks import (
    AttackConfig,
    generate_round_submissions,
    mean_cartel_pairwise_cosine,
)
from .corpus import MINORITY_DOMAIN, FederatedCorpus
from .defenses import DefenseConfig, aggregate_submissions
from .model import (
    NUM_PARAMS,
    evaluate_asr,
    evaluate_clean,
    init_params,
)

DEFAULT_ROUNDS: int = 16
DEFAULT_LR: float = 0.95
DEFAULT_LOCAL_STEPS: int = 3
DEFAULT_BATCH_SIZE: int = 16


@dataclass(frozen=True)
class SimulationResult:
    """Complete auditable metrics from one federated simulation run."""

    scenario_id: str
    attack_name: str
    fraction_label: str
    staleness_tau: int
    defense_name: str
    rounds: int
    clean_accuracy: float
    clean_loss: float
    per_domain_accuracy: dict[str, float]
    minority_domain_accuracy: float
    quality_loss_vs_oracle: float
    minority_quality_loss_vs_oracle: float
    asr: float
    frr: float
    minority_honest_acceptance_rate: float
    honest_straggler_frr: float
    attacker_detection_rate: float
    mean_honest_coordinate_retention: float
    mean_minority_coordinate_retention: float
    quorum_stall_rounds: int
    mean_cartel_cosine: float

    def to_dict(self) -> dict[str, object]:
        """Serialize to deterministic rounded JSON-compatible dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "attack_name": self.attack_name,
            "fraction_label": self.fraction_label,
            "staleness_tau": self.staleness_tau,
            "defense_name": self.defense_name,
            "rounds": self.rounds,
            "clean_accuracy": round(self.clean_accuracy, 4),
            "clean_loss": round(self.clean_loss, 6),
            "per_domain_accuracy": {
                k: round(v, 4) for k, v in self.per_domain_accuracy.items()
            },
            "minority_domain_accuracy": round(self.minority_domain_accuracy, 4),
            "quality_loss_vs_oracle": round(self.quality_loss_vs_oracle, 4),
            "minority_quality_loss_vs_oracle": round(
                self.minority_quality_loss_vs_oracle, 4
            ),
            "asr": round(self.asr, 4),
            "frr": round(self.frr, 4),
            "minority_honest_acceptance_rate": round(
                self.minority_honest_acceptance_rate, 4
            ),
            "honest_straggler_frr": round(self.honest_straggler_frr, 4),
            "attacker_detection_rate": round(self.attacker_detection_rate, 4),
            "mean_honest_coordinate_retention": round(
                self.mean_honest_coordinate_retention, 4
            ),
            "mean_minority_coordinate_retention": round(
                self.mean_minority_coordinate_retention, 4
            ),
            "quorum_stall_rounds": self.quorum_stall_rounds,
            "mean_cartel_cosine": round(self.mean_cartel_cosine, 6),
        }


def run_federated_experiment(
    scenario_id: str,
    corpus: FederatedCorpus,
    raw_text_lookup: dict[str, str],
    attack: AttackConfig,
    defense: DefenseConfig,
    seed: int = 1350,
    rounds: int = DEFAULT_ROUNDS,
    lr: float = DEFAULT_LR,
    local_steps: int = DEFAULT_LOCAL_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    oracle_clean_accuracy: float | None = None,
    oracle_minority_accuracy: float | None = None,
) -> SimulationResult:
    """Execute one deterministic N=12 federated training run and audit metrics."""
    theta = init_params(seed)
    checkpoints: list[list[float]] = [list(theta)]

    honest_total = 0
    honest_rejected = 0
    honest_coord_ret_sum = 0.0

    minority_total = 0
    minority_accepted = 0
    minority_coord_ret_sum = 0.0

    straggler_total = 0
    straggler_rejected = 0

    malicious_total = 0
    malicious_rejected = 0

    quorum_stall_rounds = 0
    cartel_cos_samples: list[float] = []

    for round_idx in range(rounds):
        submissions = generate_round_submissions(
            round_idx=round_idx,
            checkpoints=checkpoints,
            corpus=corpus,
            raw_text_lookup=raw_text_lookup,
            attack=attack,
            lr=lr,
            local_steps=local_steps,
            batch_size=batch_size,
            seed=seed,
        )
        mal_cos = mean_cartel_pairwise_cosine(submissions)
        if mal_cos != 0.0:
            cartel_cos_samples.append(mal_cos)

        agg_res = aggregate_submissions(
            submissions=submissions,
            config=defense,
            global_params=theta,
            eval_val=corpus.eval_val,
        )

        if agg_res.quorum_stalled:
            quorum_stall_rounds += 1
        else:
            agg_d = agg_res.aggregated_delta
            for idx in range(NUM_PARAMS):
                theta[idx] += agg_d[idx]

        checkpoints.append(list(theta))

        for dec in agg_res.decisions:
            if dec.is_malicious:
                malicious_total += 1
                if not dec.accepted:
                    malicious_rejected += 1
            else:
                honest_total += 1
                honest_coord_ret_sum += dec.coordinate_retention
                if not dec.accepted:
                    honest_rejected += 1
                if dec.is_minority_specialist:
                    minority_total += 1
                    minority_coord_ret_sum += dec.coordinate_retention
                    if dec.accepted:
                        minority_accepted += 1
                if dec.is_honest_straggler:
                    straggler_total += 1
                    if not dec.accepted:
                        straggler_rejected += 1

    clean_report = evaluate_clean(theta, corpus.eval_clean)
    asr = evaluate_asr(theta, corpus.eval_trigger)
    minority_acc = clean_report.per_domain_accuracy[MINORITY_DOMAIN]

    oracle_acc = (
        clean_report.accuracy
        if oracle_clean_accuracy is None
        else oracle_clean_accuracy
    )
    oracle_min = (
        minority_acc
        if oracle_minority_accuracy is None
        else oracle_minority_accuracy
    )

    frr = 100.0 * honest_rejected / honest_total if honest_total > 0 else 0.0
    minority_accept_rate = (
        100.0 * minority_accepted / minority_total if minority_total > 0 else 0.0
    )
    straggler_frr = (
        100.0 * straggler_rejected / straggler_total if straggler_total > 0 else 0.0
    )
    attacker_det_rate = (
        100.0 * malicious_rejected / malicious_total if malicious_total > 0 else 0.0
    )
    mean_honest_ret = (
        100.0 * honest_coord_ret_sum / honest_total if honest_total > 0 else 0.0
    )
    mean_minority_ret = (
        100.0 * minority_coord_ret_sum / minority_total if minority_total > 0 else 0.0
    )
    mean_cartel_cos = (
        sum(cartel_cos_samples) / len(cartel_cos_samples) if cartel_cos_samples else 0.0
    )

    return SimulationResult(
        scenario_id=scenario_id,
        attack_name=attack.name,
        fraction_label=attack.fraction_label,
        staleness_tau=attack.staleness_tau,
        defense_name=defense.name,
        rounds=rounds,
        clean_accuracy=clean_report.accuracy,
        clean_loss=clean_report.loss,
        per_domain_accuracy=clean_report.per_domain_accuracy,
        minority_domain_accuracy=minority_acc,
        quality_loss_vs_oracle=oracle_acc - clean_report.accuracy,
        minority_quality_loss_vs_oracle=oracle_min - minority_acc,
        asr=asr,
        frr=frr,
        minority_honest_acceptance_rate=minority_accept_rate,
        honest_straggler_frr=straggler_frr,
        attacker_detection_rate=attacker_det_rate,
        mean_honest_coordinate_retention=mean_honest_ret,
        mean_minority_coordinate_retention=mean_minority_ret,
        quorum_stall_rounds=quorum_stall_rounds,
        mean_cartel_cosine=mean_cartel_cos,
    )
