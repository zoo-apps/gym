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
"""Byte-reproducible benchmark runner for HIP-1350 (zoo-apps/gym#5).

Usage:
    python3 -m examples.adversarial_fl.run_benchmark --seed 1350
    python3 -m examples.adversarial_fl.run_benchmark --seed 1350 \
        --out examples/adversarial_fl/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

from . import __hip__, __issue__, __version__
from .attacks import (
    AttackConfig,
    generate_round_submissions,
)
from .corpus import (
    BACKDOOR_TARGET_DOMAIN,
    DOMAINS,
    INPUT_DIM,
    MINORITY_CLIENT_IDS,
    MINORITY_DOMAIN,
    NUM_CLIENTS,
    TRIGGER_TOKENS,
    load_federated_corpus,
)
from .defenses import DEFENSE_NAMES, DefenseConfig, aggregate_submissions
from .flower_adapter import (
    FLOWER_AVAILABLE,
    AdversarialDefenseStrategy,
    decode_flat_params,
    submissions_to_flower_results,
)
from .model import (
    B1_OFFSET,
    HIDDEN_DIM,
    NUM_PARAMS,
    W2_OFFSET,
    finite_difference_grad,
    init_params,
    loss_and_grad,
)
from .simulation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_STEPS,
    DEFAULT_LR,
    run_federated_experiment,
)

BENCHMARK_ROUNDS: int = 14

WHAT_REMAINS_UNPROVEN: list[str] = [
    (
        "7B-70B LLM federated fine-tuning: this reference suite trains a "
        "2,024-parameter 2-layer MLP token classifier on CPU; gradient geometry, "
        "low-rank LoRA subspace collisions, and transformer attention backdoors "
        "at 7B-70B scale remain unproven."
    ),
    (
        "Live Flower gRPC/mTLS transport: flower_adapter.py verifies exact "
        "Strategy contract parity via a pure-stdlib wire shim offline; live "
        "multi-node Flower gRPC/mTLS networking, TLS certificate revocation, "
        "and WAN packet-loss jitter are not exercised."
    ),
    (
        "DP-SGD certified backdoor robustness: no (epsilon, delta)-differential "
        "privacy noise or certified randomized-smoothing robustness bounds are "
        "proven against arbitrary trigger patterns."
    ),
    (
        "Adaptive white-box RL attackers: threats evaluate static and "
        "gradient-aware adversaries (including unmonitored stealth backdoors), "
        "not online reinforcement-learning attackers jointly optimizing against "
        "dynamic coordinator validation sets over hundreds of rounds."
    ),
    (
        "Sybil identity-cost economics: adversary fractions f in {0%, 17%, 33%, "
        "50%} are fixed exogenously; stake-slashing, proof-of-contribution "
        "bonding curves, and economic Sybil resistance remain out of scope."
    ),
]


def _verify_gradients(seed: int, eval_val: Any) -> dict[str, Any]:
    params = init_params(seed)
    batch = eval_val[:8]
    _loss, anal = loss_and_grad(params, batch)
    check_indices = (
        list(range(0, 64))
        + list(range(B1_OFFSET, W2_OFFSET))
        + list(range(W2_OFFSET, NUM_PARAMS))
    )
    fd = finite_difference_grad(params, batch, eps=1e-5, indices=check_indices)
    max_err = max(abs(anal[i] - fd[i]) for i in check_indices)
    if max_err >= 1e-6:
        raise RuntimeError(f"Gradient verification failed: max_err={max_err}")
    return {
        "checked_parameters": len(check_indices),
        "total_parameters": NUM_PARAMS,
        "finite_diff_eps": 1e-5,
        "max_abs_error": float(f"{max_err:.3e}"),
        "passed_lt_1e6": max_err < 1e-6,
    }


def _verify_flower_adapter_parity(
    seed: int,
    corpus: Any,
    raw_lookup: dict[str, str],
) -> dict[str, Any]:
    theta = init_params(seed)
    submissions = generate_round_submissions(
        round_idx=0,
        checkpoints=[theta],
        corpus=corpus,
        raw_text_lookup=raw_lookup,
        attack=AttackConfig("colluding_inner_product_cartel", "33%"),
        lr=DEFAULT_LR,
        local_steps=DEFAULT_LOCAL_STEPS,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=seed,
    )
    flower_results = submissions_to_flower_results(submissions, theta)
    max_diff_all = 0.0
    for dname in DEFENSE_NAMES:
        dcfg = DefenseConfig(dname)
        native = aggregate_submissions(
            submissions, dcfg, global_params=theta, eval_val=corpus.eval_val
        )
        strategy = AdversarialDefenseStrategy(
            initial_params=theta,
            defense_config=dcfg,
            eval_val=corpus.eval_val,
            eval_clean=corpus.eval_clean,
        )
        wire_params, _metrics = strategy.aggregate_fit(
            server_round=0, results=flower_results, failures=[]
        )
        assert wire_params is not None
        updated = decode_flat_params(wire_params)
        expected = [theta[i] + native.aggregated_delta[i] for i in range(NUM_PARAMS)]
        diff = max(abs(a - b) for a, b in zip(updated, expected, strict=True))
        max_diff_all = max(max_diff_all, diff)
    return {
        "flower_installed": FLOWER_AVAILABLE,
        "runtime_mode": "flwr" if FLOWER_AVAILABLE else "stdlib_shim",
        "defenses_verified": list(DEFENSE_NAMES),
        "max_abs_param_diff": float(f"{max_diff_all:.3e}"),
        "parity_verified": max_diff_all < 1e-12,
    }


def build_benchmark_artifact(
    seed: int = 1350,
    rounds: int = BENCHMARK_ROUNDS,
) -> dict[str, Any]:
    """Run the complete deterministic benchmark suite and return JSON payload."""
    corpus, raw_lookup = load_federated_corpus(seed=seed)

    grad_check = _verify_gradients(seed, corpus.eval_val)
    flower_check = _verify_flower_adapter_parity(seed, corpus, raw_lookup)

    # 1. Clean Oracle (f = 0%, undefended FedAvg)
    oracle_res = run_federated_experiment(
        scenario_id="oracle_benign_0pct_fedavg",
        corpus=corpus,
        raw_text_lookup=raw_lookup,
        attack=AttackConfig("benign", "0%"),
        defense=DefenseConfig("fedavg_undefended"),
        seed=seed,
        rounds=rounds,
    )
    oracle_acc = oracle_res.clean_accuracy
    oracle_min_acc = oracle_res.minority_domain_accuracy

    def run_matrix_block(
        block_prefix: str,
        attack: AttackConfig,
        defenses: tuple[str, ...] = DEFENSE_NAMES,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dname in defenses:
            res = run_federated_experiment(
                scenario_id=f"{block_prefix}__{dname}",
                corpus=corpus,
                raw_text_lookup=raw_lookup,
                attack=attack,
                defense=DefenseConfig(dname),
                seed=seed,
                rounds=rounds,
                oracle_clean_accuracy=oracle_acc,
                oracle_minority_accuracy=oracle_min_acc,
            )
            rows.append(res.to_dict())
        return rows

    # 2. Benign 0% across all 6 defenses (also serves as Negative Regime 1!)
    benign_0pct = run_matrix_block("benign_0pct", AttackConfig("benign", "0%"))

    # 3. Four Mandatory Threat Scenarios across f in {0%, 17%, 33%, 50%}
    poisoning_matrix: dict[str, list[dict[str, Any]]] = {"0%": benign_0pct}
    for frac in ("17%", "33%", "50%"):
        poisoning_matrix[frac] = run_matrix_block(
            f"poisoning_{frac.rstrip('%')}pct",
            AttackConfig("label_flip_and_sign_flip", frac),
        )

    backdoor_matrix: dict[str, list[dict[str, Any]]] = {"0%": benign_0pct}
    for frac in ("17%", "33%", "50%"):
        backdoor_matrix[frac] = run_matrix_block(
            f"backdoor_{frac.rstrip('%')}pct",
            AttackConfig("targeted_trigger_backdoor", frac),
        )

    stale_honest_matrix: dict[str, list[dict[str, Any]]] = {"0%": benign_0pct}
    for tau in (3, 6):
        for frac in ("17%", "33%", "50%"):
            key = f"tau{tau}_{frac}"
            stale_honest_matrix[key] = run_matrix_block(
                f"stale_honest_tau{tau}_{frac.rstrip('%')}pct",
                AttackConfig("stale_honest_straggler", frac, staleness_tau=tau),
            )

    stale_malicious_matrix: dict[str, list[dict[str, Any]]] = {"0%": benign_0pct}
    for tau in (3, 6):
        for frac in ("17%", "33%", "50%"):
            key = f"tau{tau}_{frac}"
            stale_malicious_matrix[key] = run_matrix_block(
                f"stale_malicious_tau{tau}_{frac.rstrip('%')}pct",
                AttackConfig("stale_malicious_replay", frac, staleness_tau=tau),
            )

    collusion_matrix: dict[str, list[dict[str, Any]]] = {"0%": benign_0pct}
    for frac in ("17%", "33%", "50%"):
        collusion_matrix[frac] = run_matrix_block(
            f"collusion_{frac.rstrip('%')}pct",
            AttackConfig("colluding_inner_product_cartel", frac),
        )

    # 4. Three Explicit Honest Negative Regimes
    neg_stealth_17 = run_matrix_block(
        "neg_stealth_backdoor_17pct",
        AttackConfig("stealth_unmonitored_backdoor", "17%"),
    )
    neg_stealth_33 = run_matrix_block(
        "neg_stealth_backdoor_33pct",
        AttackConfig("stealth_unmonitored_backdoor", "33%"),
    )

    strict_bft_50 = run_federated_experiment(
        scenario_id="neg_50pct_collusion__dso_strict_bft_quorum",
        corpus=corpus,
        raw_text_lookup=raw_lookup,
        attack=AttackConfig("colluding_inner_product_cartel", "50%"),
        defense=DefenseConfig("dso_hybrid_eval_cosine", dso_strict_bft_quorum=True),
        seed=seed,
        rounds=rounds,
        oracle_clean_accuracy=oracle_acc,
        oracle_minority_accuracy=oracle_min_acc,
    ).to_dict()

    negative_regimes: dict[str, Any] = {
        "neg_non_iid_honest_frr": {
            "description": (
                "Zero attackers (f = 0%) under Non-IID vaults with 2 honest "
                "minority infra specialists. Geometric defenses (coordinate_median, "
                "trimmed_mean, multi_krum) suffer severe minority false rejection "
                "and infra accuracy collapse, whereas dso_hybrid_eval_cosine "
                "preserves minority acceptance via domain-stratified validation."
            ),
            "results": benign_0pct,
        },
        "neg_unmonitored_backdoor_blindness": {
            "description": (
                "Stealth backdoor attackers preserve clean validation loss "
                "bit-for-bit by restricting trigger weights to out-of-corpus "
                "unmonitored trigger slots absent from eval_val. Even "
                "dso_hybrid_eval_cosine accepts the updates and suffers high ASR, "
                "proving clean validation gates alone cannot stop unmonitored "
                "targeted triggers."
            ),
            "results_by_fraction": {
                "17%": neg_stealth_17,
                "33%": neg_stealth_33,
            },
        },
        "neg_fifty_percent_collusion_breakdown": {
            "description": (
                "At f = 50% (6/12 colluding workers, cos > 0.99), classical "
                "geometric Byzantine-robust rules (coordinate_median, "
                "trimmed_mean, multi_krum) break down completely (BFT f < N/2 "
                "limit). Enforcing strict majority BFT quorum (> N/2 accepted) "
                "causes total liveness stall (100% quorum stall rounds)."
            ),
            "results": collusion_matrix["50%"],
            "dso_strict_bft_quorum_result": strict_bft_50,
        },
    }

    # Construct core deterministic payload (strictly free of wall-clock or host paths)
    core_payload: dict[str, Any] = {
        "metadata": {
            "specification": __hip__,
            "issue": __issue__,
            "suite_version": __version__,
            "license": "Apache-2.0",
            "seed": seed,
            "num_clients": NUM_CLIENTS,
            "minority_client_ids": list(MINORITY_CLIENT_IDS),
            "minority_domain": MINORITY_DOMAIN,
            "backdoor_target_domain": BACKDOOR_TARGET_DOMAIN,
            "trigger_tokens": list(TRIGGER_TOKENS),
            "domains": list(DOMAINS),
            "rounds": rounds,
            "lr": DEFAULT_LR,
            "local_steps": DEFAULT_LOCAL_STEPS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "model_architecture": {
                "type": "2-layer-mlp-tanh-softmax",
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_classes": len(DOMAINS),
                "num_params": NUM_PARAMS,
            },
            "defenses": list(DEFENSE_NAMES),
        },
        "corpus_manifest": {
            "file_count": len(corpus.manifest),
            "total_bytes": corpus.total_bytes,
            "manifest_sha256": corpus.manifest_sha256,
            "splits": {
                "client_vaults_total": sum(len(c.examples) for c in corpus.clients),
                "eval_val": len(corpus.eval_val),
                "eval_clean": len(corpus.eval_clean),
                "eval_trigger": len(corpus.eval_trigger),
            },
            "client_vault_summary": [
                {
                    "client_id": c.client_id,
                    "role": c.role,
                    "primary_domain": c.primary_domain,
                    "is_minority_specialist": c.is_minority_specialist,
                    "num_examples": len(c.examples),
                    "domain_counts": c.domain_counts,
                }
                for c in corpus.clients
            ],
            "files": [
                {
                    "rel_path": e.rel_path,
                    "domain": e.domain,
                    "size_bytes": e.size_bytes,
                    "sha256": e.sha256,
                    "num_chunks": e.num_chunks,
                }
                for e in corpus.manifest
            ],
        },
        "verifications": {
            "analytical_gradient_check": grad_check,
            "flower_adapter_parity": flower_check,
        },
        "clean_oracle": oracle_res.to_dict(),
        "threat_matrices": {
            "label_flip_and_sign_flip": poisoning_matrix,
            "targeted_trigger_backdoor": backdoor_matrix,
            "stale_honest_straggler": stale_honest_matrix,
            "stale_malicious_replay": stale_malicious_matrix,
            "colluding_inner_product_cartel": collusion_matrix,
        },
        "negative_regimes": negative_regimes,
        "governance_separation": {
            "policy": (
                "Scientific claims require independently checkable evidence. "
                "AI agents may run reproducible benchmarks, flag conflicts, "
                "and prepare technical recommendations, but must not redefine "
                "voting rules, authorize treasury spending, or self-certify "
                "governance outcomes."
            ),
            "artifact_license": "Apache-2.0",
            "external_runtime_dependencies": [],
        },
        "what_remains_unproven": WHAT_REMAINS_UNPROVEN,
    }

    canonical_bytes = json.dumps(
        core_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    result = dict(core_payload)
    result["independent_evaluation_receipt"] = {
        "receipt_schema": "hip-1350-eval-receipt-v1",
        "canonical_payload_sha256": payload_sha256,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "seed": seed,
        "reproduce_command": (
            f"python3 -m examples.adversarial_fl.run_benchmark --seed {seed}"
        ),
        "self_certification_disclaimer": (
            "Hash attests to deterministic execution of the pinned code and "
            "corpus only; independent human/maintainer verification is required "
            "for governance or production acceptance."
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the HIP-1350 adversarial federated-learning benchmark."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1350,
        help="Deterministic PRNG seed (default: 1350).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=BENCHMARK_ROUNDS,
        help=f"Federated communication rounds per run (default: {BENCHMARK_ROUNDS}).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("examples/adversarial_fl/results.json"),
        help="Output path for deterministic JSON artifact.",
    )
    args = parser.parse_args(argv)

    artifact = build_benchmark_artifact(seed=args.seed, rounds=args.rounds)
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized, encoding="utf-8")

    receipt = artifact["independent_evaluation_receipt"]
    oracle = artifact["clean_oracle"]
    print(
        f"[HIP-1350] Wrote {args.out} | seed={args.seed} | rounds={args.rounds} | "
        f"oracle_acc={oracle['clean_accuracy']:.2f}% | "
        f"receipt_sha256={receipt['canonical_payload_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
