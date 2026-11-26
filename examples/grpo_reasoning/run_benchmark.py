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
"""Byte-reproducible benchmark runner for HIP-1350 (zoo-apps/gym#7).

Compares four primary regimes and three honest negative regimes under a
matched rollout/trace compute budget on the strictly disjoint independent
held-out test split (in-distribution + out-of-distribution longer chains).

Usage::

    python3 -m examples.grpo_reasoning.run_benchmark --seed 1350
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from . import __hip__, __version__
from .dataset import (
    DOMAINS,
    ReasoningProblem,
    build_reasoning_dataset,
    load_license_audit,
)
from .grpo import (
    GRPOConfig,
    StructuredCoTPolicy,
    serialize_policy_summary,
    train_grpo,
    train_reinforce_baseline,
    train_sft_baseline,
)
from .recipes.validator import validate_all_recipes
from .rewards import (
    VERIFIER_VERSION,
    RewardAuditTrail,
    evaluate_completion,
    naive_proxy_reward,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results.json"


def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. (2021) unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / float(math.comb(n, k)))


def evaluate_policy_held_out(
    regime_name: str,
    policy: StructuredCoTPolicy,
    ref_policy: StructuredCoTPolicy,
    held_out_problems: tuple[ReasoningProblem, ...],
    eval_seed: int,
    num_samples_per_problem: int = 8,
) -> tuple[dict[str, Any], list[float], list[float]]:
    """Evaluate a trained policy on the independent held-out test split.

    Returns ``(metrics_dict, per_problem_pass1_list, per_problem_ood_pass1_list)``.
    """
    rng = random.Random(eval_seed)
    audit = RewardAuditTrail(f"{regime_name}:held_out_test")

    per_problem_pass1: list[float] = []
    per_problem_pass4: list[float] = []
    per_problem_ood_pass1: list[float] = []
    per_problem_id_pass1: list[float] = []
    per_problem_ood_pass4: list[float] = []
    per_problem_id_pass4: list[float] = []
    greedy_correct = 0

    domain_pass1: dict[str, list[float]] = {d: [] for d in DOMAINS}
    proxy_sum = 0.0
    total_rollouts = 0

    for prob in held_out_problems:
        # 1. Greedy decoding evaluation
        g_traj = policy.sample_trajectory(prob, rng, greedy=True)
        g_ev = evaluate_completion(prob, g_traj.completion_text)
        if g_ev.is_exact_correct:
            greedy_correct += 1

        # 2. Stochastic rollouts for unbiased pass@1 and pass@4
        correct_in_n = 0
        for _ in range(num_samples_per_problem):
            traj = policy.sample_trajectory(prob, rng, greedy=False)
            ev = evaluate_completion(prob, traj.completion_text)
            audit.record(ev)
            proxy_sum += naive_proxy_reward(prob, traj.completion_text)
            total_rollouts += 1
            if ev.is_exact_correct:
                correct_in_n += 1

        p1 = _unbiased_pass_at_k(num_samples_per_problem, correct_in_n, 1)
        p4 = _unbiased_pass_at_k(num_samples_per_problem, correct_in_n, 4)
        per_problem_pass1.append(p1)
        per_problem_pass4.append(p4)
        domain_pass1[prob.domain].append(p1)
        if prob.is_ood:
            per_problem_ood_pass1.append(p1)
            per_problem_ood_pass4.append(p4)
        else:
            per_problem_id_pass1.append(p1)
            per_problem_id_pass4.append(p4)

    kl_rng = random.Random(eval_seed + 77)
    kl_stats = policy.measure_kl_and_entropy(ref_policy, held_out_problems, kl_rng)
    audit_summary = audit.summary()

    n_probs = max(1, len(held_out_problems))
    metrics = {
        "pass_at_1": round(sum(per_problem_pass1) / n_probs, 6),
        "pass_at_4": round(sum(per_problem_pass4) / n_probs, 6),
        "pass_at_1_greedy": round(greedy_correct / n_probs, 6),
        "id_pass_at_1": round(
            sum(per_problem_id_pass1) / max(1, len(per_problem_id_pass1)), 6
        ),
        "id_pass_at_4": round(
            sum(per_problem_id_pass4) / max(1, len(per_problem_id_pass4)), 6
        ),
        "ood_pass_at_1": round(
            sum(per_problem_ood_pass1) / max(1, len(per_problem_ood_pass1)), 6
        ),
        "ood_pass_at_4": round(
            sum(per_problem_ood_pass4) / max(1, len(per_problem_ood_pass4)), 6
        ),
        "domain_pass_at_1": {
            d: round(sum(vals) / max(1, len(vals)), 6)
            for d, vals in domain_pass1.items()
        },
        "format_valid_rate": audit_summary["format_valid_rate"],
        "mean_verified_step_accuracy": audit_summary["mean_verified_step_fraction"],
        "mean_auditable_reward": audit_summary["mean_total_reward"],
        "mean_naive_proxy_reward": round(proxy_sum / max(1, total_rollouts), 6),
        "flagged_hacking_rate": audit_summary["flagged_rate"],
        "hacking_flag_counts": audit_summary["hacking_flag_counts"],
        "mean_exact_kl_nats": kl_stats["mean_exact_kl_nats"],
        "mean_entropy_nats": kl_stats["mean_entropy_nats"],
        "held_out_audit_chain_sha256": audit_summary["audit_chain_sha256"],
        "policy_summary": serialize_policy_summary(policy),
    }
    return metrics, per_problem_pass1, per_problem_ood_pass1


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    seed: int = 1350,
    num_resamples: int = 1000,
) -> dict[str, Any]:
    """Deterministic paired bootstrap 95% CI and two-sided sign/bootstrap test."""
    n = len(scores_a)
    if n != len(scores_b) or n == 0:
        raise ValueError("Paired score lists must be non-empty and equal length.")
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    obs_diff = sum(diffs) / float(n)
    wins = sum(1 for d in diffs if d > 1e-12)
    losses = sum(1 for d in diffs if d < -1e-12)
    ties = n - wins - losses

    rng = random.Random(seed)
    boot_means: list[float] = []
    non_positive = 0
    for _ in range(num_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        m = s / float(n)
        boot_means.append(m)
        if m <= 0.0:
            non_positive += 1

    boot_means.sort()
    ci_low = boot_means[int(0.025 * num_resamples)]
    ci_high = boot_means[int(0.975 * num_resamples)]
    one_sided_p = (non_positive + 1) / float(num_resamples + 1)
    return {
        "delta_mean": round(obs_diff, 6),
        "ci95_low": round(ci_low, 6),
        "ci95_high": round(ci_high, 6),
        "one_sided_bootstrap_p": round(one_sided_p, 6),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
    }


def compute_receipt_sha256(receipt_body: dict[str, Any]) -> str:
    payload = json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def verify_evaluation_receipt(results: dict[str, Any]) -> bool:
    """Independently verify integrity of a benchmark results receipt."""
    receipt = results.get("independent_evaluation_receipt")
    if not isinstance(receipt, dict):
        return False
    expected_sha = receipt.get("receipt_sha256")
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    return compute_receipt_sha256(body) == expected_sha


def run_benchmark(
    seed: int = 1350,
    total_rollout_budget: int = 2880,
) -> dict[str, Any]:
    """Execute full HIP-1350 benchmark suite deterministically for ``seed``."""
    dataset = build_reasoning_dataset(seed=seed)
    leak_report = dataset.verify_disjointness()
    base_policy = StructuredCoTPolicy.create_pinned_base(seed=seed)
    base_sha = base_policy.weight_sha256()

    eval_seed = seed + 9001

    # 1. Pinned base model (zero-shot)
    trained: dict[str, tuple[StructuredCoTPolicy, dict[str, Any]]] = {
        "pinned_base_model": (
            base_policy.clone(),
            {
                "regime_name": "pinned_base_model",
                "rollouts_used": 0,
                "optimizer_steps": 0,
                "final_weight_sha256": base_sha,
            },
        )
    }

    # 2. SFT-only baseline (matched 96 traces/batch, 2880 total traces)
    trained["sft_only_baseline"] = train_sft_baseline(
        base_policy=base_policy,
        train_problems=dataset.train,
        learning_rate=0.055,
        batch_prompts=96,
        total_trace_budget=total_rollout_budget,
        teacher_noise_rate=0.14,
        seed=seed + 101,
    )

    # 3. Vanilla REINFORCE without group-relative normalization (96 rollouts/batch)
    trained["reinforce_no_group_norm"] = train_reinforce_baseline(
        base_policy=base_policy,
        train_problems=dataset.train,
        kl_beta=0.04,
        learning_rate=0.045,
        batch_prompts=96,
        total_rollout_budget=total_rollout_budget,
        seed=seed + 202,
    )

    # 4. Full GRPO with G=8, 12 prompts/batch (96 rollouts/batch), clipped surrogate, k3 KL
    trained["grpo_kl_regularized"] = train_grpo(
        base_policy=base_policy,
        train_problems=dataset.train,
        config=GRPOConfig(
            regime_name="grpo_kl_regularized",
            group_size=8,
            num_iterations=2,
            clip_eps=0.2,
            kl_beta=0.04,
            learning_rate=0.04,
            weight_decay=1e-4,
            batch_prompts=12,
            total_rollout_budget=total_rollout_budget,
            use_naive_proxy_reward=False,
            seed=seed + 303,
        ),
    )

    # 5. Honest Negative Regime 1: Format+Length Proxy Reward Hacking
    trained["neg_format_only_reward_hacking"] = train_grpo(
        base_policy=base_policy,
        train_problems=dataset.train,
        config=GRPOConfig(
            regime_name="neg_format_only_reward_hacking",
            group_size=8,
            num_iterations=2,
            clip_eps=0.2,
            kl_beta=0.04,
            learning_rate=0.05,
            weight_decay=1e-4,
            batch_prompts=12,
            total_rollout_budget=total_rollout_budget,
            use_naive_proxy_reward=True,
            seed=seed + 404,
        ),
    )

    # 6. Honest Negative Regime 2: Zero-KL Collapse (beta = 0.0, unregularized)
    trained["neg_grpo_zero_kl_collapse"] = train_grpo(
        base_policy=base_policy,
        train_problems=dataset.train,
        config=GRPOConfig(
            regime_name="neg_grpo_zero_kl_collapse",
            group_size=8,
            num_iterations=4,
            clip_eps=0.3,
            kl_beta=0.0,
            learning_rate=0.22,
            weight_decay=0.0,
            batch_prompts=12,
            total_rollout_budget=total_rollout_budget,
            use_naive_proxy_reward=False,
            seed=seed + 505,
        ),
    )

    # 7. Honest Negative Regime 3: Tiny Group Size (G = 2, matched 96 rollouts/batch = 48 prompts)
    trained["neg_grpo_group_size_2"] = train_grpo(
        base_policy=base_policy,
        train_problems=dataset.train,
        config=GRPOConfig(
            regime_name="neg_grpo_group_size_2",
            group_size=2,
            num_iterations=2,
            clip_eps=0.2,
            kl_beta=0.04,
            learning_rate=0.04,
            weight_decay=1e-4,
            batch_prompts=48,
            total_rollout_budget=total_rollout_budget,
            use_naive_proxy_reward=False,
            seed=seed + 606,
        ),
    )

    # Evaluate every regime independently on held_out_test with identical eval_seed
    regime_results: dict[str, Any] = {}
    pass1_vectors: dict[str, list[float]] = {}
    ood_pass1_vectors: dict[str, list[float]] = {}

    for name, (pol, train_stats) in trained.items():
        held_metrics, p1_vec, ood_p1_vec = evaluate_policy_held_out(
            regime_name=name,
            policy=pol,
            ref_policy=base_policy,
            held_out_problems=dataset.held_out_test,
            eval_seed=eval_seed,
            num_samples_per_problem=8,
        )
        regime_results[name] = {
            "category": "negative_regime" if name.startswith("neg_") else "primary",
            "train_summary": train_stats,
            "held_out_metrics": held_metrics,
        }
        pass1_vectors[name] = p1_vec
        ood_pass1_vectors[name] = ood_p1_vec

    # Paired statistical comparisons against grpo_kl_regularized
    comparisons = {
        "grpo_vs_pinned_base_pass1": paired_bootstrap_ci(
            pass1_vectors["grpo_kl_regularized"],
            pass1_vectors["pinned_base_model"],
            seed=seed + 11,
        ),
        "grpo_vs_sft_pass1": paired_bootstrap_ci(
            pass1_vectors["grpo_kl_regularized"],
            pass1_vectors["sft_only_baseline"],
            seed=seed + 22,
        ),
        "grpo_vs_reinforce_pass1": paired_bootstrap_ci(
            pass1_vectors["grpo_kl_regularized"],
            pass1_vectors["reinforce_no_group_norm"],
            seed=seed + 33,
        ),
        "grpo_vs_neg_reward_hacking_pass1": paired_bootstrap_ci(
            pass1_vectors["grpo_kl_regularized"],
            pass1_vectors["neg_format_only_reward_hacking"],
            seed=seed + 44,
        ),
        "grpo_vs_neg_zero_kl_ood_pass1": paired_bootstrap_ci(
            ood_pass1_vectors["grpo_kl_regularized"],
            ood_pass1_vectors["neg_grpo_zero_kl_collapse"],
            seed=seed + 55,
        ),
        "grpo_vs_neg_group2_pass1": paired_bootstrap_ci(
            pass1_vectors["grpo_kl_regularized"],
            pass1_vectors["neg_grpo_group_size_2"],
            seed=seed + 66,
        ),
    }

    recipe_checks = validate_all_recipes()
    license_audit = load_license_audit()

    receipt_body = {
        "hip": __hip__,
        "pipeline_version": __version__,
        "seed": seed,
        "verifier_version": VERIFIER_VERSION,
        "base_model_sha256": base_sha,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "split_sha256": dataset.split_sha256,
        "zero_leakage_verified": leak_report["zero_leakage"],
        "matched_rollout_budget": total_rollout_budget,
        "regime_fingerprints": {
            name: {
                "final_weight_sha256": data["held_out_metrics"]["policy_summary"][
                    "weight_sha256"
                ],
                "held_out_audit_chain_sha256": data["held_out_metrics"][
                    "held_out_audit_chain_sha256"
                ],
                "pass_at_1": data["held_out_metrics"]["pass_at_1"],
                "pass_at_4": data["held_out_metrics"]["pass_at_4"],
                "ood_pass_at_1": data["held_out_metrics"]["ood_pass_at_1"],
                "mean_exact_kl_nats": data["held_out_metrics"]["mean_exact_kl_nats"],
            }
            for name, data in regime_results.items()
        },
        "recipes_all_valid": all(r["valid"] for r in recipe_checks),
        "governance_notice": (
            "Machine-checkable receipt for independent external verification. "
            "Self-reported agent outputs are NOT self-certifying; independent "
            "reproduction and authorized human review are required."
        ),
    }
    receipt_body["receipt_sha256"] = compute_receipt_sha256(receipt_body)

    return {
        "metadata": {
            "hip": __hip__,
            "issue": "zoo-apps/gym#7",
            "version": __version__,
            "seed": seed,
            "matched_rollout_budget": total_rollout_budget,
            "verifier_version": VERIFIER_VERSION,
            "source_license": "Apache-2.0",
            "open_r1_used": False,
        },
        "dataset": {
            "counts": {
                "train": len(dataset.train),
                "val": len(dataset.val),
                "held_out_test": len(dataset.held_out_test),
                "held_out_id": sum(1 for p in dataset.held_out_test if not p.is_ood),
                "held_out_ood": sum(1 for p in dataset.held_out_test if p.is_ood),
            },
            "domains": list(DOMAINS),
            "split_sha256": dataset.split_sha256,
            "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            "disjointness_check": leak_report,
        },
        "base_model": serialize_policy_summary(base_policy),
        "regimes": regime_results,
        "statistical_comparisons": comparisons,
        "recipe_validation": recipe_checks,
        "license_audit_summary": {
            "schema_version": license_audit["schema_version"],
            "artifact_count": len(license_audit["artifacts"]),
            "separate_review_artifacts": [
                a["id"]
                for a in license_audit["artifacts"]
                if a.get("separate_license_review_required")
            ],
            "excluded_historical_artifacts": [
                a["id"]
                for a in license_audit["artifacts"]
                if a.get("status") == "HISTORICAL_REFERENCE_ONLY_DO_NOT_USE"
            ],
        },
        "independent_evaluation_receipt": receipt_body,
    }


def _print_summary_table(results: dict[str, Any]) -> None:
    print(f"HIP-1350 Open Reasoning GRPO Benchmark (seed={results['metadata']['seed']})")
    print(f"Base Model SHA-256 : {results['base_model']['weight_sha256'][:16]}...")
    print(
        f"Dataset SHA-256    : {results['dataset']['dataset_manifest_sha256'][:16]}... "
        f"(zero_leakage={results['dataset']['disjointness_check']['zero_leakage']})"
    )
    print(
        f"Receipt SHA-256    : {results['independent_evaluation_receipt']['receipt_sha256'][:16]}..."
    )
    print()
    header = (
        f"{'Regime':<32} {'pass@1':>7} {'pass@4':>7} {'ID@1':>7} {'OOD@1':>7} "
        f"{'Fmt':>6} {'Step':>6} {'Proxy':>6} {'Hack%':>6} {'KL':>7} {'Ent':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, data in results["regimes"].items():
        m = data["held_out_metrics"]
        print(
            f"{name:<32} "
            f"{m['pass_at_1']:7.3f} {m['pass_at_4']:7.3f} "
            f"{m['id_pass_at_1']:7.3f} {m['ood_pass_at_1']:7.3f} "
            f"{m['format_valid_rate']:6.3f} {m['mean_verified_step_accuracy']:6.3f} "
            f"{m['mean_naive_proxy_reward']:6.3f} {m['flagged_hacking_rate']:6.3f} "
            f"{m['mean_exact_kl_nats']:7.4f} {m['mean_entropy_nats']:6.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run byte-reproducible HIP-1350 Open Reasoning GRPO benchmark."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1350,
        help="Deterministic master PRNG seed (default: 1350).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2880,
        help="Matched total rollout/trace budget per training regime (default: 2880).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for canonical JSON results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary table.",
    )
    args = parser.parse_args(argv)

    results = run_benchmark(seed=args.seed, total_rollout_budget=args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    args.out.write_text(serialized, encoding="utf-8")

    if not args.quiet:
        _print_summary_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
