# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Reproducible matched benchmark for Federated LoRA / QLoRA across 3 workers.

Usage:
    python3 -m examples.federated_lora_zen.run_benchmark \\
        --seed 1350 \\
        --out examples/federated_lora_zen/results.json

Produces byte-identical JSON output across repeated runs with the same seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .corpus import DOMAINS, WORKER_IDS, build_federated_corpus
from .federation import (
    FederatedConfig,
    NodeLossSchedule,
    run_centralized_baseline,
    run_federated_experiment,
)
from .lora_model import (
    FrozenBaseLM,
    ModelConfig,
    init_lora_adapter,
)


def _extract_summary_row(
    result: dict[str, Any],
    ref_central_loss: float,
    ref_healthy_fed_loss: float | None = None,
) -> dict[str, Any]:
    init_ov = result["initial_eval"]["overall"]
    fin_ov = result["final_eval"]["overall"]
    fin_dom = result["final_eval"]["per_domain"]
    mem = result["memory_footprint"]
    comm = result["communication"]
    rec = result["recovery_events"]

    row: dict[str, Any] = {
        "regime": result["regime"],
        "mode": result["mode"],
        "partition_mode": result["partition_mode"],
        "quantization_mode": result["quantization_mode"],
        "aggregation_strategy": result["aggregation_strategy"],
        "lora_rank": result["lora_rank"],
        "initial_test_loss": round(float(init_ov["loss"]), 6),
        "final_test_loss": round(float(fin_ov["loss"]), 6),
        "loss_improvement": round(
            float(init_ov["loss"]) - float(fin_ov["loss"]), 6
        ),
        "final_test_perplexity": round(float(fin_ov["perplexity"]), 4),
        "final_test_accuracy": round(float(fin_ov["accuracy"]), 6),
        "delta_loss_vs_centralized_lora": round(
            float(fin_ov["loss"]) - ref_central_loss, 6
        ),
        "per_domain_test_loss": {
            d: round(float(fin_dom[d]["loss"]), 6) for d in DOMAINS
        },
        "base_active_bytes": int(mem["base_active_bytes"]),
        "lora_fp32_bytes": int(mem["lora_fp32_bytes"]),
        "base_memory_savings_vs_fp32": round(
            float(mem["base_memory_savings_vs_fp32"]), 3
        ),
        "trainable_param_ratio": round(float(mem["trainable_param_ratio"]), 4),
        "total_wire_bytes_fp32": int(comm["total_wire_bytes_fp32"]),
        "total_wire_bytes_fp16": int(comm["total_wire_bytes_fp16"]),
        "raw_data_bytes_shared": int(comm["raw_data_bytes_shared"]),
        "base_frozen_verified": bool(result["base_frozen_verified"]),
        "recovery_count": len(rec),
        "final_adapter_sha256": str(result["final_eval"]["adapter_sha256"]),
    }
    if ref_healthy_fed_loss is not None:
        row["delta_loss_vs_healthy_fed3_qlora"] = round(
            float(fin_ov["loss"]) - ref_healthy_fed_loss, 6
        )
    if "aggregation_diagnostics" in result:
        row["mean_naive_cross_term_frobenius"] = result[
            "aggregation_diagnostics"
        ]["mean_naive_cross_term_frobenius"]
        row["mean_svd_truncation_residual_frobenius"] = result[
            "aggregation_diagnostics"
        ]["mean_svd_truncation_residual_frobenius"]
    if rec:
        row["simulated_recovery_time_ms"] = rec[0]["simulated_recovery_time_ms"]
        row["recovery_checkpoint_bytes"] = rec[0]["checkpoint_bytes"]
    return row


def run_all_benchmarks(
    seed: int = 1350,
    num_rounds: int = 12,
    local_steps: int = 8,
    worker_batch_size: int = 16,
    lr: float = 0.006,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Execute all 6 matched main regimes + 3 honest negative regimes."""
    # 1. Build SHA-256 pinned real corpus splits (identical base_pretrain & eval_test).
    split_mixed = build_federated_corpus(
        seed=seed,
        partition_mode="mixed_non_iid",
        repo_root=repo_root,
    )
    split_extreme = build_federated_corpus(
        seed=seed,
        partition_mode="extreme_non_iid",
        repo_root=repo_root,
    )

    # 2. Pretrain one shared Zen-style compact base LM, then freeze in FP32 & NF4.
    model_cfg = ModelConfig()
    base_fp32 = FrozenBaseLM.create_initialized(model_cfg, seed=seed)
    pretrain_stats = base_fp32.pretrain_on_split(
        split_mixed.base_pretrain,
        steps=60,
        batch_size=32,
        lr=0.10,
        seed=seed,
    )
    base_nf4 = base_fp32.to_quantization_mode("nf4")

    # Shared initial adapters (exact Hu et al. zero-init B, deterministic A).
    init_adapter_r8 = init_lora_adapter(
        model_cfg, rank=8, alpha=16.0, seed=seed + 101
    )
    init_adapter_r1 = init_lora_adapter(
        model_cfg, rank=1, alpha=2.0, seed=seed + 101
    )

    global_bs = worker_batch_size * len(WORKER_IDS)

    # =========================================================================
    # MAIN 6 MATCHED REGIMES (mixed_non_iid institutional 3-worker vaults)
    # =========================================================================
    regimes: dict[str, dict[str, Any]] = {}

    # Regime 1: Centralized FP32 LoRA upper bound (pooled union of 3 vaults).
    regimes["centralized_lora_baseline"] = run_centralized_baseline(
        regime_name="centralized_lora_baseline",
        split=split_mixed,
        frozen_base=base_fp32,
        initial_adapter=init_adapter_r8,
        num_rounds=num_rounds,
        local_steps=local_steps,
        global_batch_size=global_bs,
        lr=lr,
        seed=seed,
    )

    # Regime 2: Centralized 4-bit NF4 QLoRA upper bound.
    regimes["centralized_qlora_nf4_baseline"] = run_centralized_baseline(
        regime_name="centralized_qlora_nf4_baseline",
        split=split_mixed,
        frozen_base=base_nf4,
        initial_adapter=init_adapter_r8,
        num_rounds=num_rounds,
        local_steps=local_steps,
        global_batch_size=global_bs,
        lr=lr,
        seed=seed,
    )

    # Regime 3: 3-Worker FP32 LoRA with naive FedAvg(A, B).
    regimes["fed3_lora_fedavg_ba"] = run_federated_experiment(
        regime_name="fed3_lora_fedavg_ba",
        split=split_mixed,
        frozen_base=base_fp32,
        initial_adapter=init_adapter_r8,
        config=FederatedConfig(
            num_rounds=num_rounds,
            local_steps=local_steps,
            batch_size=worker_batch_size,
            lr=lr,
            aggregation_strategy="fedavg_lora_ba",
            seed=seed,
        ),
    )

    # Regime 4: 3-Worker FP32 LoRA with Exact Delta-W SVD aggregation.
    regimes["fed3_lora_exact_svd"] = run_federated_experiment(
        regime_name="fed3_lora_exact_svd",
        split=split_mixed,
        frozen_base=base_fp32,
        initial_adapter=init_adapter_r8,
        config=FederatedConfig(
            num_rounds=num_rounds,
            local_steps=local_steps,
            batch_size=worker_batch_size,
            lr=lr,
            aggregation_strategy="exact_delta_svd_fedlora",
            seed=seed,
        ),
    )

    # Regime 5: 3-Worker 4-bit NF4 QLoRA with Exact Delta-W SVD aggregation.
    regimes["fed3_qlora_nf4_exact_svd"] = run_federated_experiment(
        regime_name="fed3_qlora_nf4_exact_svd",
        split=split_mixed,
        frozen_base=base_nf4,
        initial_adapter=init_adapter_r8,
        config=FederatedConfig(
            num_rounds=num_rounds,
            local_steps=local_steps,
            batch_size=worker_batch_size,
            lr=lr,
            aggregation_strategy="exact_delta_svd_fedlora",
            seed=seed,
        ),
    )

    # Regime 6: 3-Worker QLoRA + Real Mid-Run Node Loss (worker-beta drops
    # offline for 3 rounds: rounds 4, 5, 6; recovers cold from server
    # checkpoint at round 7 and finishes cleanly).
    drop_start = max(2, num_rounds // 3)
    max_drop_end = num_rounds - 1 if num_rounds <= 4 else num_rounds - 2
    drop_end = min(max_drop_end, drop_start + 2)
    beta_drop_schedule = NodeLossSchedule(
        offline_by_round={
            r: ("worker-beta",) for r in range(drop_start, drop_end + 1)
        },
        min_quorum=2,
    )
    regimes["fed3_qlora_node_loss_recovery"] = run_federated_experiment(
        regime_name="fed3_qlora_node_loss_recovery",
        split=split_mixed,
        frozen_base=base_nf4,
        initial_adapter=init_adapter_r8,
        config=FederatedConfig(
            num_rounds=num_rounds,
            local_steps=local_steps,
            batch_size=worker_batch_size,
            lr=lr,
            aggregation_strategy="exact_delta_svd_fedlora",
            seed=seed,
        ),
        node_loss=beta_drop_schedule,
    )

    # =========================================================================
    # HONEST NEGATIVE REGIMES
    # =========================================================================
    negative_regimes: dict[str, dict[str, Any]] = {}

    # Negative 1 (paired): Extreme Non-IID (100% disjoint single-domain vaults,
    # 16 local steps/round) exposing naive FedAvg(A, B) cross-term interference
    # vs Exact Delta-W SVD aggregation.
    extreme_local_steps = max(local_steps * 2, 16)
    negative_regimes["neg_extreme_non_iid_naive_fedavg"] = (
        run_federated_experiment(
            regime_name="neg_extreme_non_iid_naive_fedavg",
            split=split_extreme,
            frozen_base=base_nf4,
            initial_adapter=init_adapter_r8,
            config=FederatedConfig(
                num_rounds=num_rounds,
                local_steps=extreme_local_steps,
                batch_size=worker_batch_size,
                lr=lr,
                aggregation_strategy="fedavg_lora_ba",
                seed=seed,
            ),
        )
    )
    negative_regimes["neg_extreme_non_iid_exact_svd_ref"] = (
        run_federated_experiment(
            regime_name="neg_extreme_non_iid_exact_svd_ref",
            split=split_extreme,
            frozen_base=base_nf4,
            initial_adapter=init_adapter_r8,
            config=FederatedConfig(
                num_rounds=num_rounds,
                local_steps=extreme_local_steps,
                batch_size=worker_batch_size,
                lr=lr,
                aggregation_strategy="exact_delta_svd_fedlora",
                seed=seed,
            ),
        )
    )

    # Negative 2: Rank-1 under-parameterized bottleneck (r=1 vs r=8).
    negative_regimes["neg_rank1_bottleneck"] = run_federated_experiment(
        regime_name="neg_rank1_bottleneck",
        split=split_mixed,
        frozen_base=base_nf4,
        initial_adapter=init_adapter_r1,
        config=FederatedConfig(
            num_rounds=num_rounds,
            local_steps=local_steps,
            batch_size=worker_batch_size,
            lr=lr,
            aggregation_strategy="exact_delta_svd_fedlora",
            seed=seed,
        ),
    )

    # Negative 3: Permanent 2-of-3 worker attrition (67% fleet loss from
    # round 3 onward under extreme Non-IID -> only worker-alpha/docs remains).
    perm_loss_start = min(3, num_rounds)
    perm_loss_schedule = NodeLossSchedule(
        offline_by_round={
            r: ("worker-beta", "worker-gamma")
            for r in range(perm_loss_start, num_rounds + 1)
        },
        min_quorum=1,
    )
    negative_regimes["neg_permanent_two_of_three_loss"] = (
        run_federated_experiment(
            regime_name="neg_permanent_two_of_three_loss",
            split=split_extreme,
            frozen_base=base_nf4,
            initial_adapter=init_adapter_r8,
            config=FederatedConfig(
                num_rounds=num_rounds,
                local_steps=local_steps,
                batch_size=worker_batch_size,
                lr=lr,
                aggregation_strategy="exact_delta_svd_fedlora",
                seed=seed,
            ),
            node_loss=perm_loss_schedule,
        )
    )

    # Build deterministic summaries.
    central_loss = float(
        regimes["centralized_lora_baseline"]["final_eval"]["overall"]["loss"]
    )
    healthy_fed_qlora_loss = float(
        regimes["fed3_qlora_nf4_exact_svd"]["final_eval"]["overall"]["loss"]
    )

    summary_main = [
        _extract_summary_row(
            regimes[name],
            ref_central_loss=central_loss,
            ref_healthy_fed_loss=healthy_fed_qlora_loss,
        )
        for name in (
            "centralized_lora_baseline",
            "centralized_qlora_nf4_baseline",
            "fed3_lora_fedavg_ba",
            "fed3_lora_exact_svd",
            "fed3_qlora_nf4_exact_svd",
            "fed3_qlora_node_loss_recovery",
        )
    ]

    summary_negative = [
        _extract_summary_row(
            negative_regimes[name],
            ref_central_loss=central_loss,
            ref_healthy_fed_loss=healthy_fed_qlora_loss,
        )
        for name in (
            "neg_extreme_non_iid_naive_fedavg",
            "neg_extreme_non_iid_exact_svd_ref",
            "neg_rank1_bottleneck",
            "neg_permanent_two_of_three_loss",
        )
    ]

    return {
        "schema_version": "1.0.0",
        "issue": "zoo-apps/gym#3",
        "hip": "HIP-1350",
        "seed": seed,
        "corpus": {
            "manifest_sha256": split_mixed.corpus_manifest_sha256,
            "total_files": split_mixed.total_files,
            "total_bytes": split_mixed.total_bytes,
            "vocab_size": split_mixed.vocab.size,
            "context_len": split_mixed.context_len,
            "chunk_chars": split_mixed.chunk_chars,
            "base_pretrain_examples": len(split_mixed.base_pretrain),
            "centralized_train_examples": len(split_mixed.centralized_train),
            "eval_test_examples": len(split_mixed.eval_test),
            "eval_test_by_domain_counts": {
                d: len(split_mixed.eval_test_by_domain[d]) for d in DOMAINS
            },
            "mixed_non_iid_vaults": {
                wid: {
                    "examples": len(split_mixed.worker_vaults[wid].examples),
                    "domain_chunk_counts": split_mixed.worker_vaults[
                        wid
                    ].domain_counts,
                    "vault_sha256": split_mixed.worker_vaults[wid].vault_sha256,
                    "admin_key_fingerprint": split_mixed.worker_vaults[
                        wid
                    ].admin_key_fingerprint,
                }
                for wid in WORKER_IDS
            },
            "extreme_non_iid_vaults": {
                wid: {
                    "examples": len(split_extreme.worker_vaults[wid].examples),
                    "domain_chunk_counts": split_extreme.worker_vaults[
                        wid
                    ].domain_counts,
                    "vault_sha256": split_extreme.worker_vaults[
                        wid
                    ].vault_sha256,
                }
                for wid in WORKER_IDS
            },
        },
        "base_model": {
            "vocab_size": model_cfg.vocab_size,
            "context_len": model_cfg.context_len,
            "embed_dim": model_cfg.embed_dim,
            "hidden_dim": model_cfg.hidden_dim,
            "nf4_block_size": model_cfg.nf4_block_size,
            "pretrain_stats": {
                k: round(v, 6) for k, v in pretrain_stats.items()
            },
            "fp32_frozen_sha256": base_fp32.frozen_sha256,
            "nf4_frozen_sha256": base_nf4.frozen_sha256,
        },
        "summary_main_regimes": summary_main,
        "summary_negative_regimes": summary_negative,
        "regimes": regimes,
        "negative_regimes": negative_regimes,
    }


def _print_report(payload: dict[str, Any], wall_sec: float) -> None:
    print(
        f"\n=== Federated LoRA/QLoRA Zen Benchmark (seed={payload['seed']}) ==="
    )
    print(
        f"Corpus: {payload['corpus']['total_files']} pinned files "
        f"({payload['corpus']['total_bytes']} bytes), "
        f"eval_test={payload['corpus']['eval_test_examples']} examples"
    )
    print("\n--- Main 6 Matched Regimes ---")
    header = (
        f"{'Regime':<33} {'Quant':<5} {'Agg':<23} "
        f"{'Loss':>7} {'PPL':>7} {'Acc%':>6} {'dLoss':>8} "
        f"{'BaseB':>7} {'WireB(F32)':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["summary_main_regimes"]:
        print(
            f"{row['regime']:<33} {row['quantization_mode']:<5} "
            f"{row['aggregation_strategy']:<23} "
            f"{row['final_test_loss']:>7.4f} "
            f"{row['final_test_perplexity']:>7.2f} "
            f"{row['final_test_accuracy'] * 100:>6.2f} "
            f"{row['delta_loss_vs_centralized_lora']:>+8.4f} "
            f"{row['base_active_bytes']:>7d} "
            f"{row['total_wire_bytes_fp32']:>10d}"
        )

    print("\n--- Honest Negative Regimes ---")
    nheader = (
        f"{'Regime':<35} {'Rank':>4} {'Loss':>7} {'PPL':>7} "
        f"{'docs':>7} {'yaml':>7} {'python':>7} {'CrossF':>8}"
    )
    print(nheader)
    print("-" * len(nheader))
    for row in payload["summary_negative_regimes"]:
        pd = row["per_domain_test_loss"]
        cf = row.get("mean_naive_cross_term_frobenius", 0.0)
        print(
            f"{row['regime']:<35} {row['lora_rank']:>4d} "
            f"{row['final_test_loss']:>7.4f} "
            f"{row['final_test_perplexity']:>7.2f} "
            f"{pd['docs']:>7.4f} {pd['yaml']:>7.4f} {pd['python']:>7.4f} "
            f"{cf:>8.4f}"
        )
    print(f"\nHost wall time (excluded from results.json): {wall_sec:.2f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible Federated LoRA/QLoRA Zen benchmark."
    )
    parser.add_argument(
        "--seed", type=int, default=1350, help="Master deterministic seed."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=12,
        help="Number of federation rounds (default: 12).",
    )
    parser.add_argument(
        "--local-steps",
        type=int,
        default=8,
        help="Local AdamW steps per worker per round (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-worker minibatch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.006,
        help="AdamW learning rate for LoRA adapters (default: 0.006).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    payload = run_all_benchmarks(
        seed=args.seed,
        num_rounds=args.rounds,
        local_steps=args.local_steps,
        worker_batch_size=args.batch_size,
        lr=args.lr,
    )
    wall_sec = time.perf_counter() - t0

    serialized = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(serialized)
    digest = hashlib.sha256(serialized).hexdigest()

    _print_report(payload, wall_sec=wall_sec)
    print(f"Wrote {args.out} ({len(serialized)} bytes, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
