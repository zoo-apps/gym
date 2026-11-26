# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Reproducible CLI benchmark for Internet-scale low-communication training (zoo-apps/gym#4).

Usage:
    python3 examples/diloco_wan/run_benchmark.py --seed 1350
    python3 examples/diloco_wan/run_benchmark.py --seed 1350 --output examples/diloco_wan/results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

# Allow direct script execution (`python3 examples/diloco_wan/run_benchmark.py`)
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from diloco_wan import __version__  # noqa: E402
from diloco_wan.corpus import CONTEXT_LEN, VOCAB_SIZE, load_corpus  # noqa: E402
from diloco_wan.diloco import RegimeConfig, RegimeResult, run_regime  # noqa: E402
from diloco_wan.model import EMBED_DIM, HIDDEN_DIM, PARAM_COUNT  # noqa: E402
from diloco_wan.quantize import (  # noqa: E402
    CODEC_FP32,
    CODEC_INT4_BLOCK,
    CODEC_INT8_BLOCK,
    CODEC_SIGN_1BIT_NO_EF,
)
from diloco_wan.wan import WanProfile  # noqa: E402


def build_benchmark_regimes(total_steps: int = 512) -> List[RegimeConfig]:
    """Construct the 11 matched benchmark regimes (8 core/ablation + 3 negative)."""
    wan_100ms_clean = WanProfile(
        name="wan_100ms_50mbps_clean",
        rtt_ms=100.0,
        jitter_ms=12.0,
        bandwidth_mbps=50.0,
        packet_loss_rate=0.0,
        churn_drop_prob=0.0,
        straggler_prob=0.0,
        round_timeout_s=5.0,
        min_quorum=2,
        compute_ms_per_step=4.0,
    )
    wan_100ms_elastic = WanProfile(
        name="wan_100ms_50mbps_loss2pct_churn",
        rtt_ms=100.0,
        jitter_ms=15.0,
        bandwidth_mbps=50.0,
        packet_loss_rate=0.02,
        churn_drop_prob=0.10,
        straggler_prob=0.10,
        straggler_multiplier=2.5,
        round_timeout_s=4.0,
        min_quorum=2,
        compute_ms_per_step=4.0,
        elastic_join_round=4,
    )
    wan_250ms_severe = WanProfile(
        name="wan_250ms_10mbps_loss20pct_churn60pct",
        rtt_ms=250.0,
        jitter_ms=40.0,
        bandwidth_mbps=10.0,
        packet_loss_rate=0.20,
        max_retransmits=2,
        churn_drop_prob=0.60,
        straggler_prob=0.25,
        straggler_multiplier=4.0,
        round_timeout_s=1.8,
        min_quorum=3,
        compute_ms_per_step=4.0,
    )

    return [
        # 1. Single-node centralized upper bound
        RegimeConfig(
            name="centralized_sgd_ceiling",
            mode="centralized",
            description="Single-node AdamW upper bound on pooled training shards (0 WAN latency/bytes).",
            total_steps_per_worker=total_steps,
            local_steps_h=1,
            inner_lr=0.012,
            wan=WanProfile(name="local_nvlink", rtt_ms=0.0, bandwidth_mbps=0.0, compute_ms_per_step=4.0),
        ),
        # 2. Synchronous step-by-step full-gradient AllReduce over WAN (H=1)
        RegimeConfig(
            name="allreduce_every_step_wan",
            mode="allreduce",
            description="Synchronous H=1 FP32 full-gradient AllReduce over 100ms/50Mbps WAN.",
            total_steps_per_worker=total_steps,
            local_steps_h=1,
            inner_lr=0.012,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan_100ms_clean,
        ),
        # 3. Prime DiLoCo H=16 uncompressed FP32
        RegimeConfig(
            name="diloco_h16_fp32_wan",
            mode="diloco",
            description="Prime DiLoCo (H=16 local AdamW steps, outer Nesterov momentum, FP32) over 100ms WAN.",
            total_steps_per_worker=total_steps,
            local_steps_h=16,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan_100ms_clean,
        ),
        # 4. Prime DiLoCo H=64 uncompressed FP32
        RegimeConfig(
            name="diloco_h64_fp32_wan",
            mode="diloco",
            description="Prime DiLoCo (H=64 local AdamW steps, outer Nesterov momentum, FP32) over 100ms WAN.",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan_100ms_clean,
        ),
        # 5. Clean compression control: DiLoCo H=64 + INT8 block quantization (0% churn/attack)
        RegimeConfig(
            name="diloco_h64_int8_clean_wan",
            mode="diloco",
            description="DiLoCo H=64 + INT8 block quantization over clean 100ms WAN (matched compression control).",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            wan=wan_100ms_clean,
        ),
        # 6. DiLoCo H=64 + INT8 block quantization + elastic churn + Byzantine attack + VERIFIED
        RegimeConfig(
            name="diloco_h64_int8_verified_wan",
            mode="diloco",
            description="DiLoCo H=64 + INT8 block quant + 2% loss + churn/join + Byzantine poison WITH verification gate.",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            verify_updates=True,
            byzantine_attack="mixed_poison",
            wan=wan_100ms_elastic,
        ),
        # 6. DiLoCo H=64 + INT4 block quantization + elastic churn + Byzantine attack + VERIFIED
        RegimeConfig(
            name="diloco_h64_int4_verified_wan",
            mode="diloco",
            description="DiLoCo H=64 + INT4 block quant + 2% loss + churn/join + Byzantine poison WITH verification gate.",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT4_BLOCK,
            broadcast_codec=CODEC_INT4_BLOCK,
            verify_updates=True,
            byzantine_attack="mixed_poison",
            wan=wan_100ms_elastic,
        ),
        # 7. Security Ablation: Exact same attack WITHOUT verification gate
        RegimeConfig(
            name="diloco_h64_int8_unverified_attack",
            mode="diloco",
            description="Ablation control: same Byzantine poison attack as #5 but with verification gate DISABLED.",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            verify_updates=False,
            byzantine_attack="mixed_poison",
            wan=wan_100ms_elastic,
        ),
        # 8. Negative Regime 1: Extreme H=512 + Non-IID worker shards (client drift)
        RegimeConfig(
            name="neg_extreme_h512_non_iid",
            mode="diloco",
            description="Negative Regime 1: Extreme H=512 sync period + non-IID target-skewed private shards (client drift).",
            is_negative_regime=True,
            partition_mode="non_iid",
            total_steps_per_worker=total_steps,
            local_steps_h=total_steps,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_FP32,
            broadcast_codec=CODEC_FP32,
            wan=wan_100ms_clean,
        ),
        # 9. Negative Regime 2: 20% packet loss + 60% churn over 250ms/10Mbps WAN
        RegimeConfig(
            name="neg_high_loss_severe_churn",
            mode="diloco",
            description="Negative Regime 2: 20% packet loss + 60% mid-round churn over 250ms/10Mbps transcontinental WAN.",
            is_negative_regime=True,
            partition_mode="balanced",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_INT8_BLOCK,
            broadcast_codec=CODEC_INT8_BLOCK,
            verify_updates=True,
            wan=wan_250ms_severe,
        ),
        # 10. Negative Regime 3: 1-bit sign compression without error feedback
        RegimeConfig(
            name="neg_1bit_sign_no_error_feedback",
            mode="diloco",
            description="Negative Regime 3: Ultra-aggressive 1-bit sign quantization without error feedback (H=64).",
            is_negative_regime=True,
            partition_mode="balanced",
            total_steps_per_worker=total_steps,
            local_steps_h=64,
            inner_lr=0.012,
            outer_lr=0.7,
            outer_momentum=0.6,
            codec=CODEC_SIGN_1BIT_NO_EF,
            broadcast_codec=CODEC_SIGN_1BIT_NO_EF,
            wan=wan_100ms_clean,
        ),
    ]


def run_full_benchmark(seed: int = 1350, total_steps: int = 512) -> Dict[str, object]:
    """Execute all 10 benchmark regimes and return a deterministic JSON-serializable report."""
    corpus_balanced = load_corpus(
        num_workers=6, partition_mode="balanced", seed=seed
    )
    corpus_non_iid = load_corpus(
        num_workers=6, partition_mode="non_iid", seed=seed
    )

    regimes = build_benchmark_regimes(total_steps=total_steps)
    results: List[RegimeResult] = []
    for cfg in regimes:
        corpus = (
            corpus_non_iid if cfg.partition_mode == "non_iid" else corpus_balanced
        )
        res = run_regime(cfg, corpus=corpus, seed=seed)
        results.append(res)

    allreduce_res = next(r for r in results if r.name == "allreduce_every_step_wan")
    baseline_bytes = max(1, allreduce_res.total_wire_bytes)
    baseline_wall = max(1e-9, allreduce_res.simulated_wall_time_s)

    enriched_results = []
    for r in results:
        entry = r.to_dict()
        if r.total_wire_bytes > 0:
            entry["wire_byte_reduction_vs_allreduce_x"] = round(
                baseline_bytes / r.total_wire_bytes, 2
            )
        else:
            entry["wire_byte_reduction_vs_allreduce_x"] = None
        entry["wall_speedup_vs_allreduce_x"] = round(
            baseline_wall / max(r.simulated_wall_time_s, 1e-9), 2
        )
        enriched_results.append(entry)

    return {
        "schema_version": __version__,
        "issue": "zoo-apps/gym#4",
        "hip": "HIP-1350",
        "seed": seed,
        "total_steps_per_worker": total_steps,
        "model_spec": {
            "architecture": "Bengio Residual CausalCharLM (pure Python stdlib)",
            "vocab_size": VOCAB_SIZE,
            "context_len": CONTEXT_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "param_count": PARAM_COUNT,
            "fp32_param_bytes": PARAM_COUNT * 4,
        },
        "corpus_manifest": {
            "num_files": len(corpus_balanced.file_pins),
            "total_raw_bytes": corpus_balanced.total_raw_bytes,
            "manifest_sha256": corpus_balanced.manifest_sha256,
            "private_train_examples_total": corpus_balanced.vault.total_train_examples(),
            "verifier_val_examples": len(corpus_balanced.verifier_val),
            "global_test_examples": len(corpus_balanced.global_test),
            "per_worker_balanced_counts": {
                wid: len(exs)
                for wid, exs in sorted(corpus_balanced.vault.worker_shards.items())
            },
            "per_worker_non_iid_counts": {
                wid: len(exs)
                for wid, exs in sorted(corpus_non_iid.vault.worker_shards.items())
            },
            "pinned_files": [
                {
                    "rel_path": p.rel_path,
                    "byte_size": p.byte_size,
                    "sha256": p.sha256,
                    "domain": p.domain,
                }
                for p in corpus_balanced.file_pins
            ],
        },
        "regimes": enriched_results,
    }


def _print_summary_table(report: Dict[str, object]) -> None:
    print("\n=== Zoo Gym #4 (HIP-1350) DiLoCo WAN Benchmark Summary ===")
    manifest = report["corpus_manifest"]
    print(
        f"Seed: {report['seed']} | Corpus: {manifest['num_files']} pinned files "
        f"({manifest['total_raw_bytes']:,} B, sha256={manifest['manifest_sha256'][:16]}...)"
    )
    print(
        f"Model: {report['model_spec']['param_count']:,} params "
        f"({report['model_spec']['fp32_param_bytes']:,} B FP32)\n"
    )
    header = (
        f"{'Regime':<36} {'TestLoss':>8} {'PPL':>7} {'Top1%':>6} "
        f"{'Wall(s)':>8} {'SpdUp':>6} {'WireMB':>8} {'ByteRed':>8} {'PoisRej':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in report["regimes"]:
        red = (
            f"{r['wire_byte_reduction_vs_allreduce_x']:.1f}x"
            if r["wire_byte_reduction_vs_allreduce_x"] is not None
            else "-"
        )
        pois = (
            f"{r['poisoned_rejection_rate']*100:.0f}%"
            if r["poisoned_rejection_rate"] is not None
            else "-"
        )
        print(
            f"{r['name']:<36} {r['final_test_loss']:>8.4f} {r['final_test_perplexity']:>7.2f} "
            f"{r['final_test_top1_acc']*100:>6.2f} {r['simulated_wall_time_s']:>8.2f} "
            f"{r['wall_speedup_vs_allreduce_x']:>5.1f}x {r['total_wire_mb']:>8.3f} "
            f"{red:>8} {pois:>8}"
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic DiLoCo WAN distributed training benchmark (zoo-apps/gym#4)."
    )
    parser.add_argument("--seed", type=int, default=1350, help="Deterministic RNG seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=512,
        help="Total local optimization steps per worker (default: 512).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_THIS_DIR / "results.json"),
        help="Output JSON path (default: examples/diloco_wan/results.json).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress stdout summary table."
    )
    args = parser.parse_args(argv)

    report = run_full_benchmark(seed=args.seed, total_steps=args.steps)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.quiet:
        _print_summary_table(report)
        print(f"\nWrote deterministic results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
