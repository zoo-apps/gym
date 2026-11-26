# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Elastic DiLoCo & matched WAN distributed training engine.

Implements:
  - Centralized AdamW ceiling (single-node pooled data, 0 WAN bytes)
  - Synchronous step-by-step WAN AllReduce (H = 1, FP32 full gradients)
  - Prime DiLoCo (inner AdamW for H local steps + outer Nesterov momentum)
  - Block-quantized pseudo-gradient wire codecs (FP32, INT8, INT4, 1-bit sign)
  - Elastic worker churn, straggler timeout, packet loss, and fresh-worker join
  - Cryptographic envelope auth + independent held-out benefit/safety gate
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import CorpusBundle, Example
from .model import (
    PARAM_COUNT,
    CausalCharLM,
    InnerAdamW,
    flat_to_grads_dict,
    grads_dict_to_flat,
)
from .quantize import (
    CODEC_FP32,
    decode_tensor,
    encode_tensor,
)
from .verify import (
    ContributorKeyring,
    RoundVerificationReceipt,
    SignedUpdateEnvelope,
    UpdateVerifier,
)
from .wan import WanProfile, WanSimulator


@dataclass(frozen=True)
class RegimeConfig:
    """Declarative specification for one benchmark training regime."""

    name: str
    mode: str  # "centralized" | "allreduce" | "diloco"
    description: str
    is_negative_regime: bool = False
    partition_mode: str = "balanced"
    num_workers: int = 6
    total_steps_per_worker: int = 512
    local_steps_h: int = 64
    batch_size: int = 4
    inner_lr: float = 0.012
    inner_weight_decay: float = 1e-4
    outer_lr: float = 0.7
    outer_momentum: float = 0.6
    codec: str = CODEC_FP32
    broadcast_codec: str = CODEC_FP32
    verify_updates: bool = False
    byzantine_attack: str = "none"  # "none" | "mixed_poison"
    wan: WanProfile = field(default_factory=lambda: WanProfile(name="local", rtt_ms=0.0, bandwidth_mbps=0.0))


@dataclass
class RegimeResult:
    """Complete quantitative output for one executed training regime."""

    name: str
    mode: str
    description: str
    is_negative_regime: bool
    partition_mode: str
    local_steps_h: int
    codec: str
    verify_updates: bool
    byzantine_attack: str
    sync_rounds_attempted: int
    sync_rounds_applied: int
    total_train_examples: int
    # Held-out global test quality (unseen during training AND verification!)
    init_test_loss: float
    final_test_loss: float
    test_loss_improvement: float
    final_test_perplexity: float
    final_test_top1_acc: float
    final_test_top5_acc: float
    # Held-out verifier validation quality
    final_val_loss: float
    # Throughput & timing
    simulated_compute_time_s: float
    simulated_wan_sync_time_s: float
    simulated_wall_time_s: float
    compute_utilization_pct: float
    throughput_examples_per_s: float
    throughput_steps_per_s: float
    # Exact network byte accounting
    total_wire_bytes: int
    total_wire_mb: float
    upload_bytes: int
    download_bytes: int
    retransmit_bytes: int
    checkpoint_sync_bytes: int
    packets_sent: int
    packets_lost: int
    # Elastic churn & verification telemetry
    transfers_dropped_loss: int
    transfers_dropped_churn: int
    transfers_dropped_timeout: int
    quorum_misses: int
    elastic_joins: int
    poisoned_submitted: int
    poisoned_rejected: int
    poisoned_rejection_rate: Optional[float]
    honest_submitted: int
    honest_accepted: int
    honest_acceptance_rate: Optional[float]
    aggregate_rollbacks: int
    privacy_audits_passed: int
    private_data_leaked_bytes: int
    # Compact loss curve (every round or sampled points)
    val_loss_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _sample_batch(
    shard: Sequence[Example], cursor: int, batch_size: int
) -> Tuple[List[Example], int]:
    """Deterministically slice a cyclic minibatch from a private shard."""
    n = len(shard)
    batch: List[Example] = []
    for _ in range(batch_size):
        batch.append(shard[cursor % n])
        cursor += 1
    return batch, cursor


def _corrupt_delta(
    delta: Sequence[float],
    poison_type: str,
    rng: random.Random,
) -> List[float]:
    """Generate a Byzantine/adversarial pseudo-gradient update."""
    if poison_type == "sign_flip":
        # Classic gradient-ascent poisoning: valid norm scale, opposite direction.
        return [-2.5 * x for x in delta]
    if poison_type == "scaled_noise":
        # High-energy random noise attack disguised as a valid signed update.
        return [rng.gauss(0.0, 0.35) for _ in delta]
    if poison_type == "scale_explosion":
        return [12.0 * x for x in delta]
    return list(delta)


def _run_centralized(
    cfg: RegimeConfig,
    corpus: CorpusBundle,
    seed: int,
) -> RegimeResult:
    model = CausalCharLM(seed=seed)
    init_test = model.evaluate(corpus.global_test)
    opt = InnerAdamW(lr=cfg.inner_lr, weight_decay=cfg.inner_weight_decay)

    eff_batch = cfg.num_workers * cfg.batch_size
    pooled = corpus.pooled_train
    cursor = 0
    val_hist: List[float] = []
    probe_val = corpus.verifier_val[:320]
    val_hist.append(round(model.evaluate(probe_val).loss, 5))

    for step in range(1, cfg.total_steps_per_worker + 1):
        batch, cursor = _sample_batch(pooled, cursor, eff_batch)
        _loss, grads = model.compute_batch_gradients(batch)
        opt.step(model, grads)
        if step % max(1, cfg.total_steps_per_worker // 8) == 0:
            val_hist.append(round(model.evaluate(probe_val).loss, 5))

    final_test = model.evaluate(corpus.global_test)
    final_val = model.evaluate(corpus.verifier_val)
    total_examples = cfg.total_steps_per_worker * eff_batch
    # Centralized runs effective batch W*B on 1 node (or tightly coupled NVLink cluster)
    # Use the exact same per-step compute baseline as distributed workers for fair speedup comparison.
    compute_s = (cfg.total_steps_per_worker * cfg.wan.compute_ms_per_step) / 1000.0

    return RegimeResult(
        name=cfg.name,
        mode=cfg.mode,
        description=cfg.description,
        is_negative_regime=cfg.is_negative_regime,
        partition_mode=cfg.partition_mode,
        local_steps_h=1,
        codec=CODEC_FP32,
        verify_updates=False,
        byzantine_attack="none",
        sync_rounds_attempted=0,
        sync_rounds_applied=0,
        total_train_examples=total_examples,
        init_test_loss=round(init_test.loss, 5),
        final_test_loss=round(final_test.loss, 5),
        test_loss_improvement=round(init_test.loss - final_test.loss, 5),
        final_test_perplexity=round(final_test.perplexity, 4),
        final_test_top1_acc=round(final_test.top1_acc, 5),
        final_test_top5_acc=round(final_test.top5_acc, 5),
        final_val_loss=round(final_val.loss, 5),
        simulated_compute_time_s=round(compute_s, 4),
        simulated_wan_sync_time_s=0.0,
        simulated_wall_time_s=round(compute_s, 4),
        compute_utilization_pct=100.0,
        throughput_examples_per_s=round(total_examples / max(compute_s, 1e-9), 2),
        throughput_steps_per_s=round(
            cfg.total_steps_per_worker / max(compute_s, 1e-9), 2
        ),
        total_wire_bytes=0,
        total_wire_mb=0.0,
        upload_bytes=0,
        download_bytes=0,
        retransmit_bytes=0,
        checkpoint_sync_bytes=0,
        packets_sent=0,
        packets_lost=0,
        transfers_dropped_loss=0,
        transfers_dropped_churn=0,
        transfers_dropped_timeout=0,
        quorum_misses=0,
        elastic_joins=0,
        poisoned_submitted=0,
        poisoned_rejected=0,
        poisoned_rejection_rate=None,
        honest_submitted=0,
        honest_accepted=0,
        honest_acceptance_rate=None,
        aggregate_rollbacks=0,
        privacy_audits_passed=0,
        private_data_leaked_bytes=0,
        val_loss_history=val_hist,
    )


def _run_allreduce(
    cfg: RegimeConfig,
    corpus: CorpusBundle,
    seed: int,
) -> RegimeResult:
    """Synchronous H=1 step-by-step full-gradient AllReduce over simulated WAN."""
    model = CausalCharLM(seed=seed)
    init_test = model.evaluate(corpus.global_test)
    opt = InnerAdamW(lr=cfg.inner_lr, weight_decay=cfg.inner_weight_decay)
    wan = WanSimulator(cfg.wan, seed=seed, vault=corpus.vault)

    worker_ids = corpus.vault.worker_ids()
    cursors = {wid: 0 for wid in worker_ids}
    val_hist: List[float] = []
    probe_val = corpus.verifier_val[:320]
    val_hist.append(round(model.evaluate(probe_val).loss, 5))

    rounds_applied = 0
    total_examples = 0

    for step in range(1, cfg.total_steps_per_worker + 1):
        step_compute_s = 0.0
        step_upload_s = 0.0
        received_grads: List[List[float]] = []

        for wid in worker_ids:
            batch, cursors[wid] = _sample_batch(
                corpus.vault.worker_shards[wid], cursors[wid], cfg.batch_size
            )
            total_examples += len(batch)
            _loss, grads = model.compute_batch_gradients(batch)
            flat_g = grads_dict_to_flat(grads)

            w_comp_s, is_straggler = wan.sample_worker_compute_time_s(1)
            if w_comp_s > step_compute_s:
                step_compute_s = w_comp_s

            if wan.check_mid_round_churn_drop():
                continue

            payload = encode_tensor(flat_g, codec=cfg.codec)
            tr = wan.simulate_transfer(
                payload,
                direction="upload",
                extra_wire_overhead=128,
                is_straggler=is_straggler,
            )
            if tr.elapsed_s > step_upload_s:
                step_upload_s = tr.elapsed_s
            if tr.delivered:
                _c, dec_g = decode_tensor(payload)
                received_grads.append(dec_g)

        wan.stats.total_compute_time_s += step_compute_s

        if len(received_grads) >= cfg.wan.min_quorum:
            inv_m = 1.0 / len(received_grads)
            avg_g = [sum(col) * inv_m for col in zip(*received_grads, strict=True)]
            # Broadcast aggregated gradient / parameter update back to every worker
            bcast_payload = encode_tensor(avg_g, codec=cfg.broadcast_codec)
            step_download_s = 0.0
            for _wid in worker_ids:
                tr_dl = wan.simulate_transfer(
                    bcast_payload, direction="download", extra_wire_overhead=128
                )
                if tr_dl.elapsed_s > step_download_s:
                    step_download_s = tr_dl.elapsed_s
            wan.stats.total_wan_sync_time_s += step_upload_s + step_download_s
            opt.step(model, flat_to_grads_dict(avg_g))
            rounds_applied += 1
        else:
            wan.stats.quorum_misses += 1
            wan.stats.total_wan_sync_time_s += step_upload_s

        if step % max(1, cfg.total_steps_per_worker // 8) == 0:
            val_hist.append(round(model.evaluate(probe_val).loss, 5))

    final_test = model.evaluate(corpus.global_test)
    final_val = model.evaluate(corpus.verifier_val)
    wall_s = wan.stats.total_wall_time_s
    comp_s = wan.stats.total_compute_time_s
    sync_s = wan.stats.total_wan_sync_time_s
    util_pct = (100.0 * comp_s / wall_s) if wall_s > 0 else 100.0

    return RegimeResult(
        name=cfg.name,
        mode=cfg.mode,
        description=cfg.description,
        is_negative_regime=cfg.is_negative_regime,
        partition_mode=cfg.partition_mode,
        local_steps_h=1,
        codec=cfg.codec,
        verify_updates=False,
        byzantine_attack="none",
        sync_rounds_attempted=cfg.total_steps_per_worker,
        sync_rounds_applied=rounds_applied,
        total_train_examples=total_examples,
        init_test_loss=round(init_test.loss, 5),
        final_test_loss=round(final_test.loss, 5),
        test_loss_improvement=round(init_test.loss - final_test.loss, 5),
        final_test_perplexity=round(final_test.perplexity, 4),
        final_test_top1_acc=round(final_test.top1_acc, 5),
        final_test_top5_acc=round(final_test.top5_acc, 5),
        final_val_loss=round(final_val.loss, 5),
        simulated_compute_time_s=round(comp_s, 4),
        simulated_wan_sync_time_s=round(sync_s, 4),
        simulated_wall_time_s=round(wall_s, 4),
        compute_utilization_pct=round(util_pct, 2),
        throughput_examples_per_s=round(total_examples / max(wall_s, 1e-9), 2),
        throughput_steps_per_s=round(
            cfg.total_steps_per_worker / max(wall_s, 1e-9), 2
        ),
        total_wire_bytes=wan.stats.total_wire_bytes,
        total_wire_mb=round(wan.stats.total_wire_bytes / (1024.0 * 1024.0), 4),
        upload_bytes=wan.stats.total_upload_bytes,
        download_bytes=wan.stats.total_download_bytes,
        retransmit_bytes=wan.stats.total_retransmit_bytes,
        checkpoint_sync_bytes=wan.stats.total_checkpoint_sync_bytes,
        packets_sent=wan.stats.packets_sent,
        packets_lost=wan.stats.packets_lost,
        transfers_dropped_loss=wan.stats.transfers_dropped_loss,
        transfers_dropped_churn=wan.stats.transfers_dropped_churn,
        transfers_dropped_timeout=wan.stats.transfers_dropped_timeout,
        quorum_misses=wan.stats.quorum_misses,
        elastic_joins=wan.stats.elastic_joins,
        poisoned_submitted=0,
        poisoned_rejected=0,
        poisoned_rejection_rate=None,
        honest_submitted=0,
        honest_accepted=0,
        honest_acceptance_rate=None,
        aggregate_rollbacks=0,
        privacy_audits_passed=wan.stats.privacy_audits_passed,
        private_data_leaked_bytes=0,
        val_loss_history=val_hist,
    )


def _run_diloco(
    cfg: RegimeConfig,
    corpus: CorpusBundle,
    seed: int,
) -> RegimeResult:
    """Prime DiLoCo with optional quantization, elastic churn, Byzantine injection, and verification gate."""
    global_model = CausalCharLM(seed=seed)
    init_test = global_model.evaluate(corpus.global_test)
    wan = WanSimulator(cfg.wan, seed=seed, vault=corpus.vault)
    keyring = ContributorKeyring(master_seed=seed)
    verifier = UpdateVerifier(keyring=keyring, verifier_val=corpus.verifier_val)
    attack_rng = random.Random(seed + 4242)

    all_worker_ids = corpus.vault.worker_ids()
    for wid in all_worker_ids:
        keyring.register(wid)

    # Elastic late-join: if configured, withhold the last worker until elastic_join_round
    active_workers = list(all_worker_ids)
    late_joiner: Optional[str] = None
    if cfg.wan.elastic_join_round is not None and len(active_workers) >= 4:
        late_joiner = active_workers.pop()

    cursors = {wid: 0 for wid in all_worker_ids}
    # Prime DiLoCo Algorithm 1: each worker persists its inner AdamW state across rounds.
    inner_opts = {
        wid: InnerAdamW(lr=cfg.inner_lr, weight_decay=cfg.inner_weight_decay)
        for wid in all_worker_ids
    }
    outer_velocity = [0.0] * PARAM_COUNT

    num_rounds = max(1, cfg.total_steps_per_worker // cfg.local_steps_h)
    val_hist: List[float] = []
    probe_val = corpus.verifier_val[:240]
    val_hist.append(round(global_model.evaluate(probe_val).loss, 5))

    rounds_applied = 0
    total_examples = 0
    poisoned_submitted = 0
    poisoned_rejected = 0
    honest_submitted = 0
    honest_accepted = 0
    aggregate_rollbacks = 0

    for round_id in range(1, num_rounds + 1):
        # Elastic join event: fresh worker joins and downloads full FP32 checkpoint over WAN
        if (
            late_joiner is not None
            and cfg.wan.elastic_join_round == round_id
            and late_joiner not in active_workers
        ):
            ckpt_bytes = encode_tensor(global_model.get_flat_params(), codec=CODEC_FP32)
            tr_join = wan.simulate_transfer(
                ckpt_bytes,
                direction="download",
                extra_wire_overhead=128,
                is_checkpoint_catchup=True,
            )
            wan.stats.total_wan_sync_time_s += tr_join.elapsed_s
            wan.stats.elastic_joins += 1
            active_workers.append(late_joiner)

        theta_start = global_model.get_flat_params()
        round_compute_s = 0.0
        round_upload_s = 0.0
        arrived_envelopes: List[SignedUpdateEnvelope] = []

        for w_idx, wid in enumerate(active_workers):
            # Local H-step inner AdamW loop strictly on worker's private shard
            local_model = global_model.clone()
            inner_opt = inner_opts[wid]
            shard = corpus.vault.worker_shards[wid]
            cursor = cursors[wid]
            for _ in range(cfg.local_steps_h):
                batch, cursor = _sample_batch(shard, cursor, cfg.batch_size)
                total_examples += len(batch)
                _loss, grads = local_model.compute_batch_gradients(batch)
                inner_opt.step(local_model, grads)
            cursors[wid] = cursor

            w_comp_s, is_straggler = wan.sample_worker_compute_time_s(cfg.local_steps_h)
            if w_comp_s > round_compute_s:
                round_compute_s = w_comp_s

            # Simulate mid-round node dropout due to elastic churn
            if wan.check_mid_round_churn_drop():
                continue

            theta_end = local_model.get_flat_params()
            delta = [p0 - p1 for p0, p1 in zip(theta_start, theta_end, strict=True)]

            # Optional Byzantine / poisoned update injection
            is_poison = False
            poison_type = "none"
            tamper_sig = False
            if cfg.byzantine_attack == "mixed_poison":
                # Deterministic 2-of-6 (33%) Byzantine attackers + round-3 signature tamper
                if w_idx == 0:
                    is_poison = True
                    poison_type = "sign_flip"
                    delta = _corrupt_delta(delta, poison_type, attack_rng)
                elif w_idx == 1:
                    is_poison = True
                    poison_type = "scaled_noise"
                    delta = _corrupt_delta(delta, poison_type, attack_rng)
                elif w_idx == 2 and round_id == 3:
                    is_poison = True
                    poison_type = "corrupted_signature"
                    tamper_sig = True

            payload = encode_tensor(delta, codec=cfg.codec)
            env = keyring.sign(
                round_id=round_id,
                contributor_id=wid,
                codec=cfg.codec,
                payload_bytes=payload,
                is_injected_poison=is_poison,
                poison_type=poison_type,
            )
            if tamper_sig:
                # Corrupt signature hex while keeping payload bytes intact
                bad_sig = ("0" if env.signature_hex[0] != "0" else "f") + env.signature_hex[1:]
                env = SignedUpdateEnvelope(
                    round_id=env.round_id,
                    contributor_id=env.contributor_id,
                    codec=env.codec,
                    payload_bytes=env.payload_bytes,
                    payload_sha256=env.payload_sha256,
                    signature_hex=bad_sig,
                    is_injected_poison=True,
                    poison_type="corrupted_signature",
                )

            tr = wan.simulate_transfer(
                env.payload_bytes,
                direction="upload",
                extra_wire_overhead=128,
                is_straggler=is_straggler,
            )
            if tr.elapsed_s > round_upload_s:
                round_upload_s = tr.elapsed_s
            if tr.delivered:
                arrived_envelopes.append(env)

        wan.stats.total_compute_time_s += round_compute_s

        # Verification gate vs unverified blind averaging
        accepted_deltas: List[List[float]] = []
        receipt: Optional[RoundVerificationReceipt] = None

        if cfg.verify_updates:
            accepted_deltas, receipt = verifier.verify_round(
                round_id=round_id,
                base_model=global_model,
                envelopes=arrived_envelopes,
            )
            for dec in receipt.decisions:
                if dec.is_injected_poison:
                    poisoned_submitted += 1
                    if not dec.accepted:
                        poisoned_rejected += 1
                else:
                    honest_submitted += 1
                    if dec.accepted:
                        honest_accepted += 1
        else:
            for env in arrived_envelopes:
                if env.is_injected_poison:
                    poisoned_submitted += 1
                else:
                    honest_submitted += 1
                    honest_accepted += 1
                _c, dec_delta = decode_tensor(env.payload_bytes)
                accepted_deltas.append(dec_delta)

        if len(accepted_deltas) < cfg.wan.min_quorum:
            wan.stats.quorum_misses += 1
            wan.stats.total_wan_sync_time_s += round_upload_s
            val_hist.append(round(global_model.evaluate(probe_val).loss, 5))
            continue

        # Average accepted pseudo-gradients & compute Prime DiLoCo Outer Nesterov step
        inv_m = 1.0 / len(accepted_deltas)
        mean_delta = [sum(col) * inv_m for col in zip(*accepted_deltas, strict=True)]

        mu = cfg.outer_momentum
        lr_out = cfg.outer_lr
        new_velocity = [
            mu * v + g for v, g in zip(outer_velocity, mean_delta, strict=True)
        ]
        nesterov_grad = [
            g + mu * v for g, v in zip(mean_delta, new_velocity, strict=True)
        ]
        candidate_theta = [
            p - lr_out * ng for p, ng in zip(theta_start, nesterov_grad, strict=True)
        ]

        candidate_model = global_model.clone()
        candidate_model.set_flat_params(candidate_theta)

        apply_step = True
        if cfg.verify_updates and receipt is not None:
            if not verifier.check_aggregate_safety(candidate_model, receipt):
                apply_step = False
                aggregate_rollbacks += 1

        if apply_step:
            outer_velocity = new_velocity
            global_model = candidate_model
            rounds_applied += 1

            # Broadcast compressed outer update delta back to all active workers
            outer_delta = [
                p0 - p1 for p0, p1 in zip(theta_start, candidate_theta, strict=True)
            ]
            bcast_payload = encode_tensor(outer_delta, codec=cfg.broadcast_codec)
            round_download_s = 0.0
            for _wid in active_workers:
                tr_dl = wan.simulate_transfer(
                    bcast_payload, direction="download", extra_wire_overhead=128
                )
                if tr_dl.elapsed_s > round_download_s:
                    round_download_s = tr_dl.elapsed_s
            wan.stats.total_wan_sync_time_s += round_upload_s + round_download_s
        else:
            wan.stats.total_wan_sync_time_s += round_upload_s

        val_hist.append(round(global_model.evaluate(probe_val).loss, 5))

    final_test = global_model.evaluate(corpus.global_test)
    final_val = global_model.evaluate(corpus.verifier_val)
    wall_s = wan.stats.total_wall_time_s
    comp_s = wan.stats.total_compute_time_s
    sync_s = wan.stats.total_wan_sync_time_s
    util_pct = (100.0 * comp_s / wall_s) if wall_s > 0 else 100.0

    poison_rate = (
        round(poisoned_rejected / poisoned_submitted, 4)
        if poisoned_submitted > 0
        else None
    )
    honest_rate = (
        round(honest_accepted / honest_submitted, 4)
        if honest_submitted > 0
        else None
    )

    return RegimeResult(
        name=cfg.name,
        mode=cfg.mode,
        description=cfg.description,
        is_negative_regime=cfg.is_negative_regime,
        partition_mode=cfg.partition_mode,
        local_steps_h=cfg.local_steps_h,
        codec=cfg.codec,
        verify_updates=cfg.verify_updates,
        byzantine_attack=cfg.byzantine_attack,
        sync_rounds_attempted=num_rounds,
        sync_rounds_applied=rounds_applied,
        total_train_examples=total_examples,
        init_test_loss=round(init_test.loss, 5),
        final_test_loss=round(final_test.loss, 5),
        test_loss_improvement=round(init_test.loss - final_test.loss, 5),
        final_test_perplexity=round(final_test.perplexity, 4),
        final_test_top1_acc=round(final_test.top1_acc, 5),
        final_test_top5_acc=round(final_test.top5_acc, 5),
        final_val_loss=round(final_val.loss, 5),
        simulated_compute_time_s=round(comp_s, 4),
        simulated_wan_sync_time_s=round(sync_s, 4),
        simulated_wall_time_s=round(wall_s, 4),
        compute_utilization_pct=round(util_pct, 2),
        throughput_examples_per_s=round(total_examples / max(wall_s, 1e-9), 2),
        throughput_steps_per_s=round(
            cfg.total_steps_per_worker / max(wall_s, 1e-9), 2
        ),
        total_wire_bytes=wan.stats.total_wire_bytes,
        total_wire_mb=round(wan.stats.total_wire_bytes / (1024.0 * 1024.0), 4),
        upload_bytes=wan.stats.total_upload_bytes,
        download_bytes=wan.stats.total_download_bytes,
        retransmit_bytes=wan.stats.total_retransmit_bytes,
        checkpoint_sync_bytes=wan.stats.total_checkpoint_sync_bytes,
        packets_sent=wan.stats.packets_sent,
        packets_lost=wan.stats.packets_lost,
        transfers_dropped_loss=wan.stats.transfers_dropped_loss,
        transfers_dropped_churn=wan.stats.transfers_dropped_churn,
        transfers_dropped_timeout=wan.stats.transfers_dropped_timeout,
        quorum_misses=wan.stats.quorum_misses,
        elastic_joins=wan.stats.elastic_joins,
        poisoned_submitted=poisoned_submitted,
        poisoned_rejected=poisoned_rejected,
        poisoned_rejection_rate=poison_rate,
        honest_submitted=honest_submitted,
        honest_accepted=honest_accepted,
        honest_acceptance_rate=honest_rate,
        aggregate_rollbacks=aggregate_rollbacks,
        privacy_audits_passed=wan.stats.privacy_audits_passed,
        private_data_leaked_bytes=0,
        val_loss_history=val_hist,
    )


def run_regime(
    cfg: RegimeConfig,
    corpus: CorpusBundle,
    seed: int = 1350,
) -> RegimeResult:
    """Dispatch and execute a single configured training regime deterministically."""
    if cfg.mode == "centralized":
        return _run_centralized(cfg, corpus, seed=seed)
    if cfg.mode == "allreduce":
        return _run_allreduce(cfg, corpus, seed=seed)
    if cfg.mode == "diloco":
        return _run_diloco(cfg, corpus, seed=seed)
    raise ValueError(f"Unsupported mode: {cfg.mode}")
