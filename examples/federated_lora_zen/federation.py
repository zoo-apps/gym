# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""3-worker independent federation, cryptographic envelopes, aggregation, and recovery.

Implements:
- ``AdapterEnvelope``: Signed wire payload containing only low-rank ``(A, B)``
  adapter weights and sample count — zero raw worker vault data.
- ``FederatedWorker``: Independently administered worker owning a private
  ``WorkerVault``, local HMAC-SHA256 signing key, local disk checkpoint
  (``worker_state.json``), and cold-restart checkpoint recovery.
- ``aggregate_envelopes``: Supports both naive ``fedavg_lora_ba`` (averaging
  ``A`` and ``B`` separately, exposing cross-term interference) and
  ``exact_delta_svd_fedlora`` (aggregating exact ``sum w_k (B_k @ A_k)`` and
  refactoring via deterministic rank-``r`` SVD).
- ``FederatedCoordinator`` & ``run_federated_experiment``: Orchestrates multi-
  round 3-worker runs, mid-run node-loss fault tolerance, and cold recovery.
- ``run_centralized_baseline``: Matched pooled-data single-node baseline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .corpus import (
    FederatedCorpusSplit,
    LMExample,
    WorkerVault,
    sign_payload_hmac,
)
from .lora_model import (
    LAYER_NAMES,
    AdapterState,
    FrozenBaseLM,
    LoRALanguageModel,
    matmul,
    matrix_frobenius_diff,
    svd_factor_delta,
)

# Reference WAN link model for deterministic simulated recovery latency reporting.
REFERENCE_WAN_MBPS: float = 100.0
REFERENCE_COLD_BOOT_MS: float = 250.0
REFERENCE_SHA_VERIFY_MS: float = 5.0


class SecurityError(RuntimeError):
    """Raised when an adapter envelope or checkpoint fails cryptographic verification."""


@dataclass(frozen=True)
class AdapterEnvelope:
    """Signed wire-protocol envelope transmitted from a worker to the coordinator."""

    round_idx: int
    worker_id: str
    rank: int
    alpha: float
    n_samples: int
    local_loss: float
    adapter_state: AdapterState
    adapter_sha256: str
    wire_bytes_fp32: int
    wire_bytes_fp16: int
    hmac_signature: str

    def header_bytes(self) -> bytes:
        return (
            f"v1|{self.round_idx}|{self.worker_id}|{self.rank}|"
            f"{self.alpha:.8f}|{self.n_samples}|{self.adapter_sha256}"
        ).encode("utf-8")


@dataclass(frozen=True)
class AggregationReport:
    """Exact mathematical diagnostics for one round of adapter aggregation."""

    strategy: str
    participating_workers: tuple[str, ...]
    total_samples: int
    naive_cross_term_frobenius: float
    svd_truncation_residual_frobenius: float
    applied_effective_error_frobenius: float
    new_adapter_sha256: str


@dataclass(frozen=True)
class RecoveryEvent:
    """Audit record for a worker cold-restart recovery after node loss."""

    worker_id: str
    rejoined_at_round: int
    recovered_from_checkpoint_round: int
    missed_rounds: tuple[int, ...]
    checkpoint_bytes: int
    checkpoint_sha256: str
    verified_integrity: bool
    simulated_recovery_time_ms: float


@dataclass(frozen=True)
class RoundMetrics:
    """Per-round telemetry recorded by the coordinator."""

    round_idx: int
    participating_workers: tuple[str, ...]
    offline_workers: tuple[str, ...]
    mean_worker_local_loss: float
    eval_loss: float
    eval_perplexity: float
    eval_accuracy: float
    naive_cross_term_frobenius: float
    svd_truncation_residual_frobenius: float
    applied_effective_error_frobenius: float
    round_uplink_bytes_fp32: int
    round_downlink_bytes_fp32: int
    server_checkpoint_bytes: int
    global_adapter_sha256: str


@dataclass(frozen=True)
class NodeLossSchedule:
    """Deterministic schedule of offline worker IDs per round (1-indexed)."""

    offline_by_round: dict[int, tuple[str, ...]]
    min_quorum: int = 1

    def offline_for(self, round_idx: int) -> tuple[str, ...]:
        return self.offline_by_round.get(round_idx, ())


@dataclass(frozen=True)
class FederatedConfig:
    """Configuration for a federated LoRA/QLoRA training run."""

    num_rounds: int = 12
    local_steps: int = 8
    batch_size: int = 16
    lr: float = 0.018
    weight_decay: float = 1e-4
    aggregation_strategy: str = "exact_delta_svd_fedlora"
    seed: int = 1350


def _atomic_write_bytes(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, path)
    return len(payload)


def sign_adapter_envelope(
    *,
    round_idx: int,
    worker_id: str,
    admin_key_hex: str,
    n_samples: int,
    local_loss: float,
    adapter_state: AdapterState,
) -> AdapterEnvelope:
    """Create and HMAC-SHA256 sign an ``AdapterEnvelope``."""
    adapter_sha = adapter_state.sha256()
    env = AdapterEnvelope(
        round_idx=round_idx,
        worker_id=worker_id,
        rank=adapter_state.rank,
        alpha=adapter_state.alpha,
        n_samples=n_samples,
        local_loss=local_loss,
        adapter_state=adapter_state.clone(),
        adapter_sha256=adapter_sha,
        wire_bytes_fp32=adapter_state.wire_bytes_fp32(),
        wire_bytes_fp16=adapter_state.wire_bytes_fp16(),
        hmac_signature="",
    )
    msg = env.header_bytes() + env.adapter_state.to_canonical_bytes()
    sig = sign_payload_hmac(admin_key_hex, msg)
    return AdapterEnvelope(
        round_idx=env.round_idx,
        worker_id=env.worker_id,
        rank=env.rank,
        alpha=env.alpha,
        n_samples=env.n_samples,
        local_loss=env.local_loss,
        adapter_state=env.adapter_state,
        adapter_sha256=env.adapter_sha256,
        wire_bytes_fp32=env.wire_bytes_fp32,
        wire_bytes_fp16=env.wire_bytes_fp16,
        hmac_signature=sig,
    )


def verify_adapter_envelope(
    envelope: AdapterEnvelope,
    expected_round: int,
    registered_admin_keys: dict[str, str],
) -> None:
    """Verify worker identity, round match, adapter digest, and HMAC signature."""
    if envelope.worker_id not in registered_admin_keys:
        raise SecurityError(f"Unregistered worker_id={envelope.worker_id!r}")
    if envelope.round_idx != expected_round:
        raise SecurityError(
            f"Envelope round mismatch for {envelope.worker_id}: "
            f"expected {expected_round}, got {envelope.round_idx}"
        )
    recomputed_sha = envelope.adapter_state.sha256()
    if not hmac.compare_digest(recomputed_sha, envelope.adapter_sha256):
        raise SecurityError(
            f"Adapter SHA-256 mismatch for {envelope.worker_id} at round {expected_round}"
        )
    admin_key = registered_admin_keys[envelope.worker_id]
    msg = envelope.header_bytes() + envelope.adapter_state.to_canonical_bytes()
    expected_sig = sign_payload_hmac(admin_key, msg)
    if not hmac.compare_digest(expected_sig, envelope.hmac_signature):
        raise SecurityError(
            f"Invalid HMAC signature for {envelope.worker_id} at round {expected_round}"
        )


def aggregate_envelopes(
    envelopes: Sequence[AdapterEnvelope],
    expected_round: int,
    registered_admin_keys: dict[str, str],
    strategy: str = "exact_delta_svd_fedlora",
) -> tuple[AdapterState, AggregationReport]:
    """Verify and aggregate signed worker adapter envelopes.

    Strategies:
        - ``"fedavg_lora_ba"``: Naive FedAvg averaging ``A`` and ``B`` matrices
          separately. Exposes mathematical cross-term discrepancy
          ``mean(B) @ mean(A) != mean(B @ A)``.
        - ``"exact_delta_svd_fedlora"``: Computes exact sample-weighted
          effective delta ``DeltaW = sum_k w_k (B_k @ A_k)`` and refactors it
          into rank-``r`` ``(B_new, A_new)`` via exact deterministic truncated
          SVD, eliminating cross-term interference.
    """
    if strategy not in ("fedavg_lora_ba", "exact_delta_svd_fedlora"):
        raise ValueError(f"Unsupported aggregation strategy={strategy!r}")
    if not envelopes:
        raise ValueError("Cannot aggregate empty envelope list")

    seen_workers: set[str] = set()
    for env in envelopes:
        verify_adapter_envelope(env, expected_round, registered_admin_keys)
        if env.worker_id in seen_workers:
            raise SecurityError(
                f"Duplicate envelope from {env.worker_id} in round {expected_round}"
            )
        seen_workers.add(env.worker_id)

    # Sort deterministically by worker_id so floating-point summation order is invariant.
    ordered = sorted(envelopes, key=lambda e: e.worker_id)
    total_samples = sum(e.n_samples for e in ordered)
    if total_samples <= 0:
        raise ValueError("Total samples across participating workers must be > 0")
    weights = [e.n_samples / float(total_samples) for e in ordered]

    rank = ordered[0].rank
    alpha = ordered[0].alpha
    scale = alpha / float(rank)

    naive_a: dict[str, list[list[float]]] = {}
    naive_b: dict[str, list[list[float]]] = {}
    svd_a: dict[str, list[list[float]]] = {}
    svd_b: dict[str, list[list[float]]] = {}

    cross_term_sq_sum = 0.0
    svd_trunc_sq_sum = 0.0

    for name in LAYER_NAMES:
        a0 = ordered[0].adapter_state.a_mats[name]
        b0 = ordered[0].adapter_state.b_mats[name]
        d_out = len(b0)
        d_in = len(a0[0])

        # 1. Naive separate average of A and B.
        avg_a = [[0.0] * d_in for _ in range(rank)]
        avg_b = [[0.0] * rank for _ in range(d_out)]
        for w, env in zip(weights, ordered, strict=True):
            a_k = env.adapter_state.a_mats[name]
            b_k = env.adapter_state.b_mats[name]
            for i in range(rank):
                for j in range(d_in):
                    avg_a[i][j] += w * a_k[i][j]
            for i in range(d_out):
                for j in range(rank):
                    avg_b[i][j] += w * b_k[i][j]
        naive_a[name] = avg_a
        naive_b[name] = avg_b

        # 2. Exact sample-weighted effective delta: sum_k w_k (B_k @ A_k).
        exact_delta = [[0.0] * d_in for _ in range(d_out)]
        for w, env in zip(weights, ordered, strict=True):
            ba_k = matmul(
                env.adapter_state.b_mats[name],
                env.adapter_state.a_mats[name],
            )
            for i in range(d_out):
                row_d = exact_delta[i]
                row_ba = ba_k[i]
                for j in range(d_in):
                    row_d[j] += w * row_ba[j]

        # Measure exact cross-term discrepancy of naive FedAvg(A, B).
        naive_delta = matmul(avg_b, avg_a)
        cross_err = scale * matrix_frobenius_diff(naive_delta, exact_delta)
        cross_term_sq_sum += cross_err * cross_err

        if strategy == "exact_delta_svd_fedlora":
            if len(ordered) == 1:
                # Single participating worker: exact identity, zero truncation.
                svd_b[name] = [row[:] for row in ordered[0].adapter_state.b_mats[name]]
                svd_a[name] = [row[:] for row in ordered[0].adapter_state.a_mats[name]]
                trunc_err = 0.0
            else:
                b_new, a_new, raw_residual = svd_factor_delta(
                    exact_delta, rank=rank
                )
                svd_b[name] = b_new
                svd_a[name] = a_new
                trunc_err = scale * raw_residual
            svd_trunc_sq_sum += trunc_err * trunc_err

    cross_frob = cross_term_sq_sum ** 0.5
    svd_frob = svd_trunc_sq_sum ** 0.5

    if strategy == "fedavg_lora_ba":
        new_adapter = AdapterState(
            rank=rank, alpha=alpha, a_mats=naive_a, b_mats=naive_b
        )
        applied_err = cross_frob
    else:
        new_adapter = AdapterState(
            rank=rank, alpha=alpha, a_mats=svd_a, b_mats=svd_b
        )
        applied_err = svd_frob

    report = AggregationReport(
        strategy=strategy,
        participating_workers=tuple(e.worker_id for e in ordered),
        total_samples=total_samples,
        naive_cross_term_frobenius=cross_frob,
        svd_truncation_residual_frobenius=svd_frob,
        applied_effective_error_frobenius=applied_err,
        new_adapter_sha256=new_adapter.sha256(),
    )
    return new_adapter, report


class FederatedWorker:
    """Independently administered worker with private vault and local checkpoint."""

    def __init__(
        self,
        vault: WorkerVault,
        frozen_base: FrozenBaseLM,
        initial_adapter: AdapterState,
        state_dir: Path,
        seed: int = 1350,
    ) -> None:
        self.worker_id = vault.worker_id
        self._vault = vault
        self._seed = seed
        self.state_dir = state_dir / self.worker_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "worker_state.json"
        self.model = LoRALanguageModel(frozen_base, initial_adapter)
        self.last_synced_round = 0
        self.missed_rounds: list[int] = []
        self.recovery_events: list[RecoveryEvent] = []
        self._samples_seen_total = 0

    @property
    def admin_key_hex(self) -> str:
        return self._vault.admin_key_hex

    @property
    def vault_sha256(self) -> str:
        return self._vault.vault_sha256

    @property
    def vault_size(self) -> int:
        return len(self._vault.examples)

    def mark_offline(self, round_idx: int) -> None:
        """Record an offline round and drop volatile in-memory adapter state."""
        self.missed_rounds.append(round_idx)

    def cold_recover_from_checkpoint(
        self, checkpoint_path: Path, rejoin_round: int
    ) -> RecoveryEvent:
        """Simulate cold node restart: wipe RAM state, reload & verify server checkpoint."""
        raw = checkpoint_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        checkpoint_round = int(payload["round_idx"])
        expected_sha = str(payload["adapter_sha256"])
        adapter = AdapterState.from_json_dict(payload["adapter"])
        actual_sha = adapter.sha256()
        if not hmac.compare_digest(actual_sha, expected_sha):
            raise SecurityError(
                f"Corrupted server checkpoint at {checkpoint_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        # Cold restart: reset local optimizer state and hydrate verified global adapter.
        self.model.load_adapter(adapter, reset_optimizer=True)
        self.last_synced_round = checkpoint_round

        # Deterministic simulated WAN transfer + cold-boot + SHA-256 verification latency.
        transfer_ms = (len(raw) * 8.0) / (REFERENCE_WAN_MBPS * 1_000.0)
        sim_ms = REFERENCE_COLD_BOOT_MS + REFERENCE_SHA_VERIFY_MS + transfer_ms

        event = RecoveryEvent(
            worker_id=self.worker_id,
            rejoined_at_round=rejoin_round,
            recovered_from_checkpoint_round=checkpoint_round,
            missed_rounds=tuple(self.missed_rounds),
            checkpoint_bytes=len(raw),
            checkpoint_sha256=actual_sha,
            verified_integrity=True,
            simulated_recovery_time_ms=round(sim_ms, 4),
        )
        self.recovery_events.append(event)
        self.missed_rounds.clear()
        self._persist_local_state(
            round_idx=checkpoint_round,
            local_loss=0.0,
            status="recovered_from_checkpoint",
        )
        return event

    def _persist_local_state(
        self, round_idx: int, local_loss: float, status: str
    ) -> int:
        payload = {
            "worker_id": self.worker_id,
            "round_idx": round_idx,
            "status": status,
            "vault_sha256": self._vault.vault_sha256,
            "samples_seen_total": self._samples_seen_total,
            "last_local_loss": local_loss,
            "adapter_sha256": self.model.adapter.sha256(),
            "adapter": self.model.adapter.to_json_dict(),
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return _atomic_write_bytes(self.state_file, raw)

    def train_local_round(
        self,
        round_idx: int,
        global_adapter: AdapterState,
        local_steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float = 1e-4,
    ) -> AdapterEnvelope:
        """Sync global adapter, train locally on private vault, persist & sign envelope."""
        self.model.load_adapter(global_adapter, reset_optimizer=True)

        # Deterministic per-worker, per-round minibatch RNG.
        seed_bytes = hashlib.sha256(
            struct.pack(">I", self._seed)
            + self.worker_id.encode("utf-8")
            + struct.pack(">I", round_idx)
        ).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))

        examples = self._vault.examples
        n = len(examples)
        loss_acc = 0.0
        for _ in range(local_steps):
            batch = [examples[rng.randrange(n)] for _ in range(batch_size)]
            loss_acc += self.model.train_step_adamw(
                batch, lr=lr, weight_decay=weight_decay
            )

        mean_loss = loss_acc / max(1, local_steps)
        n_samples = local_steps * batch_size
        self._samples_seen_total += n_samples
        self.last_synced_round = round_idx
        self._persist_local_state(
            round_idx=round_idx, local_loss=mean_loss, status="active"
        )

        return sign_adapter_envelope(
            round_idx=round_idx,
            worker_id=self.worker_id,
            admin_key_hex=self._vault.admin_key_hex,
            n_samples=n_samples,
            local_loss=mean_loss,
            adapter_state=self.model.export_adapter(),
        )


class FederatedCoordinator:
    """Coordinator managing global LoRA adapter state, checkpoints, and aggregation."""

    def __init__(
        self,
        initial_adapter: AdapterState,
        registered_admin_keys: dict[str, str],
        checkpoint_dir: Path,
        aggregation_strategy: str = "exact_delta_svd_fedlora",
    ) -> None:
        self.global_adapter = initial_adapter.clone()
        self.registered_admin_keys = dict(registered_admin_keys)
        self.checkpoint_dir = checkpoint_dir / "coordinator"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.aggregation_strategy = aggregation_strategy
        self.current_round = 0
        self.checkpoint_paths: dict[int, Path] = {}
        self.persist_checkpoint(round_idx=0, participating=())

    def persist_checkpoint(
        self, round_idx: int, participating: tuple[str, ...]
    ) -> tuple[Path, int]:
        path = self.checkpoint_dir / f"server_checkpoint_r{round_idx:03d}.json"
        payload = {
            "round_idx": round_idx,
            "aggregation_strategy": self.aggregation_strategy,
            "participating_workers": list(participating),
            "adapter_sha256": self.global_adapter.sha256(),
            "adapter": self.global_adapter.to_json_dict(),
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        n_bytes = _atomic_write_bytes(path, raw)
        self.checkpoint_paths[round_idx] = path
        return path, n_bytes

    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_paths[self.current_round]

    def step_round(
        self, round_idx: int, envelopes: Sequence[AdapterEnvelope]
    ) -> tuple[AggregationReport, int]:
        new_adapter, report = aggregate_envelopes(
            envelopes=envelopes,
            expected_round=round_idx,
            registered_admin_keys=self.registered_admin_keys,
            strategy=self.aggregation_strategy,
        )
        self.global_adapter = new_adapter
        self.current_round = round_idx
        _, ckpt_bytes = self.persist_checkpoint(
            round_idx=round_idx,
            participating=report.participating_workers,
        )
        return report, ckpt_bytes


def run_federated_experiment(
    *,
    regime_name: str,
    split: FederatedCorpusSplit,
    frozen_base: FrozenBaseLM,
    initial_adapter: AdapterState,
    config: FederatedConfig,
    node_loss: NodeLossSchedule | None = None,
    work_dir: Path | None = None,
) -> dict[str, object]:
    """Execute a complete 3-worker federated LoRA/QLoRA run with optional node loss."""
    schedule = node_loss if node_loss is not None else NodeLossSchedule({})

    if work_dir is not None:
        return _run_federated_in_dir(
            regime_name=regime_name,
            split=split,
            frozen_base=frozen_base,
            initial_adapter=initial_adapter,
            config=config,
            schedule=schedule,
            base_dir=work_dir,
        )

    with tempfile.TemporaryDirectory(prefix="fed_lora_zen_") as tmp:
        return _run_federated_in_dir(
            regime_name=regime_name,
            split=split,
            frozen_base=frozen_base,
            initial_adapter=initial_adapter,
            config=config,
            schedule=schedule,
            base_dir=Path(tmp),
        )


def _stratified_eval_probe(
    split: FederatedCorpusSplit, per_domain: int = 200
) -> tuple[LMExample, ...]:
    """Deterministic stratified subset for intermediate round tracking."""
    probe: list[LMExample] = []
    for dom in sorted(split.eval_test_by_domain.keys()):
        exs = split.eval_test_by_domain[dom]
        if len(exs) <= per_domain:
            probe.extend(exs)
        else:
            step = max(1, len(exs) // per_domain)
            probe.extend(exs[::step][:per_domain])
    return tuple(probe)


def _run_federated_in_dir(
    *,
    regime_name: str,
    split: FederatedCorpusSplit,
    frozen_base: FrozenBaseLM,
    initial_adapter: AdapterState,
    config: FederatedConfig,
    schedule: NodeLossSchedule,
    base_dir: Path,
) -> dict[str, object]:
    workers: dict[str, FederatedWorker] = {}
    admin_keys: dict[str, str] = {}
    for wid in sorted(split.worker_vaults.keys()):
        vault = split.worker_vaults[wid]
        admin_keys[wid] = vault.admin_key_hex
        workers[wid] = FederatedWorker(
            vault=vault,
            frozen_base=frozen_base,
            initial_adapter=initial_adapter,
            state_dir=base_dir / "workers",
            seed=config.seed,
        )

    coordinator = FederatedCoordinator(
        initial_adapter=initial_adapter,
        registered_admin_keys=admin_keys,
        checkpoint_dir=base_dir,
        aggregation_strategy=config.aggregation_strategy,
    )

    eval_model = LoRALanguageModel(frozen_base, initial_adapter)
    initial_eval = eval_model.evaluate_by_domain(
        split.eval_test, split.eval_test_by_domain
    )
    eval_probe = _stratified_eval_probe(split, per_domain=200)
    footprint = eval_model.memory_footprint()

    round_history: list[RoundMetrics] = []
    recovery_events: list[RecoveryEvent] = []
    total_uplink_fp32 = 0
    total_downlink_fp32 = 0
    total_uplink_fp16 = 0
    total_downlink_fp16 = 0
    total_samples_processed = 0

    for r in range(1, config.num_rounds + 1):
        offline_set = set(schedule.offline_for(r))
        active_ids = [wid for wid in sorted(workers.keys()) if wid not in offline_set]
        if len(active_ids) < schedule.min_quorum:
            raise RuntimeError(
                f"Quorum lost at round {r}: {len(active_ids)} < {schedule.min_quorum}"
            )

        for wid in sorted(offline_set):
            if wid in workers:
                workers[wid].mark_offline(r)

        envelopes: list[AdapterEnvelope] = []
        round_uplink_fp32 = 0
        round_downlink_fp32 = 0

        for wid in active_ids:
            worker = workers[wid]
            if worker.missed_rounds:
                # Worker crashed/dropped earlier -> cold-recover from latest server checkpoint.
                event = worker.cold_recover_from_checkpoint(
                    checkpoint_path=coordinator.latest_checkpoint_path(),
                    rejoin_round=r,
                )
                recovery_events.append(event)

            # Downlink: coordinator sends current global adapter to active worker.
            down_fp32 = coordinator.global_adapter.wire_bytes_fp32()
            down_fp16 = coordinator.global_adapter.wire_bytes_fp16()
            round_downlink_fp32 += down_fp32
            total_downlink_fp32 += down_fp32
            total_downlink_fp16 += down_fp16

            env = worker.train_local_round(
                round_idx=r,
                global_adapter=coordinator.global_adapter,
                local_steps=config.local_steps,
                batch_size=config.batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
            envelopes.append(env)
            round_uplink_fp32 += env.wire_bytes_fp32
            total_uplink_fp32 += env.wire_bytes_fp32
            total_uplink_fp16 += env.wire_bytes_fp16
            total_samples_processed += env.n_samples

        report, ckpt_bytes = coordinator.step_round(r, envelopes)

        eval_model.load_adapter(coordinator.global_adapter, reset_optimizer=True)
        if r == config.num_rounds:
            final_eval = eval_model.evaluate_by_domain(
                split.eval_test, split.eval_test_by_domain
            )
            r_eval = final_eval["overall"]
        else:
            r_eval = eval_model.evaluate(eval_probe)
        mean_local_loss = sum(e.local_loss for e in envelopes) / len(envelopes)

        round_history.append(
            RoundMetrics(
                round_idx=r,
                participating_workers=report.participating_workers,
                offline_workers=tuple(sorted(offline_set)),
                mean_worker_local_loss=round(mean_local_loss, 6),
                eval_loss=round(float(r_eval["loss"]), 6),  # type: ignore[index]
                eval_perplexity=round(float(r_eval["perplexity"]), 4),  # type: ignore[index]
                eval_accuracy=round(float(r_eval["accuracy"]), 6),  # type: ignore[index]
                naive_cross_term_frobenius=round(
                    report.naive_cross_term_frobenius, 6
                ),
                svd_truncation_residual_frobenius=round(
                    report.svd_truncation_residual_frobenius, 6
                ),
                applied_effective_error_frobenius=round(
                    report.applied_effective_error_frobenius, 6
                ),
                round_uplink_bytes_fp32=round_uplink_fp32,
                round_downlink_bytes_fp32=round_downlink_fp32,
                server_checkpoint_bytes=ckpt_bytes,
                global_adapter_sha256=report.new_adapter_sha256,
            )
        )

    mean_cross = sum(
        m.naive_cross_term_frobenius for m in round_history
    ) / len(round_history)
    max_cross = max(m.naive_cross_term_frobenius for m in round_history)
    mean_svd_res = sum(
        m.svd_truncation_residual_frobenius for m in round_history
    ) / len(round_history)

    return {
        "regime": regime_name,
        "mode": "federated",
        "partition_mode": split.partition_mode,
        "quantization_mode": frozen_base.quantization_mode,
        "aggregation_strategy": config.aggregation_strategy,
        "lora_rank": initial_adapter.rank,
        "lora_alpha": initial_adapter.alpha,
        "num_rounds": config.num_rounds,
        "local_steps_per_round": config.local_steps,
        "worker_batch_size": config.batch_size,
        "total_samples_processed": total_samples_processed,
        "base_frozen_verified": frozen_base.verify_frozen(),
        "memory_footprint": asdict(footprint),
        "communication": {
            "total_uplink_bytes_fp32": total_uplink_fp32,
            "total_downlink_bytes_fp32": total_downlink_fp32,
            "total_wire_bytes_fp32": total_uplink_fp32 + total_downlink_fp32,
            "total_uplink_bytes_fp16": total_uplink_fp16,
            "total_downlink_bytes_fp16": total_downlink_fp16,
            "total_wire_bytes_fp16": total_uplink_fp16 + total_downlink_fp16,
            "raw_data_bytes_shared": 0,
        },
        "aggregation_diagnostics": {
            "mean_naive_cross_term_frobenius": round(mean_cross, 6),
            "max_naive_cross_term_frobenius": round(max_cross, 6),
            "mean_svd_truncation_residual_frobenius": round(mean_svd_res, 6),
        },
        "recovery_events": [asdict(ev) for ev in recovery_events],
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "rounds": [asdict(m) for m in round_history],
    }


def run_centralized_baseline(
    *,
    regime_name: str,
    split: FederatedCorpusSplit,
    frozen_base: FrozenBaseLM,
    initial_adapter: AdapterState,
    num_rounds: int = 12,
    local_steps: int = 8,
    global_batch_size: int = 48,
    lr: float = 0.018,
    weight_decay: float = 1e-4,
    seed: int = 1350,
) -> dict[str, object]:
    """Run matched centralized LoRA/QLoRA upper bound on exact pooled worker data."""
    model = LoRALanguageModel(frozen_base, initial_adapter)
    initial_eval = model.evaluate_by_domain(
        split.eval_test, split.eval_test_by_domain
    )
    footprint = model.memory_footprint()

    eval_probe = _stratified_eval_probe(split, per_domain=200)
    rng = random.Random(seed + 991)
    examples: Sequence[LMExample] = split.centralized_train
    n = len(examples)
    total_samples = 0
    eval_curve: list[dict[str, object]] = []

    for r in range(1, num_rounds + 1):
        loss_acc = 0.0
        for _ in range(local_steps):
            batch = [examples[rng.randrange(n)] for _ in range(global_batch_size)]
            loss_acc += model.train_step_adamw(
                batch, lr=lr, weight_decay=weight_decay
            )
            total_samples += global_batch_size
        if r == num_rounds:
            final_eval = model.evaluate_by_domain(
                split.eval_test, split.eval_test_by_domain
            )
            r_eval = final_eval["overall"]
        else:
            r_eval = model.evaluate(eval_probe)
        eval_curve.append(
            {
                "round_idx": r,
                "train_loss": round(loss_acc / local_steps, 6),
                "eval_loss": round(float(r_eval["loss"]), 6),  # type: ignore[index]
                "eval_perplexity": round(float(r_eval["perplexity"]), 4),  # type: ignore[index]
                "eval_accuracy": round(float(r_eval["accuracy"]), 6),  # type: ignore[index]
                "adapter_sha256": model.adapter.sha256(),
            }
        )

    return {
        "regime": regime_name,
        "mode": "centralized",
        "partition_mode": split.partition_mode,
        "quantization_mode": frozen_base.quantization_mode,
        "aggregation_strategy": "none_centralized",
        "lora_rank": initial_adapter.rank,
        "lora_alpha": initial_adapter.alpha,
        "num_rounds": num_rounds,
        "local_steps_per_round": local_steps,
        "global_batch_size": global_batch_size,
        "total_samples_processed": total_samples,
        "base_frozen_verified": frozen_base.verify_frozen(),
        "memory_footprint": asdict(footprint),
        "communication": {
            "total_uplink_bytes_fp32": 0,
            "total_downlink_bytes_fp32": 0,
            "total_wire_bytes_fp32": 0,
            "total_uplink_bytes_fp16": 0,
            "total_downlink_bytes_fp16": 0,
            "total_wire_bytes_fp16": 0,
            "raw_data_bytes_shared": split.total_bytes,
        },
        "recovery_events": [],
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "rounds": eval_curve,
    }
