# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Deterministic WAN transport, packet-loss, and elastic worker churn simulator.

Models wide-area link RTT, jitter, bandwidth serialization, MTU packetization,
per-frame packet loss with bounded retransmissions, straggler timeouts,
mid-round node dropouts, and elastic fresh-worker checkpoint catchup.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from .corpus import PrivateShardVault


@dataclass(frozen=True)
class WanProfile:
    """Physical and elastic churn parameters for a simulated WAN cluster."""

    name: str
    rtt_ms: float = 100.0
    jitter_ms: float = 15.0
    bandwidth_mbps: float = 50.0
    packet_loss_rate: float = 0.0
    mtu_bytes: int = 1400
    max_retransmits: int = 3
    churn_drop_prob: float = 0.0
    straggler_prob: float = 0.0
    straggler_multiplier: float = 3.5
    round_timeout_s: float = 5.0
    min_quorum: int = 2
    compute_ms_per_step: float = 4.0
    elastic_join_round: Optional[int] = None


@dataclass
class TransferResult:
    """Outcome of one simulated WAN transfer (upload or download)."""

    delivered: bool
    reason: str
    elapsed_s: float
    payload_bytes: int
    retransmit_bytes: int
    packets_sent: int
    packets_lost: int


@dataclass
class WanStats:
    """Exact network, churn, and wall-clock telemetry accumulated over a run."""

    total_upload_bytes: int = 0
    total_download_bytes: int = 0
    total_retransmit_bytes: int = 0
    total_checkpoint_sync_bytes: int = 0
    packets_sent: int = 0
    packets_lost: int = 0
    packets_retransmitted: int = 0
    transfers_attempted: int = 0
    transfers_succeeded: int = 0
    transfers_dropped_loss: int = 0
    transfers_dropped_churn: int = 0
    transfers_dropped_timeout: int = 0
    quorum_misses: int = 0
    elastic_joins: int = 0
    total_compute_time_s: float = 0.0
    total_wan_sync_time_s: float = 0.0
    privacy_audits_passed: int = 0

    @property
    def total_wire_bytes(self) -> int:
        return (
            self.total_upload_bytes
            + self.total_download_bytes
            + self.total_retransmit_bytes
            + self.total_checkpoint_sync_bytes
        )

    @property
    def total_wall_time_s(self) -> float:
        return self.total_compute_time_s + self.total_wan_sync_time_s


class WanSimulator:
    """Stateful deterministic WAN channel and churn event engine."""

    def __init__(
        self,
        profile: WanProfile,
        seed: int = 1350,
        vault: Optional[PrivateShardVault] = None,
    ) -> None:
        self.profile = profile
        self.rng = random.Random(seed + 7919)
        self.vault = vault
        self.stats = WanStats()

    def sample_worker_compute_time_s(self, local_steps: int) -> Tuple[float, bool]:
        """Return (simulated_compute_seconds, is_straggler) for a worker's local H steps."""
        base_s = (local_steps * self.profile.compute_ms_per_step) / 1000.0
        if self.profile.rtt_ms == 0.0 and self.profile.bandwidth_mbps <= 0.0:
            return base_s, False
        is_straggler = (
            self.profile.straggler_prob > 0.0
            and self.rng.random() < self.profile.straggler_prob
        )
        jitter = 1.0 + self.rng.uniform(-0.08, 0.08)
        mult = self.profile.straggler_multiplier if is_straggler else 1.0
        return base_s * jitter * mult, is_straggler

    def check_mid_round_churn_drop(self) -> bool:
        """Return True if a worker drops out mid-round due to node churn."""
        if self.profile.churn_drop_prob <= 0.0:
            return False
        dropped = self.rng.random() < self.profile.churn_drop_prob
        if dropped:
            self.stats.transfers_dropped_churn += 1
        return dropped

    def simulate_transfer(
        self,
        wire_payload: bytes,
        direction: str,
        extra_wire_overhead: int = 0,
        is_checkpoint_catchup: bool = False,
        is_straggler: bool = False,
    ) -> TransferResult:
        """Simulate transmitting real `wire_payload` bytes across the WAN link.

        Audits `wire_payload` against the PrivateShardVault before transmission so
        zero raw private worker corpus bytes can ever touch the simulated wire.
        """
        if self.vault is not None:
            self.vault.assert_payload_contains_no_private_data(wire_payload)
            self.stats.privacy_audits_passed += 1

        total_bytes = len(wire_payload) + extra_wire_overhead
        self.stats.transfers_attempted += 1

        if self.profile.rtt_ms <= 0.0 and self.profile.bandwidth_mbps <= 0.0:
            # Local single-node memory path (no WAN)
            self.stats.transfers_succeeded += 1
            return TransferResult(
                delivered=True,
                reason="local",
                elapsed_s=0.0,
                payload_bytes=0,
                retransmit_bytes=0,
                packets_sent=0,
                packets_lost=0,
            )

        mtu = max(256, self.profile.mtu_bytes)
        num_packets = max(1, math.ceil(total_bytes / mtu))
        base_rtt_s = (
            self.profile.rtt_ms
            + self.rng.uniform(-self.profile.jitter_ms, self.profile.jitter_ms)
        ) / 1000.0
        base_rtt_s = max(0.001, base_rtt_s)
        if is_straggler:
            base_rtt_s *= 1.8

        bw_bps = max(1e5, self.profile.bandwidth_mbps * 1_000_000.0)
        loss_p = self.profile.packet_loss_rate

        packets_sent = 0
        packets_lost = 0
        retransmit_bytes = 0
        max_rounds_needed = 1
        failed_loss = False

        for pkt_idx in range(num_packets):
            pkt_size = (
                mtu
                if pkt_idx < num_packets - 1
                else max(64, total_bytes - pkt_idx * mtu)
            )
            delivered_pkt = False
            for attempt in range(self.profile.max_retransmits + 1):
                packets_sent += 1
                if attempt > 0:
                    retransmit_bytes += pkt_size
                    self.stats.packets_retransmitted += 1
                if loss_p > 0.0 and self.rng.random() < loss_p:
                    packets_lost += 1
                    continue
                delivered_pkt = True
                if attempt + 1 > max_rounds_needed:
                    max_rounds_needed = attempt + 1
                break
            if not delivered_pkt:
                failed_loss = True
                break

        # One-way propagation + serialization + retransmission RTT penalty
        serialization_s = ((total_bytes + retransmit_bytes) * 8.0) / bw_bps
        elapsed_s = 0.5 * base_rtt_s + serialization_s + (max_rounds_needed - 1) * base_rtt_s

        self.stats.packets_sent += packets_sent
        self.stats.packets_lost += packets_lost
        self.stats.total_retransmit_bytes += retransmit_bytes

        if is_checkpoint_catchup:
            self.stats.total_checkpoint_sync_bytes += total_bytes
        elif direction == "upload":
            self.stats.total_upload_bytes += total_bytes
        else:
            self.stats.total_download_bytes += total_bytes

        if failed_loss:
            self.stats.transfers_dropped_loss += 1
            return TransferResult(
                delivered=False,
                reason="packet_loss_exhausted",
                elapsed_s=elapsed_s,
                payload_bytes=total_bytes,
                retransmit_bytes=retransmit_bytes,
                packets_sent=packets_sent,
                packets_lost=packets_lost,
            )

        if elapsed_s > self.profile.round_timeout_s:
            self.stats.transfers_dropped_timeout += 1
            return TransferResult(
                delivered=False,
                reason="wan_timeout",
                elapsed_s=self.profile.round_timeout_s,
                payload_bytes=total_bytes,
                retransmit_bytes=retransmit_bytes,
                packets_sent=packets_sent,
                packets_lost=packets_lost,
            )

        self.stats.transfers_succeeded += 1
        return TransferResult(
            delivered=True,
            reason="ok",
            elapsed_s=elapsed_s,
            payload_bytes=total_bytes,
            retransmit_bytes=retransmit_bytes,
            packets_sent=packets_sent,
            packets_lost=packets_lost,
        )
