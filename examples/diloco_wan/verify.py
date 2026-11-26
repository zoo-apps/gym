# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Cryptographic envelope authentication & independent held-out benefit/safety gate.

Directly addresses the core research question of zoo-apps/gym#4:
"how do you show their updates are beneficial and safe, not just that their GPUs computed?"

Verification operates strictly on wire-serialized pseudo-gradients and an
independent coordinator-held validation split (`verifier_val`). No worker's
private dataset is ever requested, transmitted, or inspected.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .corpus import Example
from .model import CausalCharLM
from .quantize import decode_tensor


@dataclass(frozen=True)
class SignedUpdateEnvelope:
    """Wire-transmitted contributor update with identity & integrity metadata."""

    round_id: int
    contributor_id: str
    codec: str
    payload_bytes: bytes
    payload_sha256: str
    signature_hex: str
    is_injected_poison: bool = False  # Ground-truth audit tag for benchmark scoring only
    poison_type: str = "none"

    @property
    def wire_bytes(self) -> int:
        # Payload bytes + fixed 128-byte envelope header (round, id, digest, 64B sig)
        return len(self.payload_bytes) + 128


@dataclass(frozen=True)
class WorkerDecision:
    """Per-worker audit record emitted by UpdateVerifier."""

    contributor_id: str
    accepted: bool
    reason: str
    l2_norm: float
    cosine_to_median: float
    val_loss_delta: float
    is_injected_poison: bool
    poison_type: str


@dataclass
class RoundVerificationReceipt:
    """Immutable-style auditable verification summary for one outer sync round."""

    round_id: int
    val_loss_before: float
    val_loss_after: float
    accepted_workers: List[str] = field(default_factory=list)
    rejected_workers: List[str] = field(default_factory=list)
    decisions: List[WorkerDecision] = field(default_factory=list)
    aggregate_rolled_back: bool = False


class ContributorKeyring:
    """Deterministic contributor authentication registry.

    Uses HMAC-SHA256 over the canonical update transcript in pure Python stdlib.
    Production internet-scale deployments substitute asymmetric Ed25519 over the
    identical canonical transcript bytes (see SPEC.md Section 5).
    """

    def __init__(self, master_seed: int = 1350) -> None:
        self._master = f"zoo-gym-diloco-keyring-{master_seed}".encode("utf-8")
        self._keys: Dict[str, bytes] = {}

    def register(self, contributor_id: str) -> str:
        """Deterministically derive and register a contributor key, returning its pubkey ID."""
        key = hmac.new(
            self._master, contributor_id.encode("utf-8"), hashlib.sha256
        ).digest()
        self._keys[contributor_id] = key
        return hashlib.sha256(b"pub:" + key).hexdigest()[:16]

    def _canonical_transcript(
        self, round_id: int, contributor_id: str, codec: str, payload_sha256: str
    ) -> bytes:
        return f"v1|{round_id}|{contributor_id}|{codec}|{payload_sha256}".encode(
            "utf-8"
        )

    def sign(
        self,
        round_id: int,
        contributor_id: str,
        codec: str,
        payload_bytes: bytes,
        is_injected_poison: bool = False,
        poison_type: str = "none",
    ) -> SignedUpdateEnvelope:
        if contributor_id not in self._keys:
            self.register(contributor_id)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        transcript = self._canonical_transcript(round_id, contributor_id, codec, digest)
        sig = hmac.new(self._keys[contributor_id], transcript, hashlib.sha256).hexdigest()
        return SignedUpdateEnvelope(
            round_id=round_id,
            contributor_id=contributor_id,
            codec=codec,
            payload_bytes=payload_bytes,
            payload_sha256=digest,
            signature_hex=sig,
            is_injected_poison=is_injected_poison,
            poison_type=poison_type,
        )

    def verify_signature(self, env: SignedUpdateEnvelope) -> Tuple[bool, str]:
        key = self._keys.get(env.contributor_id)
        if key is None:
            return False, "unknown_contributor"
        computed_digest = hashlib.sha256(env.payload_bytes).hexdigest()
        if not hmac.compare_digest(computed_digest, env.payload_sha256):
            return False, "digest_mismatch"
        transcript = self._canonical_transcript(
            env.round_id, env.contributor_id, env.codec, computed_digest
        )
        expected_sig = hmac.new(key, transcript, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, env.signature_hex):
            return False, "bad_signature"
        return True, "ok"


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na * nb)
    if denom <= 1e-18:
        return 0.0
    return dot / denom


def _coordinate_median(vectors: Sequence[Sequence[float]]) -> List[float]:
    """Compute coordinate-wise median across candidate pseudo-gradients."""
    if not vectors:
        return []
    n = len(vectors[0])
    m = len(vectors)
    if m == 1:
        return list(vectors[0])
    out = [0.0] * n
    mid = m // 2
    for i in range(n):
        col = sorted(vec[i] for vec in vectors)
        if m % 2 == 1:
            out[i] = col[mid]
        else:
            out[i] = 0.5 * (col[mid - 1] + col[mid])
    return out


class UpdateVerifier:
    """Four-stage independent verification gate for untrusted WAN pseudo-gradients."""

    def __init__(
        self,
        keyring: ContributorKeyring,
        verifier_val: Sequence[Example],
        norm_ratio_cap: float = 3.5,
        abs_norm_cap: float = 50.0,
        min_cosine_sim: float = 0.05,
        probe_lr: float = 0.4,
        max_val_regression: float = 0.015,
        aggregate_val_tol: float = 0.025,
        probe_examples_limit: int = 240,
    ) -> None:
        self.keyring = keyring
        self.verifier_val = tuple(verifier_val)
        self.probe_subset = tuple(verifier_val[:probe_examples_limit])
        self.norm_ratio_cap = norm_ratio_cap
        self.abs_norm_cap = abs_norm_cap
        self.min_cosine_sim = min_cosine_sim
        self.probe_lr = probe_lr
        self.max_val_regression = max_val_regression
        self.aggregate_val_tol = aggregate_val_tol

    def verify_round(
        self,
        round_id: int,
        base_model: CausalCharLM,
        envelopes: Sequence[SignedUpdateEnvelope],
    ) -> Tuple[List[List[float]], RoundVerificationReceipt]:
        """Verify all arriving envelopes for `round_id`, returning accepted pseudo-gradients."""
        base_flat = base_model.get_flat_params()
        val_before = base_model.evaluate(self.probe_subset).loss
        receipt = RoundVerificationReceipt(
            round_id=round_id,
            val_loss_before=val_before,
            val_loss_after=val_before,
        )

        # Stage 1: Authenticate envelope, check round freshness, decode wire bytes.
        seen_contributors: set[str] = set()
        decoded: List[Tuple[SignedUpdateEnvelope, List[float], float]] = []
        for env in envelopes:
            if env.round_id != round_id or env.contributor_id in seen_contributors:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="replay_or_wrong_round",
                        l2_norm=0.0,
                        cosine_to_median=0.0,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
                continue
            seen_contributors.add(env.contributor_id)

            ok_sig, sig_reason = self.keyring.verify_signature(env)
            if not ok_sig:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason=sig_reason,
                        l2_norm=0.0,
                        cosine_to_median=0.0,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
                continue

            try:
                _codec, delta = decode_tensor(env.payload_bytes)
            except ValueError:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="decode_error",
                        l2_norm=0.0,
                        cosine_to_median=0.0,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
                continue

            if len(delta) != len(base_flat) or any(not math.isfinite(x) for x in delta):
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="non_finite_or_shape_mismatch",
                        l2_norm=float("nan"),
                        cosine_to_median=0.0,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
                continue

            decoded.append((env, delta, _l2_norm(delta)))

        if not decoded:
            return [], receipt

        # Stage 2: Relative & absolute L2-norm safety bound.
        norms = [item[2] for item in decoded]
        med_norm = statistics.median(norms)
        norm_cap = min(
            self.abs_norm_cap,
            max(med_norm * self.norm_ratio_cap, 1e-6),
        )
        norm_safe: List[Tuple[SignedUpdateEnvelope, List[float], float]] = []
        for env, delta, nrm in decoded:
            if nrm > norm_cap or nrm <= 1e-12:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="norm_anomaly",
                        l2_norm=nrm,
                        cosine_to_median=0.0,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
            else:
                norm_safe.append((env, delta, nrm))

        if not norm_safe:
            return [], receipt

        # Stage 3: Geometric cosine anomaly filter against peer coordinate median.
        peer_median = _coordinate_median([item[1] for item in norm_safe])
        cos_safe: List[Tuple[SignedUpdateEnvelope, List[float], float, float]] = []
        for env, delta, nrm in norm_safe:
            cos_sim = _cosine_similarity(delta, peer_median)
            if len(norm_safe) >= 3 and cos_sim < self.min_cosine_sim:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="cosine_anomaly",
                        l2_norm=nrm,
                        cosine_to_median=cos_sim,
                        val_loss_delta=0.0,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
            else:
                cos_safe.append((env, delta, nrm, cos_sim))

        # Stage 4: Independent held-out validation benefit check (never sees private data!).
        accepted_deltas: List[List[float]] = []
        probe_model = base_model.clone()
        for env, delta, nrm, cos_sim in cos_safe:
            # Pseudo-gradient convention: delta = theta_0 - theta_H
            # Candidate step: theta_probe = theta_0 - probe_lr * delta
            probe_flat = [
                p - self.probe_lr * d for p, d in zip(base_flat, delta, strict=True)
            ]
            probe_model.set_flat_params(probe_flat)
            probe_loss = probe_model.evaluate(self.probe_subset).loss
            loss_delta = probe_loss - val_before
            if loss_delta > self.max_val_regression:
                receipt.rejected_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=False,
                        reason="val_loss_regression",
                        l2_norm=nrm,
                        cosine_to_median=cos_sim,
                        val_loss_delta=loss_delta,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
            else:
                receipt.accepted_workers.append(env.contributor_id)
                receipt.decisions.append(
                    WorkerDecision(
                        contributor_id=env.contributor_id,
                        accepted=True,
                        reason="verified_beneficial",
                        l2_norm=nrm,
                        cosine_to_median=cos_sim,
                        val_loss_delta=loss_delta,
                        is_injected_poison=env.is_injected_poison,
                        poison_type=env.poison_type,
                    )
                )
                accepted_deltas.append(delta)

        return accepted_deltas, receipt

    def check_aggregate_safety(
        self,
        candidate_model: CausalCharLM,
        receipt: RoundVerificationReceipt,
    ) -> bool:
        """Verify that the aggregated outer step does not regress held-out validation loss."""
        val_after = candidate_model.evaluate(self.probe_subset).loss
        receipt.val_loss_after = val_after
        if val_after > receipt.val_loss_before + self.aggregate_val_tol:
            receipt.aggregate_rolled_back = True
            return False
        return True
