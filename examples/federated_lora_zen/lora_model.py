# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Clean-room frozen base LM with FP32 LoRA and 4-bit NF4 QLoRA adapters.

Implements:
- ``NF4BlockQuantizedMatrix``: Dettmers et al. (2023) 16-level NormalFloat 4-bit
  block-wise quantization with real packed 4-bit nibbles (2 weights/byte),
  per-block absmax scaling, exact dequantization, and reconstruction metrics.
- ``FrozenBaseLM``: Compact 3-layer causal neural language model (embedding +
  2-layer residual MLP + LM head) pretrained once on the base split and frozen.
- ``LoRALanguageModel``: Attaches trainable low-rank adapters
  ``DeltaW = (alpha / r) * (B @ A)`` to all three linear projections on top of
  either an FP32 frozen base (LoRA) or a 4-bit NF4 frozen base (QLoRA).
- ``svd_factor_delta``: Deterministic exact truncated rank-``r`` SVD via
  subspace iteration + Rayleigh-Ritz Jacobi eigendecomposition for cross-term-
  free federated adapter aggregation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass
from operator import mul
from typing import Sequence

from .corpus import LMExample

try:
    from math import sumprod as _sumprod
except ImportError:  # pragma: no cover - Python 3.10/3.11 fallback

    def _sumprod(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(map(mul, a, b)))


# Canonical 16-level NF4 (NormalFloat 4-bit) quantization codebook from
# Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs".
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

# Precomputed midpoints between adjacent NF4 codebook entries for fast lookup.
_NF4_MIDPOINTS: tuple[float, ...] = tuple(
    0.5 * (NF4_CODEBOOK[i] + NF4_CODEBOOK[i + 1]) for i in range(15)
)

LAYER_NAMES: tuple[str, ...] = ("proj_up", "proj_mid", "lm_head")


def _quantize_nf4_scalar(val: float) -> int:
    """Return nearest NF4 4-bit code index in [0, 15] for normalized val in [-1, 1]."""
    lo = 0
    hi = 15
    while lo < hi:
        mid = (lo + hi) // 2
        if val > _NF4_MIDPOINTS[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class NF4BlockQuantizedMatrix:
    """4-bit NF4 block-quantized matrix stored as packed 4-bit nibbles + scales."""

    rows: int
    cols: int
    block_size: int
    packed_nibbles: bytes
    block_scales: tuple[float, ...]
    rmse: float
    max_abs_err: float

    @property
    def num_params(self) -> int:
        return self.rows * self.cols

    @property
    def packed_nibble_bytes(self) -> int:
        return len(self.packed_nibbles)

    @property
    def scale_bytes_fp16(self) -> int:
        # 2 bytes (FP16) per block absmax scale.
        return len(self.block_scales) * 2

    @property
    def total_quantized_bytes(self) -> int:
        return self.packed_nibble_bytes + self.scale_bytes_fp16

    @property
    def fp32_bytes(self) -> int:
        return self.num_params * 4

    def dequantize(self) -> list[list[float]]:
        """Unpack 4-bit nibbles and dequantize all blocks back to FP32 matrix."""
        total = self.rows * self.cols
        codes = NF4_CODEBOOK
        scales = self.block_scales
        bsize = self.block_size
        packed = self.packed_nibbles

        flat: list[float] = [0.0] * total
        idx = 0
        for byte_val in packed:
            hi = (byte_val >> 4) & 0x0F
            lo = byte_val & 0x0F
            if idx < total:
                flat[idx] = codes[hi] * scales[idx // bsize]
                idx += 1
            if idx < total:
                flat[idx] = codes[lo] * scales[idx // bsize]
                idx += 1

        cols = self.cols
        return [flat[r * cols : (r + 1) * cols] for r in range(self.rows)]

    def sha256(self) -> str:
        """Return canonical SHA-256 digest of packed nibbles and block scales."""
        h = hashlib.sha256()
        h.update(struct.pack(">III", self.rows, self.cols, self.block_size))
        h.update(self.packed_nibbles)
        for s in self.block_scales:
            h.update(struct.pack(">d", float(s)))
        return h.hexdigest()


def quantize_matrix_nf4(
    matrix: list[list[float]], block_size: int = 64
) -> NF4BlockQuantizedMatrix:
    """Quantize a 2-D float matrix into packed 4-bit NF4 blocks."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    flat = [v for row in matrix for v in row]
    total = len(flat)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    nibbles: list[int] = [0] * total
    scales: list[float] = []
    sq_err = 0.0
    max_err = 0.0

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        block = flat[start:end]
        absmax = max((abs(v) for v in block), default=0.0)
        scale = absmax if absmax > 1e-12 else 1.0
        scales.append(scale)
        inv_scale = 1.0 / scale
        for i in range(start, end):
            v = flat[i]
            q = _quantize_nf4_scalar(v * inv_scale)
            nibbles[i] = q
            recon = NF4_CODEBOOK[q] * scale
            err = abs(v - recon)
            sq_err += err * err
            if err > max_err:
                max_err = err

    # Pack two 4-bit nibbles per byte (high nibble = even index, low = odd).
    packed = bytearray((total + 1) // 2)
    for i in range(0, total, 2):
        hi = nibbles[i] & 0x0F
        lo = nibbles[i + 1] & 0x0F if (i + 1) < total else 0
        packed[i // 2] = (hi << 4) | lo

    rmse = math.sqrt(sq_err / max(1, total))
    return NF4BlockQuantizedMatrix(
        rows=rows,
        cols=cols,
        block_size=block_size,
        packed_nibbles=bytes(packed),
        block_scales=tuple(scales),
        rmse=rmse,
        max_abs_err=max_err,
    )


def transpose_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Return the transpose of a 2-D float matrix."""
    if not mat:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply 2-D matrices ``a (m x k)`` and ``b (k x n)``."""
    b_t = transpose_matrix(b)
    return [[_sumprod(row, col) for col in b_t] for row in a]


def matrix_frobenius_norm(mat: list[list[float]]) -> float:
    """Compute Frobenius norm ``sqrt(sum_ij mat[i][j]^2)``."""
    return math.sqrt(sum(_sumprod(row, row) for row in mat))


def matrix_frobenius_diff(
    a: list[list[float]], b: list[list[float]]
) -> float:
    """Compute Frobenius norm ``||a - b||_F``."""
    acc = 0.0
    for ra, rb in zip(a, b, strict=True):
        for va, vb in zip(ra, rb, strict=True):
            d = va - vb
            acc += d * d
    return math.sqrt(acc)


def _jacobi_symmetric_eig(
    mat: list[list[float]], max_sweeps: int = 80, tol: float = 1e-13
) -> tuple[list[float], list[list[float]]]:
    """Deterministic cyclic Jacobi eigendecomposition for a small symmetric matrix.

    Returns ``(eigenvalues_desc, eigenvectors_cols)`` where ``eigenvectors_cols[j]``
    is the unit eigenvector corresponding to ``eigenvalues_desc[j]``.
    """
    n = len(mat)
    a = [row[:] for row in mat]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        max_off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) > max_off:
                    max_off = abs(apq)
                if abs(apq) <= tol:
                    continue
                app = a[p][p]
                aqq = a[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.hypot(1.0, tau))
                else:
                    t = -1.0 / (-tau + math.hypot(1.0, tau))
                c = 1.0 / math.hypot(1.0, t)
                s = t * c

                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0

                for r in range(n):
                    if r != p and r != q:
                        arp = a[r][p]
                        arq = a[r][q]
                        new_rp = c * arp - s * arq
                        new_rq = s * arp + c * arq
                        a[r][p] = new_rp
                        a[p][r] = new_rp
                        a[r][q] = new_rq
                        a[q][r] = new_rq

                for r in range(n):
                    vrp = v[r][p]
                    vrq = v[r][q]
                    v[r][p] = c * vrp - s * vrq
                    v[r][q] = s * vrp + c * vrq
        if max_off <= tol:
            break

    pairs = []
    for j in range(n):
        col = [v[i][j] for i in range(n)]
        # Canonical sign: largest-magnitude entry is positive.
        max_idx = max(range(n), key=lambda idx: abs(col[idx]))
        if col[max_idx] < 0.0:
            col = [-x for x in col]
        pairs.append((a[j][j], col))

    pairs.sort(key=lambda item: -item[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _orthonormalize_columns(cols: list[list[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt orthonormalization dropping near-zero columns."""
    basis: list[list[float]] = []
    for raw in cols:
        vec = raw[:]
        # Two passes of MGS for numerical stability.
        for _ in range(2):
            for b in basis:
                dot = _sumprod(vec, b)
                vec = [v - dot * bi for v, bi in zip(vec, b, strict=True)]
        norm = math.sqrt(_sumprod(vec, vec))
        if norm > 1e-12:
            inv = 1.0 / norm
            basis.append([v * inv for v in vec])
    return basis


def svd_factor_delta(
    delta_w: list[list[float]], rank: int
) -> tuple[list[list[float]], list[list[float]], float]:
    """Deterministic exact truncated rank-``r`` SVD factoring ``DeltaW ≈ B @ A``.

    Uses exact Gram-matrix subspace iteration with Rayleigh-Ritz Jacobi
    eigendecomposition so that ``B @ A`` is the optimal Frobenius rank-``r``
    approximation to ``delta_w`` with balanced singular values:
    ``B = U_r @ Sigma_r^(1/2)``, ``A = Sigma_r^(1/2) @ V_r^T``.

    Returns:
        ``(B, A, truncation_residual_frobenius)``.
    """
    m = len(delta_w)
    n = len(delta_w[0])
    r = min(rank, m, n)

    if matrix_frobenius_norm(delta_w) <= 1e-15:
        zero_b = [[0.0] * r for _ in range(m)]
        zero_a = [[0.0] * n for _ in range(r)]
        return zero_b, zero_a, 0.0

    # Form symmetric Gram matrix G = DeltaW^T @ DeltaW (n x n).
    dw_t = transpose_matrix(delta_w)
    gram = [[_sumprod(dw_t[i], dw_t[j]) for j in range(n)] for i in range(n)]

    # Build deterministic initial subspace covering top-energy columns + basis.
    # For K=3 workers of rank r <= 8, DeltaW has rank <= 3*r <= 24, so p <= 28
    # captures the full exact nonzero range of DeltaW.
    subspace_dim = min(n, max(2 * r + 8, 24))
    col_norms = [(_sumprod(dw_t[j], dw_t[j]), j) for j in range(n)]
    col_norms.sort(key=lambda item: (-item[0], item[1]))

    init_cols: list[list[float]] = []
    # Seed with top-energy rows of DeltaW (which live directly in row space).
    row_norms = [(_sumprod(delta_w[i], delta_w[i]), i) for i in range(m)]
    row_norms.sort(key=lambda item: (-item[0], item[1]))
    for norm_sq, row_idx in row_norms[:subspace_dim]:
        if norm_sq > 1e-18:
            init_cols.append(delta_w[row_idx][:])

    # Pad with canonical basis vectors in descending column energy order.
    for _, col_idx in col_norms:
        if len(init_cols) >= subspace_dim:
            break
        e_vec = [0.0] * n
        e_vec[col_idx] = 1.0
        init_cols.append(e_vec)

    q_cols = _orthonormalize_columns(init_cols)[:subspace_dim]
    if not q_cols:
        zero_b = [[0.0] * r for _ in range(m)]
        zero_a = [[0.0] * n for _ in range(r)]
        return zero_b, zero_a, matrix_frobenius_norm(delta_w)

    # Subspace iteration on G: 12 iterations gives machine precision on low-rank G.
    for _ in range(12):
        y_cols = [[_sumprod(gram[i], q) for i in range(n)] for q in q_cols]
        q_next = _orthonormalize_columns(y_cols)
        if not q_next:
            break
        q_cols = q_next

    p = len(q_cols)
    # Rayleigh-Ritz projection: T = Q^T G Q (p x p symmetric).
    gq_cols = [[_sumprod(gram[i], q) for i in range(n)] for q in q_cols]
    t_mat = [[_sumprod(q_cols[i], gq_cols[j]) for j in range(p)] for i in range(p)]
    # Symmetrize explicitly against floating-point rounding.
    for i in range(p):
        for j in range(i + 1, p):
            avg = 0.5 * (t_mat[i][j] + t_mat[j][i])
            t_mat[i][j] = avg
            t_mat[j][i] = avg

    eigvals, eigvecs = _jacobi_symmetric_eig(t_mat)

    b_mat: list[list[float]] = [[0.0] * r for _ in range(m)]
    a_mat: list[list[float]] = [[0.0] * n for _ in range(r)]

    for k in range(r):
        if k >= len(eigvals) or eigvals[k] <= 1e-16:
            continue
        sigma = math.sqrt(eigvals[k])
        # Right singular vector v_k = Q @ eigvecs[k].
        coeffs = eigvecs[k]
        v_k = [
            sum(coeffs[j] * q_cols[j][i] for j in range(p)) for i in range(n)
        ]
        v_norm = math.sqrt(_sumprod(v_k, v_k))
        if v_norm <= 1e-14:
            continue
        v_k = [x / v_norm for x in v_k]
        # Canonicalize sign of v_k deterministically.
        max_idx = max(range(n), key=lambda idx: abs(v_k[idx]))
        if v_k[max_idx] < 0.0:
            v_k = [-x for x in v_k]

        # Left singular vector u_k = (DeltaW @ v_k) / sigma.
        u_k = [_sumprod(delta_w[i], v_k) / sigma for i in range(m)]
        sqrt_sigma = math.sqrt(sigma)
        for i in range(m):
            b_mat[i][k] = u_k[i] * sqrt_sigma
        for j in range(n):
            a_mat[k][j] = sqrt_sigma * v_k[j]

    recon = matmul(b_mat, a_mat)
    residual = matrix_frobenius_diff(delta_w, recon)
    return b_mat, a_mat, residual


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the compact Zen-style causal base LM."""

    vocab_size: int = 80
    context_len: int = 4
    embed_dim: int = 16
    hidden_dim: int = 64
    nf4_block_size: int = 64

    @property
    def input_dim(self) -> int:
        return self.context_len * self.embed_dim


@dataclass(frozen=True)
class MemoryFootprint:
    """Exact byte-level parameter and memory accounting for LoRA vs QLoRA."""

    quantization_mode: str
    lora_rank: int
    base_total_params: int
    base_linear_params: int
    base_non_linear_params: int
    trainable_lora_params: int
    trainable_param_ratio: float
    base_fp32_bytes: int
    base_fp16_bytes: int
    base_active_bytes: int
    lora_fp32_bytes: int
    total_model_bytes: int
    base_memory_savings_vs_fp32: float
    base_memory_savings_vs_fp16: float
    adapter_wire_bytes_fp32: int
    adapter_wire_bytes_fp16: int
    nf4_avg_rmse: float
    nf4_max_abs_err: float


class FrozenBaseLM:
    """Pretrained frozen base LM with optional 4-bit NF4 linear weight storage."""

    def __init__(
        self,
        config: ModelConfig,
        embeddings: list[list[float]],
        weights: dict[str, list[list[float]]],
        biases: dict[str, list[float]],
        quantization_mode: str = "fp32",
    ) -> None:
        if quantization_mode not in ("fp32", "nf4"):
            raise ValueError(
                f"Unsupported quantization_mode={quantization_mode!r}"
            )
        self.config = config
        self.quantization_mode = quantization_mode
        self.embeddings = [row[:] for row in embeddings]
        self.biases = {k: v[:] for k, v in biases.items()}
        self._fp32_weights = {
            k: [row[:] for row in weights[k]] for k in LAYER_NAMES
        }
        self.nf4_weights: dict[str, NF4BlockQuantizedMatrix] = {}
        self.active_weights: dict[str, list[list[float]]] = {}
        self.active_weights_t: dict[str, list[list[float]]] = {}

        if quantization_mode == "nf4":
            for name in LAYER_NAMES:
                q_mat = quantize_matrix_nf4(
                    self._fp32_weights[name], block_size=config.nf4_block_size
                )
                self.nf4_weights[name] = q_mat
                deq = q_mat.dequantize()
                self.active_weights[name] = deq
                self.active_weights_t[name] = transpose_matrix(deq)
        else:
            for name in LAYER_NAMES:
                mat = [row[:] for row in self._fp32_weights[name]]
                self.active_weights[name] = mat
                self.active_weights_t[name] = transpose_matrix(mat)

        self.frozen_sha256 = self.compute_base_sha256()

    @classmethod
    def create_initialized(
        cls, config: ModelConfig, seed: int
    ) -> FrozenBaseLM:
        """Create a deterministically Xavier-initialized base LM (FP32)."""
        rng = random.Random(seed)
        d_in = config.input_dim
        d_hid = config.hidden_dim
        v = config.vocab_size

        def init_mat(rows: int, cols: int, scale: float) -> list[list[float]]:
            return [
                [rng.gauss(0.0, scale) for _ in range(cols)]
                for _ in range(rows)
            ]

        embeddings = init_mat(v, config.embed_dim, 0.25)
        weights = {
            "proj_up": init_mat(d_hid, d_in, math.sqrt(1.0 / d_in)),
            "proj_mid": init_mat(d_hid, d_hid, math.sqrt(0.5 / d_hid)),
            "lm_head": init_mat(v, d_hid, math.sqrt(1.0 / d_hid)),
        }
        biases = {
            "proj_up": [0.0] * d_hid,
            "proj_mid": [0.0] * d_hid,
            "lm_head": [0.0] * v,
        }
        return cls(
            config=config,
            embeddings=embeddings,
            weights=weights,
            biases=biases,
            quantization_mode="fp32",
        )

    def to_quantization_mode(self, mode: str) -> FrozenBaseLM:
        """Return a fresh frozen copy in ``'fp32'`` or ``'nf4'`` mode."""
        return FrozenBaseLM(
            config=self.config,
            embeddings=self.embeddings,
            weights=self._fp32_weights,
            biases=self.biases,
            quantization_mode=mode,
        )

    def compute_base_sha256(self) -> str:
        """Compute canonical SHA-256 over all frozen base parameters."""
        h = hashlib.sha256()
        h.update(self.quantization_mode.encode("utf-8"))
        for row in self.embeddings:
            for v in row:
                h.update(struct.pack(">d", float(v)))
        for name in LAYER_NAMES:
            h.update(name.encode("utf-8"))
            if self.quantization_mode == "nf4":
                h.update(self.nf4_weights[name].sha256().encode("utf-8"))
            else:
                for row in self.active_weights[name]:
                    for v in row:
                        h.update(struct.pack(">d", float(v)))
            for b in self.biases[name]:
                h.update(struct.pack(">d", float(b)))
        return h.hexdigest()

    def verify_frozen(self) -> bool:
        """Verify frozen base weights have not mutated since construction."""
        return self.compute_base_sha256() == self.frozen_sha256

    def pretrain_on_split(
        self,
        examples: Sequence[LMExample],
        steps: int = 80,
        batch_size: int = 32,
        lr: float = 0.08,
        seed: int = 1350,
    ) -> dict[str, float]:
        """Pretrain the FP32 base weights once before freezing for LoRA/QLoRA."""
        if self.quantization_mode != "fp32":
            raise RuntimeError("Base pretraining must run in FP32 before freezing")
        rng = random.Random(seed)
        n = len(examples)
        d_emb = self.config.embed_dim
        d_hid = self.config.hidden_dim
        v_size = self.config.vocab_size

        w1 = self._fp32_weights["proj_up"]
        w2 = self._fp32_weights["proj_mid"]
        w3 = self._fp32_weights["lm_head"]
        b1 = self.biases["proj_up"]
        b2 = self.biases["proj_mid"]
        b3 = self.biases["lm_head"]
        emb = self.embeddings

        initial_loss = 0.0
        final_loss = 0.0

        for step in range(steps):
            batch = [examples[rng.randrange(n)] for _ in range(batch_size)]
            gw1 = [[0.0] * len(w1[0]) for _ in range(d_hid)]
            gw2 = [[0.0] * d_hid for _ in range(d_hid)]
            gw3 = [[0.0] * d_hid for _ in range(v_size)]
            gb1 = [0.0] * d_hid
            gb2 = [0.0] * d_hid
            gb3 = [0.0] * v_size
            gemb: dict[int, list[float]] = {}

            w3_t = transpose_matrix(w3)
            w2_t = transpose_matrix(w2)
            w1_t = transpose_matrix(w1)

            loss_sum = 0.0
            for ex in batch:
                x0: list[float] = []
                for tok in ex.context:
                    x0.extend(emb[tok])

                h1 = [
                    math.tanh(_sumprod(w1[i], x0) + b1[i])
                    for i in range(d_hid)
                ]
                a2 = [
                    math.tanh(_sumprod(w2[i], h1) + b2[i])
                    for i in range(d_hid)
                ]
                h2 = [h1[i] + a2[i] for i in range(d_hid)]
                logits = [
                    _sumprod(w3[i], h2) + b3[i] for i in range(v_size)
                ]

                max_l = max(logits)
                exps = [math.exp(l_v - max_l) for l_v in logits]
                inv_z = 1.0 / sum(exps)
                probs = [e * inv_z for e in exps]
                loss_sum += -math.log(max(probs[ex.target], 1e-12))

                dz3 = probs
                dz3[ex.target] -= 1.0
                for i in range(v_size):
                    g = dz3[i]
                    gb3[i] += g
                    row_gw3 = gw3[i]
                    for j in range(d_hid):
                        row_gw3[j] += g * h2[j]

                dh2 = [_sumprod(w3_t[j], dz3) for j in range(d_hid)]
                dz2 = [
                    dh2[i] * (1.0 - a2[i] * a2[i]) for i in range(d_hid)
                ]
                for i in range(d_hid):
                    g = dz2[i]
                    gb2[i] += g
                    row_gw2 = gw2[i]
                    for j in range(d_hid):
                        row_gw2[j] += g * h1[j]

                dh1 = [
                    dh2[j] + _sumprod(w2_t[j], dz2) for j in range(d_hid)
                ]
                dz1 = [
                    dh1[i] * (1.0 - h1[i] * h1[i]) for i in range(d_hid)
                ]
                for i in range(d_hid):
                    g = dz1[i]
                    gb1[i] += g
                    row_gw1 = gw1[i]
                    for j in range(len(x0)):
                        row_gw1[j] += g * x0[j]

                dx0 = [_sumprod(w1_t[j], dz1) for j in range(len(x0))]
                for pos, tok in enumerate(ex.context):
                    acc = gemb.get(tok)
                    if acc is None:
                        acc = [0.0] * d_emb
                        gemb[tok] = acc
                    base_off = pos * d_emb
                    for k in range(d_emb):
                        acc[k] += dx0[base_off + k]

            step_loss = loss_sum / batch_size
            if step == 0:
                initial_loss = step_loss
            final_loss = step_loss

            scale = lr / batch_size
            for i in range(v_size):
                b3[i] -= scale * gb3[i]
                for j in range(d_hid):
                    w3[i][j] -= scale * gw3[i][j]
            for i in range(d_hid):
                b2[i] -= scale * gb2[i]
                b1[i] -= scale * gb1[i]
                for j in range(d_hid):
                    w2[i][j] -= scale * gw2[i][j]
                for j in range(len(w1[0])):
                    w1[i][j] -= scale * gw1[i][j]
            for tok, ge in gemb.items():
                row_e = emb[tok]
                for k in range(d_emb):
                    row_e[k] -= scale * ge[k]

        # Refresh active weights & freeze hash after pretraining completes.
        for name in LAYER_NAMES:
            self.active_weights[name] = [
                row[:] for row in self._fp32_weights[name]
            ]
            self.active_weights_t[name] = transpose_matrix(
                self.active_weights[name]
            )
        self.frozen_sha256 = self.compute_base_sha256()
        return {
            "pretrain_steps": float(steps),
            "pretrain_initial_loss": initial_loss,
            "pretrain_final_loss": final_loss,
        }


@dataclass
class AdapterState:
    """Serializable LoRA adapter weights ``(A, B)`` across all adapted layers."""

    rank: int
    alpha: float
    a_mats: dict[str, list[list[float]]]
    b_mats: dict[str, list[list[float]]]

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    def clone(self) -> AdapterState:
        return AdapterState(
            rank=self.rank,
            alpha=self.alpha,
            a_mats={
                k: [row[:] for row in self.a_mats[k]] for k in LAYER_NAMES
            },
            b_mats={
                k: [row[:] for row in self.b_mats[k]] for k in LAYER_NAMES
            },
        )

    def num_params(self) -> int:
        total = 0
        for name in LAYER_NAMES:
            a = self.a_mats[name]
            b = self.b_mats[name]
            total += len(a) * len(a[0]) + len(b) * len(b[0])
        return total

    def wire_bytes_fp32(self) -> int:
        return self.num_params() * 4

    def wire_bytes_fp16(self) -> int:
        return self.num_params() * 2

    def to_canonical_bytes(self) -> bytes:
        """Deterministic binary encoding of adapter metadata and weights."""
        buf = bytearray()
        buf.extend(struct.pack(">Id", self.rank, float(self.alpha)))
        for name in LAYER_NAMES:
            buf.extend(name.encode("utf-8"))
            a = self.a_mats[name]
            b = self.b_mats[name]
            buf.extend(
                struct.pack(">IIII", len(a), len(a[0]), len(b), len(b[0]))
            )
            for row in a:
                for v in row:
                    buf.extend(struct.pack(">d", float(v)))
            for row in b:
                for v in row:
                    buf.extend(struct.pack(">d", float(v)))
        return bytes(buf)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "a_mats": {k: self.a_mats[k] for k in LAYER_NAMES},
            "b_mats": {k: self.b_mats[k] for k in LAYER_NAMES},
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> AdapterState:
        raw_a = payload["a_mats"]
        raw_b = payload["b_mats"]
        if not isinstance(raw_a, dict) or not isinstance(raw_b, dict):
            raise ValueError("Invalid adapter JSON payload")
        return cls(
            rank=int(payload["rank"]),  # type: ignore[arg-type]
            alpha=float(payload["alpha"]),  # type: ignore[arg-type]
            a_mats={
                k: [[float(v) for v in row] for row in raw_a[k]]
                for k in LAYER_NAMES
            },
            b_mats={
                k: [[float(v) for v in row] for row in raw_b[k]]
                for k in LAYER_NAMES
            },
        )

    def effective_deltas(self) -> dict[str, list[list[float]]]:
        """Compute ``DeltaW_l = (alpha / r) * (B_l @ A_l)`` for every layer."""
        s = self.scale
        out: dict[str, list[list[float]]] = {}
        for name in LAYER_NAMES:
            ba = matmul(self.b_mats[name], self.a_mats[name])
            out[name] = [[v * s for v in row] for row in ba]
        return out


def init_lora_adapter(
    config: ModelConfig, rank: int, alpha: float, seed: int
) -> AdapterState:
    """Initialize Hu et al. (2021) LoRA adapters: ``A ~ N(0, 1/d_in)``, ``B = 0``."""
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    rng = random.Random(seed)
    shapes = {
        "proj_up": (config.hidden_dim, config.input_dim),
        "proj_mid": (config.hidden_dim, config.hidden_dim),
        "lm_head": (config.vocab_size, config.hidden_dim),
    }
    a_mats: dict[str, list[list[float]]] = {}
    b_mats: dict[str, list[list[float]]] = {}
    for name in LAYER_NAMES:
        d_out, d_in = shapes[name]
        std = math.sqrt(1.0 / d_in)
        a_mats[name] = [
            [rng.gauss(0.0, std) for _ in range(d_in)] for _ in range(rank)
        ]
        b_mats[name] = [[0.0] * rank for _ in range(d_out)]
    return AdapterState(rank=rank, alpha=alpha, a_mats=a_mats, b_mats=b_mats)


class LoRALanguageModel:
    """Frozen base LM (FP32 or NF4) paired with trainable FP32 LoRA adapters."""

    def __init__(
        self, base: FrozenBaseLM, adapter: AdapterState
    ) -> None:
        self.base = base
        self.adapter = adapter.clone()
        self._adam_step = 0
        self._reset_optimizer()

    def _reset_optimizer(self) -> None:
        self._adam_step = 0
        self._m_a = {
            k: [[0.0] * len(self.adapter.a_mats[k][0]) for _ in self.adapter.a_mats[k]]
            for k in LAYER_NAMES
        }
        self._v_a = {
            k: [[0.0] * len(self.adapter.a_mats[k][0]) for _ in self.adapter.a_mats[k]]
            for k in LAYER_NAMES
        }
        self._m_b = {
            k: [[0.0] * len(self.adapter.b_mats[k][0]) for _ in self.adapter.b_mats[k]]
            for k in LAYER_NAMES
        }
        self._v_b = {
            k: [[0.0] * len(self.adapter.b_mats[k][0]) for _ in self.adapter.b_mats[k]]
            for k in LAYER_NAMES
        }

    def load_adapter(
        self, adapter: AdapterState, reset_optimizer: bool = True
    ) -> None:
        self.adapter = adapter.clone()
        if reset_optimizer:
            self._reset_optimizer()

    def export_adapter(self) -> AdapterState:
        return self.adapter.clone()

    def memory_footprint(self) -> MemoryFootprint:
        """Compute exact parameter and byte-level footprint for LoRA vs QLoRA."""
        cfg = self.base.config
        emb_params = cfg.vocab_size * cfg.embed_dim
        bias_params = cfg.hidden_dim + cfg.hidden_dim + cfg.vocab_size
        non_lin = emb_params + bias_params

        lin_params = 0
        nf4_bytes = 0
        rmse_list: list[float] = []
        max_err = 0.0
        for name in LAYER_NAMES:
            w = self.base.active_weights[name]
            lin_params += len(w) * len(w[0])
            if self.base.quantization_mode == "nf4":
                qm = self.base.nf4_weights[name]
                nf4_bytes += qm.total_quantized_bytes
                rmse_list.append(qm.rmse)
                if qm.max_abs_err > max_err:
                    max_err = qm.max_abs_err

        base_total = lin_params + non_lin
        lora_params = self.adapter.num_params()

        base_fp32_bytes = base_total * 4
        base_fp16_bytes = base_total * 2
        if self.base.quantization_mode == "nf4":
            # Linear weights in 4-bit NF4 + FP16 block scales; embeddings/biases FP16.
            base_active_bytes = nf4_bytes + non_lin * 2
        else:
            base_active_bytes = base_fp32_bytes

        lora_fp32_bytes = lora_params * 4
        return MemoryFootprint(
            quantization_mode=self.base.quantization_mode,
            lora_rank=self.adapter.rank,
            base_total_params=base_total,
            base_linear_params=lin_params,
            base_non_linear_params=non_lin,
            trainable_lora_params=lora_params,
            trainable_param_ratio=lora_params / float(base_total),
            base_fp32_bytes=base_fp32_bytes,
            base_fp16_bytes=base_fp16_bytes,
            base_active_bytes=base_active_bytes,
            lora_fp32_bytes=lora_fp32_bytes,
            total_model_bytes=base_active_bytes + lora_fp32_bytes,
            base_memory_savings_vs_fp32=base_fp32_bytes / float(base_active_bytes),
            base_memory_savings_vs_fp16=base_fp16_bytes / float(base_active_bytes),
            adapter_wire_bytes_fp32=self.adapter.wire_bytes_fp32(),
            adapter_wire_bytes_fp16=self.adapter.wire_bytes_fp16(),
            nf4_avg_rmse=(sum(rmse_list) / len(rmse_list)) if rmse_list else 0.0,
            nf4_max_abs_err=max_err,
        )

    def _effective_weights(
        self,
    ) -> tuple[
        dict[str, list[list[float]]],
        dict[str, list[list[float]]],
    ]:
        """Compute current effective matrices ``W_eff = W_base + s * B @ A``."""
        s = self.adapter.scale
        eff: dict[str, list[list[float]]] = {}
        eff_t: dict[str, list[list[float]]] = {}
        for name in LAYER_NAMES:
            w0 = self.base.active_weights[name]
            ba = matmul(self.adapter.b_mats[name], self.adapter.a_mats[name])
            w_eff = [
                [w0[i][j] + s * ba[i][j] for j in range(len(w0[0]))]
                for i in range(len(w0))
            ]
            eff[name] = w_eff
            eff_t[name] = transpose_matrix(w_eff)
        return eff, eff_t

    def compute_batch_loss_and_grads(
        self, batch: Sequence[LMExample]
    ) -> tuple[
        float,
        dict[str, list[list[float]]],
        dict[str, list[list[float]]],
    ]:
        """Exact forward + reverse-mode backprop for LoRA ``(A, B)`` only."""
        cfg = self.base.config
        d_hid = cfg.hidden_dim
        v_size = cfg.vocab_size
        rank = self.adapter.rank
        scale = self.adapter.scale

        emb = self.base.embeddings
        b1 = self.base.biases["proj_up"]
        b2 = self.base.biases["proj_mid"]
        b3 = self.base.biases["lm_head"]

        w1 = self.base.active_weights["proj_up"]
        w2 = self.base.active_weights["proj_mid"]
        w3 = self.base.active_weights["lm_head"]

        a1 = self.adapter.a_mats["proj_up"]
        a2_m = self.adapter.a_mats["proj_mid"]
        a3 = self.adapter.a_mats["lm_head"]

        b1_m = self.adapter.b_mats["proj_up"]
        b2_m = self.adapter.b_mats["proj_mid"]
        b3_m = self.adapter.b_mats["lm_head"]

        b3_t = transpose_matrix(b3_m)
        b2_t = transpose_matrix(b2_m)
        b1_t = transpose_matrix(b1_m)
        a3_t = transpose_matrix(a3)
        a2_t = transpose_matrix(a2_m)
        w3_t = self.base.active_weights_t["lm_head"]
        w2_t = self.base.active_weights_t["proj_mid"]

        ga = {
            k: [[0.0] * len(self.adapter.a_mats[k][0]) for _ in range(rank)]
            for k in LAYER_NAMES
        }
        gb = {
            k: [[0.0] * rank for _ in range(len(self.adapter.b_mats[k]))]
            for k in LAYER_NAMES
        }

        ga1, ga2, ga3 = ga["proj_up"], ga["proj_mid"], ga["lm_head"]
        gb1, gb2, gb3 = gb["proj_up"], gb["proj_mid"], gb["lm_head"]

        loss_sum = 0.0
        for ex in batch:
            x0: list[float] = []
            for tok in ex.context:
                x0.extend(emb[tok])

            # Layer 1: proj_up
            u1 = [_sumprod(a1[k], x0) for k in range(rank)]
            h1 = [
                math.tanh(
                    _sumprod(w1[i], x0)
                    + scale * _sumprod(b1_m[i], u1)
                    + b1[i]
                )
                for i in range(d_hid)
            ]

            # Layer 2: proj_mid (residual)
            u2 = [_sumprod(a2_m[k], h1) for k in range(rank)]
            act2 = [
                math.tanh(
                    _sumprod(w2[i], h1)
                    + scale * _sumprod(b2_m[i], u2)
                    + b2[i]
                )
                for i in range(d_hid)
            ]
            h2 = [h1[i] + act2[i] for i in range(d_hid)]

            # Layer 3: lm_head
            u3 = [_sumprod(a3[k], h2) for k in range(rank)]
            logits = [
                _sumprod(w3[i], h2)
                + scale * _sumprod(b3_m[i], u3)
                + b3[i]
                for i in range(v_size)
            ]

            max_l = max(logits)
            exps = [math.exp(val - max_l) for val in logits]
            inv_z = 1.0 / sum(exps)
            probs = [e * inv_z for e in exps]
            loss_sum += -math.log(max(probs[ex.target], 1e-12))

            # Backward Layer 3
            dz3 = probs
            dz3[ex.target] -= 1.0
            for i in range(v_size):
                g = scale * dz3[i]
                row_gb3 = gb3[i]
                for k in range(rank):
                    row_gb3[k] += g * u3[k]
            du3 = [scale * _sumprod(b3_t[k], dz3) for k in range(rank)]
            for k in range(rank):
                g = du3[k]
                row_ga3 = ga3[k]
                for j in range(d_hid):
                    row_ga3[j] += g * h2[j]

            dh2 = [
                _sumprod(w3_t[j], dz3) + _sumprod(a3_t[j], du3)
                for j in range(d_hid)
            ]

            # Backward Layer 2
            dz2 = [dh2[i] * (1.0 - act2[i] * act2[i]) for i in range(d_hid)]
            for i in range(d_hid):
                g = scale * dz2[i]
                row_gb2 = gb2[i]
                for k in range(rank):
                    row_gb2[k] += g * u2[k]
            du2 = [scale * _sumprod(b2_t[k], dz2) for k in range(rank)]
            for k in range(rank):
                g = du2[k]
                row_ga2 = ga2[k]
                for j in range(d_hid):
                    row_ga2[j] += g * h1[j]

            dh1 = [
                dh2[j] + _sumprod(w2_t[j], dz2) + _sumprod(a2_t[j], du2)
                for j in range(d_hid)
            ]

            # Backward Layer 1
            dz1 = [dh1[i] * (1.0 - h1[i] * h1[i]) for i in range(d_hid)]
            for i in range(d_hid):
                g = scale * dz1[i]
                row_gb1 = gb1[i]
                for k in range(rank):
                    row_gb1[k] += g * u1[k]
            du1 = [scale * _sumprod(b1_t[k], dz1) for k in range(rank)]
            for k in range(rank):
                g = du1[k]
                row_ga1 = ga1[k]
                for j in range(len(x0)):
                    row_ga1[j] += g * x0[j]

        inv_bs = 1.0 / max(1, len(batch))
        for name in LAYER_NAMES:
            for row in ga[name]:
                for j in range(len(row)):
                    row[j] *= inv_bs
            for row in gb[name]:
                for j in range(len(row)):
                    row[j] *= inv_bs

        return loss_sum * inv_bs, ga, gb

    def train_step_adamw(
        self,
        batch: Sequence[LMExample],
        lr: float = 0.015,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 2.0,
    ) -> float:
        """Perform one AdamW optimization step on trainable LoRA ``(A, B)`` only."""
        loss, ga, gb = self.compute_batch_loss_and_grads(batch)

        # Global gradient norm clipping over LoRA parameters.
        sq_norm = 0.0
        for name in LAYER_NAMES:
            for row in ga[name]:
                sq_norm += _sumprod(row, row)
            for row in gb[name]:
                sq_norm += _sumprod(row, row)
        gnorm = math.sqrt(sq_norm)
        clip_scale = (grad_clip / gnorm) if gnorm > grad_clip else 1.0

        self._adam_step += 1
        bc1 = 1.0 - (beta1 ** self._adam_step)
        bc2 = 1.0 - (beta2 ** self._adam_step)

        for name in LAYER_NAMES:
            for mat, g_mat, m_mat, v_mat in (
                (
                    self.adapter.a_mats[name],
                    ga[name],
                    self._m_a[name],
                    self._v_a[name],
                ),
                (
                    self.adapter.b_mats[name],
                    gb[name],
                    self._m_b[name],
                    self._v_b[name],
                ),
            ):
                for i in range(len(mat)):
                    row_w = mat[i]
                    row_g = g_mat[i]
                    row_m = m_mat[i]
                    row_v = v_mat[i]
                    for j in range(len(row_w)):
                        g = row_g[j] * clip_scale
                        m_val = beta1 * row_m[j] + (1.0 - beta1) * g
                        v_val = beta2 * row_v[j] + (1.0 - beta2) * g * g
                        row_m[j] = m_val
                        row_v[j] = v_val
                        m_hat = m_val / bc1
                        v_hat = v_val / bc2
                        row_w[j] = row_w[j] * (1.0 - lr * weight_decay) - (
                            lr * m_hat / (math.sqrt(v_hat) + eps)
                        )
        return loss

    def evaluate(
        self, examples: Sequence[LMExample]
    ) -> dict[str, float]:
        """Evaluate cross-entropy loss, perplexity, and top-1 next-token accuracy."""
        if not examples:
            return {"loss": 0.0, "perplexity": 1.0, "accuracy": 0.0}

        eff, _ = self._effective_weights()
        w1 = eff["proj_up"]
        w2 = eff["proj_mid"]
        w3 = eff["lm_head"]
        b1 = self.base.biases["proj_up"]
        b2 = self.base.biases["proj_mid"]
        b3 = self.base.biases["lm_head"]
        emb = self.base.embeddings
        d_hid = self.base.config.hidden_dim
        v_size = self.base.config.vocab_size
        tanh = math.tanh
        exp = math.exp
        log = math.log

        loss_sum = 0.0
        correct = 0
        for ex in examples:
            t0, t1, t2, t3 = ex.context
            x0 = emb[t0] + emb[t1] + emb[t2] + emb[t3]
            h1 = [tanh(_sumprod(w1[i], x0) + b1[i]) for i in range(d_hid)]
            h2 = [
                h1[i] + tanh(_sumprod(w2[i], h1) + b2[i])
                for i in range(d_hid)
            ]
            logits = [_sumprod(w3[i], h2) + b3[i] for i in range(v_size)]

            max_l = max(logits)
            exps = [exp(val - max_l) for val in logits]
            z = sum(exps)
            p_tgt = exps[ex.target] / z
            loss_sum += -log(p_tgt if p_tgt > 1e-12 else 1e-12)

            if logits[ex.target] == max_l and logits.index(max_l) == ex.target:
                correct += 1

        avg_loss = loss_sum / len(examples)
        return {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20.0)),
            "accuracy": correct / len(examples),
        }

    def evaluate_by_domain(
        self,
        eval_all: Sequence[LMExample],
        eval_by_domain: dict[str, Sequence[LMExample]],
    ) -> dict[str, object]:
        """Evaluate overall and per-domain held-out test metrics in a single pass."""
        per_domain: dict[str, dict[str, float]] = {}
        total_n = 0
        loss_weighted = 0.0
        acc_weighted = 0.0
        for dom, exs in eval_by_domain.items():
            metrics = self.evaluate(exs)
            per_domain[dom] = metrics
            n_dom = len(exs)
            total_n += n_dom
            loss_weighted += metrics["loss"] * n_dom
            acc_weighted += metrics["accuracy"] * n_dom

        if total_n > 0 and total_n == len(eval_all):
            avg_loss = loss_weighted / total_n
            overall = {
                "loss": avg_loss,
                "perplexity": math.exp(min(avg_loss, 20.0)),
                "accuracy": acc_weighted / total_n,
            }
        else:
            overall = self.evaluate(eval_all)

        return {
            "overall": overall,
            "per_domain": per_domain,
            "base_frozen_verified": self.base.verify_frozen(),
            "adapter_sha256": self.adapter.sha256(),
        }


def save_adapter_json(path: str, adapter: AdapterState) -> int:
    """Persist adapter JSON to disk and return exact byte count written."""
    data = json.dumps(
        adapter.to_json_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
