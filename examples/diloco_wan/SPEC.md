<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors. -->

# HIP-1350 Protocol Specification: Internet-Scale Low-Communication Elastic Training (`v0.1.0`)

- **Tracking Issue**: `zoo-apps/gym#4` (`[p1, training, research, bounty, hip-1350]`)
- **Specification Version**: `0.1.0`
- **License**: Apache-2.0 (source code, zero new runtime dependencies)
- **References**:
  - Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models* (2023)
  - Jaghouar et al., *OpenDiLoCo / INTELLECT-1* (Prime Intellect, 2024)
  - Ryabinin & Gusev, *Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts (Hivemind)* (2020)

---

## 1. Research Question

> **Can geographically distributed contributors improve a shared model without continuously transmitting full gradients or sharing private datasets, and how do you show their updates are beneficial and safe, not just that their GPUs computed?**

This specification defines a self-contained, byte-deterministic reference implementation (`examples/diloco_wan/`) and empirical benchmark harness designed to evaluate both halves of this question simultaneously:

1. **Communication & Elasticity**: Compare synchronous step-by-step full-gradient AllReduce ($H = 1$) against Prime DiLoCo ($H \in \{16, 64\}$) and block-quantized pseudo-gradient wire codecs (`int8_block`, `int4_block`) under realistic WAN latency, bandwidth caps, packet loss, stragglers, mid-round worker dropouts, and elastic fresh-worker joins.
2. **Benefit & Safety Verification**: Demonstrate why proof-of-compute alone is insufficient against valid-compute Byzantine attacks (gradient sign-flipping, scaled noise) and specify a 4-stage verification gate that rejects harmful updates using only wire-transmitted pseudo-gradients and an independent coordinator-held validation split—never inspecting worker private datasets.

---

## 2. Corpus Provenance & Strict Split Isolation

### 2.1 Pinned Repository Corpus
All experiments train on real versioned text files from the `zoo-apps/gym` repository:
- Top-level documentation: `README.md`, `LLM.md`, `FAQS.md`
- Core documentation: `docs/*.qmd` and `docs/dataset-formats/*.qmd`
- Distributed parallel training configs: `examples/distributed-parallel/*`

Each file is pinned by exact relative POSIX path, byte length, and SHA-256 digest in `results.json` (`corpus_manifest`), together with an aggregate SHA-256 `manifest_sha256` over the sorted file table.

### 2.2 Fixed Leakage-Free Vocabulary
To prevent any held-out test leakage into tokenizer construction, the task uses a fixed 64-symbol vocabulary (`VOCAB_SIZE = 64`): `<UNK>` (ID `0`) plus 63 lowercase alphanumeric, whitespace, and Markdown/YAML punctuation symbols covering >99.8% of repository bytes.

### 2.3 Three-Way Split & Zero Private-Data Transmission Invariant
Each document of length $N$ is partitioned with guard gaps of $C + 2$ characters ($C = 4$ causal context window) so zero $n$-grams cross split boundaries:
- **`[0 : 74%]` — Private Worker Training Shards (`PrivateShardVault`)**: Partitioned into strictly disjoint per-worker shards across $W = 6$ workers:
  - `balanced`: disjoint 320-character blocks assigned round-robin across workers.
  - `non_iid`: extreme domain and target-class partitioning where each worker holds a disjoint target-symbol partition (`target % W`), maximizing client drift.
- **`[75% : 87%]` — Coordinator Verification Validation Split (`verifier_val`)**: Exclusively held by the coordinator's `UpdateVerifier`; never exposed to any worker during local training.
- **`[88% : 100%]` — Unseen Global Test Split (`global_test`)**: Never exposed to workers *or* to `UpdateVerifier`; evaluated strictly for final quality reporting to prevent verifier overfitting.

**Privacy Wire Audit**: Every simulated WAN payload passes through `PrivateShardVault.assert_payload_contains_no_private_data(wire_payload)`, verifying that no private worker canary substrings appear on the wire and that only binary-serialized pseudo-gradient envelopes traverse the network.

---

## 3. Model & Optimization Specification

### 3.1 Neural Language Model (`CausalCharLM`)
A pure-Python stdlib Bengio-style Residual Causal Neural Language Model predicts the next symbol $y_t \in \{0,\dots,63\}$ from context $(y_{t-4}, y_{t-3}, y_{t-2}, y_{t-1})$:
- Token embeddings $E \in \mathbb{R}^{64 \times 8}$ (`512` params)
- Hidden layer $h = \tanh(W_1 [E_{c_0}; E_{c_1}; E_{c_2}; E_{c_3}] + b_1)$, $W_1 \in \mathbb{R}^{16 \times 32}, b_1 \in \mathbb{R}^{16}$ (`528` params)
- Output projection $W_2 \in \mathbb{R}^{64 \times 16}, b_2 \in \mathbb{R}^{64}$ (`1,088` params)
- Direct bigram skip residuals $W_{\text{skip1}}, W_{\text{skip2}} \in \mathbb{R}^{64 \times 64}$ (`8,192` params)
- **Total trainable parameters**: $P = 10{,}320$ (`41,280` bytes in FP32).

### 3.2 Prime DiLoCo Algorithm (Persistent Inner AdamW + Outer Nesterov)
At outer synchronization round $r \in \{1, \dots, R\}$ with $R = T / H$:
1. Each active worker $i$ initializes local weights $\theta_{i,0}^{(r)} \leftarrow \theta^{(r)}$ and executes $H$ steps of **Inner AdamW** ($\text{lr}=0.012, \beta_1=0.9, \beta_2=0.999, \lambda=10^{-4}$, batch size $B=4$) exclusively on its private shard $D_i$. Per Prime DiLoCo Algorithm 1, worker $i$ **persists** its inner AdamW moment buffers $(m_i, v_i, t_i)$ across outer rounds.
2. Worker $i$ forms the pseudo-gradient:
   $$\Delta_i^{(r)} = \theta^{(r)} - \theta_{i,H}^{(r)}$$
3. Worker $i$ encodes $\Delta_i^{(r)}$ into binary wire bytes via the selected codec, signs the canonical envelope, and transmits it over the simulated WAN.
4. Let $S_r$ be the set of updates arriving before `round_timeout_s` that pass `UpdateVerifier` (or all arriving updates when verification is disabled). If $|S_r| < q_{\min}$, the round is counted as a `quorum_miss` and $\theta^{(r+1)} = \theta^{(r)}$.
5. Otherwise, the coordinator computes the mean pseudo-gradient and applies **Outer SGD with Nesterov Momentum** ($\eta_{\text{out}} = 0.7, \mu = 0.6$):
   $$\bar{\Delta}^{(r)} = \frac{1}{|S_r|} \sum_{i \in S_r} \tilde{\Delta}_i^{(r)}$$
   $$v^{(r)} = \mu v^{(r-1)} + \bar{\Delta}^{(r)}$$
   $$\theta_{\text{cand}}^{(r+1)} = \theta^{(r)} - \eta_{\text{out}}\left(\bar{\Delta}^{(r)} + \mu v^{(r)}\right)$$
6. When verification is enabled, the coordinator runs the aggregate safety check on `verifier_val` (Section 5.5), committing $\theta^{(r+1)} = \theta_{\text{cand}}^{(r+1)}$ or rolling back to $\theta^{(r)}$.
7. On commit, the coordinator broadcasts the compressed outer delta $\delta_{\text{out}}^{(r)} = \theta^{(r)} - \theta^{(r+1)}$ back to all active workers.

---

## 4. Binary Wire Protocol & Quantization Codecs (`quantize.py`)

Every tensor transmitted over the simulated WAN is serialized into a deterministic binary frame:

```
Offset  Size  Field
0       4B    Magic ("DLCW")
4       1B    Wire version (1)
5       1B    Codec ID (1=fp32, 2=int8_block, 3=int4_block, 4=sign_1bit_no_ef)
6       2B    Block size (uint16 LE, default 64)
8       4B    Element count N (uint32 LE, 10,320)
12      ...   Codec-specific payload
```

Plus a fixed `128`-byte envelope overhead per message (round ID, contributor ID, 32-byte payload SHA-256 digest, 64-byte signature).

| Codec | Representation | Payload Bytes ($P=10{,}320$) | Compression vs FP32 |
|---|---|---:|---:|
| `fp32` | IEEE-754 float32 | `41,292` B | `1.00x` |
| `int8_block` | Per-64-block FP32 scale + signed int8 `[-127, 127]` | `10,980` B | `3.76x` |
| `int4_block` | Per-64-block FP32 scale + packed signed 4-bit `[-7, 7]` | `5,820` B | `7.09x` |
| `sign_1bit_no_ef` | Global L1 mean scale + packed 1-bit signs (no error feedback) | `1,306` B | `31.62x` |

---

## 5. WAN Physics, Elastic Churn & Verification Gate

### 5.1 Deterministic WAN Transport (`wan.py`)
Each transfer of $B_{\text{wire}}$ bytes is packetized into `MTU = 1,400`-byte frames. Each frame independently experiences packet loss probability $p_{\text{loss}}$ with up to `max_retransmits` retries. Total transfer latency is:
$$t_{\text{xfer}} = 0.5 \cdot \text{RTT}_{\text{jittered}} + \frac{8(B_{\text{wire}} + B_{\text{retx}})}{\text{BW}_{\text{bps}}} + (k_{\max} - 1)\cdot \text{RTT}_{\text{jittered}}$$
where $k_{\max}$ is the maximum transmission attempts required across frames. Transfers failing all retries are dropped (`packet_loss_exhausted`); transfers exceeding `round_timeout_s` are dropped (`wan_timeout`). Mid-round worker dropouts (`churn_drop_prob`) and fresh elastic joins (`elastic_join_round`, downloading a full FP32 checkpoint) are explicitly accounted in `total_wire_bytes` and `simulated_wall_time_s`.

### 5.2 Four-Stage Independent Benefit & Safety Verification (`verify.py`)
Proof-of-compute receipts prove only that FLOPs were spent, not that the resulting update improves the model or is safe. `UpdateVerifier` enforces four sequential checks without ever reading private worker data:

1. **Stage 1 — Envelope Authentication & Freshness**: Verifies `round_id` freshness, duplicate suppression, payload SHA-256 digest, and constant-time signature check over canonical transcript `v1|<round>|<id>|<codec>|<sha256>`. *(Reference stdlib implementation uses HMAC-SHA256; production WAN deployments replace the MAC primitive with asymmetric Ed25519 over the identical canonical transcript.)*
2. **Stage 2 — Finite & Relative L2-Norm Bound**: Rejects NaN/Inf values, zero-norm no-ops, and magnitude-explosion attacks exceeding $\min(\tau_{\text{abs}}, 3.5 \cdot \text{median}_j \|\Delta_j\|_2)$.
3. **Stage 3 — Geometric Peer-Median Cosine Filter**: Computes coordinate-wise median $\Delta_{\text{med}}$ across norm-safe submissions and rejects anti-aligned / sign-flipped vectors with $\cos(\Delta_i, \Delta_{\text{med}}) < 0.05$.
4. **Stage 4 — Independent Held-Out Validation Benefit Gate**: Probes $\theta_{\text{probe}} = \theta^{(r)} - \eta_{\text{probe}} \Delta_i$ ($\eta_{\text{probe}} = 0.4$) on `verifier_val` (240 examples, strictly disjoint from worker data and `global_test`). Rejects any update with $L_{\text{val}}(\theta_{\text{probe}}) - L_{\text{val}}(\theta^{(r)}) > 0.015$.
5. **Stage 5 — Aggregate Outer-Step Rollback Safeguard**: Evaluates $\theta_{\text{cand}}^{(r+1)}$ on `verifier_val`; if the aggregated outer step regresses validation loss beyond `0.025`, the coordinator rolls back to $\theta^{(r)}$.

---

## 6. Governance, Scientific Integrity & Licensing

- **Independently Checkable Evidence**: Every claim in `BENCHMARK.md` maps directly to `results.json`, regenerable byte-for-byte via `python3 examples/diloco_wan/run_benchmark.py --seed 1350`.
- **Governance Separation**: Per the Zoo x Hanzo open research program charter, this artifact provides reproducible engineering evidence and protocol analysis only; AI agents may not redefine voting rules, authorize treasury spending, or self-certify production readiness.
- **License**: All source code under `examples/diloco_wan/` is Apache-2.0 with zero new runtime dependencies (`setup.py`, `pyproject.toml`, and `requirements.txt` are unmodified).

---

## 7. What Remains Unproven / Open Research Gaps

This reference benchmark establishes exact, reproducible mechanics at small scale, but **does not prove** the following properties, which remain open research and engineering milestones:

1. **Transformer Scale (7B–70B Parameters)**: Our pure-Python neural LM has $P = 10{,}320$ parameters (`41.3 KB` FP32). Real frontier models (e.g., 7B–70B Zen / Llama transformers, 14–140 GB BF16) exhibit vastly different curvature, layer-wise gradient norm variance, activation-checkpointing memory tradeoffs, and bandwidth-to-compute ratios where multi-gigabyte WAN transfers saturate TCP congestion control over minutes rather than milliseconds.
2. **Real Internet Transport (TCP/QUIC/libp2p)**: `wan.py` models RTT, jitter, bandwidth serialization, and independent per-packet loss deterministically. Real transcontinental links exhibit bursty correlated packet loss (Gilbert-Elliott dynamics), TCP slow-start/BBR congestion dynamics, NAT traversal, BGP routing jitter, and asymmetric home uplink speeds (`5–20 Mbps` up vs `100–1000 Mbps` down).
3. **Gradient Inversion & Membership Inference Privacy**: While zero raw private corpus bytes ever touch the wire, pseudo-gradients $\Delta_i = \theta^{(r)} - \theta_{i,H}^{(r)}$ can still leak statistical information about private training batches under sophisticated gradient-inversion or membership-inference attacks. Differential privacy (DP-SGD / local Gaussian noise + clipping) and secure aggregation (SecAgg) are not implemented here and remain necessary for formal $(\varepsilon, \delta)$-privacy guarantees.
4. **Adaptive & Sybil Byzantine Collusion ($\ge 33\%$ / Targeted Backdoors)**: Our threat model tests non-adaptive sign-flip, scaled noise, and corrupted signatures up to 33% malicious workers. Adaptive adversaries who craft subtle **targeted backdoor triggers** that preserve average loss on `verifier_val`, or Sybil swarms exceeding 50% of submissions (which can poison the coordinate median), require decentralized reputation/staking, rotating private red-team canary probes, and multi-verifier quorum attestation.
5. **Verifier Overfitting & Coordinator Trust**: Holding `verifier_val` on a single coordinator introduces a centralization point and potential long-horizon overfitting if contributors adaptively hill-climb the same static validation split over thousands of rounds. Rotating blind validation shards across an independent committee of verifiers remains future work.
6. **Asynchronous / Stale-Gradient DiLoCo & Error Feedback**: We evaluate synchronous-round elastic quorums (dropping stragglers past `round_timeout_s`) and 1-bit sign quantization *without* error feedback. Asynchronous stale-update staleness compensation (e.g., Delay-Compensated Async DiLoCo) and EF21/momentum error-feedback accumulators for 1-bit/2-bit codecs remain to be evaluated on multi-GPU clusters.
