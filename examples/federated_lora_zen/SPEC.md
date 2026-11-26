# Federated LoRA / QLoRA for Zen — Protocol & Reference Specification (v1.0.0)

- **Issue**: `zoo-apps/gym#3`
- **Track**: HIP-1350 (`p1`, `training`, `research`, `bounty`)
- **Package**: `examples/federated_lora_zen/`
- **License**: Apache-2.0
- **Runtime Dependencies**: Python >= 3.10 standard library only (zero additions to `setup.py`, `pyproject.toml`, or `requirements.txt`)

---

## 1. Scope and Objectives

This specification defines a self-contained, reproducible reference protocol and benchmark harness for federated low-rank adaptation (LoRA and 4-bit NF4 QLoRA) across **three independently administered workers** (`worker-alpha`, `worker-beta`, `worker-gamma`) targeting the Zen model family within Zoo Gym.

The specification establishes four verifiable engineering invariants:

1. **Data Sovereignty (Zero Raw Data Sharing)**: Each worker owns a cryptographically hashed private data vault (`WorkerVault`) and an independent HMAC-SHA256 administrative identity. Raw text, token sequences, and chunk IDs never leave the worker boundary; only signed low-rank adapter matrices `(A, B)` and scalar sample counts traverse the wire.
2. **Frozen Base Immutability & 4-bit NF4 QLoRA Parity**: Base model weights are pretrained once and cryptographically frozen (verified via canonical SHA-256 before and after training). In QLoRA mode, all adapted linear projections are stored as real packed 4-bit NormalFloat (NF4) nibbles (2 weights/byte) with per-block FP16 absmax scales (`block_size = 64`), dequantized during forward and backward propagation while trainable `(A, B)` adapters remain in FP32.
3. **Cross-Term-Free Exact Adapter Aggregation**: Standard naive FedAvg averages `A` and `B` separately (`mean(B) @ mean(A)`), introducing mathematical cross-term interference relative to the exact sample-weighted effective update `sum_k w_k (B_k @ A_k)`. This specification implements and compares both `fedavg_lora_ba` and `exact_delta_svd_fedlora` (exact effective-delta reconstruction followed by deterministic truncated rank-`r` SVD refactoring).
4. **Fault Tolerance & Cold Node-Loss Recovery**: Mid-run crash or disconnect of a worker does not stall federation as long as `active_workers >= min_quorum`. Surviving workers continue aggregating with re-normalized sample weights. Upon restart, the lost worker discards volatile RAM state, downloads and SHA-256-verifies the coordinator's latest disk checkpoint, synchronizes, and rejoins cleanly.

---

## 2. Corpus Grounding and Disjoint Partitioning (`corpus.py`)

All benchmark evaluations are grounded in **47 real files (133,751 bytes)** pinned by exact SHA-256 digests from the Zoo Gym repository across three distinct technical domains:

| Domain | Source Files | Bytes | Role |
| :--- | :---: | ---: | :--- |
| `docs` | 9 Markdown / Quarto files (`README.md`, `LLM.md`, `docs/*.qmd`) | 40,606 | Natural-language technical documentation |
| `yaml` | 32 LoRA/QLoRA model recipes (`examples/*/*.y*ml`) | 38,796 | Structured YAML training configurations |
| `python` | 6 CLI and trainer builder modules (`src/gym/**/*.py`) | 54,349 | Python source code |

### Disjoint Splitting Rules
- Every file is sliced into contiguous `128`-character non-overlapping chunks (`TextChunk`).
- Causal next-token windows (`context_len = 4`, `stride = 6`) are extracted strictly *within* chunk boundaries so no context window ever crosses a split boundary.
- Per domain, chunks are deterministically shuffled with `--seed 1350` and partitioned disjointly:
  - `15%` -> `base_pretrain` (used once to pretrain the shared frozen base LM).
  - `20%` -> `eval_test` (strictly held-out evaluation split, scored overall and per-domain).
  - `65%` -> training pool, split without overlap across the three worker vaults (`worker-alpha`, `worker-beta`, `worker-gamma`).
- **Centralized Parity**: `centralized_train` equals the exact union of the three worker vaults (`alpha ∪ beta ∪ gamma`), guaranteeing zero extra training data for the centralized upper bound.
- **Partition Modes**:
  - `mixed_non_iid` (main regimes): institutional domain skew (`60%` primary domain + `20%` + `20%` secondary domains per worker).
  - `extreme_non_iid` (negative regimes): `100%` single-domain isolation (`worker-alpha` = `docs`, `worker-beta` = `yaml`, `worker-gamma` = `python`).

---

## 3. Model, LoRA, and 4-bit NF4 QLoRA Architecture (`lora_model.py`)

### 3.1 Frozen Base Causal LM
- Fixed deterministic `80`-token character vocabulary built from the pinned corpus.
- Context window `C = 4`, embedding dimension `d_emb = 16` (`d_in = 64`), hidden dimension `d_hid = 64`:
  - `proj_up`: `64 x 64` linear projection + `tanh`
  - `proj_mid`: `64 x 64` residual linear projection (`h2 = h1 + tanh(z2)`)
  - `lm_head`: `80 x 64` output projection -> softmax cross-entropy loss
- Total base parameters: `14,800` (`13,312` linear projection weights + `1,488` embeddings/biases).

### 3.2 Low-Rank Adapters (LoRA)
Every linear projection `l ∈ {proj_up, proj_mid, lm_head}` with frozen weight `W_l` is augmented as:

```
W_eff,l = W_l + (alpha / r) * (B_l @ A_l)
```

- `A_l ∈ R^{r x d_in}` initialized from `N(0, 1 / d_in)`.
- `B_l ∈ R^{d_out x r}` initialized to exact `0` (Hu et al., 2021), guaranteeing `DeltaW_l = 0` at round `0`.
- Trainable parameters: `3,200` at `r = 8` (`21.62%` of base), `1,600` at `r = 4` (`10.81%`), `400` at `r = 1` (`2.70%`).
- Gradients are computed exclusively for `(A_l, B_l)` via exact reverse-mode autodiff and verified against numerical finite differences (`< 1e-7` max relative error).

### 3.3 4-bit NormalFloat Quantization (QLoRA)
In `nf4` mode, each frozen linear projection matrix `W_l` is block-quantized (`block_size = 64`):
- Uses the canonical 16-level Dettmers et al. (2023) `NF4_CODEBOOK` in `[-1.0, 1.0]`.
- Each 64-weight block stores one absmax scale `s_b = max(|w|)` (accounted at FP16 = 2 bytes/block) and 32 bytes of packed 4-bit nibbles (`(q_even << 4) | q_odd`).
- Effective linear weight storage: `0.53125 bytes/param` vs `4.0 bytes/param` (FP32) and `2.0 bytes/param` (FP16).
- Total active base storage drops from `59,200 bytes` (FP32) to `10,048 bytes` (NF4 linear + FP16 embeddings/biases) — a **5.89x reduction vs FP32** (**2.95x vs FP16**).

---

## 4. Federation Wire Protocol and Security (`federation.py`)

### 4.1 Signed `AdapterEnvelope` Schema
Each active worker transmits an `AdapterEnvelope` at round `t`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `round_idx` | `int` | 1-indexed federation round number |
| `worker_id` | `str` | Registered worker identity (`worker-alpha`, `worker-beta`, `worker-gamma`) |
| `rank`, `alpha` | `int`, `float` | LoRA rank and scaling factor |
| `n_samples` | `int` | Local training examples processed in round `t` |
| `local_loss` | `float` | Mean local minibatch cross-entropy loss |
| `adapter_state` | `(A, B)` | Low-rank matrices across `{proj_up, proj_mid, lm_head}` |
| `adapter_sha256` | `str` | SHA-256 of canonical IEEE-754 big-endian encoding of `adapter_state` |
| `hmac_signature` | `str` | `HMAC-SHA256(admin_key, header_bytes \|\| canonical_adapter_bytes)` |

The coordinator verifies registration, round number, duplicate-submission prevention, `adapter_sha256`, and constant-time `hmac.compare_digest` before any matrix enters aggregation. Any violation raises `SecurityError`.

### 4.2 Aggregation Strategies
Let `w_k = n_k / sum_j n_j` be the normalized sample weight of participating worker `k`.

1. **Naive `fedavg_lora_ba`**:
   ```
   B_bar = sum_k w_k B_k
   A_bar = sum_k w_k A_k
   DeltaW_naive = (alpha / r) * (B_bar @ A_bar)
   ```
   Because `B_bar @ A_bar = sum_j sum_k w_j w_k (B_j @ A_k)`, this introduces an unwanted cross-term error:
   ```
   E_cross = (alpha / r) * || (B_bar @ A_bar) - sum_k w_k (B_k @ A_k) ||_F
   ```

2. **Exact Effective-Delta SVD (`exact_delta_svd_fedlora`)**:
   - Reconstructs the exact weighted adapter update `DeltaW_exact = sum_k w_k (B_k @ A_k)` on the coordinator from received `(A_k, B_k)` envelopes (wire payload remains strictly rank-`r`).
   - Computes the exact truncated rank-`r` SVD `DeltaW_exact ≈ U_r Sigma_r V_r^T` via deterministic Gram-matrix subspace iteration (`G = DeltaW^T DeltaW`) + Rayleigh-Ritz cyclic Jacobi eigendecomposition with canonical sign fixing.
   - Factors symmetrically for balanced AdamW gradient conditioning:
     ```
     B_new = U_r @ Sigma_r^(1/2)
     A_new = Sigma_r^(1/2) @ V_r^T
     ```
   - Eliminates cross-term interference while keeping the broadcast global adapter rank-`r`.

---

## 5. Node-Loss Fault Tolerance and Cold Recovery State Machine

```
[ACTIVE] --(dropout at round R_drop)--> [OFFLINE]
   ^                                        |
   |  (rejoin at round R_rejoin)            | (surviving workers >= min_quorum
   |  1. Wipe volatile RAM state            |  continue aggregating & persisting
   |  2. Fetch server_checkpoint_r{R-1}.json|  server_checkpoint_r{t}.json)
   |  3. Verify adapter_sha256 integrity    |
   |  4. Hydrate model & reset local AdamW  |
   +--------------[COLD_RECOVER] <----------+
```

- Every round `t >= 0`, the coordinator atomically writes `coordinator/server_checkpoint_r{t:03d}.json` with `adapter_sha256`.
- Every active worker atomically writes `workers/{worker_id}/worker_state.json`.
- Deterministic simulated recovery latency is modeled as:
  `T_recovery_ms = 250.0 (cold boot) + 5.0 (SHA-256 verify) + (checkpoint_bytes * 8) / (100 Mbps * 1000)`
  ensuring reproducible telemetry in `results.json` independent of host scheduling jitter.

---

## 6. Mapping to Flower and Production Zoo Gym / Zen Models

| Reference Component | Production Flower + Zoo Gym / Zen Mapping |
| :--- | :--- |
| `FrozenBaseLM` (`nf4`) | Zen 4B–70B loaded via `bitsandbytes` `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` |
| `AdapterState` (`A, B`) | PEFT `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])` |
| `AdapterEnvelope` + HMAC | Flower `FitRes` / `Parameters` over mTLS gRPC (`flwr.client.NumPyClient`) with per-org client certificates |
| `exact_delta_svd_fedlora` | Custom Flower `Strategy.aggregate_fit` performing GPU batched SVD on `DeltaW_exact` per adapted module |
| `server_checkpoint_r{t}.json` | Versioned S3/OCI adapter checkpoint ledger (`adapter_model.safetensors` + manifest SHA-256) |
