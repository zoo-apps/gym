# Federated LoRA / QLoRA for Zen — Reproducible Benchmark Report

- **Issue**: `zoo-apps/gym#3` (HIP-1350)
- **Seed**: `--seed 1350`
- **Artifact**: [`results.json`](./results.json) (byte-identical across consecutive runs; SHA-256: `bc578b1c6e68b3eac4df5798532cc514659b553672f8215304718889abf455fe`)
- **Reproduce**:
  ```bash
  python3 -m examples.federated_lora_zen.run_benchmark \
    --seed 1350 \
    --out examples/federated_lora_zen/results.json
  ```

---

## 1. Experimental Setup & Matched Budget

| Property | Value |
| :--- | :--- |
| Pinned Corpus | 47 real Zoo Gym files (`133,751` bytes; manifest SHA-256 `5f01c5ef...7b65e888`) |
| Domains | `docs` (9 files, 40,606 B), `yaml` (32 files, 38,796 B), `python` (6 files, 54,349 B) |
| Splits (strictly disjoint 128-char chunks) | `base_pretrain`: 3,213 ex · `centralized_train`: 13,965 ex (= `alpha ∪ beta ∪ gamma`) · `eval_test`: 4,284 ex (`docs` 1,302, `yaml` 1,197, `python` 1,785) |
| Frozen Base Model | 3-layer residual causal LM (`V=80`, `ctx=4`, `d_emb=16`, `d_hid=64`, `14,800` base params, pretrained 60 steps on `base_pretrain`, then cryptographically frozen) |
| LoRA / QLoRA Config (Main Regimes) | `r = 8`, `alpha = 16.0` (`3,200` trainable params = `21.62%` of base), NF4 `block_size = 64` |
| Matched Compute Budget | `12` rounds × `8` steps × `16` batch/worker × `3` workers = **`4,608` sample-updates** (`96` sequential AdamW steps, `lr = 0.006`) vs Centralized `96` steps × `48` global batch = **`4,608` sample-updates** |
| Host Wall Time (excluded from `results.json`) | `166.22s` on single CPU for all 10 regimes (6 main + 4 negative-regime runs) |

---

## 2. Main 6 Matched Regimes (`mixed_non_iid` 3-Worker Vaults)

Initial held-out test loss at round 0 (zero-initialized LoRA `B = 0`): **`3.5078`** (FP32 base) and **`3.5052`** (4-bit NF4 base; quantization RMSE = `0.01048`, max abs err = `0.06348`).

| Regime | Base Quant | Aggregation | Final Test Loss | Test PPL | Top-1 Acc | ΔLoss vs Central LoRA | Base Active RAM | Adapter Wire (FP32 / FP16) | Raw Data Shared |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `centralized_lora_baseline` | `fp32` | `none_centralized` | **3.1675** | **23.75** | **21.73%** | `+0.0000` | 59,200 B (`1.00×`) | 0 B / 0 B | 133,751 B |
| `centralized_qlora_nf4_baseline` | `nf4` | `none_centralized` | **3.1695** | **23.80** | 21.41% | `+0.0020` | **10,048 B (`5.89×`)** | 0 B / 0 B | 133,751 B |
| `fed3_lora_fedavg_ba` | `fp32` | `fedavg_lora_ba` | 3.2217 | 25.07 | 20.10% | `+0.0542` | 59,200 B (`1.00×`) | 921.6 KB / 460.8 KB | **0 B** |
| `fed3_lora_exact_svd` | `fp32` | `exact_delta_svd_fedlora` | **3.2082** | **24.74** | 19.93% | **`+0.0407`** | 59,200 B (`1.00×`) | 921.6 KB / 460.8 KB | **0 B** |
| `fed3_qlora_nf4_exact_svd` | `nf4` | `exact_delta_svd_fedlora` | **3.2094** | **24.76** | 20.03% | **`+0.0418`** | **10,048 B (`5.89×`)** | 921.6 KB / 460.8 KB | **0 B** |
| `fed3_qlora_node_loss_recovery` | `nf4` | `exact_delta_svd_fedlora` | **3.2080** | **24.73** | **20.14%** | **`+0.0405`** | **10,048 B (`5.89×`)** | **844.8 KB / 422.4 KB** | **0 B** |

### Per-Domain Held-Out Loss & Aggregation Diagnostics (Main Regimes)

| Regime | `docs` Loss | `yaml` Loss | `python` Loss | Mean Naive Cross-Term `‖E_cross‖_F` | Mean SVD Truncation Residual `‖E_svd‖_F` | Frozen Base Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `centralized_lora_baseline` | 3.5921 | 3.2423 | 2.8076 | — | — | `true` |
| `centralized_qlora_nf4_baseline` | 3.5967 | 3.2446 | 2.8077 | — | — | `true` |
| `fed3_lora_fedavg_ba` | 3.6274 | 3.3402 | 2.8464 | `0.155508` (applied) | — | `true` |
| `fed3_lora_exact_svd` | 3.6367 | **3.3012** | **2.8333** | `0.147956` (avoided) | **`0.075587`** (`2.06×` lower) | `true` |
| `fed3_qlora_nf4_exact_svd` | 3.6368 | **3.3020** | **2.8354** | `0.150588` (avoided) | **`0.075999`** (`1.98×` lower) | `true` |
| `fed3_qlora_node_loss_recovery` | 3.6382 | 3.3050 | **2.8292** | `0.148573` (avoided) | **`0.077920`** (`1.91×` lower) | `true` |

### Key Takeaways from Main Regimes
1. **3-Worker Federation vs Centralized Baseline**: With zero raw data shared (`0 B` vs `133,751 B`), 3-worker federated LoRA (`exact_delta_svd_fedlora`) reaches **`3.2082`** vs centralized **`3.1675`** (`+0.0407` nats / `+0.99` PPL gap across 8 local steps between syncs and heterogeneous institutional vaults).
2. **Exact Effective-Delta SVD Beats Naive FedAvg(A, B)**: `fed3_lora_exact_svd` (`3.2082`, PPL `24.74`) improves over naive `fed3_lora_fedavg_ba` (`3.2217`, PPL `25.07`) by **`-0.0135` nats (`-0.33` PPL)** — with especially strong gains on structured YAML (`3.3012` vs `3.3402`, `-0.0390` nats) — while cutting effective aggregation Frobenius error by **`2.06×`** (`0.1555 -> 0.0756`).
3. **4-bit NF4 QLoRA Memory Savings at Near-Zero Quality Cost**: Quantizing the frozen base to packed 4-bit NF4 reduces base weight memory by **`5.89×` vs FP32** (`59,200 B -> 10,048 B`) and **`2.95×` vs FP16** (`29,600 B -> 10,048 B`), while costing only **`+0.0020` nats** centralized (`3.1675 -> 3.1695`) and **`+0.0012` nats** federated (`3.2082 -> 3.2094`).
4. **Mid-Run Node-Loss & Cold Recovery Audit**:
   - `worker-beta` went offline for **3 consecutive rounds (`rounds 4, 5, 6`)** while surviving workers (`worker-alpha`, `worker-gamma`) continued aggregating without stalling (`min_quorum = 2`).
   - At **Round 7**, `worker-beta` performed a cold restart from persisted coordinator checkpoint `server_checkpoint_r006.json` (`67,961` bytes, SHA-256 `aa2e09ae...8fcd`, integrity verified `true`).
   - Deterministic simulated recovery latency (100 Mbps WAN + 250 ms cold boot + 5 ms SHA-256 verification): **`260.44 ms`**.
   - Final test loss after recovery: **`3.2080`** (PPL **`24.73`**) vs healthy 3-worker QLoRA **`3.2094`** (PPL **`24.76`**) — **`-0.0013` nats delta** (full quality parity, zero corruption, `-8.3%` total wire traffic during outage).

---

## 3. Honest Negative Regimes

Every negative regime below was executed with the same pinned corpus, frozen NF4 base, and `--seed 1350`:

| Negative Regime | Partition | Rank | Local Steps/Rd | Final Loss | Test PPL | `docs` | `yaml` | `python` | Mean Cross-Term `‖E_cross‖_F` | Mean SVD Residual `‖E_svd‖_F` | Wire (FP32) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `neg_extreme_non_iid_naive_fedavg` | `extreme_non_iid` | 8 | 16\* | 3.1511 | 23.36 | 3.5695 | 3.1883 | 2.8210 | `0.345302` | — | 921.6 KB |
| `neg_extreme_non_iid_exact_svd_ref` | `extreme_non_iid` | 8 | 16\* | **3.1499** | **23.33** | **3.5641** | **3.1717** | 2.8332 | `0.351902` (avoided) | **`0.161975`** | 921.6 KB |
| `neg_rank1_bottleneck` | `mixed_non_iid` | **1** | 8 | **3.3280** | **27.88** | 3.7212 | 3.4364 | 2.9686 | `0.039928` | `0.037829` | **115.2 KB** (`8×` less) |
| `neg_permanent_two_of_three_loss` | `extreme_non_iid` | 8 | 8 | **3.4108** | **30.29** | 3.5928 | **3.5116** | **3.2105** | `0.033704` | `0.013171` | 409.6 KB |

*\*Note: The paired extreme Non-IID stress test runs `16` local steps/round (`9,216` total sample-updates, 2× the main 8-step compute budget) specifically to amplify multi-step local subspace drift; its two rows should be compared directly against each other rather than against the 8-step main regimes.*

### Failure-Mode Analysis
1. **Negative #1 — Extreme 100% Single-Domain Non-IID Cross-Term Interference**:
   - Under 100% disjoint single-domain vaults (`worker-alpha=docs`, `worker-beta=yaml`, `worker-gamma=python`) and 16 local steps/round, naive `FedAvg(A, B)` exhibits **`2.22×` larger mathematical cross-term Frobenius discrepancy** (`0.3453` vs `0.1555` in `mixed_non_iid`).
   - Exact effective-delta SVD cuts aggregation Frobenius error by **`2.17×`** (`0.3519 -> 0.1620`), improves top-1 accuracy from `21.55% -> 22.04%`, lowers overall loss (`3.1499` vs `3.1511`), and improves `yaml` loss by `-0.0166` nats (`3.1883 -> 3.1717`).
   - Honestly, we also observe that at this compact model scale (`d = 64`), the overall loss gap under `extreme_non_iid` is modest (`-0.0012` nats) because rank-8 SVD itself incurs a nonzero rank-truncation residual (`0.1620`) when projecting a rank-24 sum of three orthogonal domain updates back to rank-8.
2. **Negative #2 — Rank-1 Under-Parameterized Bottleneck (`r = 1` vs `r = 8`)**:
   - Dropping rank to `r = 1` slashes adapter wire traffic by **`8×`** (`115.2 KB` vs `921.6 KB` FP32) and trainable parameters to `2.70%`, but creates a severe representational bottleneck: held-out loss degrades by **`+0.1186` nats** (`3.2094 -> 3.3280`, PPL **`24.76 -> 27.88`**) uniformly across `docs`, `yaml`, and `python`.
3. **Negative #3 — Permanent 2-of-3 Worker Attrition (`67%` Fleet Loss)**:
   - When `worker-beta` (`yaml`) and `worker-gamma` (`python`) fail permanently at Round 3, only `worker-alpha` (`docs`) trains through Round 12.
   - Overall held-out test loss degrades catastrophically by **`+0.2014` nats** vs healthy 3-worker QLoRA (`3.2094 -> 3.4108`, PPL **`24.76 -> 30.29`**, accuracy drops to `16.20%`).
   - Per-domain breakdown confirms severe **catastrophic forgetting on the two lost workers' distributions**: `yaml` (`3.5116`, `+0.2096` worse than healthy) and `python` (`3.2105`, `+0.3751` worse than healthy), while the sole surviving worker's `docs` domain remains well-adapted (`3.5928`).

---

## 4. What Remains Unproven / Open Research Gaps

This reference implementation and benchmark prove the mathematical mechanics, cryptographic envelope verification, 4-bit NF4 storage/dequantization fidelity, exact vs naive LoRA aggregation trade-offs, and cold node-loss recovery invariants on a real repository corpus. The following remain **explicitly unproven** and require dedicated GPU / WAN validation before production Zen 4B–70B deployment:

1. **Production 4B–70B Zen GPU QLoRA Kernels**: Pure-Python NF4 block quantization verifies the Dettmers et al. codebook, nibble packing, and quantization error math, but does **not** benchmark `bitsandbytes` CUDA NF4/FP4 fused dequantization kernels, double quantization of absmax scales, paged AdamW CPU-offloading, BF16 activation checkpointing, or FSDP+QLoRA memory fragmentation on multi-GPU H100/A100/B200 clusters.
2. **Live Multi-Host Flower gRPC / mTLS Transport Over Real WANs**: Envelopes, HMAC-SHA256 signatures, wire byte counts, and checkpoint recovery latency (`260.44 ms` modeled at 100 Mbps) are executed deterministically in-process with real disk checkpoints, not over live Flower (`flwr`) gRPC/mTLS sockets across geographically separated hosts with real packet loss, NAT traversal, firewall traversal, straggler tail latency, or asynchronous buffered aggregation (FedBuff).
3. **Differential Privacy & Inversion Resistance**: Zero raw data sharing prevents direct vault exposure, and HMAC signing prevents unauthenticated tampering, but neither provides formal `(epsilon, delta)`-differential privacy (DP-SGD per-example clipping + calibrated Gaussian noise), cryptographic Secure Aggregation (`SecAgg`), nor empirical defense against adapter-update gradient inversion or membership inference attacks.
4. **Byzantine / Compromised Worker Defenses**: An authenticated worker holding a valid admin key can still submit adversarially poisoned `(A, B)` updates. Robust aggregators (coordinate-wise trimmed mean, Krum, or held-out validation-gated acceptance) are not evaluated here.
5. **Heterogeneous Worker Ranks & Downstream LLM Benchmarks**: All three workers use identical LoRA rank `r` and target layers; heterogeneous-rank aggregation (HetLoRA / FlexLoRA across mixed GPU memory tiers), multi-adapter hot-swapping at inference time, and downstream Zen capability/safety benchmarks (MMLU, GSM8K, HumanEval, refusal/safety alignment) remain open work.
