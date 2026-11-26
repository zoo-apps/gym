<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors. -->

# Empirical Benchmark Report: Internet-Scale Low-Communication Training (`zoo-apps/gym#4`)

- **Seed**: `1350` (byte-identical across independent runs: `results.json` SHA-256 `d9defdc397029593f58aa139957f528be2d44d027e4c5620fb5908f45d79d9ad`)
- **Corpus**: `49` SHA-256 pinned versioned files from `zoo-apps/gym` (`208,532` raw bytes, aggregate `manifest_sha256 = 969f7e992b4cadd72973d0f82728a2477b34bce9b92ca3ca371ebfc846fa2a7b`)
  - Strictly disjoint private per-worker training shards (`W = 6` workers, `152,178` total private examples; never transmitted over WAN)
  - Independent coordinator verification split (`verifier_val = 1,500` examples; never seen by workers)
  - Completely unseen global evaluation split (`global_test = 1,500` examples; never seen by workers or verifier)
- **Model**: Pure-Python stdlib Bengio Residual `CausalCharLM` (`P = 10,320` trainable parameters, `41,280` B FP32, fixed 64-symbol vocabulary, context window `C = 4`, initial random test loss `4.15812`, initial PPL `63.95`)
- **Reproduction Command**:
  ```bash
  python3 examples/diloco_wan/run_benchmark.py --seed 1350
  ```

---

## 1. Master Results Table (Quality, Throughput & Network Bytes)

| Regime | Type | H | Codec | Test Loss ↓ | PPL ↓ | Top-1 ↑ | Top-5 ↑ | Wall (s) ↓ | Util % ↑ | Ex/s ↑ | Speedup vs H=1 | Wire MB ↓ | Byte Red. vs H=1 | Poison Rej. |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `centralized_sgd_ceiling` | Ceiling | 1 | `fp32` | `2.6865` | `14.68` | `27.07%` | `62.47%` | `2.05` | `100.0%` | `6,000.0` | `31.5x` | `0.000` | — | — |
| `allreduce_every_step_wan` | Baseline | 1 | `fp32` | `2.7092` | `15.02` | `27.00%` | `61.53%` | `64.56` | `3.35%` | `190.3` | `1.0x` | `242.695` | `1.0x` | — |
| `diloco_h16_fp32_wan` | Core | 16 | `fp32` | `2.7804` | `16.13` | `25.73%` | `59.27%` | `6.06` | `35.75%` | `2,027.6` | `10.7x` | `15.169` | `16.0x` | — |
| `diloco_h64_fp32_wan` | Core | 64 | `fp32` | `2.7994` | `16.43` | `25.87%` | `58.47%` | `3.15` | `69.04%` | `3,901.6` | `20.5x` | `3.792` | `64.0x` | — |
| `diloco_h64_int8_clean_wan` | Core | 64 | `int8_block` | `2.7994` | `16.44` | `25.80%` | `58.47%` | `3.07` | `70.79%` | `4,000.2` | `21.0x` | `1.017` | `238.7x` | — |
| `diloco_h64_int8_verified_wan` | **Verified** | 64 | `int8_block` | `2.8748` | `17.72` | `24.87%` | `57.80%` | `6.93` | `65.34%` | `1,662.1` | `9.3x` | `0.924` | **`262.6x`** | **`100%` (12/12)** |
| `diloco_h64_int4_verified_wan` | **Verified** | 64 | `int4_block` | `2.8723` | `17.68` | `24.80%` | `57.80%` | `4.42` | `64.75%` | `2,604.1` | `14.6x` | `0.507` | **`478.7x`** | **`100%` (15/15)** |
| `diloco_h64_int8_unverified_attack` | Ablation | 64 | `int8_block` | `3.3874` | `29.59` | `14.47%` | `42.87%` | `6.93` | `65.34%` | `1,662.1` | `9.3x` | `0.924` | `262.6x` | `0%` (0/12) |
| `neg_extreme_h512_non_iid` | **Neg. #1** | 512 | `fp32` | `3.8174` | `45.49` | `7.80%` | `36.87%` | `2.25` | `94.58%` | `5,454.6` | `28.7x` | `0.474` | `512.0x` | — |
| `neg_high_loss_severe_churn` | **Neg. #2** | 64 | `int8_block` | `3.1243` | `22.74` | `18.40%` | `49.73%` | `11.74` | `58.88%` | `1,046.6` | `5.5x` | `0.347` | `700.1x` | — |
| `neg_1bit_sign_no_error_feedback` | **Neg. #3** | 64 | `sign_1bit_no_ef` | `2.8590` | `17.44` | `24.80%` | `58.07%` | `3.05` | `71.36%` | `4,032.7` | `21.2x` | `0.131` | `1,848.6x` | — |

*(All regimes start from identical random weights `init_test_loss = 4.15812` and enforce `private_data_leaked_bytes = 0` across every wire payload audit.)*

---

## 2. Key Empirical Findings

### 2.1 WAN Bottleneck Collapse vs Prime DiLoCo
- Under `100 ms` RTT / `50 Mbps` WAN, synchronous step-by-step full-gradient AllReduce (`allreduce_every_step_wan`, $H = 1$) spends `62.40 s` of its `64.56 s` wall time waiting on network synchronization (`3.35%` compute utilization) and transmits `254,484,480` wire bytes (`242.70 MB`).
- Prime DiLoCo with persistent per-worker Inner AdamW and Outer Nesterov momentum cuts total wire bytes by exactly **`16.0x`** at $H=16$ (`15.17 MB`, `10.7x` wall speedup) and **`64.0x`** at $H=64$ (`3.79 MB`, `20.5x` wall speedup, `69.04%` compute utilization), retaining `92.3%` of the centralized test-loss reduction (`-1.3588` vs `-1.4716` nats) without ever sharing a single byte of private worker data.

### 2.2 Block-Quantized Wire Compression Fidelity
At matched $H = 64$ on a clean `100 ms` WAN (`0%` churn/attack):
- `diloco_h64_fp32_wan`: Test loss `2.79935` (`PPL 16.43`), `3,976,320` B (`64.0x` reduction vs AllReduce).
- `diloco_h64_int8_clean_wan`: Test loss `2.79942` (`PPL 16.44`, **`+0.00007` nats**), `1,066,368` B (**`238.7x`** reduction vs AllReduce, `21.0x` wall speedup).
- `diloco_h64_int4_verified_wan`: Even under 2% packet loss, 10% mid-round churn, elastic late-join (`41,420` B full checkpoint catchup), and 33% concurrent Byzantine attack, `int4_block` compresses total WAN traffic to **`531,608` B (`0.507 MB`, `478.7x` reduction vs AllReduce)** while matching `int8_block` quality (`2.8723` vs `2.8748`).

### 2.3 Security Ablation: Why Proof-of-Compute Is Not Enough
Comparing `diloco_h64_int8_verified_wan` against `diloco_h64_int8_unverified_attack` under the **exact same** 2-of-6 (33%) concurrent Byzantine fault pattern (`sign_flip`, `scaled_noise`, and round-3 `corrupted_signature`), identical churn (`10%`), identical packet loss (`2%`), and identical elastic join (`round 4`):

| Metric | Verification Gate **ON** (`int8`) | Verification Gate **ON** (`int4`) | Verification Gate **OFF** (`int8`) |
|---|---:|---:|---:|
| Poisoned updates rejected | **`12 / 12` (`100%`)** | **`15 / 15` (`100%`)** | `0 / 12` (`0%`) |
| Honest updates accepted | **`25 / 25` (`100%`)** | **`20 / 20` (`100%`)** | `25 / 25` (`100%`) |
| Unseen `global_test` Loss ↓ | **`2.8748`** | **`2.8723`** | `3.3874` (`+0.5126` regression) |
| Unseen `global_test` Perplexity ↓ | **`17.72`** | **`17.68`** | `29.59` (`+67.0%` worse) |
| Unseen `global_test` Top-1 Acc ↑ | **`24.87%`** | **`24.80%`** | `14.47%` (`-10.40 pp`) |
| Unseen `global_test` Top-5 Acc ↑ | **`57.80%`** | **`57.80%`** | `42.87%` (`-14.93 pp`) |
| Private corpus bytes leaked | `0` (`83` wire audits) | `0` (`81` wire audits) | `0` (`83` wire audits) |

Because `sign_flip` and `scaled_noise` updates carry valid cryptographic signatures and reflect genuine local GPU steps, **proof-of-compute cannot detect them**. Only the independent held-out verification gate (`UpdateVerifier`) prevents model poisoning—achieving **100% poison rejection with 0% false positives on honest contributors**.

---

## 3. Honest Negative Regimes (Failure Boundaries)

1. **Negative Regime 1 — Extreme Sync Period ($H=512$) + Non-IID Shards (`neg_extreme_h512_non_iid`)**:
   - Workers specialize on disjoint target-symbol partitions for 512 local steps before a single outer sync. Local models drift sharply apart, collapsing unseen test loss to **`3.8174`** (`PPL 45.49`, Top-1 **`7.80%`** vs `25.87%` for balanced $H=64$), despite high compute utilization (`94.58%`) and `512x` lower traffic.
2. **Negative Regime 2 — 20% Packet Loss + 60% Mid-Round Churn (`neg_high_loss_severe_churn`)**:
   - Over a `250 ms / 10 Mbps` transcontinental link with `20%` MTU packet loss and `60%` dropout per round, **`6` of `8` sync rounds miss quorum** (`min_quorum = 3`), retransmission overhead rises to `63,572` B (`17.5%` of total wire bytes), wall time inflates **`3.7x`** (`11.74 s` vs `3.15 s`), and held-out test loss degrades to **`3.1243`** (`PPL 22.74`, Top-1 `18.40%`).
3. **Negative Regime 3 — 1-Bit Sign Compression Without Error Feedback (`neg_1bit_sign_no_error_feedback`)**:
   - Even on a clean `100 ms` WAN with zero churn and zero attackers (`48/48` honest updates), discarding magnitude structure via 1-bit sign quantization without error feedback degrades held-out test loss from `2.7994` (`fp32` / `int8_block`) to **`2.8590`** (`+0.0596` nats, `PPL 17.44` vs `16.43`, Top-1 `24.80%` vs `25.87%`).

---

## 4. What Remains Unproven

See [SPEC.md](SPEC.md#7-what-remains-unproven--open-research-gaps) for the full technical analysis. In summary, this benchmark does **not** yet prove:
- Scaling behavior on 7B–70B parameter transformer models with real multi-GPU activation checkpointing and BF16/FP8 mixed precision.
- Live internet transport dynamics (TCP/QUIC/libp2p congestion control, correlated bursty packet loss, NAT traversal, asymmetric residential uplinks).
- Formal $(\varepsilon, \delta)$-differential privacy or immunity to gradient-inversion / membership-inference attacks on transmitted pseudo-gradients.
- Resistance to adaptive targeted backdoor attacks (designed to pass `verifier_val` while firing on hidden triggers) or >50% Sybil collusion poisoning the coordinate median.
- Decentralized multi-verifier validation committees to eliminate single-coordinator trust and long-horizon validation overfitting.
