# Adversarial Federated-Learning Benchmark Report (`HIP-1350`)

- **Issue**: `zoo-apps/gym#5` (`[p1, research, training, bounty, hip-1350]`)
- **Specification**: [`SPEC.md`](./SPEC.md)
- **Seed**: `--seed 1350` (`rounds = 14`, `N = 12` clients, `lr = 0.95`, `local_steps = 3`, `batch_size = 16`)
- **Artifact**: [`results.json`](./results.json) (verified byte-identical across consecutive runs via `cmp -s` and `sha256sum`)
  - File SHA-256: `b17cafbf3744558cc65b79e4ef290d8d4f8eacfa3e78af69e5f5334607684c4f`
  - Canonical Evaluation Receipt SHA-256: `c3f4203bf0404eb2dabaab4a768e09f12bf8a91a566517c8a14952dd36e085a4`
  - Corpus Manifest SHA-256: `dc274dfaba9a379ec907e913526c6b6d65af5493a3c1c3afcc953c5fa48119a4`
- **Reproduce**:
  ```bash
  python3 -m examples.adversarial_fl.run_benchmark \
    --seed 1350 \
    --out examples/adversarial_fl/results.json
  ```

---

## 1. Experimental Setup & Integrity Checks

| Property | Verified Value |
| :--- | :--- |
| Pinned Corpus | 74 real Zoo Gym repository files (`191,140` bytes; 4 domains: `docs`, `yaml`, `python`, `infra`) |
| Disjoint Splits (`120`-char chunks) | `client_vaults_total`: `1,088` (10 majority + 2 `infra` minority specialists) · `eval_val`: `156` · `eval_clean`: `311` · `eval_trigger`: `193` |
| Model Architecture | 2-layer MLP (`96 -> 20 tanh -> 4 softmax`, `2,024` float64 params, L2 decay `1e-4`) |
| Reverse-Mode Gradient Verification | `max_abs_error = 2.361e-11` (`< 1e-6` against central finite differences, `eps = 1e-5`) |
| Flower Strategy Adapter Parity | `parity_verified = true` across all 6 defenses (`max_abs_param_diff = 5.551e-17`, runtime `stdlib_shim`) |
| Clean Oracle (`f = 0%`, `fedavg_undefended`) | **Clean Acc `79.4%`** (`docs 84.7%`, `yaml 75.9%`, `python 88.3%`, **`infra 41.7%`**) · Baseline Trigger Rate `7.8%` |

---

## 2. Threat 1: Untargeted Model Poisoning (`label_flip_and_sign_flip`)

| Defense | `f=0%` Acc (`ΔQ`) | `f=17%` Acc (`ΔQ`) / TPR | `f=33%` Acc (`ΔQ`) / `infra` / FRR / TPR | `f=50%` Acc (`ΔQ`) / TPR |
| :--- | :---: | :---: | :---: | :---: |
| `fedavg_undefended` | **79.4%** (`+0.0`) | 40.8% (`+38.6`) / 0.0% | 11.6% (`+67.8`) / 100.0%* / 0.0% / 0.0% | 11.6% (`+67.8`) / 0.0% |
| `norm_clipping` | 76.2% (`+3.2`) | 69.5% (`+10.0`) / 0.0% | 27.7% (`+51.8`) / 61.1% / 0.0% / 0.0% | 11.6% (`+67.8`) / 0.0% |
| `coordinate_median` | 73.0% (`+6.4`) | 62.7% (`+16.7`) / 100.0% | 17.0% (`+62.4`) / 97.2%* / 100.0% / 100.0% | 11.6% (`+67.8`) / 100.0% |
| `trimmed_mean` | 74.6% (`+4.8`) | 63.3% (`+16.1`) / 100.0% | 11.6% (`+67.8`) / 100.0%* / 0.0% / 26.8% | 11.6% (`+67.8`) / 8.3% |
| `multi_krum` | 74.0% (`+5.5`) | 73.0% (`+6.4`) / 100.0% | **78.8%** (`+0.6`) / 36.1% / 12.5% / 100.0% | 27.7% (`+51.8`) / 83.3% |
| `dso_hybrid_eval_cosine` | **79.1%** (`+0.3`) | **79.1%** (`+0.3`) / **96.4%** | **77.5%** (`+1.9`) / **50.0%** / 14.3% / **98.2%** | **64.3%** (`+15.1`) / **75.0%** |

*\*At `11.6%` overall accuracy, collapsed models predict class 3 (`infra`, `36/311 = 11.58%`) constantly.*

---

## 3. Threat 2: Targeted Backdoor Insertion (`targeted_trigger_backdoor`)

Target class `0` (`docs`); clean baseline false-trigger rate is `7.8%`.

| Defense | `f=17%` Acc / `infra` / **ASR** / FRR / TPR | `f=33%` Acc / `infra` / **ASR** / FRR / TPR | `f=50%` Acc / **ASR** / TPR |
| :--- | :---: | :---: | :---: |
| `fedavg_undefended` | 78.8% / 38.9% / **96.9%** / 0.0% / 0.0% | 79.4% / 30.6% / **100.0%** / 0.0% / 0.0% | 61.4% / **100.0%** / 0.0% |
| `norm_clipping` | 75.2% / 11.1% / **95.3%** / 0.0% / 0.0% | 74.3% / 2.8% / **100.0%** / 0.0% / 0.0% | 74.3% / **100.0%** / 0.0% |
| `coordinate_median` | 71.7% / 0.0% / 42.5% / 100.0% / 100.0% | 69.8% / 0.0% / 58.0% / 100.0% / 100.0% | 73.0% / **100.0%** / 100.0% |
| `trimmed_mean` | 73.3% / 0.0% / 41.5% / 10.0% / 100.0% | 73.6% / 0.0% / **97.9%** / 0.0% / 73.2% | 74.0% / **100.0%** / 0.0% |
| `multi_krum` | 73.0% / 0.0% / **8.3%** / 30.0% / 100.0% | **78.8%** / 36.1% / **7.3%** / 12.5% / 100.0% | 75.2% / **97.4%** / 60.7% |
| `dso_hybrid_eval_cosine` | **79.1%** / **47.2%** / **7.3%** / **7.1%** / **100.0%** | **78.1%** / **50.0%** / **6.2%** / 14.3% / **100.0%** | 66.6% / **97.9%** / 9.5% |

**Key takeaway**: At `f = 17%` and `f = 33%`, `dso_hybrid_eval_cosine` suppresses boosted backdoor `ASR` all the way down to clean baseline (`7.3%` and `6.2%` vs `96.9%–100.0%` undefended) with `100.0%` attacker detection while keeping `infra` accuracy at `47.2%–50.0%`. At `f = 50%`, the boosted coalition shifts the `33rd`-percentile norm statistic, causing backdoor defense breakdown across all six rules.

---

## 4. Threat 3: Stale-Update / Asynchronous Straggler Drift (`stale_update_lag`, `tau in {3, 6}`)

### 4.1 Honest Slow Stragglers (`stale_honest_straggler`, `33%` delayed honest clients)

| Defense | `tau=3` Acc (`ΔQ`) | `tau=3` `infra` | `tau=3` Overall FRR | `tau=3` **Straggler FRR** | `tau=6` Acc (`ΔQ`) | `tau=6` **Straggler FRR** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fedavg_undefended` | 79.1% (`+0.3`) | 41.7% | 0.0% | **0.0%** | 78.1% (`+1.3`) | **0.0%** |
| `norm_clipping` | 76.8% (`+2.6`) | 19.4% | 0.0% | **0.0%** | 75.6% (`+3.9`) | **0.0%** |
| `coordinate_median` | 72.3% (`+7.1`) | 0.0% | 100.0% | **100.0%** | 73.0% (`+6.4`) | **100.0%** |
| `trimmed_mean` | 74.0% (`+5.5`) | 0.0% | 16.7% | 0.0% | 73.0% (`+6.4`) | 0.0% |
| `multi_krum` | 74.6% (`+4.8`) | 0.0% | 41.7% | **42.9%** | 74.6% (`+4.8`) | **50.0%** |
| `dso_hybrid_eval_cosine` | **79.7%** (`-0.3`) | **47.2%** | **7.1%** | **12.5%** | **79.1%** (`+0.3`) | **12.5%** |

### 4.2 Malicious Stale Replays (`stale_malicious_replay`, `33%` adversaries)

| Defense | `tau=3` Acc (`ΔQ`) / `infra` / TPR | `tau=6` (`f=17%`) Acc (`ΔQ`) / TPR | `tau=6` (`f=33%`) Acc (`ΔQ`) / `infra` / TPR | `tau=6` (`f=50%`) Acc (`ΔQ`) / TPR |
| :--- | :---: | :---: | :---: | :---: |
| `fedavg_undefended` | 70.1% (`+9.3`) / 0.0% / 0.0% | 70.7% (`+8.7`) / 0.0% | 68.8% (`+10.6`) / 0.0% / 0.0% | 60.8% (`+18.6`) / 0.0% |
| `norm_clipping` | 70.4% (`+9.0`) / 0.0% / 0.0% | 70.4% (`+9.0`) / 0.0% | 68.5% (`+10.9`) / 0.0% / 0.0% | 58.8% (`+20.6`) / 0.0% |
| `coordinate_median` | 68.5% (`+10.9`) / 0.0% / 100.0% | 70.1% (`+9.3`) / 100.0% | 66.9% (`+12.5`) / 0.0% / 100.0% | 52.1% (`+27.3`) / 100.0% |
| `trimmed_mean` | 69.5% (`+10.0`) / 0.0% / 17.9% | 71.4% (`+8.0`) / 100.0% | 66.9% (`+12.5`) / 0.0% / 17.9% | 47.3% (`+32.2`) / 0.0% |
| `multi_krum` | 69.5% (`+10.0`) / 0.0% / 26.8% | 73.0% (`+6.4`) / 100.0% | 74.3% (`+5.1`) / 0.0% / 57.1% | 11.6% (`+67.8`) / 1.2% |
| `dso_hybrid_eval_cosine` | **79.7%** (`-0.3`) / **47.2%** / **60.7%** | **79.1%** (`+0.3`) / **89.3%** | **77.8%** (`+1.6`) / **47.2%** / **87.5%** | **68.2%** (`+11.3`) / **33.3%** |

**Key takeaway**: `multi_krum` falsely rejects `42.9%–50.0%` of legitimate slow stragglers while failing to protect `infra` (`0.0%`). `dso_hybrid_eval_cosine` uses staleness-weighted decay + validation gating to keep honest straggler FRR low (`12.5%`, `79.1%–79.7%` clean acc) while catching `87.5%–89.3%` of malicious `tau = 6` stale replays.

---

## 5. Threat 4: Coordinated Collusion Cartel (`colluding_inner_product_cartel`, `cos = 0.999`)

Colluders craft near-collinear shifts (`mean_cartel_cosine = 0.999 > 0.99`) scaled to `0.92 * median_honest_norm` to bypass L2 clipping and form an artificially dense cluster that fools Multi-Krum.

| Defense | `f=17%` Acc (`ΔQ`) / FRR / TPR | `f=33%` Acc (`ΔQ`) / FRR / **TPR** | `f=50%` Acc (`ΔQ`) / FRR / **TPR** |
| :--- | :---: | :---: | :---: |
| `fedavg_undefended` | 67.8% (`+11.6`) / 0.0% / 0.0% | 9.0% (`+70.4`) / 0.0% / 0.0% | 17.4% (`+62.1`) / 0.0% / 0.0% |
| `norm_clipping` | 67.8% (`+11.6`) / 0.0% / 0.0% | 28.9% (`+50.5`) / 0.0% / 0.0% | 17.4% (`+62.1`) / 0.0% / 0.0% |
| `coordinate_median` | 67.5% (`+11.9`) / 100.0% / 100.0% | 39.5% (`+39.9`) / 100.0% / 100.0% | 17.4% (`+62.1`) / 100.0% / 100.0% |
| `trimmed_mean` | 69.1% (`+10.3`) / 0.0% / 100.0% | 37.6% (`+41.8`) / 0.0% / 14.3% | 17.4% (`+62.1`) / 0.0% / 0.0% |
| `multi_krum` | 74.3% (`+5.1`) / 41.4% / 42.9% | 37.9% (`+41.5`) / **54.5%** / **16.1%** | 17.4% (`+62.1`) / **83.3%** / **0.0%** |
| `dso_hybrid_eval_cosine` | **78.5%** (`+1.0`) / **7.1%** / **85.7%** | **68.5%** (`+10.9`) / 19.6% / **92.9%** | **76.2%** (`+3.2`) / 20.2% / **78.6%** |

**Key takeaway**: At `f = 33%`, `multi_krum` selects the tight `cos = 0.999` cartel over spread-out Non-IID honest workers (`TPR = 16.1%`, `FRR = 54.5%`, clean accuracy collapses to `37.9%`). `dso_hybrid_eval_cosine` detects the collinear cartel via cosine clustering + validation gating (`TPR = 92.9%`, `68.5%` clean accuracy).

---

## 6. Three Explicit Honest Negative Regimes

### 6.1 Negative Regime 1: Non-IID Honest Minority False Rejection (`neg_non_iid_honest_frr`, `f = 0%`)

With **zero attackers**, the 2 honest `infra` specialists naturally have negative cosine similarity (`~ -0.48`) against the 10 majority clients:

| Defense | Clean Acc | Quality Loss (`ΔQ`) | Minority `infra` Acc | Honest FRR | **Minority-Honest Accept** | Honest Coord Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fedavg_undefended` (Oracle) | **79.4%** | `+0.0` | **41.7%** | 0.0% | **100.0%** | 100.0% |
| `norm_clipping` | 76.2% | `+3.2` | 13.9% | 0.0% | 100.0% | 100.0% |
| `coordinate_median` | 73.0% | `+6.4` | **0.0%** | **100.0%** | **0.0%** | **16.7%** |
| `trimmed_mean` (`beta=0.2`) | 74.6% | `+4.8` | **0.0%** | **17.3%** | **0.0%** | 66.7% |
| `multi_krum` | 74.0% | `+5.5` | **0.0%** | **41.7%** | **0.0%** | 58.3% |
| `dso_hybrid_eval_cosine` | **79.1%** | **`+0.3`** | **41.7%** | **7.7%** | **92.9%** | **92.3%** |

**Finding**: `coordinate_median`, `trimmed_mean`, and `multi_krum` completely erase the honest minority domain (`0.0%` minority acceptance, `0.0%` `infra` accuracy) even when zero attackers exist. `dso_hybrid_eval_cosine` preserves `92.9%` minority acceptance and matches Oracle `infra` accuracy (`41.7%`, only `+0.3` pp overall quality loss).

### 6.2 Negative Regime 2: Stealth Backdoor Unmonitored Trigger Blindness (`neg_unmonitored_backdoor_blindness`)

Under `stealth_unmonitored_backdoor`, attackers compute a genuine clean gradient on clean features and graft exact trigger gradients exclusively onto the unmonitored trigger feature columns (`W1[:, 94:96]`), preserving `eval_val` clean loss bit-for-bit (`< 1e-12` diff):

| Defense | `f=17%` Clean Acc / **ASR** / TPR | `f=33%` Clean Acc / `infra` / **ASR** / TPR |
| :--- | :---: | :---: |
| `fedavg_undefended` | 79.4% / **66.8%** / 0.0% | 79.4% / 41.7% / **99.0%** / 0.0% |
| `norm_clipping` | 76.2% / **63.7%** / 0.0% | 76.2% / 13.9% / **97.9%** / 0.0% |
| `coordinate_median` | 73.0% / 17.1% / 100.0% | 73.0% / 0.0% / 17.1% / 100.0% |
| `trimmed_mean` | 74.6% / 14.0% / 3.6% | 74.6% / 0.0% / **90.7%** / 1.8% |
| `multi_krum` | 74.0% / **77.7%** / 28.6% | 74.0% / 0.0% / **96.9%** / 39.3% |
| `dso_hybrid_eval_cosine` | 79.1% / **66.8%** / **3.6%** | 79.1% / 41.7% / **97.9%** / **7.1%** |

**Honest Negative Finding**: Even `dso_hybrid_eval_cosine` passes the stealth backdoor updates (`TPR = 3.6%` at `f = 17%`, `7.1%` at `f = 33%`), keeping clean accuracy at `79.1%` while suffering **`66.8%`–`97.9%` `ASR`**! This proves empirically that held-out clean validation gates alone cannot defeat unmonitored targeted triggers; complementary defenses (e.g., DP-SGD noise, randomized trigger auditing, or post-training pruning/unlearning) are required.

### 6.3 Negative Regime 3: 50% Collusion Majority Breakdown (`neg_fifty_percent_collusion_breakdown`)

At `f = 50%` (`6/12` colluding workers), the classical `f < N/2` Byzantine Fault Tolerance bound is breached:

| Defense (at `f = 50%` Collusion Cartel) | Clean Acc | Quality Loss (`ΔQ`) | Honest FRR | Attacker TPR | Quorum Stall Rounds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fedavg_undefended` | **17.4%** | `+62.1` | 0.0% | 0.0% | 0 / 14 |
| `norm_clipping` | **17.4%** | `+62.1` | 0.0% | 0.0% | 0 / 14 |
| `coordinate_median` | **17.4%** | `+62.1` | 100.0% | 100.0% | 0 / 14 |
| `trimmed_mean` | **17.4%** | `+62.1` | 0.0% | **0.0%** | 0 / 14 |
| `multi_krum` | **17.4%** | `+62.1` | **83.3%** | **0.0%** | 0 / 14 |
| `dso_hybrid_eval_cosine` (relaxed quorum `>= 2`) | 76.2% | `+3.2` | **20.2%** | 78.6% | 0 / 14 |
| `dso_hybrid_eval_cosine` (**strict BFT quorum `> N/2`**) | **31.8%** | **`+47.6`** | **100.0%** | 100.0% | **14 / 14 (100% stall)** |

**Honest Negative Finding**: Every geometric Byzantine rule (`coordinate_median`, `trimmed_mean`, `multi_krum`) collapses identically to `17.4%`. While `dso_hybrid_eval_cosine` can continue learning with a relaxed quorum (`>= 2`) by leaning on its external coordinator validation set (at the cost of higher honest FRR `20.2%` and 50% throughput loss), enforcing a strict Byzantine majority quorum (`> N/2`) causes **14/14 stalled rounds** (total liveness collapse, `31.8%` accuracy).

---

## 7. What Remains Unproven

1. **7B-70B LLM Federated Fine-Tuning**: Results are measured on a `2,024`-parameter 2-layer MLP token classifier; high-dimensional transformer attention dynamics, LoRA/QLoRA adapter interference, and natural-language semantic backdoors at 7B-70B scale remain unproven.
2. **Live Flower gRPC/mTLS Transport**: `flower_adapter.py` verifies exact `Strategy` contract parity offline via pure-stdlib wire shims (`max_abs_param_diff = 5.551e-17`); live multi-node gRPC/mTLS transport, certificate rotation, and real WAN packet loss are not exercised.
3. **DP-SGD Certified Backdoor Robustness**: No `(epsilon, delta)`-differential privacy noise or certified randomized-smoothing robustness bounds are established.
4. **Adaptive White-Box RL Attackers**: Online reinforcement-learning adversaries jointly adapting to dynamic/rotating coordinator validation sets over long horizons remain out of scope.
5. **Sybil Identity-Cost Economics**: Adversary fractions are fixed exogenously; stake-slashing, reputation bonding curves, and economic Sybil costs are not modeled.
