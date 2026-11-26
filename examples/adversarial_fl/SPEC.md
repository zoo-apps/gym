# HIP-1350: Adversarial Federated-Learning Benchmark Specification

- **Document ID**: `HIP-1350`
- **Tracking Issue**: `zoo-apps/gym#5` (`[p1, research, training, bounty, hip-1350]`)
- **Program**: Zoo x Hanzo Open Research Program
- **Status**: Reference Implementation & Reproducible Benchmark (`v0.1.0`)
- **License**: Apache-2.0 (production-bound source code; zero third-party runtime dependencies)
- **References**: Flower (`flwr.server.strategy.Strategy`), Zoo DSO (Decentralized Semantic Optimization) research, Yin et al. (ICML 2018), Blanchard et al. (NeurIPS 2017), Bagdasaryan et al. (AISTATS 2020)

---

## 1. Purpose & Scope

Byzantine-robust federated aggregation rules are routinely evaluated only on IID partitions where honest gradients cluster tightly around a single mean. In realistic multi-institution federations (Non-IID vaults with minority domain specialists and heterogeneous network latency), purely geometric outlier filters introduce severe **false-rejection rates (FRR)** and **minority-domain quality loss** even when zero attackers are present (`f = 0%`). Conversely, validation-gated defenses can protect minority specialists and defeat poisoning, stale replays, and collusion cartels, yet remain fundamentally blind to targeted backdoors whose triggers are absent from the coordinator's clean validation set.

`HIP-1350` specifies a clean-room, pure-stdlib, byte-reproducible reference benchmark that jointly measures:
1. **Four mandatory threat classes** across adversary fractions `f in {0%, 17% (2/12), 33% (4/12), 50% (6/12)}`.
2. **Six aggregation defenses** evaluated head-to-head on identical data partitions, initialization, and compute budgets.
3. **Explicit safety vs. utility trade-offs**: Clean Accuracy, Per-Domain & Minority Accuracy, Quality Loss vs. Clean Oracle (`Delta Acc`), Attack Success Rate (`ASR`), Honest False-Rejection Rate (`FRR`), Honest Straggler FRR, Minority-Honest Acceptance Rate, and Attacker Detection Rate (`TPR`).
4. **Three explicit honest negative regimes** exposing the failure boundaries of both geometric and validation-gated defenses.

---

## 2. Pinned Corpus & Strictly Disjoint Non-IID Federation (`corpus.py`)

### 2.1 Grounded Repository Corpus
All training, validation, clean-test, and backdoor-trigger examples are extracted deterministically from **74 real Apache-2.0 files** inside this repository (`191,140` raw bytes), pinned by repository-relative POSIX path and per-file SHA-256 digest into a canonical `corpus_manifest`:

| Class ID | Domain | Source Files | Role |
| :---: | :--- | :---: | :--- |
| `0` | `docs` | 12 Markdown / Quarto technical docs (`README.md`, `LLM.md`, `FAQS.md`, `docs/*.qmd`) | Majority domain (also backdoor target class) |
| `1` | `yaml` | 28 YAML model training recipes (`examples/*/*.yaml`) | Majority domain |
| `2` | `python` | 10 CLI and trainer Python modules (`src/gym/**/*.py`) | Majority domain |
| `3` | `infra` | 24 DeepSpeed JSON, Dockerfiles, K8s, Compose, and CI scripts | **Explicit minority specialist domain** |

### 2.2 Disjoint Split & Non-IID Vault Partitioning
Every source file is sliced into non-overlapping `120`-character chunks (`TextChunk`) with unique `chunk_id`s. Per domain, chunks are deterministically shuffled with `--seed 1350` and partitioned without overlap:
- **`eval_clean` (`20%`)**: Strictly held-out test set (`311` chunks across all 4 domains). Never touched during vocabulary construction, client training, or coordinator validation gating.
- **`eval_val` (`10%`)**: Independent domain-stratified clean validation set (`156` chunks across all 4 domains) held by the coordinator for `dso_hybrid_eval_cosine` probing. Contains **zero** triggered samples.
- **`eval_trigger`**: Constructed exclusively from all non-target-class (`label != 0`) examples in `eval_clean` (`193` chunks) by appending the out-of-corpus rare trigger sequence `("zxq_beacon", "qzv_anchor")` (asserted to occur `0` times across all 74 raw source files) and assigning `target_class = 0` (`docs`).
- **Training Pool (`70%`) -> `N = 12` Strictly Disjoint Client Vaults**:
  - **Clients `0..3` (`majority_docs`)**: `docs`-dominant (~79% `docs`, ~10% `yaml`, ~11% `python`, **`0% infra`**).
  - **Clients `4..6` (`majority_yaml`)**: `yaml`-dominant (~59% `yaml`, ~20% `docs`, ~21% `python`, **`0% infra`**).
  - **Clients `7..9` (`majority_python`)**: `python`-dominant (~80% `python`, ~12% `docs`, ~8% `yaml`, **`0% infra`**).
  - **Clients `10..11` (`minority_infra_specialist`)**: **`100% infra`** (strictly disjoint halves of the `infra` training pool). These two clients remain honest in all adversarial regimes so minority-domain false rejection is continuously audited.

### 2.3 Leak-Free Feature Encoding
A 96-token vocabulary is built strictly from the 70% training pool by round-robin selection of top domain-discriminative tokens (`94` clean tokens balanced across all 4 domains) plus `2` reserved out-of-corpus trigger token slots (`indices 94, 95`). Every chunk is tokenized with a deterministic regex and encoded as an L2-normalized log-TF sparse vector `x in R^96`.

---

## 3. Model & Exact Analytical Gradients (`model.py`)

A pure-stdlib 2-layer MLP classifier is trained in float64:
- **Layer 1**: `h = tanh(W1 x + b1)` with `W1 in R^{20 x 96}`, `b1 in R^{20}`.
- **Layer 2**: `z = W2 h + b2` with `W2 in R^{4 x 20}`, `b2 in R^{4}`.
- **Objective**: Mean softmax cross-entropy loss plus L2 weight decay (`lambda = 1e-4` on `W1, W2`).
- **Total Parameters**: `2,024` flat float64 scalars (`W1: 1,920`, `b1: 20`, `W2: 80`, `b2: 4`).
- **Gradient Verification**: Reverse-mode analytical gradients are verified against central finite differences (`eps = 1e-5`) on every benchmark run and unit test, requiring `max_abs_err < 1e-6` (observed `< 1e-10`).

---

## 4. Mandatory Threat Scenarios (`attacks.py`)

Deterministic adversary indices preserve minority specialists `(10, 11)` as honest:
- `f = 0%` (`0/12`): `()`
- `f = 17%` (`2/12`): `(0, 4)`
- `f = 33%` (`4/12`): `(0, 4, 7, 1)`
- `f = 50%` (`6/12`): `(0, 4, 7, 1, 5, 8)`

1. **Untargeted Model Poisoning (`label_flip_and_sign_flip`)**:
   Adversaries train on cyclically label-flipped local vaults (`(y + 1) mod 4`) and combine the resulting semantic poisoning shift with sign-flipped scaled honest gradients (`-2.6 * d_hon + 1.4 * d_lf`) to collapse global clean accuracy.
2. **Targeted Backdoor Insertion (`targeted_trigger_backdoor`)**:
   Adversaries inject the rare trigger phrase into `50%`-ratio non-target local chunks mapped to `target_class = 0` (`docs`), train locally, and apply a Bagdasaryan-style model-replacement boost (`3.2x`) to survive averaging while keeping clean accuracy high.
3. **Stale-Update / Asynchronous Straggler Drift (`stale_update_lag`, `tau in {3, 6}`)**:
   Evaluated in two explicit sub-regimes against persisted coordinator checkpoint history `theta_{max(0, t - tau)}`:
   - `stale_honest_straggler`: Legitimate slow clients (`is_malicious=False`) submitting honest gradients computed on `theta_{t - tau}`. Any rejection counts directly toward `FRR` and `honest_straggler_frr`.
   - `stale_malicious_replay`: Adversaries (`is_malicious=True`) replaying amplified backward parameter drift `1.8 * (theta_{t - tau} - theta_t) + 0.8 * d_stale` to stall or revert convergence.
4. **Coordinated Collusion Cartel (`colluding_inner_product_cartel`)**:
   `k` colluding workers coordinate a shared inner-product-manipulation direction `-mean(d_honest)` plus small deterministic jitter so pairwise cosine similarity strictly satisfies `cos(d_i, d_j) > 0.99`, while rescaling every cartel vector to `0.92 * median_honest_norm` — strictly inside naive L2 clipping bounds and tightly clustered to deceive Euclidean Multi-Krum.

---

## 5. Evaluated Defenses & Explicit Metrics (`defenses.py`)

1. **`fedavg_undefended`**: Uniform mean of all `N` client deltas.
2. **`norm_clipping`**: Clips each delta to `||d_i||_2 <= 1.50` before uniform averaging.
3. **`coordinate_median`** (Yin et al., 2018): Coordinate-wise median across `N = 12` clients.
4. **`trimmed_mean`** (Yin et al., 2018, `beta = 0.20`): Trims lowest `2` and highest `2` values per coordinate, averaging the middle `8`.
5. **`multi_krum`** (Blanchard et al., 2017): Assumes `f_assumed = 3`, scores each client by squared Euclidean distance to its `N - f_assumed - 2 = 7` nearest neighbors, and averages the `m = 7` lowest-scoring clients.
6. **`dso_hybrid_eval_cosine`** (Zoo DSO-inspired hybrid defense):
   - **Staleness-Weighted Decay**: Weights accepted updates by `w_stale = 1 / (1 + 0.35 * tau)` (hard reject only if `tau > 8`), preserving legitimate slow stragglers.
   - **Lower-Order Norm Screening**: Uses the `33rd`-percentile norm `q33` as reference (`soft_ceil = 2.50 * q33`, `hard_ceil = 2.95 * q33`), resisting up to `33%` norm-boosted attacks without falsely rejecting honest minority specialists.
   - **Pairwise Cosine Collusion Discounting**: Detects tight `cos >= 0.992` clusters (`cluster_size >= 2`), rejecting clusters that lack validation benefit and discounting surviving clusters of size `>= 3` by `0.15 / cluster_size`.
   - **Domain-Stratified Held-Out Validation Gate (`eval_val`)**: Probes `theta_t + 0.08 * d_i` (matching the `~1/N` marginal step) on `eval_val`, computing both micro-averaged loss change `delta_overall` and domain-stratified **macro-averaged** loss change `delta_macro = mean_d(loss_d_probe - loss_d_base)`. Accepts an update if `delta_overall <= 0.004` OR (for fresh updates `tau == 0`) `delta_macro <= 0.004` with `delta_overall <= 0.08`, explicitly protecting minority specialists whose stratum is small in micro loss while blocking stale backward replays and harmful cartels.

### Formal Metric Definitions
- **Clean Accuracy (`%`)**: Top-1 accuracy on `eval_clean`.
- **Quality Loss vs. Clean Oracle (`Delta Acc`)**: `oracle_clean_accuracy - defense_clean_accuracy` (percentage points; positive indicates degradation).
- **Attack Success Rate (`ASR %`)**: Fraction of `eval_trigger` non-target examples predicted as `target_class = 0` (`docs`).
- **False-Rejection Rate (`FRR %`)**: Fraction of `(round, honest_client)` submissions rejected (`accepted == False`). For vector-gating rules (`multi_krum`, `dso_hybrid_eval_cosine`), `accepted` is explicit inclusion; for coordinate-wise rules (`coordinate_median`, `trimmed_mean`), a client is counted as effectively rejected if `< 50%` of its `2,024` coordinates survive trimming/selection (`coordinate_retention < 0.50`), and exact `mean_honest_coordinate_retention` is additionally reported.
- **Minority-Honest Acceptance Rate (`%`)**: Acceptance rate specifically over honest minority specialists `(clients 10, 11)`.
- **Honest Straggler FRR (`%`)**: Rejection rate specifically over honest delayed clients in `stale_honest_straggler`.
- **Attacker Detection Rate (`TPR %`)**: Fraction of `(round, malicious_client)` submissions rejected.

---

## 6. Explicit Honest Negative Regimes

1. **`neg_non_iid_honest_frr`**: At `f = 0%` (zero attackers), geometric Byzantine rules (`coordinate_median`, `trimmed_mean`, `multi_krum`) misclassify honest `infra` specialists (whose gradients have negative cosine similarity `~ -0.48` to majority clients) as outliers, driving minority acceptance to `0%` and `infra` accuracy to `0%`.
2. **`neg_unmonitored_backdoor_blindness`**: Under `stealth_unmonitored_backdoor`, attackers compute a benign clean update `d_clean` and add exact analytical gradients restricted exclusively to the unmonitored trigger feature columns `W1[:, 94:96]`. Because `eval_val` contains zero trigger tokens, clean validation loss is preserved bit-for-bit (`< 1e-12` diff); `dso_hybrid_eval_cosine` accepts the update and suffers high `ASR`, proving clean held-out validation gates alone cannot eliminate unmonitored targeted triggers.
3. **`neg_fifty_percent_collusion_breakdown`**: At `f = 50%` (`6/12` colluding workers), classical BFT limits (`f < N/2`) are violated: `coordinate_median`, `trimmed_mean`, and `multi_krum` collapse completely. Furthermore, enforcing strict BFT majority quorum (`> N/2` accepted clients) inside `dso_hybrid_eval_cosine` causes `100%` quorum stall rounds (total liveness breakdown).

---

## 7. Flower Compatibility Adapter (`flower_adapter.py`)

`AdversarialDefenseStrategy` subclasses `flwr.server.strategy.Strategy` whenever `flwr` is installed and transparently falls back to pure-stdlib protocol shims (`StdlibParameters`, `StdlibFitRes`, `StdlibClientProxy`, `StdlibEvaluateRes`) in offline environments. Wire serialization packs float64 tensors deterministically (`struct.pack("<2024d", ...)`). Every benchmark run verifies exact parity (`max_abs_param_diff < 1e-12`) across all six defenses between `AdversarialDefenseStrategy.aggregate_fit` and native `aggregate_submissions`.

---

## 8. Constitutional & Governance Separation

- Every `results.json` artifact embeds a deterministic `independent_evaluation_receipt` containing `canonical_payload_sha256` (SHA-256 over canonical sorted JSON excluding wall-clock timestamps and host paths) and `corpus_manifest_sha256`.
- Scientific claims in this specification are bounded strictly to what can be independently reproduced via `python3 -m examples.adversarial_fl.run_benchmark --seed 1350`.
- **Governance Boundary**: AI agents may author code, execute reproducible benchmarks, flag security limitations, and prepare technical recommendations, but must never redefine governance voting rules, authorize treasury disbursements, or self-certify production acceptance. Final acceptance requires independent maintainer verification.

---

## 9. What Remains Unproven

1. **7B-70B LLM Federated Fine-Tuning**: This reference suite trains a 2,024-parameter 2-layer MLP token classifier on CPU. High-dimensional transformer gradient geometry, LoRA/QLoRA low-rank adapter interference, and multi-token semantic backdoors at 7B-70B scale are not proven here.
2. **Live Flower gRPC/mTLS Transport**: `flower_adapter.py` verifies Strategy-interface parity offline via a pure-stdlib wire shim; live multi-host Flower gRPC/mTLS networking, peer authentication, TLS revocation, and asynchronous WAN jitter remain untested.
3. **DP-SGD Certified Backdoor Robustness**: This benchmark does not provide `(epsilon, delta)`-differential privacy accounting or certified randomized-smoothing robustness bounds against arbitrary trigger norms.
4. **Adaptive White-Box RL Attackers**: Although `neg_unmonitored_backdoor_blindness` demonstrates a strong orthogonal white-box stealth backdoor, online reinforcement-learning attackers jointly adapting to rotating validation sets over hundreds of rounds remain unproven.
5. **Sybil Identity-Cost Economics**: Adversary fractions `f in {0%, 17%, 33%, 50%}` are fixed exogenously; stake-slashing, proof-of-contribution bonding, and economic Sybil-resistance mechanisms are outside the scope of this specification.
