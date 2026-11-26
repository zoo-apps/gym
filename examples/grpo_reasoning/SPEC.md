# HIP-1350 Specification: Open Reasoning RL / GRPO Training Pipeline

- **Identifier**: `HIP-1350`
- **Tracking Issue**: `zoo-apps/gym#7` (*Open reasoning RL / GRPO training pipeline*)
- **Spec Version**: `0.1.0`
- **Source Code License**: `Apache-2.0`
- **Maintained External Trainers**: `huggingface/trl` (`GRPOConfig`/`GRPOTrainer`), `volcengine/verl` (Ray GRPO)
- **Historical Reference Policy**: `huggingface/open-r1` is explicitly **excluded** (upstream states it is no longer maintained).

---

## 1. Purpose & Scope

This specification defines a clean-room, byte-reproducible, openly licensed Group Relative Policy Optimization (GRPO) reasoning pipeline consisting of:

1. A **SHA-256 pinned, zero-leakage verifiable reasoning benchmark** (`dataset.py`) spanning three exact-verifier domains (`math_reasoning`, `symbolic_logic`, `code_trace`) with strictly disjoint `train`, `val`, and independent `held_out_test` splits (including longer-chain out-of-distribution generalization problems).
2. **Stateless, deterministic, cryptographically auditable reward functions** (`rewards.py`) with explicit anti-reward-hacking detectors (`RewardAuditTrail`) — strictly excluding LLM-as-a-judge.
3. A **differentiable pinned base policy and exact GRPO optimizer** (`grpo.py`) implementing group-relative advantage normalization, PPO-style clipped surrogate ratios, and the unbiased non-negative Schulman `k3` KL penalty, with exact analytical gradients verified against central finite differences (`< 1e-6` relative error across all 491 parameters).
4. **Schema-validated production recipes** (`recipes/trl_grpo_openthoughts.yaml`, `recipes/verl_grpo_openthoughts.yaml`) aligned with Zoo Gym's TRL schema (`src/gym/utils/schemas/trl.py`) and `verl`'s Ray GRPO config using pinned `Qwen/Qwen2.5-1.5B-Instruct` and `open-thoughts/OpenThoughts-114k` references.
5. A **machine-checkable Independent Evaluation Receipt** (`run_benchmark.py`, `results.json`) enabling any external reviewer to reproduce and verify every hash, metric, and negative control without trusting self-reported agent claims.

---

## 2. Formal GRPO Objective & Exact Gradients

Let $\pi_\theta$ be the trainable autoregressive reasoning policy, $\pi_{\theta_{\text{old}}}$ the rollout behavior policy snapshot at the start of a batch, and $\pi_{\text{ref}}$ the frozen pinned base checkpoint (`weight_sha256` pinned at initialization).

For each prompt $q$ in a batch, we sample a group of $G$ trajectories $\{o_1, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$, render each discrete trajectory deterministically to text, and grade the raw text with the black-box deterministic verifier $r_i = R(q, \text{text}(o_i, q)) \in [0, 1]$.

### 2.1 Group-Relative Advantage Normalization

$$\mu_r = \frac{1}{G}\sum_{j=1}^G r_j, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_r)^2}$$

$$A_i = \begin{cases} 0 & \text{if } \sigma_r < 10^{-12} \text{ (zero-variance group)} \\ \dfrac{r_i - \mu_r}{\sigma_r + \varepsilon_{\text{adv}}} & \text{otherwise} \end{cases}$$

Zero-variance groups (all rollouts receive identical reward) contribute zero surrogate gradient and are tracked via `zero_variance_group_rate`.

### 2.2 Clipped Surrogate + Unbiased Schulman `k3` KL Penalty

For each decision step $t \in \{1, \dots, |o_i|\}$ with importance ratio $\rho_{i,t}(\theta) = \exp\!\big(\log \pi_\theta(a_{i,t} \mid s_{i,t}) - \log \pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})\big)$ and log-ratio $\Delta_{i,t} = \log \pi_{\text{ref}}(a_{i,t} \mid s_{i,t}) - \log \pi_\theta(a_{i,t} \mid s_{i,t})$:

$$D_{\text{KL}}^{k3}(a_{i,t}) = \exp(\Delta_{i,t}) - \Delta_{i,t} - 1 \ge 0$$

$$J_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\!\big(\rho_{i,t} A_i,\; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\big) - \beta \, D_{\text{KL}}^{k3}(a_{i,t}) \right]$$

The exact analytical derivative with respect to $\log \pi_\theta(a_{i,t} \mid s_{i,t})$ is:

$$\frac{\partial D_{\text{KL}}^{k3}}{\partial \log \pi_\theta} = 1 - \exp\!\big(\log \pi_{\text{ref}} - \log \pi_\theta\big)$$

$$\frac{\partial J_{\text{GRPO}}}{\partial \log \pi_\theta(a_{i,t})} = \frac{1}{G \cdot |o_i|} \left( \mathbb{I}[\text{unclipped}]\,\rho_{i,t} A_i - \beta \big(1 - \exp(\Delta_{i,t})\big) \right)$$

Exact backpropagation through `log_softmax`, head projections, condition embeddings, and the `tanh` hidden encoder is verified against central finite differences (`eps = 1e-5`) across all 491 parameters with worst-case relative error `< 1e-6` (`8.79e-7` observed in `tests/test_grpo_reasoning.py`).

---

## 3. Verifiable Reasoning Benchmark & Zero-Leakage Protocol

### 3.1 Domains & Exact Solvers

| Domain | Reasoning Task | Canonical `<answer>` Schema |
|---|---|---|
| `math_reasoning` | Multi-step modular arithmetic (`MUL_ADD`, `SQUARE_SUB`, `MOD_SHIFT`) | Exact integer (`^-?\d+$`) |
| `symbolic_logic` | Multi-step Boolean gate chains (`AND_NOT`, `XOR_OR`, `IMPLIES_XOR`) | `TRUE` or `FALSE` |
| `code_trace` | Two-register state machine (`ADD_R0`, `SUB_R1`, `MUL_MOD`, `IF_EVEN_ADD`, `SWAP_INC`) | `r0=<int>,r1=<int>` |

### 3.2 Split Partitioning & Disjointness Invariants

| Split | Count | Depths | Modulus / Var Pool | Purpose |
|---|---|---|---|---|
| `train` | 180 (60/domain) | 2–3 | Math mod `{13,17,19,23,29}`; Logic vars `A..D`; Code mod `{11,13,17,19}` | Policy optimization |
| `val` | 60 (20/domain) | 2–3 | Math mod `{31,37,41}`; Logic vars `P..S`; Code mod `{23,29}` | Development checks |
| `held_out_test` (ID) | 72 (24/domain) | 2–3 | Math mod `{43,47,53,59,61}`; Logic vars `X..W`; Code mod `{31,37,41,43}` | Independent held-out ID eval |
| `held_out_test` (OOD) | 48 (16/domain) | 4–5 | Same held-out disjoint pools, longer unseen chains | Generalization & KL-collapse test |

Every split build asserts **zero intersection** across `train`, `val`, and `held_out_test` for:
- `prompt_sha256`
- `canonical_fingerprint` (SHA-256 of canonical `(domain, init, ops)`)
- `problem_id`

---

## 4. Auditable Verifier & Anti-Reward-Hacking Threat Model

`rewards.evaluate_completion(problem, completion)` grades **raw completion text only** (`total_reward in [0.0, 1.0]`):

1. **`format_reward` (`[0.0, 0.25]`)**: Requires exactly one `<think>...</think>` followed by one `<answer>...</answer>`, optional whitespace only outside tags, non-empty sections, and at least one `step_k: <expr> => <state>` entry.
2. **`correctness_reward` (`[0.0, 0.60]`)**: Strict canonical regex + exact match against `problem.canonical_answer`, plus anti-guessing verification (requires at least one verified derivation step in `<think>` and final step matching `<answer>`).
3. **`reasoning_coherence_reward` (`[0.0, 0.15]`)**: Independently verifies every intermediate `step_k` state in `<think>` against ground-truth transitions (`0.15 * verified_steps / depth`), penalizing spurious extra steps or answer mismatch.

### 4.1 Explicit Anti-Hacking Guards (`RewardAuditTrail`)

| Flag | Attack Vector Caught | Consequence |
|---|---|---|
| `DUPLICATE_TAG` | Multiple `<think>` or `<answer>` tags | `0.0` format & correctness |
| `TRAILING_GARBAGE` | Unstructured text outside XML envelope | `0.0` format & correctness |
| `LENGTH_PADDING_HACK` | Excessive length (`> 1200` chars) or repeated filler words | `0.0` total reward |
| `ANSWER_STUFFING` | Disjunctions (`or`, `/`, `|`, `;`) or multiple answers in `<answer>` | `0.0` correctness |
| `UNSUPPORTED_GUESS` | Correct guess with zero verified reasoning steps in `<think>` | Rejected as exact pass; capped |
| `STEP_MISMATCH` | `<answer>` disagrees with final derived step in `<think>` | Rejected as exact pass; penalized |

Every evaluation produces an immutable `audit_hash = SHA-256(canonical_record)`, and `RewardAuditTrail` maintains a tamper-evident hash chain `H_t = SHA-256(H_{t-1} || audit_hash_t)`.

---

## 5. Independent Evaluation Receipt Protocol

`run_benchmark.py` emits an `independent_evaluation_receipt` block inside `results.json` containing:
- `dataset_manifest_sha256`, `split_sha256`, and `zero_leakage_verified`
- `base_model_sha256` and `verifier_version`
- Per-regime `final_weight_sha256`, `held_out_audit_chain_sha256`, `pass_at_1`, `pass_at_4`, `ood_pass_at_1`, and `mean_exact_kl_nats`
- `recipes_all_valid`
- `receipt_sha256 = SHA-256(canonical_json(receipt_body))`

Any external reviewer can verify byte-identical reproduction via:

```bash
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/r1.json --quiet
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/r2.json --quiet
cmp -s /tmp/r1.json /tmp/r2.json && sha256sum /tmp/r1.json
```

---

## 6. Licensing & Governance Separation

- **Production-bound source code** under `examples/grpo_reasoning/` is licensed **Apache-2.0**.
- **External artifacts** (datasets such as `open-thoughts/OpenThoughts-114k`, reasoning traces, model weights such as `Qwen/Qwen2.5-1.5B-Instruct` or Zen checkpoints, and external SDKs such as `trl` and `verl`) are explicitly tracked in [`LICENSE_AUDIT.json`](LICENSE_AUDIT.json) with `separate_license_review_required: true`. Authorized human maintainers must complete a separate license and provenance review before production deployment or weight redistribution.
- **Governance boundary**: Scientific claims require independently checkable evidence. AI agents may recommend, flag conflicts, estimate value, and prepare implementation plans, but **must not** redefine voting rules, authorize treasury spending, or declare their own experiments valid. All benchmark receipts in this package are provided strictly for independent human review and external reproduction.
