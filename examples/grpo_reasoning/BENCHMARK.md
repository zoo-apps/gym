# HIP-1350 Reproducible Benchmark Report

- **Specification**: [`SPEC.md`](SPEC.md) (`HIP-1350 v0.1.0`, `zoo-apps/gym#7`)
- **Master Seed**: `1350`
- **Matched Compute Budget**: `2,880` rollouts / offline traces per trained regime (`96` rollouts per batch across all trained regimes)
- **Held-Out Evaluation Split**: `120` strictly disjoint problems (`72` in-distribution depth 2–3 + `48` out-of-distribution depth 4–5; `8` stochastic rollouts per problem + greedy decoding = `960` stochastic evaluations per regime)
- **Verifier**: `hip1350-verifier-v1.0.0` (100% deterministic, zero LLM-as-a-judge)

---

## 1. Exact Reproduction Commands

All code uses **pure Python >= 3.10 stdlib** with zero external runtime dependencies:

```bash
# 1. Run linter (0 errors)
ruff check examples/grpo_reasoning/

# 2. Run full offline unit, gradient, leakage & receipt test suite (0 warnings)
python3 -W error -m unittest discover -s examples/grpo_reasoning/tests -v

# 3. Reproduce benchmark twice and verify byte-identical JSON output
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/grpo_run1.json
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/grpo_run2.json --quiet
cmp -s /tmp/grpo_run1.json /tmp/grpo_run2.json && sha256sum /tmp/grpo_run1.json examples/grpo_reasoning/results.json
```

---

## 2. Independent Held-Out Results (`--seed 1350`)

### 2.1 Primary Regimes vs. Honest Negative Regimes

| Regime | Category | Held-Out `pass@1` | Held-Out `pass@4` | ID `pass@1` | OOD `pass@1` | OOD `pass@4` | Format Valid | Verified Step Acc | Naive Proxy | Hacking Flag Rate | Exact KL (nats) | Entropy (nats) | Weight L2 Norm | Zero-Var Groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pinned_base_model` | Primary | `0.083` | `0.273` | `0.106` | `0.049` | `0.185` | `0.353` | `0.137` | `0.508` | `0.411` | `0.0000` | `1.030` | `4.38` | — |
| `sft_only_baseline` | Primary | `0.491` | `0.835` | `0.642` | `0.263` | `0.655` | `0.831` | `0.570` | `0.565` | `0.074` | `0.2769` | `0.493` | `9.63` | — |
| `reinforce_no_group_norm` | Primary | `0.638` | `0.934` | `0.778` | `0.427` | `0.852` | `0.956` | `0.737` | `0.632` | `0.051` | `0.3435` | `0.327` | `8.56` | — |
| **`grpo_kl_regularized` (`G=8, β=0.04`)** | **Primary** | **`0.740`** | **`0.948`** | **`0.925`** | **`0.461`** | **`0.869`** | **`0.984`** | **`0.820`** | `0.641` | **`0.014`** | **`0.4290`** | `0.219` | `9.42` | `0.486` |
| `neg_format_only_reward_hacking` | Negative 1 | `0.006` | `0.023` | `0.000` | `0.016` | `0.058` | `0.066` | `0.016` | **`0.871`** | **`0.926`** | `0.3157` | `0.763` | `7.55` | `0.761` |
| `neg_grpo_zero_kl_collapse` (`β=0.0`) | Negative 2 | `0.732` | `0.857` | `0.983` | `0.357` | `0.642` | `1.000` | `0.802` | `0.642` | `0.000` | **`0.5681`** | **`0.130`** | **`28.17`** | `0.856` |
| `neg_grpo_group_size_2` (`G=2`) | Negative 3 | `0.574` | `0.877` | `0.769` | `0.281` | `0.696` | `0.872` | `0.664` | `0.624` | `0.111` | `0.2802` | `0.430` | `8.23` | **`0.654`** |

### 2.2 Per-Domain Held-Out `pass@1` Breakdown

| Regime | `math_reasoning` | `symbolic_logic` | `code_trace` | Greedy Held-Out `pass@1` |
|---|---:|---:|---:|---:|
| `pinned_base_model` | `0.053` | `0.144` | `0.053` | `0.958` |
| `sft_only_baseline` | `0.431` | `0.566` | `0.475` | `0.983` |
| `reinforce_no_group_norm` | `0.575` | `0.731` | `0.606` | `1.000` |
| **`grpo_kl_regularized`** | **`0.703`** | **`0.794`** | **`0.722`** | **`1.000`** |
| `neg_format_only_reward_hacking` | `0.003` | `0.016` | `0.000` | `0.000` |
| `neg_grpo_zero_kl_collapse` | `0.641` | `0.838` | `0.719` | `0.775` |
| `neg_grpo_group_size_2` | `0.538` | `0.641` | `0.544` | `1.000` |

### 2.3 Paired Bootstrap Statistical Significance (`B = 1,000`, seed `1350`)

| Comparison (Held-Out Test) | Mean Delta ($\Delta$) | 95% Bootstrap CI | One-Sided Bootstrap $p$ | Paired W / L / T |
|---|---:|---:|---:|---:|
| `grpo_kl_regularized` vs `pinned_base_model` (`pass@1`) | `+0.6563` | `[+0.6073, +0.7042]` | `0.000999` | `119 / 0 / 1` |
| `grpo_kl_regularized` vs `sft_only_baseline` (`pass@1`) | `+0.2490` | `[+0.2135, +0.2865]` | `0.000999` | `99 / 4 / 17` |
| `grpo_kl_regularized` vs `reinforce_no_group_norm` (`pass@1`) | `+0.1021` | `[+0.0667, +0.1417]` | `0.000999` | `65 / 19 / 36` |
| `grpo_kl_regularized` vs `neg_format_only_reward_hacking` (`pass@1`) | `+0.7333` | `[+0.6844, +0.7833]` | `0.000999` | `118 / 0 / 2` |
| `grpo_kl_regularized` vs `neg_grpo_zero_kl_collapse` (**OOD** `pass@1`) | `+0.1042` | `[+0.0286, +0.1849]` | `0.003996` | `27 / 16 / 5` |
| `grpo_kl_regularized` vs `neg_grpo_group_size_2` (`pass@1`) | `+0.1656` | `[+0.1365, +0.1938]` | `0.000999` | `88 / 3 / 29` |

---

## 3. Analysis of Honest Negative Regimes

1. **Reward Hacking (`neg_format_only_reward_hacking`)**:
   - Optimizing a naive format+length proxy (`naive_proxy_reward`) drives the proxy score to **`0.871`** (the highest proxy score of all 7 regimes), while **collapsing** independent held-out `pass@1` to **`0.006`** (13x worse than the un-tuned pinned base model `0.083`).
   - `RewardAuditTrail` independently catches **`92.6%`** of held-out rollouts as hacking attempts (`871` `LENGTH_PADDING_HACK`, `10` `ANSWER_STUFFING`, `10` `DUPLICATE_TAG`).
   - **Takeaway**: Format/length heuristics without strict canonical verifiers and anti-padding guards are trivially exploited by GRPO.

2. **KL Collapse & OOD Degradation (`neg_grpo_zero_kl_collapse`, `β = 0.0`)**:
   - Removing the Schulman `k3` KL anchor allows parameter L2 norm to explode 3x (`28.17` vs `9.42` for regularized GRPO), drives held-out KL up (`0.5681` vs `0.4290` nats), and collapses policy entropy (`0.130` vs `0.219` nats).
   - While in-distribution memorization climbs (`ID pass@1 = 0.983`), generalization on unseen longer chains **collapses**: OOD `pass@1` drops `-10.4` pp (`0.357` vs `0.461`, $p = 0.004$), OOD `pass@4` collapses `-22.7` pp (`0.642` vs `0.869`), and greedy held-out accuracy drops `-22.5` pp (`0.775` vs `1.000` — below even the base model's greedy mode `0.958`).
   - **Takeaway**: Positive `k3` KL regularization (`β = 0.04`) is essential to preserve pretrained reasoning generality and `pass@k` exploration diversity under distribution shift.

3. **Tiny Group / High-Variance Regime (`neg_grpo_group_size_2`, `G = 2`)**:
   - Holding both total rollout budget (`2,880`) and rollouts per optimizer batch (`96`) strictly matched (`48 prompts × G=2` vs `12 prompts × G=8`), `G = 2` wastes **`65.4%`** of prompt groups on zero-variance all-pass or all-fail outcomes (vs `48.6%` for `G = 8`).
   - Held-out `pass@1` lags `G = 8` by **`-16.6` pp** (`0.574` vs `0.740`, $p = 0.001$), and OOD `pass@1` lags by **`-18.0` pp** (`0.281` vs `0.461`).
   - **Takeaway**: Sufficient group size (`G >= 8`) is necessary for low-variance group-relative baseline estimation and effective rollout utilization.

---

## 4. Independent Evaluation Receipt (`--seed 1350`)

All hashes below are verbatim from [`results.json`](results.json) and can be verified offline via `verify_evaluation_receipt(results)`:

- **Base Model SHA-256**: `b78f4ebad78469589df47094bf28ec065c31c65fa458fb1fc8b3fa2172f1c8e8`
- **Dataset Manifest SHA-256**: `38e1d111221a4dc7eaeb6229ec543d08f04479196705fc02e5413b034766ba4a`
  - `train`: `ec9404472a5df37f49d93ecf8b751da8e1ec3de16e4b937a162d3ef4f8a58d17`
  - `val`: `37ae8a52236c22349687e13113a41b17b37a1ea80b73c444e457e20e1bbfb3bd`
  - `held_out_test`: `eb93c6a5f98311ebee068f106f67b335a584db41a4640ffa2e2e75c96c95ca91`
  - `zero_leakage_verified`: `true`
- **Overall Receipt SHA-256**: `e31eaca217274fc5b2bb89dd5d3accf4e7eb40312f57327611e7a3794ce23404`
- **Production Recipe Validation**: `trl_grpo_openthoughts.yaml` (`valid: true`), `verl_grpo_openthoughts.yaml` (`valid: true`), `open_r1_used: false`.

---

## 5. What Remains Unproven

In accordance with Zoo × Hanzo open research standards, the following claims are **explicitly NOT proven** by this offline CPU reference benchmark and require dedicated GPU validation and human review:

1. **7B–32B Transformer Scaling & vLLM Rollout Throughput**: The offline benchmark trains a compact 491-parameter differentiable autoregressive policy on CPU for byte-exact reproducibility. Token-level throughput, KV-cache memory pressure, colocate vs. server vLLM latency, and bf16/fp8 numerical drift on 1.5B–32B dense or MoE LLMs remain unproven until executed on multi-GPU clusters using the provided TRL (`recipes/trl_grpo_openthoughts.yaml`) and `verl` (`recipes/verl_grpo_openthoughts.yaml`) recipes.
2. **OpenThoughts Multi-Million-Trace Scaling & Teacher-Distillation Provenance**: Although our recipes pin `open-thoughts/OpenThoughts-114k` (Apache-2.0) and flag separate license review, empirical gain magnitudes on large-scale natural-language math/competition-coding corpora (AIME, MATH-500, LiveCodeBench, GPQA) and legal clearance of upstream teacher traces remain unproven pending GPU runs and human legal sign-off.
3. **Sandboxed Arbitrary Code Execution Verifiers**: `code_trace` uses a pure-Python deterministic two-register state machine. Running arbitrary untrusted LLM-generated Python/C++ solutions against hidden unit-test suites requires hardened container/microVM sandboxing (seccomp, memory/CPU cgroups, zero network egress), which is out of scope for this stdlib offline package.
4. **Multi-Turn Tool-Use & Agentic RL**: Our GRPO rollout engine evaluates single-turn `<think>...</think><answer>...</answer>` reasoning trajectories. Multi-turn environment feedback, tool-call credit assignment, and long-horizon trajectory masking remain unproven.
5. **Pretraining Corpus Contamination Auditing**: Disjointness is cryptographically guaranteed (`zero_leakage = true`) across our procedural splits, but evaluating pretrained public checkpoints (`Qwen2.5`, `Zen`) on external public math/code benchmarks requires independent n-gram / membership contamination audits against base-model pretraining corpora.
6. **External Human Verification**: Self-generated receipts are mechanically checkable but not self-certifying; independent third-party reproduction and authorized maintainer review are required before any scientific claim or governance action is treated as validated.
