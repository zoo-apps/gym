# Open Reasoning RL / GRPO Training Pipeline (`HIP-1350`)

Clean-room **Apache-2.0** pure-Python reference implementation, auditable verifier suite, byte-reproducible benchmark, and schema-validated **TRL** & **`verl`** production recipes for **`zoo-apps/gym#7`**.

> **Maintained Framework Policy**: Production recipes target current maintained trainers (`huggingface/trl` and `volcengine/verl`) with openly licensed data (`open-thoughts/OpenThoughts-114k`, Apache-2.0) and pinned open-weight base models (`Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0). `huggingface/open-r1` is treated strictly as an archived historical reference and is **not** used.

---

## Quickstart (Zero External Dependencies)

Requires only **Python >= 3.10** standard library:

```bash
# 1. Lint
ruff check examples/grpo_reasoning/

# 2. Run all 21 unit, gradient-check (<1e-6), leakage, recipe & receipt tests
python3 -W error -m unittest discover -s examples/grpo_reasoning/tests -v

# 3. Reproduce the seed-1350 benchmark and verify byte-identical output
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/r1.json
python3 -m examples.grpo_reasoning.run_benchmark --seed 1350 --out /tmp/r2.json --quiet
cmp -s /tmp/r1.json /tmp/r2.json && sha256sum /tmp/r1.json examples/grpo_reasoning/results.json
```

---

## Key Held-Out Results (`--seed 1350`, Matched Budget = 2,880 Rollouts)

| Regime | Held-Out `pass@1` | Held-Out `pass@4` | OOD `pass@1` | Format Valid | Verified Step Acc | Exact KL (nats) |
|---|---:|---:|---:|---:|---:|---:|
| `pinned_base_model` | `0.083` | `0.273` | `0.049` | `0.353` | `0.137` | `0.0000` |
| `sft_only_baseline` | `0.491` | `0.835` | `0.263` | `0.831` | `0.570` | `0.2769` |
| `reinforce_no_group_norm` | `0.638` | `0.934` | `0.427` | `0.956` | `0.737` | `0.3435` |
| **`grpo_kl_regularized` (`G=8, β=0.04`)** | **`0.740`** | **`0.948`** | **`0.461`** | **`0.984`** | **`0.820`** | **`0.4290`** |
| `neg_format_only_reward_hacking` | `0.006` | `0.023` | `0.016` | `0.066` | `0.016` | `0.3157` |
| `neg_grpo_zero_kl_collapse` (`β=0.0`) | `0.732` | `0.857` | `0.357` | `1.000` | `0.802` | `0.5681` |
| `neg_grpo_group_size_2` (`G=2`) | `0.574` | `0.877` | `0.281` | `0.872` | `0.664` | `0.2802` |

Full statistical comparisons (95% paired bootstrap CIs, $p \le 0.004$), domain breakdowns, cryptographic receipts, and an explicit **What Remains Unproven** section are documented in [`BENCHMARK.md`](BENCHMARK.md).

---

## Repository Layout

```text
examples/grpo_reasoning/
├── __init__.py                         # Package metadata (HIP-1350 v0.1.0)
├── README.md                           # Quickstart & architecture overview
├── SPEC.md                             # Versioned formal specification & governance boundary
├── BENCHMARK.md                        # Reproducible results, negative regimes & What Remains Unproven
├── LICENSE_AUDIT.json                  # Per-artifact license & separate-review manifest
├── dataset.py                          # 3-domain verifiable reasoning dataset + SHA-256 disjoint splits
├── rewards.py                          # Stateless deterministic verifiers, anti-hacking guards & TRL/verl hooks
├── grpo.py                             # Pinned base policy, exact GRPO + Schulman k3 KL, SFT & REINFORCE
├── run_benchmark.py                    # Byte-reproducible benchmark runner & cryptographic receipt generator
├── results.json                        # Committed seed-1350 benchmark output + receipt
├── recipes/
│   ├── __init__.py
│   ├── trl_grpo_openthoughts.yaml      # Zoo Gym + TRL GRPOTrainer GPU recipe
│   ├── verl_grpo_openthoughts.yaml     # verl Ray GRPO GPU recipe (low_var_kl / k3)
│   └── validator.py                    # Pure-stdlib YAML recipe schema & no-Open-R1 validator
└── tests/
    ├── __init__.py
    └── test_grpo_reasoning.py          # 21 offline unit, gradient (<1e-6), leakage, recipe & receipt tests
```

---

## GPU Production Handoff (TRL & `verl`)

After authorized human license review of external model weights and OpenThoughts traces (see [`LICENSE_AUDIT.json`](LICENSE_AUDIT.json)):

- **Zoo Gym + TRL (`GRPOTrainer`)**:
  ```bash
  gym train examples/grpo_reasoning/recipes/trl_grpo_openthoughts.yaml
  ```
- **`verl` (Ray GRPO)**:
  Use `examples/grpo_reasoning/recipes/verl_grpo_openthoughts.yaml` with `verl.trainer.main_ppo` and custom verifier `examples/grpo_reasoning/rewards.py:verl_compute_score`.

Validate both recipes offline at any time:

```bash
python3 -c "from examples.grpo_reasoning.recipes.validator import validate_all_recipes; import json; print(json.dumps(validate_all_recipes(), indent=2))"
```

---

## License & Governance Notice

- All source files in `examples/grpo_reasoning/` are **Apache-2.0**.
- External model weights, datasets, reasoning traces, and SDKs require a **separate human license review** prior to production release (see [`LICENSE_AUDIT.json`](LICENSE_AUDIT.json)).
- AI agents do not certify experiments, modify governance rules, or authorize spending; independent human verification is required.
