# Adversarial Federated-Learning Benchmark (`HIP-1350` / `zoo-apps/gym#5`)

Clean-room, Apache-2.0, pure-Python (`>= 3.10` stdlib, zero external runtime dependencies) reference implementation and byte-reproducible benchmark evaluating Byzantine and adversarial threats in Non-IID federated learning — with explicit **False-Rejection Rate (FRR)**, **Attack Success Rate (ASR)**, **Quality Loss vs. Clean Oracle (`Delta Acc`)**, and **Minority-Honest Acceptance Rate** metrics, plus a drop-in Flower (`flwr.server.strategy.Strategy`) adapter.

## Contents

| File | Purpose |
| :--- | :--- |
| [`SPEC.md`](./SPEC.md) | Versioned `HIP-1350` technical specification, governance separation, and *What Remains Unproven* |
| [`BENCHMARK.md`](./BENCHMARK.md) | Reproducible benchmark report with full FRR / ASR / Quality-Loss tables and 3 negative regimes |
| [`corpus.py`](./corpus.py) | 74 pinned Zoo Gym repo files (SHA-256 manifest), strictly disjoint `N=12` Non-IID vaults + `eval_val` + `eval_clean` + `eval_trigger` |
| [`model.py`](./model.py) | Pure-stdlib 2-layer MLP multi-class classifier (`2,024` params) with exact reverse-mode gradients (finite-diff verified `< 1e-6`) |
| [`attacks.py`](./attacks.py) | All 4 mandatory threats (`label_flip_and_sign_flip`, `targeted_trigger_backdoor`, `stale_update_lag` honest/malicious, `colluding_inner_product_cartel`) + `stealth_unmonitored_backdoor` |
| [`defenses.py`](./defenses.py) | 6 head-to-head defenses (`fedavg_undefended`, `norm_clipping`, `coordinate_median`, `trimmed_mean`, `multi_krum`, `dso_hybrid_eval_cosine`) |
| [`simulation.py`](./simulation.py) | Deterministic federated training loop with checkpoint history and per-client audit metrics |
| [`flower_adapter.py`](./flower_adapter.py) | Drop-in `flwr.server.strategy.Strategy` adapter with pure-stdlib offline shim |
| [`run_benchmark.py`](./run_benchmark.py) | CLI benchmark runner emitting byte-reproducible [`results.json`](./results.json) with SHA-256 evaluation receipt |
| [`tests/`](./tests/) | 25 unit & integration tests (zero warnings under `python3 -W error`) |

## Quickstart

```bash
# 1. Run linter & full unit test suite (zero third-party installs required)
ruff check examples/adversarial_fl/
python3 -W error -m unittest discover -s examples/adversarial_fl/tests -v

# 2. Reproduce canonical results.json and verify byte-for-byte determinism
python3 -m examples.adversarial_fl.run_benchmark --seed 1350 --out examples/adversarial_fl/results.json
python3 -m examples.adversarial_fl.run_benchmark --seed 1350 --out /tmp/results_verify.json
cmp -s examples/adversarial_fl/results.json /tmp/results_verify.json && echo "BYTE-IDENTICAL"
sha256sum examples/adversarial_fl/results.json /tmp/results_verify.json
```

## Flower Drop-In Usage

When `flwr` is installed, `AdversarialDefenseStrategy` subclasses `flwr.server.strategy.Strategy` directly:

```python
from examples.adversarial_fl.defenses import DefenseConfig
from examples.adversarial_fl.flower_adapter import AdversarialDefenseStrategy
from examples.adversarial_fl.model import init_params

strategy = AdversarialDefenseStrategy(
    initial_params=init_params(1350),
    defense_config=DefenseConfig("dso_hybrid_eval_cosine"),
    eval_val=corpus.eval_val,
    eval_clean=corpus.eval_clean,
)
```

When `flwr` is absent, the same class executes natively on pure-stdlib wire shims (`StdlibParameters`, `StdlibFitRes`, `StdlibClientProxy`) with exact parameter parity (`< 1e-12`).
