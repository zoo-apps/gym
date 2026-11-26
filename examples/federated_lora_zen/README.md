# Federated LoRA / QLoRA for Zen (`examples/federated_lora_zen`)

Self-contained, zero-dependency reference implementation and reproducible benchmark for **`zoo-apps/gym#3`** (HIP-1350): training LoRA and 4-bit NF4 QLoRA adapters across **three independently administered workers** (`worker-alpha`, `worker-beta`, `worker-gamma`), verifying aggregation quality against matched centralized baselines, and recovering cleanly from mid-run node loss.

## Highlights

- **Zero New Runtime Dependencies**: Pure Python >= 3.10 standard library (`setup.py`, `pyproject.toml`, and `requirements.txt` are untouched).
- **Real SHA-256 Pinned Corpus (`corpus.py`)**: 47 real Zoo Gym files (`133,751` bytes across `docs`, `yaml`, `python`), split strictly without overlap into `base_pretrain`, 3 private worker vaults, and an independent `eval_test` split (`4,284` examples).
- **Frozen Base + FP32 LoRA & 4-bit NF4 QLoRA (`lora_model.py`)**: Exact reverse-mode autodiff for trainable `(A, B)` adapters (`< 1e-7` finite-difference error), packed 4-bit Dettmers et al. NF4 block quantization (`5.89×` base RAM reduction vs FP32, `2.95×` vs FP16), and cryptographic SHA-256 frozen-base verification.
- **Independent Workers, Exact ΔW SVD Aggregation & Cold Recovery (`federation.py`)**:
  - Per-worker HMAC-SHA256 signed `AdapterEnvelope` (zero raw data leaves any worker vault) + atomic local `worker_state.json` persistence.
  - Compares naive `fedavg_lora_ba` (`mean(B) @ mean(A)`) against `exact_delta_svd_fedlora` (`sum_k w_k (B_k @ A_k)` refactored via deterministic truncated rank-`r` SVD).
  - Real mid-run `worker-beta` dropout (rounds `4..6`) with uninterrupted 2-worker quorum continuation and verified cold-restart recovery from persisted `server_checkpoint_r006.json` at round `7`.

## Quickstart

Run from the repository root:

```bash
# 1. Run the full reproducible benchmark (6 main regimes + 3 negative regimes)
python3 -m examples.federated_lora_zen.run_benchmark \
  --seed 1350 \
  --out examples/federated_lora_zen/results.json

# 2. Verify byte-identical determinism across two consecutive runs
python3 -m examples.federated_lora_zen.run_benchmark \
  --seed 1350 \
  --out /tmp/results_check.json
cmp -s examples/federated_lora_zen/results.json /tmp/results_check.json && echo "BYTE-IDENTICAL"
rm -f /tmp/results_check.json

# 3. Run unit tests (zero warnings)
python3 -W error -m unittest discover -s examples/federated_lora_zen/tests -v

# 4. Lint check
ruff check examples/federated_lora_zen/
```

## Summary Results (`--seed 1350`)

| Regime | Quant | Aggregation | Final Test Loss | Test PPL | Base RAM | Wire (FP32) | Raw Data Shared |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `centralized_lora_baseline` | `fp32` | `none_centralized` | `3.1675` | `23.75` | 59,200 B | 0 B | 133,751 B |
| `centralized_qlora_nf4_baseline` | `nf4` | `none_centralized` | `3.1695` | `23.80` | **10,048 B** | 0 B | 133,751 B |
| `fed3_lora_fedavg_ba` | `fp32` | `fedavg_lora_ba` | `3.2217` | `25.07` | 59,200 B | 921.6 KB | **0 B** |
| `fed3_lora_exact_svd` | `fp32` | `exact_delta_svd_fedlora` | `3.2082` | `24.74` | 59,200 B | 921.6 KB | **0 B** |
| `fed3_qlora_nf4_exact_svd` | `nf4` | `exact_delta_svd_fedlora` | `3.2094` | `24.76` | **10,048 B** | 921.6 KB | **0 B** |
| `fed3_qlora_node_loss_recovery` | `nf4` | `exact_delta_svd_fedlora` | `3.2080` | `24.73` | **10,048 B** | 844.8 KB | **0 B** |

## Package Structure

- [`corpus.py`](./corpus.py) — SHA-256 pinned 47-file Zoo Gym corpus and disjoint 3-worker vault splitter (`mixed_non_iid` and `extreme_non_iid`).
- [`lora_model.py`](./lora_model.py) — Frozen base causal LM, packed 4-bit NF4 quantization, LoRA/QLoRA autodiff + AdamW, and deterministic truncated SVD.
- [`federation.py`](./federation.py) — Independent worker runtime, HMAC-signed envelopes, naive vs exact-SVD aggregation, disk checkpoints, and cold node-loss recovery.
- [`run_benchmark.py`](./run_benchmark.py) — Deterministic CLI benchmark runner producing [`results.json`](./results.json).
- [`SPEC.md`](./SPEC.md) — Versioned v1.0.0 protocol specification and Flower / Zoo Gym / Zen 4B–70B mapping.
- [`BENCHMARK.md`](./BENCHMARK.md) — Complete benchmark tables, negative-regime analysis, and explicit open research gaps.
- [`tests/`](./tests/) — Unit tests covering SHA-256 pinning, vault disjointness, NF4 packing, finite-difference gradients, SVD accuracy, HMAC security, cold recovery, and byte-identical determinism.
