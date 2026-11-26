<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors. -->

# Internet-Scale Low-Communication Elastic Training (`examples/diloco_wan`)

Reference implementation, versioned protocol specification, and byte-reproducible benchmark for **`zoo-apps/gym#4` (HIP-1350)**.

> **Research Question**: Can geographically distributed contributors improve a shared model without continuously transmitting full gradients or sharing private datasets, and how do you show their updates are beneficial and safe, not just that their GPUs computed?

## Highlights

- **Zero New Runtime Dependencies**: 100% pure Python 3.10+ standard library (Apache-2.0). `setup.py`, `pyproject.toml`, and `requirements.txt` are untouched.
- **Grounded Real Corpus**: Trains a causal neural language model (`CausalCharLM`, `10,320` params) on `49` SHA-256 pinned documentation & config files from `zoo-apps/gym` (`208,532` raw bytes), strictly partitioned into disjoint private per-worker shards, an independent coordinator verification split (`verifier_val`), and an unseen global test split (`global_test`).
- **Prime DiLoCo + Binary Quantization**: Persistent per-worker Inner AdamW (`H ∈ {16, 64}`) + Outer SGD with Nesterov momentum + real binary wire serialization (`fp32`, `int8_block`, `int4_block`, `sign_1bit_no_ef`), cutting WAN bytes by **`64x`–`479x`** and speeding up wall time by **`9.3x`–`21.0x`** vs synchronous `H=1` WAN AllReduce.
- **Benefit & Safety Verification Beyond Proof-of-Compute**: 4-stage `UpdateVerifier` (authenticated envelope + finite/norm cap + peer-median cosine anomaly filter + independent held-out validation benefit gate + outer-step rollback) rejects **`100%`** of Byzantine `sign_flip`, `scaled_noise`, and `corrupted_signature` updates with **`0%`** false rejections on honest workers and **`0`** bytes of private dataset transmitted.
- **Three Honest Negative Regimes**: Explicitly characterizes failure boundaries under extreme non-IID client drift (`H=512`), severe packet loss/churn (`20%` loss / `60%` dropout), and 1-bit sign quantization without error feedback.

## Quickstart & Exact Reproduction

Run the full 11-regime deterministic benchmark (regenerates `results.json` byte-for-byte with SHA-256 `d9defdc397029593f58aa139957f528be2d44d027e4c5620fb5908f45d79d9ad`):

```bash
python3 examples/diloco_wan/run_benchmark.py --seed 1350
```

Run the 11-test verification suite with warnings treated as errors:

```bash
python3 -W error -m unittest discover -s examples/diloco_wan/tests -v
```

## Directory Layout

| File | Purpose |
|---|---|
| [`SPEC.md`](SPEC.md) | Versioned protocol specification (`v0.1.0`), wire format, threat model, governance notes, and explicit **What Remains Unproven** section |
| [`BENCHMARK.md`](BENCHMARK.md) | Complete empirical benchmark tables, security ablation, negative regimes, and open gaps |
| [`results.json`](results.json) | Byte-deterministic raw JSON output from `run_benchmark.py --seed 1350` |
| [`corpus.py`](corpus.py) | SHA-256 pinned repository corpus loader, fixed 64-symbol vocab, disjoint `PrivateShardVault` |
| [`model.py`](model.py) | Pure-Python Bengio Residual `CausalCharLM` (`10,320` params) and `InnerAdamW` |
| [`quantize.py`](quantize.py) | Real binary wire codecs (`DLCW` v1): `fp32`, `int8_block`, `int4_block`, `sign_1bit_no_ef` |
| [`wan.py`](wan.py) | Deterministic WAN transport (RTT/jitter, bandwidth, MTU packet loss/retransmit, churn, elastic joins, wire privacy auditor) |
| [`verify.py`](verify.py) | `ContributorKeyring` + 4-stage `UpdateVerifier` + auditable `RoundVerificationReceipt` |
| [`diloco.py`](diloco.py) | Matched centralized ceiling, `H=1` AllReduce, and elastic verified Prime DiLoCo engine |
| [`run_benchmark.py`](run_benchmark.py) | Reproducible CLI harness running all 11 regimes |
| [`tests/test_diloco_wan.py`](tests/test_diloco_wan.py) | 11-test `unittest` suite covering all invariants and negative regimes |
