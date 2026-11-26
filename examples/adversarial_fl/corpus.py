# Copyright 2026 Zoo Labs Foundation Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned repository corpus, SHA-256 manifest, and Non-IID federated splits.

All examples are extracted from real, Apache-2.0-licensed repository files
pinned by relative path and SHA-256 digest. Chunks are strictly disjoint across:
  - 12 federated client vaults (10 majority institutional clients + 2 minority
    ``infra`` specialist clients),
  - an independent domain-stratified coordinator validation split (``eval_val``),
  - a held-out clean evaluation split (``eval_clean``), and
  - a held-out backdoor trigger evaluation split (``eval_trigger``) derived from
    non-target ``eval_clean`` examples with a rare out-of-corpus trigger phrase.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass

DOMAINS: tuple[str, ...] = ("docs", "yaml", "python", "infra")
DOMAIN_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DOMAINS)}
NUM_CLASSES: int = len(DOMAINS)
MINORITY_DOMAIN: str = "infra"
MINORITY_CLASS_ID: int = DOMAIN_TO_ID[MINORITY_DOMAIN]
BACKDOOR_TARGET_DOMAIN: str = "docs"
BACKDOOR_TARGET_CLASS: int = DOMAIN_TO_ID[BACKDOOR_TARGET_DOMAIN]

NUM_CLIENTS: int = 12
MINORITY_CLIENT_IDS: tuple[int, ...] = (10, 11)
MAJORITY_CLIENT_IDS: tuple[int, ...] = tuple(range(10))

CHUNK_CHARS: int = 120
CLEAN_VOCAB_SIZE: int = 94
TRIGGER_TOKENS: tuple[str, str] = ("zxq_beacon", "qzv_anchor")
TRIGGER_PHRASE: str = "zxq_beacon qzv_anchor zxq_beacon qzv_anchor"
INPUT_DIM: int = CLEAN_VOCAB_SIZE + len(TRIGGER_TOKENS)

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\s]")

PINNED_CORPUS_FILES: dict[str, tuple[str, ...]] = {
    "docs": (
        "README.md",
        "LLM.md",
        "FAQS.md",
        "docs/getting-started.qmd",
        "docs/lora_optims.qmd",
        "docs/quantize.qmd",
        "docs/fsdp_qlora.qmd",
        "docs/multi-gpu.qmd",
        "docs/rlhf.qmd",
        "docs/optimizers.qmd",
        "docs/debugging.qmd",
        "docs/dataset_loading.qmd",
    ),
    "yaml": (
        "examples/LiquidAI/lfm2-350m-fft.yaml",
        "examples/LiquidAI/lfm2-vl-lora.yaml",
        "examples/alst/llama3-8b-deepspeed-alst.yaml",
        "examples/alst/llama3-8b-fsdp2-alst.yaml",
        "examples/apertus/apertus-8b-qlora.yaml",
        "examples/arcee/afm-4.5b-qlora.yaml",
        "examples/cloud/baseten.yaml",
        "examples/cloud/modal.yaml",
        "examples/deepseek-v2/fft-fsdp-16b.yaml",
        "examples/deepseek-v2/qlora-fsdp-2_5.yaml",
        "examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml",
        "examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml",
        "examples/falcon-h1/falcon-h1-1b-deep-qlora.yaml",
        "examples/falcon-h1/falcon-h1-1b-qlora.yaml",
        "examples/falcon-h1/falcon-h1-34b-qlora.yaml",
        "examples/falcon-h1/falcon-h1-3b-qlora.yaml",
        "examples/falcon-h1/falcon-h1-500m-qlora.yaml",
        "examples/falcon-h1/falcon-h1-7b-qlora.yaml",
        "examples/gemma2/reward-model.yaml",
        "examples/glm4/qlora-32b.yaml",
        "examples/gpt-oss/gpt-oss-120b-fft-fsdp2-offload.yaml",
        "examples/gpt-oss/gpt-oss-20b-fft-deepspeed-zero3.yaml",
        "examples/gpt-oss/gpt-oss-20b-fft-fsdp2-offload.yaml",
        "examples/gpt-oss/gpt-oss-20b-fft-fsdp2.yaml",
        "examples/gpt-oss/gpt-oss-20b-sft-lora-singlegpu.yaml",
        "examples/hunyuan/hunyuan-v1-dense-qlora.yaml",
        "examples/jamba/qlora.yaml",
        "examples/jamba/qlora_deepspeed.yaml",
    ),
    "python": (
        "src/gym/cli/args.py",
        "src/gym/cli/config.py",
        "src/gym/cli/train.py",
        "src/gym/cli/evaluate.py",
        "src/gym/cli/inference.py",
        "src/gym/cli/quantize.py",
        "src/gym/cli/merge_lora.py",
        "src/gym/common/architectures.py",
        "src/gym/common/datasets.py",
        "src/gym/core/builders/base.py",
    ),
    "infra": (
        "deepspeed_configs/zero1.json",
        "deepspeed_configs/zero1_torch_compile.json",
        "deepspeed_configs/zero2.json",
        "deepspeed_configs/zero2_torch_compile.json",
        "deepspeed_configs/zero3.json",
        "deepspeed_configs/zero3_bf16.json",
        "deepspeed_configs/zero3_bf16_cpuoffload_all.json",
        "deepspeed_configs/zero3_bf16_cpuoffload_params.json",
        "deploy/k8s.yaml",
        "compose.yml",
        "Dockerfile.site",
        "docker/Dockerfile",
        "docker/Dockerfile-base",
        "docker/Dockerfile-base-next",
        "docker/Dockerfile-cloud",
        "docker/Dockerfile-tests",
        "docker/Dockerfile-uv-base",
        "docker/docker-cuda/compose.yml",
        "docker/docker-rocm/compose.yml",
        "docker/docker-npu/compose.yml",
        ".runpod/Dockerfile",
        "cicd/cicd.sh",
        "cicd/multigpu.sh",
        "cicd/cleanup.sh",
    ),
}


@dataclass(frozen=True)
class ManifestEntry:
    """SHA-256 pinned record for a single repository source file."""

    rel_path: str
    domain: str
    size_bytes: int
    sha256: str
    num_chunks: int


@dataclass(frozen=True)
class Example:
    """Single tokenized and L2-normalized sparse feature example."""

    chunk_id: str
    source_path: str
    domain: str
    label: int
    features: tuple[tuple[int, float], ...]
    is_triggered: bool = False


@dataclass(frozen=True)
class ClientVault:
    """Private training vault for a single federated client."""

    client_id: int
    role: str
    primary_domain: str
    is_minority_specialist: bool
    examples: tuple[Example, ...]
    domain_counts: dict[str, int]


@dataclass(frozen=True)
class FederatedCorpus:
    """Complete strictly disjoint federated corpus and manifest."""

    seed: int
    manifest: tuple[ManifestEntry, ...]
    manifest_sha256: str
    total_bytes: int
    vocab: tuple[str, ...]
    token_to_id: dict[str, int]
    trigger_token_ids: tuple[int, int]
    clients: tuple[ClientVault, ...]
    eval_val: tuple[Example, ...]
    eval_clean: tuple[Example, ...]
    eval_trigger: tuple[Example, ...]


def repo_root() -> pathlib.Path:
    """Return repository root relative to this module without hardcoded host paths."""
    return pathlib.Path(__file__).resolve().parents[2]


def tokenize(text: str) -> list[str]:
    """Deterministic lowercased regex tokenizer preserving code/config symbols."""
    return TOKEN_PATTERN.findall(text.lower())


def encode_text(text: str, token_to_id: dict[str, int]) -> tuple[tuple[int, float], ...]:
    """Encode raw text into an L2-normalized sparse bag-of-tokens vector."""
    counts: dict[int, float] = {}
    for tok in tokenize(text):
        idx = token_to_id.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0.0) + 1.0
    if not counts:
        # Deterministic fallback weight on index 0 so vectors are never all-zero
        return ((0, 1.0),)
    # Sublinear log-tf weighting improves linear/MLP separability across domains
    weighted = {idx: 1.0 + math.log(cnt) for idx, cnt in counts.items()}
    norm = math.sqrt(sum(v * v for v in weighted.values()))
    inv = 1.0 / norm if norm > 0.0 else 1.0
    return tuple((idx, weighted[idx] * inv) for idx in sorted(weighted))


def apply_trigger_to_example(
    raw_text: str,
    chunk_id: str,
    source_path: str,
    domain: str,
    target_label: int,
    token_to_id: dict[str, int],
) -> Example:
    """Inject the rare backdoor trigger phrase into a clean chunk and re-encode."""
    triggered_text = f"{raw_text} {TRIGGER_PHRASE}"
    return Example(
        chunk_id=f"{chunk_id}::trig",
        source_path=source_path,
        domain=domain,
        label=target_label,
        features=encode_text(triggered_text, token_to_id),
        is_triggered=True,
    )


def _build_discriminative_vocab(
    train_raw_by_domain: dict[str, list[tuple[str, str, str]]],
) -> tuple[str, ...]:
    """Build a leak-free 96-token vocabulary from the training pool only.

    Selects the most domain-discriminative tokens (balanced across all 4 domains)
    strictly from training chunks, then appends the 2 reserved out-of-corpus
    trigger tokens at indices 94 and 95.
    """
    doc_freq_by_domain: dict[str, collections.Counter[str]] = {
        d: collections.Counter() for d in DOMAINS
    }
    total_chunks_by_domain: dict[str, int] = {}
    global_df: collections.Counter[str] = collections.Counter()

    for domain in DOMAINS:
        chunks = train_raw_by_domain[domain]
        total_chunks_by_domain[domain] = max(1, len(chunks))
        for _cid, _path, text in chunks:
            unique_toks = set(tokenize(text))
            for tok in unique_toks:
                if tok in TRIGGER_TOKENS:
                    continue
                doc_freq_by_domain[domain][tok] += 1
                global_df[tok] += 1

    # Score tokens per domain by smoothed log-odds ratio * sqrt(domain_rate)
    per_domain_ranked: dict[str, list[str]] = {}
    total_all = max(1, sum(total_chunks_by_domain.values()))
    for domain in DOMAINS:
        n_d = total_chunks_by_domain[domain]
        n_other = max(1, total_all - n_d)
        scored: list[tuple[float, str]] = []
        for tok, df_d in doc_freq_by_domain[domain].items():
            if df_d < 3:
                continue
            df_other = global_df[tok] - df_d
            p_d = (df_d + 0.5) / (n_d + 1.0)
            p_other = (df_other + 0.5) / (n_other + 1.0)
            contrast = math.log(p_d / p_other)
            score = contrast * math.sqrt(p_d)
            scored.append((score, tok))
        scored.sort(key=lambda item: (-item[0], item[1]))
        per_domain_ranked[domain] = [tok for _s, tok in scored]

    # Round-robin across the 4 domains so every domain (including minority infra)
    # receives balanced discriminative vocabulary slots.
    selected: list[str] = []
    seen: set[str] = set()
    cursors = {d: 0 for d in DOMAINS}
    while len(selected) < CLEAN_VOCAB_SIZE:
        progressed = False
        for domain in DOMAINS:
            if len(selected) >= CLEAN_VOCAB_SIZE:
                break
            ranked = per_domain_ranked[domain]
            while cursors[domain] < len(ranked):
                cand = ranked[cursors[domain]]
                cursors[domain] += 1
                if cand not in seen:
                    seen.add(cand)
                    selected.append(cand)
                    progressed = True
                    break
        if not progressed:
            break

    # Deterministic fallback fill if needed
    if len(selected) < CLEAN_VOCAB_SIZE:
        for tok, _cnt in sorted(global_df.items(), key=lambda kv: (-kv[1], kv[0])):
            if tok not in seen and tok not in TRIGGER_TOKENS:
                seen.add(tok)
                selected.append(tok)
                if len(selected) == CLEAN_VOCAB_SIZE:
                    break

    if len(selected) != CLEAN_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected {CLEAN_VOCAB_SIZE} clean vocab tokens, got {len(selected)}"
        )

    return tuple(selected) + TRIGGER_TOKENS


def _partition_non_iid_clients(
    train_raw_by_domain: dict[str, list[tuple[str, str, str]]],
    token_to_id: dict[str, int],
) -> tuple[tuple[ClientVault, ...], dict[str, str]]:
    """Strictly partition the 70% training pool across N=12 Non-IID clients.

    Layout:
      - Clients 0..3:  docs-dominant institutional clients (0% infra)
      - Clients 4..6:  yaml-dominant institutional clients (0% infra)
      - Clients 7..9:  python-dominant institutional clients (0% infra)
      - Clients 10..11: infra minority-domain honest specialists (100% infra)
    """
    raw_text_lookup: dict[str, str] = {}
    queues: dict[str, list[tuple[str, str, str]]] = {
        d: list(train_raw_by_domain[d]) for d in DOMAINS
    }
    client_raw: list[list[tuple[str, str, str, str]]] = [[] for _ in range(NUM_CLIENTS)]

    def take(domain: str, count: int, client_idx: int) -> None:
        pool = queues[domain]
        n = min(count, len(pool))
        for _ in range(n):
            cid, path, text = pool.pop(0)
            client_raw[client_idx].append((cid, path, domain, text))
            raw_text_lookup[cid] = text

    # Allocate primary + secondary majority domain chunks (strictly zero infra)
    for cid in range(0, 4):
        take("docs", 82, cid)
        take("yaml", 10, cid)
        take("python", 12, cid)
    for cid in range(4, 7):
        take("yaml", 40, cid)
        take("docs", 14, cid)
        take("python", 14, cid)
    for cid in range(7, 10):
        take("python", 88, cid)
        take("docs", 14, cid)
        take("yaml", 9, cid)

    # Distribute any remaining majority chunks round-robin to their primary group
    rr_groups = {
        "docs": [0, 1, 2, 3],
        "yaml": [4, 5, 6],
        "python": [7, 8, 9],
    }
    for domain, group in rr_groups.items():
        idx = 0
        while queues[domain]:
            take(domain, 1, group[idx % len(group)])
            idx += 1

    # Split 100% of infra chunks cleanly across minority specialists 10 and 11
    infra_pool = queues["infra"]
    half = len(infra_pool) // 2
    take("infra", half, 10)
    take("infra", len(queues["infra"]), 11)

    vaults: list[ClientVault] = []
    for client_id in range(NUM_CLIENTS):
        if client_id < 4:
            primary = "docs"
            role = "majority_docs"
        elif client_id < 7:
            primary = "yaml"
            role = "majority_yaml"
        elif client_id < 10:
            primary = "python"
            role = "majority_python"
        else:
            primary = "infra"
            role = "minority_infra_specialist"

        domain_counts = {d: 0 for d in DOMAINS}
        examples: list[Example] = []
        for cid, path, domain, text in client_raw[client_id]:
            domain_counts[domain] += 1
            examples.append(
                Example(
                    chunk_id=cid,
                    source_path=path,
                    domain=domain,
                    label=DOMAIN_TO_ID[domain],
                    features=encode_text(text, token_to_id),
                    is_triggered=False,
                )
            )
        vaults.append(
            ClientVault(
                client_id=client_id,
                role=role,
                primary_domain=primary,
                is_minority_specialist=(client_id in MINORITY_CLIENT_IDS),
                examples=tuple(examples),
                domain_counts=domain_counts,
            )
        )
    return tuple(vaults), raw_text_lookup


def load_federated_corpus(
    seed: int = 1350,
    root: pathlib.Path | None = None,
) -> tuple[FederatedCorpus, dict[str, str]]:
    """Load pinned repo files, verify SHA-256, and construct disjoint FL splits.

    Returns ``(corpus, raw_text_by_chunk_id)`` where ``raw_text_by_chunk_id``
    enables clean trigger injection during backdoor training without ever storing
    host-local paths.
    """
    base = root if root is not None else repo_root()
    rng = random.Random(seed)

    manifest_entries: list[ManifestEntry] = []
    raw_chunks_by_domain: dict[str, list[tuple[str, str, str]]] = {
        d: [] for d in DOMAINS
    }
    total_bytes = 0

    for domain in DOMAINS:
        for rel_path in PINNED_CORPUS_FILES[domain]:
            file_path = base / rel_path
            if not file_path.is_file():
                raise FileNotFoundError(f"Pinned corpus file missing: {rel_path}")
            raw_bytes = file_path.read_bytes()
            total_bytes += len(raw_bytes)
            digest = hashlib.sha256(raw_bytes).hexdigest()
            text = raw_bytes.decode("utf-8")

            # Assert out-of-corpus rarity of trigger tokens on real repo content
            lowered = text.lower()
            for trig in TRIGGER_TOKENS:
                if trig in lowered:
                    raise ValueError(
                        f"Trigger token {trig!r} unexpectedly found in {rel_path}"
                    )

            n_chunks = len(text) // CHUNK_CHARS
            for idx in range(n_chunks):
                chunk_text = text[idx * CHUNK_CHARS : (idx + 1) * CHUNK_CHARS]
                chunk_id = f"{domain}:{rel_path}:{idx:04d}"
                raw_chunks_by_domain[domain].append((chunk_id, rel_path, chunk_text))

            manifest_entries.append(
                ManifestEntry(
                    rel_path=rel_path,
                    domain=domain,
                    size_bytes=len(raw_bytes),
                    sha256=digest,
                    num_chunks=n_chunks,
                )
            )

    manifest_payload = json.dumps(
        [
            {
                "rel_path": e.rel_path,
                "domain": e.domain,
                "size_bytes": e.size_bytes,
                "sha256": e.sha256,
                "num_chunks": e.num_chunks,
            }
            for e in manifest_entries
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    # Deterministic stratified 20% eval_clean / 10% eval_val / 70% train_pool
    eval_clean_raw: list[tuple[str, str, str, str]] = []
    eval_val_raw: list[tuple[str, str, str, str]] = []
    train_raw_by_domain: dict[str, list[tuple[str, str, str]]] = {}

    for domain in DOMAINS:
        items = list(raw_chunks_by_domain[domain])
        rng.shuffle(items)
        n_total = len(items)
        n_clean = max(1, round(n_total * 0.20))
        n_val = max(1, round(n_total * 0.10))
        for cid, path, txt in items[:n_clean]:
            eval_clean_raw.append((cid, path, domain, txt))
        for cid, path, txt in items[n_clean : n_clean + n_val]:
            eval_val_raw.append((cid, path, domain, txt))
        train_raw_by_domain[domain] = items[n_clean + n_val :]

    vocab = _build_discriminative_vocab(train_raw_by_domain)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    trigger_token_ids = (
        token_to_id[TRIGGER_TOKENS[0]],
        token_to_id[TRIGGER_TOKENS[1]],
    )

    clients, raw_text_lookup = _partition_non_iid_clients(
        train_raw_by_domain, token_to_id
    )

    eval_val: list[Example] = []
    for cid, path, domain, txt in eval_val_raw:
        eval_val.append(
            Example(
                chunk_id=cid,
                source_path=path,
                domain=domain,
                label=DOMAIN_TO_ID[domain],
                features=encode_text(txt, token_to_id),
                is_triggered=False,
            )
        )

    eval_clean: list[Example] = []
    eval_trigger: list[Example] = []
    for cid, path, domain, txt in eval_clean_raw:
        label = DOMAIN_TO_ID[domain]
        eval_clean.append(
            Example(
                chunk_id=cid,
                source_path=path,
                domain=domain,
                label=label,
                features=encode_text(txt, token_to_id),
                is_triggered=False,
            )
        )
        if label != BACKDOOR_TARGET_CLASS:
            eval_trigger.append(
                apply_trigger_to_example(
                    raw_text=txt,
                    chunk_id=cid,
                    source_path=path,
                    domain=domain,
                    target_label=BACKDOOR_TARGET_CLASS,
                    token_to_id=token_to_id,
                )
            )

    # Strictly verify zero chunk overlap across all 14 splits
    seen_ids: set[str] = set()
    for split_examples in [*(c.examples for c in clients), eval_val, eval_clean]:
        for ex in split_examples:
            if ex.chunk_id in seen_ids:
                raise AssertionError(f"Duplicate chunk_id across splits: {ex.chunk_id}")
            seen_ids.add(ex.chunk_id)

    corpus = FederatedCorpus(
        seed=seed,
        manifest=tuple(manifest_entries),
        manifest_sha256=manifest_sha256,
        total_bytes=total_bytes,
        vocab=vocab,
        token_to_id=token_to_id,
        trigger_token_ids=trigger_token_ids,
        clients=clients,
        eval_val=tuple(eval_val),
        eval_clean=tuple(eval_clean),
        eval_trigger=tuple(eval_trigger),
    )
    return corpus, raw_text_lookup
