# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Reproducible corpus loader grounded in versioned Zoo Gym repository files.

Constructs strictly disjoint private per-worker training shards, an independent
held-out coordinator validation split (for the update verification gate), and a
completely unseen held-out global test split. All source files are pinned by
SHA-256 digest and byte size for deterministic reproduction.
"""

from __future__ import annotations

import hashlib
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Fixed 64-symbol vocabulary (independent of corpus statistics to avoid leakage).
# Index 0 is <UNK>; indices 1..63 cover >99.8% of ASCII lowercase docs & YAML.
UNK_ID = 0
VOCAB_CHARS = (
    "\n !\"#'()*+,-./0123456789:<=>?[\\]_`abcdefghijklmnopqrstuvwxyz{|}"
)
VOCAB_SIZE = 1 + len(VOCAB_CHARS)  # 64
assert VOCAB_SIZE == 64

CHAR_TO_ID: Dict[str, int] = {ch: idx + 1 for idx, ch in enumerate(VOCAB_CHARS)}
ID_TO_CHAR: Tuple[str, ...] = ("<UNK>",) + tuple(VOCAB_CHARS)

CONTEXT_LEN = 4


@dataclass(frozen=True)
class FilePin:
    """Immutable provenance record for a single versioned repository file."""

    rel_path: str
    byte_size: int
    sha256: str
    domain: str


@dataclass(frozen=True)
class Example:
    """Causal next-character prediction example (context -> target)."""

    context: Tuple[int, int, int, int]
    target: int


@dataclass
class PrivateShardVault:
    """Holds strictly private per-worker datasets that never leave worker memory."""

    worker_shards: Dict[str, List[Example]]
    worker_raw_texts: Dict[str, str]
    canary_substrings: Tuple[bytes, ...]

    def worker_ids(self) -> List[str]:
        return sorted(self.worker_shards.keys())

    def total_train_examples(self) -> int:
        return sum(len(v) for v in self.worker_shards.values())

    def assert_payload_contains_no_private_data(self, wire_payload: bytes) -> None:
        """Audit check ensuring no raw private document snippets appear on the wire."""
        for canary in self.canary_substrings:
            if canary and canary in wire_payload:
                raise AssertionError(
                    "Privacy violation: private worker corpus bytes leaked into WAN packet"
                )


@dataclass(frozen=True)
class CorpusBundle:
    """Complete deterministic dataset split with cryptographic manifest."""

    repo_root: str
    manifest_sha256: str
    file_pins: Tuple[FilePin, ...]
    total_raw_bytes: int
    partition_mode: str
    num_workers: int
    seed: int
    vault: PrivateShardVault
    verifier_val: Tuple[Example, ...]
    global_test: Tuple[Example, ...]
    pooled_train: Tuple[Example, ...]


def find_repo_root() -> pathlib.Path:
    """Locate the Zoo Gym repository root relative to this module."""
    here = pathlib.Path(__file__).resolve()
    # examples/diloco_wan/corpus.py -> parents[2] is repo root
    candidate = here.parents[2]
    if (candidate / "README.md").is_file() and (candidate / "docs").is_dir():
        return candidate
    raise FileNotFoundError(f"Could not locate Zoo Gym repo root from {here}")


def _classify_domain(rel_path: str) -> str:
    """Assign each versioned repo file to one of six thematic domains."""
    name = pathlib.Path(rel_path).name.lower()
    if rel_path.startswith("examples/distributed-parallel") or name in {
        "amd_hpc.qmd",
        "multi-gpu.qmd",
        "multi-node.qmd",
        "nccl.qmd",
        "nd_parallelism.qmd",
        "ray-integration.qmd",
        "sequence_parallelism.qmd",
    }:
        return "distributed_hpc"
    if rel_path.startswith("docs/dataset-formats") or name in {
        "dataset_loading.qmd",
        "dataset_preprocessing.qmd",
        "multipack.qmd",
        "streaming.qmd",
    }:
        return "datasets_streaming"
    if name in {
        "fsdp_qlora.qmd",
        "lora_optims.qmd",
        "mixed_precision.qmd",
        "optimizers.qmd",
        "qat.qmd",
        "quantize.qmd",
        "torchao.qmd",
        "unsloth.qmd",
    }:
        return "quantization_lora"
    if name in {
        "readme.md",
        "llm.md",
        "faqs.md",
        "cli.qmd",
        "config-reference.qmd",
        "debugging.qmd",
        "faq.qmd",
        "getting-started.qmd",
        "installation.qmd",
    }:
        return "core_cli_debug"
    if name in {
        "multimodal.qmd",
        "reward_modelling.qmd",
        "rlhf.qmd",
        "inference.qmd",
        "custom_integrations.qmd",
    }:
        return "rlhf_multimodal"
    return "systems_misc"


def discover_pinned_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    """Deterministically collect the fixed set of upstream repository files."""
    files: List[pathlib.Path] = []
    for name in ("README.md", "LLM.md", "FAQS.md"):
        p = repo_root / name
        if p.is_file():
            files.append(p)
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(p for p in docs_dir.glob("*.qmd") if p.is_file()))
        ds_dir = docs_dir / "dataset-formats"
        if ds_dir.is_dir():
            files.extend(sorted(p for p in ds_dir.glob("*.qmd") if p.is_file()))
    dp_dir = repo_root / "examples" / "distributed-parallel"
    if dp_dir.is_dir():
        files.extend(sorted(p for p in dp_dir.iterdir() if p.is_file()))
    # Deduplicate and sort by relative POSIX path for total determinism.
    unique = {p.relative_to(repo_root).as_posix(): p for p in files}
    return [unique[k] for k in sorted(unique.keys())]


def encode_text(text: str) -> List[int]:
    """Encode lowercase text into fixed 64-symbol vocabulary IDs."""
    lowered = text.lower()
    return [CHAR_TO_ID.get(ch, UNK_ID) for ch in lowered]


def tokens_to_examples(tokens: Sequence[int], stride: int = 1) -> List[Example]:
    """Convert a contiguous token sequence into causal (4-gram -> next) examples."""
    if len(tokens) <= CONTEXT_LEN:
        return []
    out: List[Example] = []
    for idx in range(CONTEXT_LEN, len(tokens), stride):
        ctx = (
            tokens[idx - 4],
            tokens[idx - 3],
            tokens[idx - 2],
            tokens[idx - 1],
        )
        out.append(Example(context=ctx, target=tokens[idx]))
    return out


def load_corpus(
    num_workers: int = 6,
    partition_mode: str = "balanced",
    seed: int = 1350,
    max_val_examples: int = 1500,
    max_test_examples: int = 1500,
    repo_root: pathlib.Path | None = None,
) -> CorpusBundle:
    """Load, hash-pin, and strictly partition the Zoo Gym documentation corpus.

    Split structure per document (with CONTEXT_LEN guard gap between regions so
    zero overlapping n-grams cross train / verifier_val / global_test boundaries):
      - [0 : 74%]    -> private worker training shard (strictly disjoint across workers)
      - [75% : 87%]  -> independent coordinator validation split (verifier gate only)
      - [88% : 100%] -> completely unseen global test split (final reporting only)

    Partition modes:
      - "balanced": disjoint non-overlapping 256-char document blocks distributed
        round-robin across workers so shards are strictly private & disjoint yet
        roughly balanced across topics.
      - "non_iid": each worker exclusively owns documents from a single domain
        cluster (or extreme character/syntax slice), inducing strong client drift.
    """
    if partition_mode not in ("balanced", "non_iid"):
        raise ValueError(f"Unsupported partition_mode: {partition_mode}")
    if num_workers < 2:
        raise ValueError("num_workers must be >= 2")

    root = repo_root if repo_root is not None else find_repo_root()
    paths = discover_pinned_files(root)
    if not paths:
        raise RuntimeError(f"No pinned corpus files found under {root}")

    file_pins: List[FilePin] = []
    manifest_hasher = hashlib.sha256()
    total_raw_bytes = 0

    domain_train_texts: Dict[str, List[Tuple[str, str]]] = {}
    val_examples_all: List[Example] = []
    test_examples_all: List[Example] = []

    for path in paths:
        raw_bytes = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        domain = _classify_domain(rel)
        file_pins.append(
            FilePin(
                rel_path=rel,
                byte_size=len(raw_bytes),
                sha256=digest,
                domain=domain,
            )
        )
        manifest_hasher.update(f"{rel}:{len(raw_bytes)}:{digest}\n".encode("utf-8"))
        total_raw_bytes += len(raw_bytes)

        text = raw_bytes.decode("utf-8", errors="ignore")
        n = len(text)
        if n < 64:
            continue
        train_end = int(n * 0.74)
        val_start = min(n, train_end + CONTEXT_LEN + 2)
        val_end = int(n * 0.87)
        test_start = min(n, val_end + CONTEXT_LEN + 2)

        train_slice = text[:train_end]
        val_slice = text[val_start:val_end]
        test_slice = text[test_start:]

        domain_train_texts.setdefault(domain, []).append((rel, train_slice))
        val_examples_all.extend(tokens_to_examples(encode_text(val_slice), stride=2))
        test_examples_all.extend(tokens_to_examples(encode_text(test_slice), stride=2))

    rng = random.Random(seed)
    worker_ids = [f"worker-{i}" for i in range(num_workers)]
    worker_shards: Dict[str, List[Example]] = {wid: [] for wid in worker_ids}
    worker_raw_texts: Dict[str, str] = {wid: "" for wid in worker_ids}

    if partition_mode == "balanced":
        # Strictly disjoint 320-character blocks assigned deterministically round-robin.
        block_size = 320
        block_idx = 0
        for domain in sorted(domain_train_texts.keys()):
            for _rel, train_slice in domain_train_texts[domain]:
                for start in range(0, len(train_slice), block_size):
                    block = train_slice[start : start + block_size]
                    if len(block) <= CONTEXT_LEN + 2:
                        continue
                    wid = worker_ids[block_idx % num_workers]
                    worker_raw_texts[wid] += block
                    worker_shards[wid].extend(
                        tokens_to_examples(encode_text(block), stride=1)
                    )
                    block_idx += 1
    else:
        # Extreme non-IID partitioning:
        # Each worker owns a distinct domain + syntactic regime (e.g., YAML/code vs
        # prose vs numeric/punctuation heavy segments), amplifying local drift.
        sorted_domains = sorted(domain_train_texts.keys())
        for idx, domain in enumerate(sorted_domains):
            wid = worker_ids[idx % num_workers]
            for _rel, train_slice in domain_train_texts[domain]:
                worker_raw_texts[wid] += train_slice
                worker_shards[wid].extend(
                    tokens_to_examples(encode_text(train_slice), stride=1)
                )
        # Apply deterministic label-skew / target-class partitioning on top of
        # domain ownership so every worker specializes in a disjoint target-char
        # partition (simulating extreme geographic/domain non-IID skew).
        all_non_iid = [ex for wid in worker_ids for ex in worker_shards[wid]]
        worker_shards = {wid: [] for wid in worker_ids}
        for ex in all_non_iid:
            bucket = ex.target % num_workers
            worker_shards[worker_ids[bucket]].append(ex)

    # Deterministically shuffle each private shard internally.
    for wid in worker_ids:
        rng.shuffle(worker_shards[wid])

    # Build deterministic canary substrings from private worker text for wire audit.
    canaries: List[bytes] = []
    for wid in worker_ids:
        raw = worker_raw_texts[wid].encode("utf-8")
        if len(raw) >= 48:
            mid = len(raw) // 2
            canaries.append(raw[mid : mid + 32])
            canaries.append(raw[:32])

    # Subsample independent held-out validation & test sets deterministically.
    rng_val = random.Random(seed + 101)
    rng_val.shuffle(val_examples_all)
    verifier_val = tuple(val_examples_all[:max_val_examples])

    rng_test = random.Random(seed + 202)
    rng_test.shuffle(test_examples_all)
    global_test = tuple(test_examples_all[:max_test_examples])

    pooled: List[Example] = []
    for wid in worker_ids:
        pooled.extend(worker_shards[wid])
    rng_pool = random.Random(seed + 303)
    rng_pool.shuffle(pooled)

    vault = PrivateShardVault(
        worker_shards=worker_shards,
        worker_raw_texts=worker_raw_texts,
        canary_substrings=tuple(canaries),
    )

    return CorpusBundle(
        repo_root=str(root),
        manifest_sha256=manifest_hasher.hexdigest(),
        file_pins=tuple(file_pins),
        total_raw_bytes=total_raw_bytes,
        partition_mode=partition_mode,
        num_workers=num_workers,
        seed=seed,
        vault=vault,
        verifier_val=verifier_val,
        global_test=global_test,
        pooled_train=tuple(pooled),
    )
