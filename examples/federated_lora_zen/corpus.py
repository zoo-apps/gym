# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""SHA-256 pinned real repository corpus and 3-worker disjoint vault splitter.

Grounds all training and evaluation examples in 47 real files from Zoo Gym across
three distinct technical domains (``docs``, ``yaml``, ``python``). Splits contiguous
chunks strictly without overlap into:

1. A small ``base_pretrain`` split used once to pretrain the frozen base LM.
2. Three independently administered worker vaults (``worker-alpha``,
   ``worker-beta``, ``worker-gamma``) whose union matches the centralized baseline.
3. An independent held-out ``eval_test`` split tracked both overall and per-domain.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DOMAINS: tuple[str, ...] = ("docs", "yaml", "python")
WORKER_IDS: tuple[str, ...] = ("worker-alpha", "worker-beta", "worker-gamma")

# Pinned (relative_path, expected_sha256) tuples from the Zoo Gym repository.
PINNED_CORPUS_FILES: dict[str, tuple[tuple[str, str], ...]] = {
    "docs": (
        (
            "README.md",
            "83b1e75ce13d2549772fb16f6727a8198104fe9e0af55976707c4854bcea2a18",
        ),
        (
            "LLM.md",
            "362ecb9b312c4856f04b796613fc3eefb50d96efd43e5e71cd0d50f13c8f3dd4",
        ),
        (
            "docs/fsdp_qlora.qmd",
            "a1cc022e8163d105db0d1dffc8b57a1804ff4eb94481489a885708a21655bc98",
        ),
        (
            "docs/lora_optims.qmd",
            "fc30b32d9cb9ce698c1f0cd7eb80bd3a3b7c7cbbafdb9ca22b112888a0632e54",
        ),
        (
            "docs/quantize.qmd",
            "f0fdae9e79af3fde1aa5e10f55472521d1c3bc7e8e16c015ebb4a9089f7bf853",
        ),
        (
            "docs/multi-node.qmd",
            "6e7bc100acd27d09c8bdf10f9ec3c6016ccb35240da2200e23dd5913d878772d",
        ),
        (
            "docs/mixed_precision.qmd",
            "2cd9dd05620d2fcc6b9e3063dcbb07eae6ced93a3894c850586f0fd8958d53aa",
        ),
        (
            "docs/optimizers.qmd",
            "45becd74fced68939ac96a9bd65f12cc5534491ec72c72eac60a267baf76d745",
        ),
        (
            "docs/getting-started.qmd",
            "2f4bda97707e19dae0111a180d408950a72e43fed74ec869f575fb967732a855",
        ),
    ),
    "yaml": (
        (
            "examples/LiquidAI/lfm2-350m-fft.yaml",
            "d8bf8d6d563bd0955b9d1b54d7d50862a0174e7d5ffc691d817bfdae612befa9",
        ),
        (
            "examples/LiquidAI/lfm2-vl-lora.yaml",
            "88fbb002e03276ed5ca9856545a4c991900d0d3090de223be66d3debf9c1f12e",
        ),
        (
            "examples/alst/llama3-8b-deepspeed-alst.yaml",
            "06a552d7e87028502dc7c3394db48d3ea541f0134cef5ff4118bb73511df219d",
        ),
        (
            "examples/alst/llama3-8b-fsdp2-alst.yaml",
            "ec24e83187a45023c73f7c09518a289069351d77a9c1038f94acb1c772bfe335",
        ),
        (
            "examples/apertus/apertus-8b-qlora.yaml",
            "dbd0f345f6f8413f8e3541095ba0edd44cfdce3ad97357ebf3501ab9ee46817a",
        ),
        (
            "examples/arcee/afm-4.5b-qlora.yaml",
            "bd341300f45c1666e50ce5e39e3b3dc68bdceb0d5eda95786cd4f87aa09357c7",
        ),
        (
            "examples/cloud/baseten.yaml",
            "4ecc9b7592936e24a5af4d01f7c48e73a94d8f506e55372e6c886b48d4ac7e1f",
        ),
        (
            "examples/cloud/modal.yaml",
            "0547687b8616f4eaa6738576c95205ea3f524bed65b5d362fc696d1150c053b0",
        ),
        (
            "examples/cohere/command-r-7b-qlora.yml",
            "76eaa6fb0471e753f06a637e4b93f69f18213a30233d1b21fab0f15e7c75fcc0",
        ),
        (
            "examples/deepcogito/cogito-v1-preview-llama-3B-lora.yml",
            "f3321093e10ccafe1d1c9f3786ed7467209c7c82934ea1458d759e4cde37a864",
        ),
        (
            "examples/deepcogito/cogito-v1-preview-qwen-14B-lora.yml",
            "ee61c8c40c5e9572614541da682d95bc3abc5b4790b63ddd19018c05f4cafeca",
        ),
        (
            "examples/deepseek-v2/fft-fsdp-16b.yaml",
            "94c5caf6c87e9938d5752259bc2397584746f735294f78aff0b55f85955c29ac",
        ),
        (
            "examples/deepseek-v2/qlora-fsdp-2_5.yaml",
            "3d1da6e8703cc4d50cb72472e7be0f4e9f1b3e03e0963d3194a0db76ea5945d5",
        ),
        (
            "examples/devstral/devstral-small-qlora.yml",
            "aa5f033cf4f39cc21ac212ed444a1cdd5afcb631837988e89e02e54865a22b05",
        ),
        (
            "examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml",
            "846235b34c161b311bb2a1b4b7986a29d67b65d4c661a788b0e71ce9eda854fe",
        ),
        (
            "examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml",
            "6352d38385a30e1d4e97f03d0194f2c8cd30f0ab9b761c70d28e8028b7e1c6f8",
        ),
        (
            "examples/falcon-h1/falcon-h1-1b-deep-qlora.yaml",
            "a3e84e8977ae57fad542f0bb29d0f960b5ae3b65b34a321d68f0a1cc87119bee",
        ),
        (
            "examples/falcon-h1/falcon-h1-1b-qlora.yaml",
            "c75ecd69aecbb8d7aac88162bdfa76ac5cb7c11405ad1ddc333818acb6b7768b",
        ),
        (
            "examples/falcon-h1/falcon-h1-34b-qlora.yaml",
            "1af1d140d8f481ea386dd740d8136ed89afec63a9026e58848af352b8b793091",
        ),
        (
            "examples/falcon-h1/falcon-h1-3b-qlora.yaml",
            "10fa98fbe89cd63279f368003ea403e6f0318367d0833fe667ddb672a197a615",
        ),
        (
            "examples/falcon-h1/falcon-h1-500m-qlora.yaml",
            "c57621667264a35b4863d181e7cf266fb015f1d76ae5b09c7a238de1f39f3e80",
        ),
        (
            "examples/falcon-h1/falcon-h1-7b-qlora.yaml",
            "e1bf8d189b42892b2f5d5c246b9a42ec7b5ba2363701cdeef5d54879d0cbd3a5",
        ),
        (
            "examples/gemma2/qlora.yml",
            "c1e3457df2af35da578df2378298dc0eadd6aae33bcfc2d379d354d1dee0f02c",
        ),
        (
            "examples/gemma2/reward-model.yaml",
            "b5eb0becbc097fb341a5c9b6b0aa22edc87ab1c6dd60a651e9363eed37b0d1bf",
        ),
        (
            "examples/gemma3/gemma-3-1b-qlora.yml",
            "4da6c15ccf35deba1ccfd57eee56d85a516aaaba3693ee8f0f77ad80de8e541e",
        ),
        (
            "examples/gemma3/gemma-3-270m-qlora.yml",
            "c6333ded59130036a63a77063c283e1bab214a1b46ebf912015cc8c36b1e8837",
        ),
        (
            "examples/gemma3/gemma-3-4b-qlora.yml",
            "2f9be3a6e19f861764862aab569b1ff5d919d1173a43cd3a8cecce11e2dfb66e",
        ),
        (
            "examples/gemma3/gemma-3-4b-vision-qlora.yml",
            "f2055da9f118df3fd76cac48794f2d04a1e36d9fa1cff7cd6e531671cef24160",
        ),
        (
            "examples/gemma3n/gemma-3n-e2b-qlora.yml",
            "b6d8d7166bd95e49738c7d144ecfd8fe155ed5cf05c4089813092a29dce17c76",
        ),
        (
            "examples/gemma3n/gemma-3n-e2b-vision-audio-qlora.yml",
            "1b708924ddb7879331e8b49defb6c2637fa0cf475df7aaf1bcba1c0736b6245c",
        ),
        (
            "examples/gemma3n/gemma-3n-e2b-vision-qlora.yml",
            "c0e9f6a1d41ba631d0f281454aadde3bb30c05e7f141a51557ea99f66c522ebd",
        ),
        (
            "examples/glm4/qlora-32b.yaml",
            "de0388f2df22d221ac9c26db709a18c9b727e66862aa4f669ca189b9fab2c403",
        ),
    ),
    "python": (
        (
            "src/gym/cli/merge_lora.py",
            "89f0bb57c07328f0a6f2709e73756fe914e0e5ab6d3ab8ae6a156940d8f28dd6",
        ),
        (
            "src/gym/cli/quantize.py",
            "a33949b35171fe9913042f7fb13419ab915174d5e77b16f7bd3dc58560f2571e",
        ),
        (
            "src/gym/cli/train.py",
            "6c29d7d6761b0cff7b0e5cb9a7df7fbe1e7e984fa1923317838110b6f8d9f8f8",
        ),
        (
            "src/gym/cli/evaluate.py",
            "6a9224eff4c091136df1fc6585cac976e4e614618bdb092ad3de0460af1c6d4a",
        ),
        (
            "src/gym/core/builders/base.py",
            "50911ca1506cc24aee6947e61f07a1a8c9fb037acc547a140729383a87e31fb0",
        ),
        (
            "src/gym/core/builders/causal.py",
            "f1e76acbbe97609fb4f68da93ce53c9456bdddac3b8bf21be73adf1866397f61",
        ),
    ),
}


@dataclass(frozen=True)
class TextChunk:
    """Contiguous non-overlapping text slice tagged with origin domain and ID."""

    chunk_id: str
    domain: str
    source_path: str
    text: str


@dataclass(frozen=True)
class CharVocab:
    """Deterministic fixed-size character vocabulary built from the pinned corpus."""

    itos: tuple[str, ...]
    stoi: dict[str, int]
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        unk = self.unk_id
        return [stoi.get(ch, unk) for ch in text]


@dataclass(frozen=True)
class LMExample:
    """Single causal language modeling example: ``context -> target``."""

    context: tuple[int, ...]
    target: int
    domain: str
    chunk_id: str


@dataclass(frozen=True)
class WorkerVault:
    """Private data vault owned by an independently administered worker."""

    worker_id: str
    admin_key_hex: str
    admin_key_fingerprint: str
    partition_mode: str
    chunk_ids: tuple[str, ...]
    domain_counts: dict[str, int]
    examples: tuple[LMExample, ...]
    vault_sha256: str


@dataclass(frozen=True)
class FederatedCorpusSplit:
    """Complete disjoint split for pretraining, 3 federated vaults, and evaluation."""

    seed: int
    partition_mode: str
    context_len: int
    chunk_chars: int
    corpus_manifest_sha256: str
    total_files: int
    total_bytes: int
    vocab: CharVocab
    base_pretrain: tuple[LMExample, ...]
    base_pretrain_chunk_ids: tuple[str, ...]
    worker_vaults: dict[str, WorkerVault]
    centralized_train: tuple[LMExample, ...]
    eval_test: tuple[LMExample, ...]
    eval_test_by_domain: dict[str, tuple[LMExample, ...]]
    eval_test_chunk_ids: tuple[str, ...]


def default_repo_root() -> Path:
    """Return the repository root directory containing ``examples/federated_lora_zen``."""
    return Path(__file__).resolve().parents[2]


def derive_admin_key(seed: int, worker_id: str) -> str:
    """Derive a deterministic per-worker HMAC administration secret (hex)."""
    material = f"zoo-gym-hip1350-v1:{seed}:{worker_id}:admin-secret".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def sign_payload_hmac(admin_key_hex: str, message: bytes) -> str:
    """Compute HMAC-SHA256 hex digest using a worker's admin secret."""
    return hmac.new(
        bytes.fromhex(admin_key_hex), message, hashlib.sha256
    ).hexdigest()


def load_pinned_chunks(
    repo_root: Path | None = None,
    chunk_chars: int = 128,
) -> tuple[dict[str, list[TextChunk]], str, int, int]:
    """Read and SHA-256 verify every pinned repository file, slicing into chunks.

    Raises:
        FileNotFoundError: If any pinned repository file is missing.
        ValueError: If any file's SHA-256 digest does not match its pinned digest.
    """
    root = repo_root if repo_root is not None else default_repo_root()
    chunks_by_domain: dict[str, list[TextChunk]] = {d: [] for d in DOMAINS}
    manifest_hasher = hashlib.sha256()
    total_files = 0
    total_bytes = 0

    for domain in DOMAINS:
        for rel_path, expected_sha in PINNED_CORPUS_FILES[domain]:
            full_path = root / rel_path
            if not full_path.is_file():
                raise FileNotFoundError(
                    f"Pinned corpus file missing: {full_path}"
                )
            raw = full_path.read_bytes()
            actual_sha = hashlib.sha256(raw).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"SHA-256 mismatch for {rel_path}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            total_files += 1
            total_bytes += len(raw)
            manifest_hasher.update(
                f"{domain}:{rel_path}:{actual_sha}:{len(raw)}\n".encode("utf-8")
            )
            text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
            n_chunks = len(text) // chunk_chars
            for idx in range(n_chunks):
                slice_text = text[idx * chunk_chars : (idx + 1) * chunk_chars]
                chunk_id = f"{domain}:{rel_path}#c{idx:04d}"
                chunks_by_domain[domain].append(
                    TextChunk(
                        chunk_id=chunk_id,
                        domain=domain,
                        source_path=rel_path,
                        text=slice_text,
                    )
                )

    return chunks_by_domain, manifest_hasher.hexdigest(), total_files, total_bytes


def build_char_vocab(
    chunks_by_domain: dict[str, list[TextChunk]], vocab_size: int = 80
) -> CharVocab:
    """Build a deterministic character vocabulary of exact size ``vocab_size``."""
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    counts: Counter[str] = Counter()
    for domain in DOMAINS:
        for chunk in chunks_by_domain[domain]:
            counts.update(chunk.text)
    # Deterministic ordering: descending frequency, then ascending Unicode codepoint.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], ord(kv[0])))
    top_chars = [ch for ch, _ in ranked[: vocab_size - 1]]
    itos = ("<UNK>", *top_chars)
    stoi = {ch: idx for idx, ch in enumerate(itos)}
    return CharVocab(itos=itos, stoi=stoi, unk_id=0)


def chunks_to_examples(
    chunks: list[TextChunk],
    vocab: CharVocab,
    context_len: int = 4,
    stride: int = 6,
) -> tuple[LMExample, ...]:
    """Convert contiguous chunks into causal ``(context, target)`` examples.

    Examples never cross chunk boundaries, ensuring zero leakage across splits.
    """
    examples: list[LMExample] = []
    for chunk in chunks:
        ids = vocab.encode(chunk.text)
        limit = len(ids) - context_len
        for pos in range(0, limit, stride):
            ctx = tuple(ids[pos : pos + context_len])
            tgt = ids[pos + context_len]
            examples.append(
                LMExample(
                    context=ctx,
                    target=tgt,
                    domain=chunk.domain,
                    chunk_id=chunk.chunk_id,
                )
            )
    return tuple(examples)


def _hash_chunk_list(chunks: list[TextChunk]) -> str:
    h = hashlib.sha256()
    for ch in chunks:
        h.update(f"{ch.chunk_id}:{len(ch.text)}\n".encode("utf-8"))
        h.update(ch.text.encode("utf-8"))
    return h.hexdigest()


def build_federated_corpus(
    seed: int = 1350,
    partition_mode: str = "mixed_non_iid",
    repo_root: Path | None = None,
    vocab_size: int = 80,
    context_len: int = 4,
    chunk_chars: int = 128,
    stride: int = 6,
) -> FederatedCorpusSplit:
    """Build deterministic disjoint splits for pretraining, 3 worker vaults, and eval.

    Args:
        seed: Master RNG seed (default 1350 per HIP-1350).
        partition_mode:
            - ``"mixed_non_iid"``: Realistic institutional skew where each worker
              holds 60% of its primary domain plus 20% of each secondary domain.
            - ``"extreme_non_iid"``: 100% disjoint single-domain vaults
              (alpha=docs, beta=yaml, gamma=python).
        repo_root: Optional repository root path.
        vocab_size: Fixed character vocabulary size (default 80).
        context_len: Number of preceding tokens per causal example (default 4).
        chunk_chars: Contiguous chunk length in characters (default 128).
        stride: Sliding window stride within each chunk (default 2).
    """
    if partition_mode not in ("mixed_non_iid", "extreme_non_iid"):
        raise ValueError(
            f"Unsupported partition_mode={partition_mode!r}; "
            "expected 'mixed_non_iid' or 'extreme_non_iid'"
        )

    chunks_by_domain, manifest_sha, total_files, total_bytes = load_pinned_chunks(
        repo_root=repo_root, chunk_chars=chunk_chars
    )
    vocab = build_char_vocab(chunks_by_domain, vocab_size=vocab_size)

    rng = random.Random(seed)

    base_chunks: list[TextChunk] = []
    eval_chunks_by_domain: dict[str, list[TextChunk]] = {}
    train_chunks_by_domain: dict[str, list[TextChunk]] = {}

    for domain in DOMAINS:
        dom_chunks = list(chunks_by_domain[domain])
        rng.shuffle(dom_chunks)
        n = len(dom_chunks)
        n_base = max(4, int(round(n * 0.15)))
        n_eval = max(6, int(round(n * 0.20)))
        base_chunks.extend(dom_chunks[:n_base])
        eval_chunks_by_domain[domain] = dom_chunks[n_base : n_base + n_eval]
        train_chunks_by_domain[domain] = dom_chunks[n_base + n_eval :]

    rng.shuffle(base_chunks)

    worker_chunks: dict[str, list[TextChunk]] = {w: [] for w in WORKER_IDS}
    primary_domain_for_worker = {
        "worker-alpha": "docs",
        "worker-beta": "yaml",
        "worker-gamma": "python",
    }

    if partition_mode == "extreme_non_iid":
        for worker_id in WORKER_IDS:
            dom = primary_domain_for_worker[worker_id]
            worker_chunks[worker_id].extend(train_chunks_by_domain[dom])
    else:
        # Realistic institutional skew: 60% primary domain, 20% + 20% secondary.
        domain_to_primary = {d: w for w, d in primary_domain_for_worker.items()}
        for domain in DOMAINS:
            dom_train = list(train_chunks_by_domain[domain])
            n_dom = len(dom_train)
            n_primary = int(round(n_dom * 0.60))
            n_sec1 = (n_dom - n_primary) // 2
            primary_w = domain_to_primary[domain]
            secondary_ws = [w for w in WORKER_IDS if w != primary_w]
            worker_chunks[primary_w].extend(dom_train[:n_primary])
            worker_chunks[secondary_ws[0]].extend(
                dom_train[n_primary : n_primary + n_sec1]
            )
            worker_chunks[secondary_ws[1]].extend(dom_train[n_primary + n_sec1 :])

    worker_vaults: dict[str, WorkerVault] = {}
    centralized_chunks: list[TextChunk] = []

    for worker_id in WORKER_IDS:
        w_chunks = worker_chunks[worker_id]
        # Deterministic per-worker shuffle so each worker sees mixed batches locally.
        w_rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"{seed}:{worker_id}:shuffle".encode("utf-8")).digest()[
                    :8
                ],
                "big",
            )
        )
        w_rng.shuffle(w_chunks)
        centralized_chunks.extend(w_chunks)

        dom_counts = {d: 0 for d in DOMAINS}
        for ch in w_chunks:
            dom_counts[ch.domain] += 1

        admin_key_hex = derive_admin_key(seed, worker_id)
        admin_fp = hashlib.sha256(bytes.fromhex(admin_key_hex)).hexdigest()[:16]
        examples = chunks_to_examples(
            w_chunks, vocab=vocab, context_len=context_len, stride=stride
        )
        worker_vaults[worker_id] = WorkerVault(
            worker_id=worker_id,
            admin_key_hex=admin_key_hex,
            admin_key_fingerprint=admin_fp,
            partition_mode=partition_mode,
            chunk_ids=tuple(ch.chunk_id for ch in w_chunks),
            domain_counts=dom_counts,
            examples=examples,
            vault_sha256=_hash_chunk_list(w_chunks),
        )

    # Centralized baseline pools exact union of the 3 worker vaults (zero extra data).
    central_rng = random.Random(seed + 17)
    central_rng.shuffle(centralized_chunks)

    eval_all_chunks: list[TextChunk] = []
    eval_examples_by_domain: dict[str, tuple[LMExample, ...]] = {}
    for domain in DOMAINS:
        d_chunks = eval_chunks_by_domain[domain]
        eval_all_chunks.extend(d_chunks)
        eval_examples_by_domain[domain] = chunks_to_examples(
            d_chunks, vocab=vocab, context_len=context_len, stride=stride
        )

    eval_all_examples = tuple(
        ex for d in DOMAINS for ex in eval_examples_by_domain[d]
    )

    return FederatedCorpusSplit(
        seed=seed,
        partition_mode=partition_mode,
        context_len=context_len,
        chunk_chars=chunk_chars,
        corpus_manifest_sha256=manifest_sha,
        total_files=total_files,
        total_bytes=total_bytes,
        vocab=vocab,
        base_pretrain=chunks_to_examples(
            base_chunks, vocab=vocab, context_len=context_len, stride=stride
        ),
        base_pretrain_chunk_ids=tuple(ch.chunk_id for ch in base_chunks),
        worker_vaults=worker_vaults,
        centralized_train=chunks_to_examples(
            centralized_chunks, vocab=vocab, context_len=context_len, stride=stride
        ),
        eval_test=eval_all_examples,
        eval_test_by_domain=eval_examples_by_domain,
        eval_test_chunk_ids=tuple(ch.chunk_id for ch in eval_all_chunks),
    )
