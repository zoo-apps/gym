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
"""Openly licensed (Apache-2.0) verifiable reasoning dataset with pinned splits.

Generates deterministic, zero-dependency, cryptographically pinned train,
validation, and independent held-out test splits across three exact-verifier
reasoning domains:

1. ``math_reasoning``: multi-step arithmetic and modular algebra chains.
2. ``symbolic_logic``: multi-step Boolean constraint & gate propagation.
3. ``code_trace``: deterministic two-register imperative state-machine traces.

Strict leakage prevention:
- Train, validation, and held-out test splits use disjoint seed namespaces,
  disjoint parameter/modulus strata, and verified zero overlap across
  ``prompt_sha256`` and ``canonical_fingerprint``.
- The independent held-out test split includes both in-distribution depths
  (2-3 steps) and out-of-distribution longer reasoning chains (4-5 steps) to
  test generalization and detect KL-collapse overfitting honestly.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOMAINS: tuple[str, ...] = ("math_reasoning", "symbolic_logic", "code_trace")

# Feature dimensionality constants (shared with grpo.py policy heads)
PROBLEM_FEATURE_DIM: int = 16
STEP_FEATURE_DIM: int = 12


def _canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _derived_seed(master_seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class ReasoningStepSpec:
    """Deterministic ground-truth specification for one reasoning step."""

    step_index: int  # 1-indexed
    op: str
    required_op_class: int  # 0, 1, or 2 (operator family required for exact step)
    args: tuple[int, ...]
    expression: str
    expected_state: str
    expected_numeric: int
    alt_slip_state: str
    alt_mod_error_state: str
    prev_state: str


@dataclass(frozen=True)
class ReasoningProblem:
    """Single verifiable multi-step reasoning problem."""

    problem_id: str
    domain: str
    split: str
    is_ood: bool
    depth: int
    prompt: str
    initial_state_str: str
    steps: tuple[ReasoningStepSpec, ...]
    canonical_answer: str
    prompt_sha256: str
    canonical_fingerprint: str
    features: tuple[float, ...]
    step_features: tuple[tuple[float, ...], ...]
    template_id: int

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "domain": self.domain,
            "split": self.split,
            "is_ood": self.is_ood,
            "depth": self.depth,
            "prompt_sha256": self.prompt_sha256,
            "canonical_fingerprint": self.canonical_fingerprint,
            "canonical_answer": self.canonical_answer,
            "template_id": self.template_id,
        }


@dataclass(frozen=True)
class ReasoningDatasetBundle:
    """Pinned train / val / independent held-out test bundle with SHA-256 hashes."""

    seed: int
    train: tuple[ReasoningProblem, ...]
    val: tuple[ReasoningProblem, ...]
    held_out_test: tuple[ReasoningProblem, ...]
    split_sha256: dict[str, str]
    dataset_manifest_sha256: str
    license_id: str = "Apache-2.0"

    def verify_disjointness(self) -> dict[str, Any]:
        """Verify zero prompt, fingerprint, or problem_id overlap across splits."""
        splits = {
            "train": self.train,
            "val": self.val,
            "held_out_test": self.held_out_test,
        }
        prompt_sets = {k: {p.prompt_sha256 for p in v} for k, v in splits.items()}
        fp_sets = {k: {p.canonical_fingerprint for p in v} for k, v in splits.items()}
        id_sets = {k: {p.problem_id for p in v} for k, v in splits.items()}

        overlaps = {
            "train_val_prompts": len(prompt_sets["train"] & prompt_sets["val"]),
            "train_test_prompts": len(
                prompt_sets["train"] & prompt_sets["held_out_test"]
            ),
            "val_test_prompts": len(prompt_sets["val"] & prompt_sets["held_out_test"]),
            "train_val_fingerprints": len(fp_sets["train"] & fp_sets["val"]),
            "train_test_fingerprints": len(
                fp_sets["train"] & fp_sets["held_out_test"]
            ),
            "val_test_fingerprints": len(fp_sets["val"] & fp_sets["held_out_test"]),
            "train_val_ids": len(id_sets["train"] & id_sets["val"]),
            "train_test_ids": len(id_sets["train"] & id_sets["held_out_test"]),
            "val_test_ids": len(id_sets["val"] & id_sets["held_out_test"]),
        }
        overlaps["zero_leakage"] = all(v == 0 for v in overlaps.values())
        return overlaps


def _bool_str(val: int) -> str:
    return "TRUE" if val else "FALSE"


def _build_problem_features(
    domain: str,
    depth: int,
    op_counts: dict[str, int],
    complexity_scalar: float,
    template_id: int,
    split: str,
    is_ood: bool,
    fingerprint_hex: str,
) -> tuple[float, ...]:
    """Build deterministic 16-d feature vector for policy conditioning.

    Includes both genuine structural features (domain one-hot, depth, operator
    profile, complexity) and surface-form features (template & split style
    projections) whose correlations shift between train and held-out OOD.
    """
    dom_math = 1.0 if domain == "math_reasoning" else 0.0
    dom_logic = 1.0 if domain == "symbolic_logic" else 0.0
    dom_code = 1.0 if domain == "code_trace" else 0.0

    norm_depth = (depth - 2.5) / 2.5
    depth_excess = max(0.0, (depth - 3.0) / 2.0)
    total_ops = max(1, sum(op_counts.values()))
    op_vals = list(op_counts.values())
    op0 = (op_vals[0] / total_ops) if len(op_vals) > 0 else 0.0
    op1 = (op_vals[1] / total_ops) if len(op_vals) > 1 else 0.0
    op2 = (op_vals[2] / total_ops) if len(op_vals) > 2 else 0.0

    # Deterministic surface-style projections from canonical fingerprint
    h_bytes = bytes.fromhex(fingerprint_hex[:16])
    surf_0 = ((h_bytes[0] / 255.0) * 2.0 - 1.0) * 0.5
    surf_1 = ((h_bytes[1] / 255.0) * 2.0 - 1.0) * 0.5
    # Surface template & style features shift between train (templates 0..2,
    # short chains) and held-out OOD (templates 3..5, longer depth 4..5 chains).
    tmpl_norm = (template_id - 2.5) / 2.5
    ood_interact = depth_excess * (1.0 if is_ood else 0.0)
    split_style = 0.9 if split == "train" else (-0.2 if not is_ood else -1.5)

    feats = (
        1.0,  # bias
        dom_math,
        dom_logic,
        dom_code,
        norm_depth,
        depth_excess,
        op0,
        op1,
        op2,
        math.tanh(complexity_scalar),
        surf_0,
        surf_1,
        tmpl_norm,
        ood_interact,
        split_style,
        depth_excess * split_style,
    )
    assert len(feats) == PROBLEM_FEATURE_DIM
    return feats


def _build_step_features(
    domain: str,
    step_idx: int,
    depth: int,
    op_class: int,
    arg_scale: float,
    is_ood: bool,
    split: str,
) -> tuple[float, ...]:
    """Build deterministic 12-d per-step feature vector.

    Dimensions 1..3 encode the true generalizable operator class one-hot
    (invariant across all splits and chain depths). Dimensions 9..11 encode
    training-corpus surface cues (e.g., short-chain phrasing shortcuts) that
    correlate with operators on the training split, vanish on held-out ID, and
    shift on OOD longer chains (depth 4-5) — exposing zero-KL overfitting.
    """
    _ = arg_scale
    dom_math = 1.0 if domain == "math_reasoning" else 0.0
    dom_logic = 1.0 if domain == "symbolic_logic" else 0.0
    dom_code = 1.0 if domain == "code_trace" else 0.0
    op_oh_0 = 1.0 if op_class == 0 else 0.0
    op_oh_1 = 1.0 if op_class == 1 else 0.0
    op_oh_2 = 1.0 if op_class == 2 else 0.0
    pos_frac = step_idx / max(1, depth)
    late_step = max(0.0, (step_idx - 3.0) / 2.0)

    if split == "train":
        cues = [1.0 if c == op_class else 0.0 for c in range(3)]
    elif not is_ood:
        cues = [0.0, 0.0, 0.0]
    else:
        shifted = (op_class + 1) % 3
        ood_scale = 1.0 + 0.18 * max(1, depth - 3)
        cues = [ood_scale if c == shifted else 0.0 for c in range(3)]

    feats = (
        1.0,
        op_oh_0,
        op_oh_1,
        op_oh_2,
        dom_math,
        dom_logic,
        dom_code,
        pos_frac,
        late_step,
        cues[0],
        cues[1],
        cues[2],
    )
    assert len(feats) == STEP_FEATURE_DIM
    return feats


def _generate_math_problem(
    rng: random.Random,
    problem_id: str,
    split: str,
    is_ood: bool,
    depth: int,
    template_id: int,
) -> ReasoningProblem:
    # Disjoint modulus pools per split guarantee strict zero-leakage separation
    mod_pools = {
        "train": (13, 17, 19, 23, 29),
        "val": (31, 37, 41),
        "held_out_test": (43, 47, 53, 59, 61),
    }
    mod = rng.choice(mod_pools[split])
    v = rng.randint(2, mod - 2)
    initial_str = f"v0={v} (mod {mod})"

    ops = ("MUL_ADD", "SQUARE_SUB", "MOD_SHIFT")
    op_counts = {op: 0 for op in ops}
    steps: list[ReasoningStepSpec] = []
    step_feats: list[tuple[float, ...]] = []
    prompt_lines: list[str] = []

    if template_id % 3 == 0:
        prompt_lines.append(
            f"[Math Modular Chain] Starting with v0 = {v}, compute each step modulo {mod}:"
        )
    elif template_id % 3 == 1:
        prompt_lines.append(
            f"Evaluate the following {depth}-step modular arithmetic sequence (mod {mod}) from v0 = {v}:"
        )
    else:
        prompt_lines.append(
            f"Modular Algebra Task (modulus {mod}): initial value v0 = {v}. Execute in order:"
        )

    canonical_ops: list[tuple[str, int, int, int]] = []
    for k in range(1, depth + 1):
        op = rng.choice(ops)
        op_counts[op] += 1
        a = rng.randint(2, 7)
        b = rng.randint(1, 11)
        prev_v = v
        if op == "MUL_ADD":
            raw = prev_v * a + b
            v = raw % mod
            expr = f"({prev_v} * {a} + {b}) mod {mod}"
            slip_v = (raw + 1) % mod
            no_mod_v = raw
            prompt_lines.append(f"  step_{k}: v{k} = (v{k - 1} * {a} + {b}) mod {mod}")
        elif op == "SQUARE_SUB":
            raw = prev_v * prev_v - a * prev_v + b
            v = raw % mod
            expr = f"({prev_v}^2 - {a}*{prev_v} + {b}) mod {mod}"
            slip_v = (raw + 2) % mod
            no_mod_v = abs(raw)
            prompt_lines.append(
                f"  step_{k}: v{k} = (v{k - 1}^2 - {a} * v{k - 1} + {b}) mod {mod}"
            )
        else:
            raw = (prev_v + a) * b
            v = raw % mod
            expr = f"(({prev_v} + {a}) * {b}) mod {mod}"
            slip_v = (raw - 1) % mod
            no_mod_v = raw
            prompt_lines.append(f"  step_{k}: v{k} = ((v{k - 1} + {a}) * {b}) mod {mod}")

        op_cls = ops.index(op)
        canonical_ops.append((op, a, b, mod))
        steps.append(
            ReasoningStepSpec(
                step_index=k,
                op=op,
                required_op_class=op_cls,
                args=(a, b, mod),
                expression=expr,
                expected_state=str(v),
                expected_numeric=v,
                alt_slip_state=str(slip_v),
                alt_mod_error_state=str(no_mod_v),
                prev_state=str(prev_v),
            )
        )
        step_feats.append(
            _build_step_features(
                domain="math_reasoning",
                step_idx=k,
                depth=depth,
                op_class=op_cls,
                arg_scale=(a + b) / float(mod),
                is_ood=is_ood,
                split=split,
            )
        )

    prompt_lines.append(
        "Show each intermediate step inside <think>...</think> and give only the final integer in <answer>...</answer>."
    )
    prompt = "\n".join(prompt_lines)
    canonical_fp = _canonical_sha256(
        {
            "domain": "math_reasoning",
            "mod": mod,
            "v0": int(steps[0].prev_state),
            "ops": canonical_ops,
        }
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    features = _build_problem_features(
        domain="math_reasoning",
        depth=depth,
        op_counts=op_counts,
        complexity_scalar=(mod * depth) / 100.0,
        template_id=template_id,
        split=split,
        is_ood=is_ood,
        fingerprint_hex=canonical_fp,
    )
    return ReasoningProblem(
        problem_id=problem_id,
        domain="math_reasoning",
        split=split,
        is_ood=is_ood,
        depth=depth,
        prompt=prompt,
        initial_state_str=initial_str,
        steps=tuple(steps),
        canonical_answer=str(v),
        prompt_sha256=prompt_hash,
        canonical_fingerprint=canonical_fp,
        features=features,
        step_features=tuple(step_feats),
        template_id=template_id,
    )


def _generate_logic_problem(
    rng: random.Random,
    problem_id: str,
    split: str,
    is_ood: bool,
    depth: int,
    template_id: int,
) -> ReasoningProblem:
    # Disjoint variable naming schemes per split for strict prompt & spec separation
    var_names_by_split = {
        "train": ("A", "B", "C", "D"),
        "val": ("P", "Q", "R", "S"),
        "held_out_test": ("X", "Y", "Z", "W"),
    }
    var_names = var_names_by_split[split]
    var_vals = [rng.randint(0, 1) for _ in range(4)]
    s = var_vals[0]
    bindings_str = ", ".join(
        f"{name}={_bool_str(val)}" for name, val in zip(var_names, var_vals, strict=True)
    )

    ops = ("AND_NOT", "XOR_OR", "IMPLIES_XOR")
    op_counts = {op: 0 for op in ops}
    steps: list[ReasoningStepSpec] = []
    step_feats: list[tuple[float, ...]] = []
    prompt_lines: list[str] = []

    if template_id % 3 == 0:
        prompt_lines.append(
            f"[Symbolic Logic Propagation] Given {bindings_str} and s0 = {var_names[0]}, evaluate:"
        )
    elif template_id % 3 == 1:
        prompt_lines.append(
            f"Boolean Constraint Chain with {bindings_str}; initialize s0 = {_bool_str(s)}:"
        )
    else:
        prompt_lines.append(
            f"Deduce the final truth value s{depth} from {bindings_str} (s0 = {_bool_str(s)}):"
        )

    canonical_ops: list[tuple[str, int, int]] = []
    for k in range(1, depth + 1):
        op = rng.choice(ops)
        op_counts[op] += 1
        i1 = rng.randint(0, 3)
        i2 = rng.randint(0, 3)
        prev_s = s
        if op == "AND_NOT":
            flip = i2 % 2
            s = int((bool(prev_s) and (not bool(var_vals[i1]))) ^ bool(flip))
            expr = f"({_bool_str(prev_s)} AND NOT {var_names[i1]}) XOR {_bool_str(flip)}"
            prompt_lines.append(
                f"  step_{k}: s{k} = (s{k - 1} AND NOT {var_names[i1]}) XOR {_bool_str(flip)}"
            )
            mod_err = int(bool(prev_s) and (not bool(var_vals[i1])))
            arg_pair = (i1, flip)
        elif op == "XOR_OR":
            s = int((bool(prev_s) ^ bool(var_vals[i1])) or bool(var_vals[i2]))
            expr = f"({_bool_str(prev_s)} XOR {var_names[i1]}) OR {var_names[i2]}"
            prompt_lines.append(
                f"  step_{k}: s{k} = (s{k - 1} XOR {var_names[i1]}) OR {var_names[i2]}"
            )
            mod_err = int(bool(prev_s) ^ bool(var_vals[i1]))
            arg_pair = (i1, i2)
        else:
            s = int(((not bool(prev_s)) or bool(var_vals[i1])) ^ bool(var_vals[i2]))
            expr = f"({_bool_str(prev_s)} -> {var_names[i1]}) XOR {var_names[i2]}"
            prompt_lines.append(
                f"  step_{k}: s{k} = (s{k - 1} -> {var_names[i1]}) XOR {var_names[i2]}"
            )
            mod_err = int((not bool(prev_s)) or bool(var_vals[i1]))
            arg_pair = (i1, i2)

        op_cls = ops.index(op)
        slip_s = 1 - s
        canonical_ops.append((op, arg_pair[0], arg_pair[1]))
        steps.append(
            ReasoningStepSpec(
                step_index=k,
                op=op,
                required_op_class=op_cls,
                args=arg_pair,
                expression=expr,
                expected_state=_bool_str(s),
                expected_numeric=s,
                alt_slip_state=_bool_str(slip_s),
                alt_mod_error_state=_bool_str(mod_err),
                prev_state=_bool_str(prev_s),
            )
        )
        step_feats.append(
            _build_step_features(
                domain="symbolic_logic",
                step_idx=k,
                depth=depth,
                op_class=op_cls,
                arg_scale=(arg_pair[0] + arg_pair[1] + 1) / 8.0,
                is_ood=is_ood,
                split=split,
            )
        )

    prompt_lines.append(
        "Show each intermediate truth value inside <think>...</think> and give only TRUE or FALSE in <answer>...</answer>."
    )
    prompt = "\n".join(prompt_lines)
    canonical_fp = _canonical_sha256(
        {
            "domain": "symbolic_logic",
            "split_vars": var_names,
            "var_vals": var_vals,
            "ops": canonical_ops,
        }
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    features = _build_problem_features(
        domain="symbolic_logic",
        depth=depth,
        op_counts=op_counts,
        complexity_scalar=(depth + sum(var_vals)) / 8.0,
        template_id=template_id,
        split=split,
        is_ood=is_ood,
        fingerprint_hex=canonical_fp,
    )
    return ReasoningProblem(
        problem_id=problem_id,
        domain="symbolic_logic",
        split=split,
        is_ood=is_ood,
        depth=depth,
        prompt=prompt,
        initial_state_str=bindings_str,
        steps=tuple(steps),
        canonical_answer=_bool_str(s),
        prompt_sha256=prompt_hash,
        canonical_fingerprint=canonical_fp,
        features=features,
        step_features=tuple(step_feats),
        template_id=template_id,
    )


def _generate_code_trace_problem(
    rng: random.Random,
    problem_id: str,
    split: str,
    is_ood: bool,
    depth: int,
    template_id: int,
) -> ReasoningProblem:
    mod_pools = {
        "train": (11, 13, 17, 19),
        "val": (23, 29),
        "held_out_test": (31, 37, 41, 43),
    }
    mod = rng.choice(mod_pools[split])
    r0 = rng.randint(1, mod - 1)
    r1 = rng.randint(1, mod - 1)
    init_str = f"r0={r0},r1={r1}"

    ops = ("ADD_R0", "SUB_R1", "MUL_MOD", "IF_EVEN_ADD", "SWAP_INC")
    op_counts = {op: 0 for op in ops}
    steps: list[ReasoningStepSpec] = []
    step_feats: list[tuple[float, ...]] = []
    prompt_lines: list[str] = []

    if template_id % 3 == 0:
        prompt_lines.append(
            f"[Code Trace] Registers start at r0={r0}, r1={r1} (all ops mod {mod}). Trace:"
        )
    elif template_id % 3 == 1:
        prompt_lines.append(
            f"State-Machine Execution (mod {mod}): init r0={r0}, r1={r1}. Execute {depth} instructions:"
        )
    else:
        prompt_lines.append(
            f"Simulate register machine mod {mod} starting from (r0={r0}, r1={r1}):"
        )

    canonical_ops: list[tuple[str, int, int]] = []
    for k in range(1, depth + 1):
        op = rng.choice(ops)
        op_counts[op] += 1
        c = rng.randint(1, 6)
        prev_r0, prev_r1 = r0, r1
        prev_state = f"r0={prev_r0},r1={prev_r1}"

        if op == "ADD_R0":
            raw_r0 = prev_r0 + prev_r1 + c
            r0, r1 = raw_r0 % mod, prev_r1
            expr = f"r0=({prev_r0}+{prev_r1}+{c})%{mod},r1={prev_r1}"
            slip_state = f"r0={(raw_r0 + 1) % mod},r1={prev_r1}"
            nomod_state = f"r0={raw_r0},r1={prev_r1}"
            prompt_lines.append(f"  step_{k}: r0 = (r0 + r1 + {c}) % {mod}")
        elif op == "SUB_R1":
            raw_r1 = prev_r1 - prev_r0 + c
            r0, r1 = prev_r0, raw_r1 % mod
            expr = f"r0={prev_r0},r1=({prev_r1}-{prev_r0}+{c})%{mod}"
            slip_state = f"r0={prev_r0},r1={(raw_r1 + 1) % mod}"
            nomod_state = f"r0={prev_r0},r1={raw_r1}"
            prompt_lines.append(f"  step_{k}: r1 = (r1 - r0 + {c}) % {mod}")
        elif op == "MUL_MOD":
            raw_r0 = prev_r0 * c + prev_r1
            r0, r1 = raw_r0 % mod, prev_r1
            expr = f"r0=({prev_r0}*{c}+{prev_r1})%{mod},r1={prev_r1}"
            slip_state = f"r0={(raw_r0 + 1) % mod},r1={prev_r1}"
            nomod_state = f"r0={raw_r0},r1={prev_r1}"
            prompt_lines.append(f"  step_{k}: r0 = (r0 * {c} + r1) % {mod}")
        elif op == "IF_EVEN_ADD":
            if prev_r0 % 2 == 0:
                raw_r1 = prev_r1 + c
                r0, r1 = prev_r0, raw_r1 % mod
                expr = f"even(r0)->r1=({prev_r1}+{c})%{mod}"
                slip_state = f"r0={(prev_r0 + c) % mod},r1={prev_r1}"
                nomod_state = f"r0={prev_r0},r1={raw_r1}"
            else:
                raw_r0 = prev_r0 + c
                r0, r1 = raw_r0 % mod, prev_r1
                expr = f"odd(r0)->r0=({prev_r0}+{c})%{mod}"
                slip_state = f"r0={prev_r0},r1={(prev_r1 + c) % mod}"
                nomod_state = f"r0={raw_r0},r1={prev_r1}"
            prompt_lines.append(
                f"  step_{k}: if r0 % 2 == 0: r1=(r1+{c})%{mod} else: r0=(r0+{c})%{mod}"
            )
        else:
            raw_r0 = prev_r1 + c
            r0, r1 = raw_r0 % mod, prev_r0 % mod
            expr = f"swap_inc({c})->r0=({prev_r1}+{c})%{mod},r1={prev_r0}"
            slip_state = f"r0={prev_r1},r1={(prev_r0 + c) % mod}"
            nomod_state = f"r0={raw_r0},r1={prev_r0}"
            prompt_lines.append(f"  step_{k}: r0, r1 = (r1 + {c}) % {mod}, r0 % {mod}")

        op_cls = 0 if op in ("ADD_R0", "SUB_R1") else (1 if op == "MUL_MOD" else 2)
        state_str = f"r0={r0},r1={r1}"
        canonical_ops.append((op, c, mod))
        steps.append(
            ReasoningStepSpec(
                step_index=k,
                op=op,
                required_op_class=op_cls,
                args=(c, mod),
                expression=expr,
                expected_state=state_str,
                expected_numeric=r0 * 100 + r1,
                alt_slip_state=slip_state,
                alt_mod_error_state=nomod_state,
                prev_state=prev_state,
            )
        )
        step_feats.append(
            _build_step_features(
                domain="code_trace",
                step_idx=k,
                depth=depth,
                op_class=op_cls,
                arg_scale=c / float(mod),
                is_ood=is_ood,
                split=split,
            )
        )

    prompt_lines.append(
        "Show each register state inside <think>...</think> and give only r0=<int>,r1=<int> in <answer>...</answer>."
    )
    prompt = "\n".join(prompt_lines)
    canonical_fp = _canonical_sha256(
        {
            "domain": "code_trace",
            "mod": mod,
            "init": init_str,
            "ops": canonical_ops,
        }
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    # Collapse 5 ops into 3 grouped buckets for feature vector parity
    grouped_counts = {
        "arith": op_counts["ADD_R0"] + op_counts["SUB_R1"],
        "mul": op_counts["MUL_MOD"],
        "ctrl": op_counts["IF_EVEN_ADD"] + op_counts["SWAP_INC"],
    }
    features = _build_problem_features(
        domain="code_trace",
        depth=depth,
        op_counts=grouped_counts,
        complexity_scalar=(mod * depth) / 100.0,
        template_id=template_id,
        split=split,
        is_ood=is_ood,
        fingerprint_hex=canonical_fp,
    )
    return ReasoningProblem(
        problem_id=problem_id,
        domain="code_trace",
        split=split,
        is_ood=is_ood,
        depth=depth,
        prompt=prompt,
        initial_state_str=init_str,
        steps=tuple(steps),
        canonical_answer=f"r0={r0},r1={r1}",
        prompt_sha256=prompt_hash,
        canonical_fingerprint=canonical_fp,
        features=features,
        step_features=tuple(step_feats),
        template_id=template_id,
    )


_DOMAIN_GENERATORS = {
    "math_reasoning": _generate_math_problem,
    "symbolic_logic": _generate_logic_problem,
    "code_trace": _generate_code_trace_problem,
}


def build_reasoning_dataset(
    seed: int = 1350,
    train_per_domain: int = 60,
    val_per_domain: int = 20,
    test_id_per_domain: int = 24,
    test_ood_per_domain: int = 16,
) -> ReasoningDatasetBundle:
    """Construct pinned, strictly disjoint train / val / held-out test splits.

    Default sizes:
    - ``train``: 180 problems (60 per domain, depth 2-3)
    - ``val``: 60 problems (20 per domain, depth 2-3)
    - ``held_out_test``: 120 problems (72 in-distribution depth 2-3 + 48 OOD depth 4-5)
    """
    seen_prompts: set[str] = set()
    seen_fingerprints: set[str] = set()

    def _gen_split(
        split_name: str,
        per_domain_id: int,
        per_domain_ood: int,
    ) -> tuple[ReasoningProblem, ...]:
        rng = random.Random(_derived_seed(seed, f"split:{split_name}"))
        problems: list[ReasoningProblem] = []
        template_range = (
            (0, 1, 2) if split_name == "train" else (3, 4, 5)
        )
        for domain in DOMAINS:
            gen_fn = _DOMAIN_GENERATORS[domain]
            # In-distribution problems (depth 2..3)
            count = 0
            attempt = 0
            while count < per_domain_id:
                attempt += 1
                if attempt > per_domain_id * 50:
                    raise RuntimeError(f"Exhausted retries generating {split_name}:{domain}:id")
                depth = rng.choice((2, 3))
                tmpl = rng.choice(template_range)
                pid = f"{split_name}_{domain}_id_{count:04d}"
                prob = gen_fn(rng, pid, split_name, False, depth, tmpl)
                if (
                    prob.prompt_sha256 in seen_prompts
                    or prob.canonical_fingerprint in seen_fingerprints
                ):
                    continue
                seen_prompts.add(prob.prompt_sha256)
                seen_fingerprints.add(prob.canonical_fingerprint)
                problems.append(prob)
                count += 1

            # Out-of-distribution longer-chain problems (depth 4..5)
            count_ood = 0
            attempt = 0
            while count_ood < per_domain_ood:
                attempt += 1
                if attempt > max(1, per_domain_ood) * 50:
                    raise RuntimeError(f"Exhausted retries generating {split_name}:{domain}:ood")
                depth = rng.choice((4, 5))
                tmpl = rng.choice(template_range)
                pid = f"{split_name}_{domain}_ood_{count_ood:04d}"
                prob = gen_fn(rng, pid, split_name, True, depth, tmpl)
                if (
                    prob.prompt_sha256 in seen_prompts
                    or prob.canonical_fingerprint in seen_fingerprints
                ):
                    continue
                seen_prompts.add(prob.prompt_sha256)
                seen_fingerprints.add(prob.canonical_fingerprint)
                problems.append(prob)
                count_ood += 1

        # Deterministic interleave across domains
        rng.shuffle(problems)
        return tuple(problems)

    train = _gen_split("train", train_per_domain, 0)
    val = _gen_split("val", val_per_domain, 0)
    held_out = _gen_split("held_out_test", test_id_per_domain, test_ood_per_domain)

    split_sha256 = {
        "train": _canonical_sha256([p.to_manifest_record() for p in train]),
        "val": _canonical_sha256([p.to_manifest_record() for p in val]),
        "held_out_test": _canonical_sha256([p.to_manifest_record() for p in held_out]),
    }
    manifest_sha256 = _canonical_sha256(
        {
            "seed": seed,
            "license": "Apache-2.0",
            "splits": split_sha256,
            "counts": {
                "train": len(train),
                "val": len(val),
                "held_out_test": len(held_out),
            },
        }
    )

    bundle = ReasoningDatasetBundle(
        seed=seed,
        train=train,
        val=val,
        held_out_test=held_out,
        split_sha256=split_sha256,
        dataset_manifest_sha256=manifest_sha256,
    )
    leak_report = bundle.verify_disjointness()
    if not leak_report["zero_leakage"]:
        raise AssertionError(f"Dataset leakage detected: {leak_report}")
    return bundle


def load_license_audit(path: Path | None = None) -> dict[str, Any]:
    """Load the repository's LICENSE_AUDIT.json manifest."""
    audit_path = path or (Path(__file__).resolve().parent / "LICENSE_AUDIT.json")
    with audit_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
