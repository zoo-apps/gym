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
"""Auditable deterministic verifiers and reward functions for GRPO reasoning.

All reward functions are pure, stateless, and deterministic — no LLM-as-a-judge.
Every evaluation returns a canonical ``RewardEvaluation`` containing:
- ``total_reward`` in ``[0.0, 1.0]``
- component breakdown (``format_reward``, ``correctness_reward``,
  ``reasoning_coherence_reward``)
- explicit anti-reward-hacking flags
- cryptographic ``audit_hash`` (SHA-256 over canonical evaluation record)

Also provides ``RewardAuditTrail`` for tamper-evident chained auditing and
importable adapter hooks for TRL (``GRPOTrainer``) and ``verl``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .dataset import ReasoningProblem

VERIFIER_VERSION = "hip1350-verifier-v1.0.0"

# Reward weights sum strictly to 1.00
WEIGHT_FORMAT = 0.25
WEIGHT_CORRECTNESS = 0.60
WEIGHT_COHERENCE = 0.15

MAX_COMPLETION_CHARS = 1200

# Canonical strict answer regexes per domain
_MATH_ANSWER_RE = re.compile(r"^-?\d+$")
_LOGIC_ANSWER_RE = re.compile(r"^(TRUE|FALSE)$")
_CODE_ANSWER_RE = re.compile(r"^r0=-?\d+,r1=-?\d+$")

# Strict outer XML envelope: only optional whitespace around <think> and <answer>
_STRICT_ENVELOPE_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)

# Step extractor inside <think>: matches "step_1: <expr> => <state>"
_STEP_ENTRY_RE = re.compile(
    r"step_(?P<idx>\d+)\s*:\s*(?P<expr>[^;\n]+?)\s*=>\s*(?P<state>[^;\n]+)"
)

_DOMAIN_OPERATORS = frozenset(
    {"true", "false", "and", "or", "not", "xor", "mod", "r0", "r1", "even", "odd"}
)


@dataclass(frozen=True)
class RewardEvaluation:
    """Immutable, cryptographically hashed deterministic reward record."""

    problem_id: str
    total_reward: float
    format_reward: float
    correctness_reward: float
    coherence_reward: float
    is_exact_correct: bool
    is_format_valid: bool
    verified_step_fraction: float
    hacking_flags: tuple[str, ...]
    extracted_answer: str
    audit_hash: str
    completion_sha256: str

    def as_tuple(self) -> tuple[float, dict[str, Any], str]:
        """Return standard ``(total_reward, breakdown, audit_hash)`` triple."""
        breakdown = {
            "format_reward": self.format_reward,
            "correctness_reward": self.correctness_reward,
            "coherence_reward": self.coherence_reward,
            "is_exact_correct": self.is_exact_correct,
            "is_format_valid": self.is_format_valid,
            "verified_step_fraction": self.verified_step_fraction,
            "hacking_flags": list(self.hacking_flags),
            "extracted_answer": self.extracted_answer,
            "verifier_version": VERIFIER_VERSION,
        }
        return self.total_reward, breakdown, self.audit_hash


def _detect_hacking_flags(
    completion: str,
    think_body: str | None,
    answer_body: str | None,
    domain: str,
) -> list[str]:
    """Run deterministic anti-reward-hacking inspections on raw completion text."""
    flags: list[str] = []

    # 1. Duplicate or malformed XML tags
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if completion.count(tag) > 1:
            flags.append("DUPLICATE_TAG")
            break

    # 2. Trailing/leading garbage or non-XML text outside envelope
    if (
        "<think>" in completion
        and "</think>" in completion
        and "<answer>" in completion
        and "</answer>" in completion
        and _STRICT_ENVELOPE_RE.match(completion) is None
    ):
        flags.append("TRAILING_GARBAGE")

    # 3. Length padding / repetitive token stuffing
    if len(completion) > MAX_COMPLETION_CHARS:
        flags.append("LENGTH_PADDING_HACK")
    elif think_body is not None:
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", think_body.lower())
        non_op_words = [
            w for w in words if w not in _DOMAIN_OPERATORS and not w.startswith("step_")
        ]
        if non_op_words:
            top_word_count = Counter(non_op_words).most_common(1)[0][1]
            if top_word_count >= 6:
                flags.append("LENGTH_PADDING_HACK")
            elif len(non_op_words) >= 20:
                unique_ratio = len(set(non_op_words)) / float(len(non_op_words))
                if unique_ratio < 0.30:
                    flags.append("LENGTH_PADDING_HACK")

    # 4. Answer stuffing (multiple candidate values or disjunctions in <answer>)
    if answer_body is not None:
        ans = answer_body.strip()
        if any(sep in ans.lower() for sep in (" or ", " / ", "|", ";")):
            flags.append("ANSWER_STUFFING")
        elif domain == "math_reasoning":
            if len(re.findall(r"-?\d+", ans)) > 1:
                flags.append("ANSWER_STUFFING")
        elif domain == "symbolic_logic":
            if len(re.findall(r"\b(?:TRUE|FALSE)\b", ans.upper())) > 1:
                flags.append("ANSWER_STUFFING")
        elif domain == "code_trace":
            if len(re.findall(r"r0=-?\d+,r1=-?\d+", ans)) > 1:
                flags.append("ANSWER_STUFFING")

    return sorted(set(flags))


def _parse_steps(think_body: str) -> dict[int, tuple[str, str]]:
    """Extract step_k -> (expr, state) mapping; detects duplicate step indices."""
    parsed: dict[int, tuple[str, str]] = {}
    for match in _STEP_ENTRY_RE.finditer(think_body):
        idx = int(match.group("idx"))
        expr = match.group("expr").strip()
        state = match.group("state").strip()
        if idx not in parsed:
            parsed[idx] = (expr, state)
    return parsed


def format_reward(
    completion: str,
    domain: str = "math_reasoning",
) -> tuple[float, bool, tuple[str, ...]]:
    """Evaluate strict XML structure and non-empty multi-step reasoning envelope.

    Returns ``(score_in_0_to_0.25, is_valid, flags)``.
    """
    match = _STRICT_ENVELOPE_RE.match(completion)
    think_body = match.group("think") if match else None
    answer_body = match.group("answer") if match else None
    flags = _detect_hacking_flags(completion, think_body, answer_body, domain)

    if match is None or flags:
        return 0.0, False, tuple(flags)

    assert think_body is not None and answer_body is not None
    think_stripped = think_body.strip()
    ans_stripped = answer_body.strip()
    if not think_stripped or not ans_stripped:
        return 0.0, False, ("EMPTY_SECTION",)

    steps = _parse_steps(think_stripped)
    if len(steps) < 1:
        # Valid tags, but missing structured step derivations
        return round(WEIGHT_FORMAT * 0.4, 6), False, ("MISSING_STEP_MARKERS",)

    return WEIGHT_FORMAT, True, ()


def correctness_reward(
    problem: ReasoningProblem,
    completion: str,
) -> tuple[float, bool, str, tuple[str, ...]]:
    """Exact canonical domain verifier with anti-reward-hacking guards.

    Returns ``(score_in_0_to_0.60, is_exact_correct, extracted_answer, flags)``.
    """
    match = _STRICT_ENVELOPE_RE.match(completion)
    think_body = match.group("think") if match else None
    answer_body = match.group("answer") if match else None
    flags = _detect_hacking_flags(completion, think_body, answer_body, problem.domain)

    if match is None:
        return 0.0, False, "", tuple(flags)

    assert think_body is not None and answer_body is not None
    extracted = answer_body.strip()

    # Reject any hacking attempt immediately (zero correctness credit)
    if any(
        f in flags
        for f in ("DUPLICATE_TAG", "TRAILING_GARBAGE", "ANSWER_STUFFING", "LENGTH_PADDING_HACK")
    ):
        return 0.0, False, extracted, tuple(flags)

    # Strict canonical format check per domain
    if problem.domain == "math_reasoning":
        canonical_ok = _MATH_ANSWER_RE.match(extracted) is not None
    elif problem.domain == "symbolic_logic":
        canonical_ok = _LOGIC_ANSWER_RE.match(extracted) is not None
    elif problem.domain == "code_trace":
        canonical_ok = _CODE_ANSWER_RE.match(extracted) is not None
    else:
        raise ValueError(f"Unknown reasoning domain: {problem.domain}")

    if not canonical_ok:
        return 0.0, False, extracted, tuple(flags)

    if extracted != problem.canonical_answer:
        return 0.0, False, extracted, tuple(flags)

    # Anti-unsupported-guess guard: must have at least one valid step in <think>
    parsed_steps = _parse_steps(think_body)
    verified_count = sum(
        1
        for spec in problem.steps
        if spec.step_index in parsed_steps
        and parsed_steps[spec.step_index][1] == spec.expected_state
    )
    if verified_count == 0:
        flags.append("UNSUPPORTED_GUESS")
        # Heavily penalize guessing without derivation (not credited as verified pass)
        return round(WEIGHT_CORRECTNESS * 0.15, 6), False, extracted, tuple(sorted(set(flags)))

    # Check consistency between final derived step and <answer>
    last_step_idx = max(parsed_steps.keys())
    if parsed_steps[last_step_idx][1] != extracted:
        flags.append("STEP_MISMATCH")
        return round(WEIGHT_CORRECTNESS * 0.25, 6), False, extracted, tuple(sorted(set(flags)))

    return WEIGHT_CORRECTNESS, True, extracted, tuple(sorted(set(flags)))


def reasoning_coherence_reward(
    problem: ReasoningProblem,
    completion: str,
) -> tuple[float, float, tuple[str, ...]]:
    """Deterministic intermediate step-by-step verification reward.

    Independently checks each ``step_k`` derivation in ``<think>`` against the
    problem's deterministic ground-truth state transition trace.
    Returns ``(score_in_0_to_0.15, verified_step_fraction, flags)``.
    """
    match = _STRICT_ENVELOPE_RE.match(completion)
    if match is None:
        return 0.0, 0.0, ()

    think_body = match.group("think")
    answer_body = match.group("answer").strip()
    flags = _detect_hacking_flags(completion, think_body, answer_body, problem.domain)
    if flags:
        return 0.0, 0.0, tuple(flags)

    parsed_steps = _parse_steps(think_body)
    if not parsed_steps:
        return 0.0, 0.0, ("MISSING_STEP_MARKERS",)

    verified = 0
    for spec in problem.steps:
        if spec.step_index in parsed_steps:
            _, derived_state = parsed_steps[spec.step_index]
            if derived_state == spec.expected_state:
                verified += 1

    # Penalize spurious extra steps beyond problem depth
    extra_steps = [idx for idx in parsed_steps if idx < 1 or idx > problem.depth]
    if extra_steps:
        flags.append("EXTRA_SPURIOUS_STEPS")

    frac = verified / float(max(1, problem.depth))
    if extra_steps:
        frac = max(0.0, frac - 0.25 * len(extra_steps))

    # Consistency check with final answer
    last_idx = max(parsed_steps.keys())
    if parsed_steps[last_idx][1] != answer_body:
        flags.append("STEP_MISMATCH")
        frac *= 0.5

    score = round(WEIGHT_COHERENCE * frac, 6)
    return score, round(frac, 6), tuple(sorted(set(flags)))


def evaluate_completion(
    problem: ReasoningProblem,
    completion: str,
) -> RewardEvaluation:
    """Compute full auditable deterministic reward and SHA-256 audit hash."""
    fmt_score, fmt_valid, fmt_flags = format_reward(completion, problem.domain)
    corr_score, exact_ok, extracted, corr_flags = correctness_reward(
        problem, completion
    )
    coh_score, step_frac, coh_flags = reasoning_coherence_reward(problem, completion)

    all_flags = tuple(sorted(set(fmt_flags) | set(corr_flags) | set(coh_flags)))
    total = round(min(1.0, max(0.0, fmt_score + corr_score + coh_score)), 6)

    comp_sha = hashlib.sha256(completion.encode("utf-8")).hexdigest()
    audit_payload = {
        "verifier_version": VERIFIER_VERSION,
        "problem_id": problem.problem_id,
        "completion_sha256": comp_sha,
        "format_reward": fmt_score,
        "correctness_reward": corr_score,
        "coherence_reward": coh_score,
        "total_reward": total,
        "is_exact_correct": exact_ok,
        "is_format_valid": fmt_valid,
        "verified_step_fraction": step_frac,
        "hacking_flags": list(all_flags),
    }
    audit_hash = hashlib.sha256(
        json.dumps(audit_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return RewardEvaluation(
        problem_id=problem.problem_id,
        total_reward=total,
        format_reward=fmt_score,
        correctness_reward=corr_score,
        coherence_reward=coh_score,
        is_exact_correct=exact_ok,
        is_format_valid=fmt_valid,
        verified_step_fraction=step_frac,
        hacking_flags=all_flags,
        extracted_answer=extracted,
        audit_hash=audit_hash,
        completion_sha256=comp_sha,
    )


def naive_proxy_reward(problem: ReasoningProblem, completion: str) -> float:
    """Intentionally flawed format+length proxy reward (NEGATIVE REGIME ONLY).

    Demonstrates Goodhart's Law / reward hacking when GRPO optimizes a proxy
    that rewards XML tag presence, verbose length, and candidate-heavy answers
    without strict exact correctness or anti-padding verification.
    """
    _ = problem
    score = 0.0
    if "<think>" in completion and "</think>" in completion:
        score += 0.30
    if "<answer>" in completion and "</answer>" in completion:
        score += 0.25
    # Vulnerable length bonus: rewards padded/repetitive completions!
    length_bonus = min(0.35, len(completion) / 650.0 * 0.35)
    score += length_bonus
    # Vulnerable multiple-answer / tag density bonus
    if completion.count("<answer>") > 1 or " or " in completion:
        score += 0.10
    return round(min(1.0, score), 6)


class RewardAuditTrail:
    """Append-only cryptographic audit trail with hacking telemetry."""

    def __init__(self, regime_name: str) -> None:
        self.regime_name = regime_name
        self._chain_hash = hashlib.sha256(
            f"{VERIFIER_VERSION}:{regime_name}".encode("utf-8")
        ).hexdigest()
        self.total_evaluated = 0
        self.exact_correct_count = 0
        self.format_valid_count = 0
        self.sum_total_reward = 0.0
        self.sum_step_fraction = 0.0
        self.flag_counts: Counter[str] = Counter()
        self.flagged_completions = 0

    def record(self, evaluation: RewardEvaluation) -> None:
        self.total_evaluated += 1
        if evaluation.is_exact_correct:
            self.exact_correct_count += 1
        if evaluation.is_format_valid:
            self.format_valid_count += 1
        self.sum_total_reward += evaluation.total_reward
        self.sum_step_fraction += evaluation.verified_step_fraction
        if evaluation.hacking_flags:
            self.flagged_completions += 1
            for flag in evaluation.hacking_flags:
                self.flag_counts[flag] += 1
        self._chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{evaluation.audit_hash}".encode("utf-8")
        ).hexdigest()

    @property
    def chain_sha256(self) -> str:
        return self._chain_hash

    def summary(self) -> dict[str, Any]:
        denom = max(1, self.total_evaluated)
        return {
            "regime_name": self.regime_name,
            "verifier_version": VERIFIER_VERSION,
            "total_evaluated": self.total_evaluated,
            "exact_correct_rate": round(self.exact_correct_count / denom, 6),
            "format_valid_rate": round(self.format_valid_count / denom, 6),
            "mean_total_reward": round(self.sum_total_reward / denom, 6),
            "mean_verified_step_fraction": round(self.sum_step_fraction / denom, 6),
            "flagged_rate": round(self.flagged_completions / denom, 6),
            "hacking_flag_counts": dict(sorted(self.flag_counts.items())),
            "audit_chain_sha256": self._chain_hash,
        }


# ---------------------------------------------------------------------------
# Importable adapter hooks referenced by TRL & verl production YAML recipes
# ---------------------------------------------------------------------------


def trl_format_reward_func(
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """TRL GRPOTrainer-compatible format reward callable."""
    domain = str(kwargs.get("domain", "math_reasoning"))
    return [format_reward(c, domain=domain)[0] for c in completions]


def trl_correctness_reward_func(
    completions: list[str],
    ground_truth: list[str] | None = None,
    domain: str = "math_reasoning",
    **kwargs: Any,
) -> list[float]:
    """TRL GRPOTrainer-compatible strict canonical correctness callable."""
    _ = kwargs
    if ground_truth is None:
        return [0.0 for _ in completions]
    scores: list[float] = []
    for comp, truth in zip(completions, ground_truth, strict=True):
        match = _STRICT_ENVELOPE_RE.match(comp)
        if match is None:
            scores.append(0.0)
            continue
        flags = _detect_hacking_flags(
            comp, match.group("think"), match.group("answer"), domain
        )
        if flags:
            scores.append(0.0)
            continue
        ans = match.group("answer").strip()
        scores.append(WEIGHT_CORRECTNESS if ans == str(truth).strip() else 0.0)
    return scores


def verl_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """verl custom_reward_function.name compatible deterministic scorer."""
    _ = data_source
    domain = (extra_info or {}).get("domain", "math_reasoning")
    fmt_s, _, fmt_flags = format_reward(solution_str, domain=domain)
    if fmt_flags:
        return 0.0
    match = _STRICT_ENVELOPE_RE.match(solution_str)
    if match is None:
        return 0.0
    ans = match.group("answer").strip()
    corr_s = WEIGHT_CORRECTNESS if ans == str(ground_truth).strip() else 0.0
    return round(fmt_s + corr_s, 6)
