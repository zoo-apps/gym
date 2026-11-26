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
"""Pinned base reasoning policy and exact Group Relative Policy Optimization (GRPO).

Pure Python >= 3.10 stdlib implementation of:
1. ``StructuredCoTPolicy``: a differentiable 2-layer neural autoregressive
   chain-of-thought policy with SHA-256 pinned initial checkpoint weights
   (``weight_sha256``), exact categorical log-probabilities, exact entropy,
   and exact analytical KL divergence against a frozen reference policy.
2. Deterministic black-box text renderer turning sampled discrete reasoning
   trajectories into XML ``<think>...</think><answer>...</answer>`` strings
   with genuine state-transition error propagation.
3. Exact GRPO objective (Shao et al., DeepSeekMath) with:
   - Group-relative advantage normalization ``A_i = (r_i - mean(r)) / (std(r) + eps)``
   - Token/decision-level PPO clipped surrogate ratio
   - Unbiased Schulman ``k3`` KL penalty ``exp(log_ref - log_pi) - (log_ref - log_pi) - 1``
   - Exact analytical backpropagation verified against finite differences (< 1e-6)
   - AdamW optimizer
4. Matched-budget SFT and vanilla REINFORCE baselines using the exact same
   policy architecture and optimizer.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import Callable

from .dataset import (
    PROBLEM_FEATURE_DIM,
    STEP_FEATURE_DIM,
    ReasoningProblem,
)
from .rewards import RewardAuditTrail, evaluate_completion, naive_proxy_reward

HIDDEN_DIM = 12
NUM_FMT_ACTIONS = 4
NUM_STRAT_ACTIONS = 4
NUM_STEP_ACTIONS = 4
NUM_ANS_ACTIONS = 3

# Format action semantics
FMT_STRICT_XML = 0
FMT_PLAIN_TEXT = 1
FMT_LENGTH_PADDED = 2
FMT_ANSWER_STUFFED = 3

# Strategy action semantics
STRAT_FULL_DECOMP = 0
STRAT_SHALLOW_SHORTCUT = 1
STRAT_WRONG_PRECEDENCE = 2
STRAT_UNSUPPORTED_GUESS = 3

# Step micro-action semantics
STEP_EXACT = 0
STEP_SLIP = 1
STEP_MOD_OMIT = 2
STEP_STAGNATE = 3

# Answer extraction action semantics
ANS_EXTRACT_FINAL = 0
ANS_EXTRACT_EARLY = 1
ANS_FALLBACK_CONST = 2


def _log_softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    log_sum = m + math.log(sum(exps))
    return [z - log_sum for z in logits]


def _categorical_entropy(log_probs: list[float]) -> float:
    return -sum(math.exp(lp) * lp for lp in log_probs)


def _exact_categorical_kl(log_p: list[float], log_q: list[float]) -> float:
    """Exact KL(p || q) = sum_a p(a) * (log p(a) - log q(a))."""
    return sum(math.exp(lp) * (lp - lq) for lp, lq in zip(log_p, log_q, strict=True))


def schulman_k3_kl(log_pi: float, log_ref: float) -> float:
    """Unbiased non-negative Schulman k3 KL estimator: exp(d) - d - 1."""
    delta = max(-20.0, min(20.0, log_ref - log_pi))
    return math.exp(delta) - delta - 1.0


def schulman_k3_kl_grad_wrt_log_pi(log_pi: float, log_ref: float) -> float:
    """Exact analytical derivative d(k3_kl)/d(log_pi) = 1 - exp(log_ref - log_pi)."""
    delta = max(-20.0, min(20.0, log_ref - log_pi))
    return 1.0 - math.exp(delta)


def _sample_from_log_probs(log_probs: list[float], rng: random.Random) -> int:
    u = rng.random()
    cdf = 0.0
    for idx, lp in enumerate(log_probs):
        cdf += math.exp(lp)
        if u <= cdf:
            return idx
    return len(log_probs) - 1


@dataclass(frozen=True)
class Trajectory:
    """Discrete autoregressive reasoning trajectory sampled from policy."""

    a_fmt: int
    a_strat: int
    a_steps: tuple[int, ...]
    a_ans: int
    old_log_probs: tuple[float, ...]  # [fmt, strat, step_1..step_K, ans]
    completion_text: str

    @property
    def num_decisions(self) -> int:
        return len(self.old_log_probs)

    def action_sequence(self) -> tuple[int, ...]:
        return (self.a_fmt, self.a_strat, *self.a_steps, self.a_ans)


def _exec_math_step(prev_val: int, op: str, args: tuple[int, ...]) -> tuple[int, int, int]:
    a, b, mod = args
    if op == "MUL_ADD":
        raw = prev_val * a + b
        return raw % mod, (raw + 1) % mod, raw
    if op == "SQUARE_SUB":
        raw = prev_val * prev_val - a * prev_val + b
        return raw % mod, (raw + 2) % mod, abs(raw)
    raw = (prev_val + a) * b
    return raw % mod, (raw - 1) % mod, raw


def _exec_code_step(
    prev_r0: int, prev_r1: int, op: str, args: tuple[int, ...]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    c, mod = args
    if op == "ADD_R0":
        raw = prev_r0 + prev_r1 + c
        return (raw % mod, prev_r1), ((raw + 1) % mod, prev_r1), (raw, prev_r1)
    if op == "SUB_R1":
        raw = prev_r1 - prev_r0 + c
        return (prev_r0, raw % mod), (prev_r0, (raw + 1) % mod), (prev_r0, raw)
    if op == "MUL_MOD":
        raw = prev_r0 * c + prev_r1
        return (raw % mod, prev_r1), ((raw + 1) % mod, prev_r1), (raw, prev_r1)
    if op == "IF_EVEN_ADD":
        if prev_r0 % 2 == 0:
            raw = prev_r1 + c
            return (prev_r0, raw % mod), ((prev_r0 + c) % mod, prev_r1), (prev_r0, raw)
        raw = prev_r0 + c
        return (raw % mod, prev_r1), (prev_r0, (prev_r1 + c) % mod), (raw, prev_r1)
    raw = prev_r1 + c
    return (raw % mod, prev_r0 % mod), (prev_r1, (prev_r0 + c) % mod), (raw, prev_r0)


def _parse_code_state(state_str: str) -> tuple[int, int]:
    left, right = state_str.split(",")
    return int(left.split("=")[1]), int(right.split("=")[1])


def render_trajectory(
    problem: ReasoningProblem,
    a_fmt: int,
    a_strat: int,
    a_steps: tuple[int, ...],
    a_ans: int,
) -> str:
    """Deterministically render discrete reasoning actions into completion text.

    Executes the chosen reasoning transitions step-by-step with real error
    propagation so the black-box verifier in ``rewards.py`` grades pure text.
    """
    derived_states: list[str] = []
    step_lines: list[str] = []

    if a_strat == STRAT_UNSUPPORTED_GUESS:
        think_body = "intuition without intermediate derivation"
        # Guess from initial state or constant fallback
        if a_ans == ANS_EXTRACT_FINAL:
            final_ans = problem.steps[0].expected_state
        elif a_ans == ANS_EXTRACT_EARLY:
            final_ans = problem.steps[0].alt_slip_state
        else:
            final_ans = _domain_fallback(problem.domain)
    elif a_strat == STRAT_SHALLOW_SHORTCUT:
        spec0 = problem.steps[0]
        act0 = a_steps[0]
        s0 = _apply_step_action_first(spec0, act0)
        step_lines.append(f"step_1: {spec0.expression} => {s0}")
        derived_states.append(s0)
        think_body = "; ".join(step_lines)
        final_ans = s0 if a_ans != ANS_FALLBACK_CONST else _domain_fallback(problem.domain)
    else:
        # Full step-by-step execution (or systematic precedence error when a_strat == 2)
        corrupted = False
        curr_state = problem.steps[0].prev_state
        for k_idx, (spec, act) in enumerate(zip(problem.steps, a_steps, strict=True)):
            eff_act = (
                (spec.required_op_class + 2) % 3
                if a_strat == STRAT_WRONG_PRECEDENCE
                else act
            )
            if not corrupted:
                next_state = _apply_step_action_first(spec, eff_act)
                if next_state != spec.expected_state:
                    corrupted = True
            else:
                next_state = _propagate_corrupted_state(
                    problem.domain, curr_state, spec, eff_act
                )
            curr_state = next_state
            derived_states.append(curr_state)
            step_lines.append(f"step_{k_idx + 1}: {spec.expression} => {curr_state}")
        think_body = "; ".join(step_lines)

        if a_ans == ANS_EXTRACT_FINAL:
            final_ans = derived_states[-1]
        elif a_ans == ANS_EXTRACT_EARLY:
            final_ans = derived_states[0]
        else:
            final_ans = _domain_fallback(problem.domain)

    # Apply chosen formatting protocol
    if a_fmt == FMT_STRICT_XML:
        return f"<think>{think_body}</think><answer>{final_ans}</answer>"
    if a_fmt == FMT_PLAIN_TEXT:
        return f"Reasoning: {think_body}. Final answer is {final_ans}"
    if a_fmt == FMT_LENGTH_PADDED:
        filler = " ".join(["verify_reasoning_padding_step"] * 22)
        return f"<think>{think_body}; {filler}</think><answer>{final_ans}</answer>"
    # FMT_ANSWER_STUFFED: duplicate <answer> tags + multiple candidate answers
    alt = problem.steps[-1].alt_slip_state
    return (
        f"<think>{think_body}</think>"
        f"<answer>{final_ans} or {alt}</answer>"
        f"<answer>{final_ans}</answer>"
    )


def _domain_fallback(domain: str) -> str:
    if domain == "math_reasoning":
        return "0"
    if domain == "symbolic_logic":
        return "FALSE"
    return "r0=0,r1=0"


def _apply_step_action_first(spec, act: int) -> str:
    if act == spec.required_op_class:
        return spec.expected_state
    if act == STEP_STAGNATE:
        return spec.prev_state
    if (act - spec.required_op_class) % 3 == 1:
        return spec.alt_slip_state
    return spec.alt_mod_error_state


def _propagate_corrupted_state(
    domain: str,
    curr_state: str,
    spec,
    act: int,
) -> str:
    if act == STEP_STAGNATE:
        return curr_state
    is_exact_op = act == spec.required_op_class
    is_slip_op = (act - spec.required_op_class) % 3 == 1
    if domain == "math_reasoning":
        try:
            val = int(curr_state)
        except ValueError:
            val = 0
        exact_v, slip_v, nomod_v = _exec_math_step(val, spec.op, spec.args)
        if is_exact_op:
            return str(exact_v)
        if is_slip_op:
            return str(slip_v)
        return str(nomod_v)
    if domain == "code_trace":
        r0, r1 = _parse_code_state(curr_state)
        exact_s, slip_s, nomod_s = _exec_code_step(r0, r1, spec.op, spec.args)
        chosen = exact_s if is_exact_op else (slip_s if is_slip_op else nomod_s)
        return f"r0={chosen[0]},r1={chosen[1]}"
    # symbolic_logic: corrupted boolean state flips expected outcome unless stagnating
    if is_exact_op:
        return spec.alt_slip_state
    if is_slip_op:
        return spec.expected_state
    return spec.alt_mod_error_state


class StructuredCoTPolicy:
    """Differentiable 2-layer neural autoregressive reasoning policy.

    All parameter tensors are stored in flat contiguous float64 vectors with
    exact manual backpropagation and SHA-256 checkpoint fingerprinting.
    """

    # Parameter layout: name -> (rows, cols) or (size,)
    _SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("W_enc", (HIDDEN_DIM, PROBLEM_FEATURE_DIM)),
        ("b_enc", (HIDDEN_DIM,)),
        ("W_fmt", (NUM_FMT_ACTIONS, HIDDEN_DIM)),
        ("b_fmt", (NUM_FMT_ACTIONS,)),
        ("W_strat", (NUM_STRAT_ACTIONS, HIDDEN_DIM)),
        ("E_fmt_strat", (NUM_FMT_ACTIONS, NUM_STRAT_ACTIONS)),
        ("b_strat", (NUM_STRAT_ACTIONS,)),
        ("W_step_x", (NUM_STEP_ACTIONS, STEP_FEATURE_DIM)),
        ("W_step_h", (NUM_STEP_ACTIONS, HIDDEN_DIM)),
        ("E_strat_step", (NUM_STRAT_ACTIONS, NUM_STEP_ACTIONS)),
        ("b_step", (NUM_STEP_ACTIONS,)),
        ("W_ans", (NUM_ANS_ACTIONS, HIDDEN_DIM)),
        ("E_strat_ans", (NUM_STRAT_ACTIONS, NUM_ANS_ACTIONS)),
        ("b_ans", (NUM_ANS_ACTIONS,)),
    )

    def __init__(self, params: dict[str, list[float]]) -> None:
        self.params = {k: list(v) for k, v in params.items()}
        self._total_params = sum(len(v) for v in self.params.values())

    @classmethod
    def create_pinned_base(cls, seed: int = 1350) -> StructuredCoTPolicy:
        """Create a deterministic pinned base policy simulating an un-tuned base LLM.

        The base policy has realistic pretrained priors:
        - Moderate strict XML format compliance (~45-55%) alongside plain text &
          formatting glitches.
        - Weakly aligned pretrained operator routing on ``W_step_x`` (~55-60%
          per-step accuracy across all chain lengths, yielding ~15-25% end-to-end
          held-out ``pass@1`` and substantial headroom for GRPO).
        """
        rng = random.Random(seed ^ 0x5A00_1350)
        params: dict[str, list[float]] = {}
        for name, shape in cls._SPEC:
            size = shape[0] if len(shape) == 1 else shape[0] * shape[1]
            scale = 0.12 if len(shape) == 2 else 0.0
            params[name] = [rng.gauss(0.0, scale) for _ in range(size)]

        # Pretrained operator-to-action alignment in W_step_x (features 1, 2, 3 are op one-hot)
        # Surface-cue columns (9, 10, 11) start at 0.0 in the pinned base model.
        wx = params["W_step_x"]
        for a in range(NUM_STEP_ACTIONS):
            for cue_col in (9, 10, 11):
                wx[a * STEP_FEATURE_DIM + cue_col] = 0.0
        for op_cls in range(3):
            wx[op_cls * STEP_FEATURE_DIM + (1 + op_cls)] += 1.75

        # Calibrated pretrained base biases (realistic base-model behavior)
        params["b_fmt"] = [0.55, 0.30, -0.45, -0.55]
        params["b_strat"] = [0.50, 0.20, -0.25, -0.40]
        params["b_step"] = [0.0, 0.0, 0.0, -0.55]
        params["b_ans"] = [0.95, -0.25, -0.55]

        return cls(params)

    def clone(self) -> StructuredCoTPolicy:
        return StructuredCoTPolicy({k: list(v) for k, v in self.params.items()})

    @property
    def num_parameters(self) -> int:
        return self._total_params

    def weight_sha256(self) -> str:
        """Exact IEEE-754 float64 SHA-256 checkpoint fingerprint."""
        hasher = hashlib.sha256()
        for name, _ in self._SPEC:
            hasher.update(name.encode("utf-8"))
            vec = self.params[name]
            hasher.update(struct.pack(f"<{len(vec)}d", *vec))
        return hasher.hexdigest()

    def zero_grads(self) -> dict[str, list[float]]:
        return {name: [0.0] * len(vec) for name, vec in self.params.items()}

    def _encode_hidden(self, x_prob: tuple[float, ...]) -> list[float]:
        w = self.params["W_enc"]
        b = self.params["b_enc"]
        h = [0.0] * HIDDEN_DIM
        for i in range(HIDDEN_DIM):
            row_off = i * PROBLEM_FEATURE_DIM
            s = b[i]
            for j in range(PROBLEM_FEATURE_DIM):
                s += w[row_off + j] * x_prob[j]
            h[i] = math.tanh(s)
        return h

    def _fmt_logits(self, h: list[float]) -> list[float]:
        w = self.params["W_fmt"]
        b = self.params["b_fmt"]
        return [
            b[a] + sum(w[a * HIDDEN_DIM + j] * h[j] for j in range(HIDDEN_DIM))
            for a in range(NUM_FMT_ACTIONS)
        ]

    def _strat_logits(self, h: list[float], a_fmt: int) -> list[float]:
        w = self.params["W_strat"]
        e = self.params["E_fmt_strat"]
        b = self.params["b_strat"]
        e_off = a_fmt * NUM_STRAT_ACTIONS
        return [
            b[a]
            + e[e_off + a]
            + sum(w[a * HIDDEN_DIM + j] * h[j] for j in range(HIDDEN_DIM))
            for a in range(NUM_STRAT_ACTIONS)
        ]

    def _step_logits(
        self,
        h: list[float],
        x_step: tuple[float, ...],
        a_strat: int,
    ) -> list[float]:
        wx = self.params["W_step_x"]
        wh = self.params["W_step_h"]
        e = self.params["E_strat_step"]
        b = self.params["b_step"]
        e_off = a_strat * NUM_STEP_ACTIONS
        return [
            b[a]
            + e[e_off + a]
            + sum(wx[a * STEP_FEATURE_DIM + j] * x_step[j] for j in range(STEP_FEATURE_DIM))
            + sum(wh[a * HIDDEN_DIM + j] * h[j] for j in range(HIDDEN_DIM))
            for a in range(NUM_STEP_ACTIONS)
        ]

    def _ans_logits(self, h: list[float], a_strat: int) -> list[float]:
        w = self.params["W_ans"]
        e = self.params["E_strat_ans"]
        b = self.params["b_ans"]
        e_off = a_strat * NUM_ANS_ACTIONS
        return [
            b[a]
            + e[e_off + a]
            + sum(w[a * HIDDEN_DIM + j] * h[j] for j in range(HIDDEN_DIM))
            for a in range(NUM_ANS_ACTIONS)
        ]

    def sample_trajectory(
        self,
        problem: ReasoningProblem,
        rng: random.Random,
        greedy: bool = False,
    ) -> Trajectory:
        """Sample (or greedily decode) a full autoregressive reasoning trajectory."""
        h = self._encode_hidden(problem.features)

        lp_fmt = _log_softmax(self._fmt_logits(h))
        a_fmt = (
            max(range(NUM_FMT_ACTIONS), key=lambda i: lp_fmt[i])
            if greedy
            else _sample_from_log_probs(lp_fmt, rng)
        )

        lp_strat = _log_softmax(self._strat_logits(h, a_fmt))
        a_strat = (
            max(range(NUM_STRAT_ACTIONS), key=lambda i: lp_strat[i])
            if greedy
            else _sample_from_log_probs(lp_strat, rng)
        )

        a_steps: list[int] = []
        step_lps: list[float] = []
        for x_step in problem.step_features:
            lp_s = _log_softmax(self._step_logits(h, x_step, a_strat))
            a_s = (
                max(range(NUM_STEP_ACTIONS), key=lambda i: lp_s[i])
                if greedy
                else _sample_from_log_probs(lp_s, rng)
            )
            a_steps.append(a_s)
            step_lps.append(lp_s[a_s])

        lp_ans = _log_softmax(self._ans_logits(h, a_strat))
        a_ans = (
            max(range(NUM_ANS_ACTIONS), key=lambda i: lp_ans[i])
            if greedy
            else _sample_from_log_probs(lp_ans, rng)
        )

        a_steps_t = tuple(a_steps)
        old_lps = (lp_fmt[a_fmt], lp_strat[a_strat], *step_lps, lp_ans[a_ans])
        text = render_trajectory(problem, a_fmt, a_strat, a_steps_t, a_ans)
        return Trajectory(
            a_fmt=a_fmt,
            a_strat=a_strat,
            a_steps=a_steps_t,
            a_ans=a_ans,
            old_log_probs=old_lps,
            completion_text=text,
        )

    def evaluate_trajectory_log_probs(
        self,
        problem: ReasoningProblem,
        traj: Trajectory,
    ) -> tuple[list[float], list[list[float]], list[float]]:
        """Compute per-decision log-probs, full distributions, and hidden state."""
        h = self._encode_hidden(problem.features)
        dist_fmt = _log_softmax(self._fmt_logits(h))
        dist_strat = _log_softmax(self._strat_logits(h, traj.a_fmt))
        step_dists = [
            _log_softmax(self._step_logits(h, x_step, traj.a_strat))
            for x_step in problem.step_features
        ]
        dist_ans = _log_softmax(self._ans_logits(h, traj.a_strat))

        selected_lps = [
            dist_fmt[traj.a_fmt],
            dist_strat[traj.a_strat],
            *[d[a] for d, a in zip(step_dists, traj.a_steps, strict=True)],
            dist_ans[traj.a_ans],
        ]
        all_dists = [dist_fmt, dist_strat, *step_dists, dist_ans]
        return selected_lps, all_dists, h

    def backprop_selected_log_probs(
        self,
        problem: ReasoningProblem,
        traj: Trajectory,
        d_log_probs: list[float],
        grads: dict[str, list[float]],
    ) -> None:
        """Accumulate exact analytical gradient of sum_t d_log_probs[t] * log_pi(a_t).

        Uses exact chain rule through log_softmax, linear projections, embeddings,
        and tanh hidden encoder.
        """
        _, all_dists, h = self.evaluate_trajectory_log_probs(problem, traj)
        actions = traj.action_sequence()
        dh = [0.0] * HIDDEN_DIM

        # Helper: d/dz_j (c * log_softmax(z)[a]) = c * (1_{j==a} - softmax(z)_j)
        def _logit_grad(dist: list[float], action: int, coeff: float) -> list[float]:
            return [
                coeff * ((1.0 if j == action else 0.0) - math.exp(dist[j]))
                for j in range(len(dist))
            ]

        # 1. Format head (index 0)
        dz_fmt = _logit_grad(all_dists[0], actions[0], d_log_probs[0])
        w_fmt = self.params["W_fmt"]
        gw_fmt = grads["W_fmt"]
        gb_fmt = grads["b_fmt"]
        for a in range(NUM_FMT_ACTIONS):
            g = dz_fmt[a]
            gb_fmt[a] += g
            row_off = a * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                gw_fmt[row_off + j] += g * h[j]
                dh[j] += g * w_fmt[row_off + j]

        # 2. Strategy head (index 1)
        dz_strat = _logit_grad(all_dists[1], actions[1], d_log_probs[1])
        w_strat = self.params["W_strat"]
        gw_strat = grads["W_strat"]
        ge_fs = grads["E_fmt_strat"]
        gb_strat = grads["b_strat"]
        e_fs_off = traj.a_fmt * NUM_STRAT_ACTIONS
        for a in range(NUM_STRAT_ACTIONS):
            g = dz_strat[a]
            gb_strat[a] += g
            ge_fs[e_fs_off + a] += g
            row_off = a * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                gw_strat[row_off + j] += g * h[j]
                dh[j] += g * w_strat[row_off + j]

        # 3. Step heads (indices 2 .. 2+K-1)
        w_step_h = self.params["W_step_h"]
        gw_sx = grads["W_step_x"]
        gw_sh = grads["W_step_h"]
        ge_ss = grads["E_strat_step"]
        gb_step = grads["b_step"]
        e_ss_off = traj.a_strat * NUM_STEP_ACTIONS
        for k_idx, x_step in enumerate(problem.step_features):
            t = 2 + k_idx
            dz_step = _logit_grad(all_dists[t], actions[t], d_log_probs[t])
            for a in range(NUM_STEP_ACTIONS):
                g = dz_step[a]
                gb_step[a] += g
                ge_ss[e_ss_off + a] += g
                row_x = a * STEP_FEATURE_DIM
                for j in range(STEP_FEATURE_DIM):
                    gw_sx[row_x + j] += g * x_step[j]
                row_h = a * HIDDEN_DIM
                for j in range(HIDDEN_DIM):
                    gw_sh[row_h + j] += g * h[j]
                    dh[j] += g * w_step_h[row_h + j]

        # 4. Answer extraction head (last index)
        t_ans = 2 + len(problem.step_features)
        dz_ans = _logit_grad(all_dists[t_ans], actions[t_ans], d_log_probs[t_ans])
        w_ans = self.params["W_ans"]
        gw_ans = grads["W_ans"]
        ge_sa = grads["E_strat_ans"]
        gb_ans = grads["b_ans"]
        e_sa_off = traj.a_strat * NUM_ANS_ACTIONS
        for a in range(NUM_ANS_ACTIONS):
            g = dz_ans[a]
            gb_ans[a] += g
            ge_sa[e_sa_off + a] += g
            row_off = a * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                gw_ans[row_off + j] += g * h[j]
                dh[j] += g * w_ans[row_off + j]

        # 5. Backprop through tanh hidden encoder
        gw_enc = grads["W_enc"]
        gb_enc = grads["b_enc"]
        x_prob = problem.features
        for i in range(HIDDEN_DIM):
            dtanh = dh[i] * (1.0 - h[i] * h[i])
            gb_enc[i] += dtanh
            row_off = i * PROBLEM_FEATURE_DIM
            for j in range(PROBLEM_FEATURE_DIM):
                gw_enc[row_off + j] += dtanh * x_prob[j]

    def measure_kl_and_entropy(
        self,
        ref_policy: StructuredCoTPolicy,
        problems: tuple[ReasoningProblem, ...],
        rng: random.Random,
    ) -> dict[str, float]:
        """Compute exact mean per-decision KL(pi || pi_ref) and policy entropy."""
        total_kl = 0.0
        total_ent = 0.0
        total_decisions = 0
        for prob in problems:
            traj = self.sample_trajectory(prob, rng, greedy=False)
            _, pi_dists, _ = self.evaluate_trajectory_log_probs(prob, traj)
            _, ref_dists, _ = ref_policy.evaluate_trajectory_log_probs(prob, traj)
            for p_d, r_d in zip(pi_dists, ref_dists, strict=True):
                total_kl += _exact_categorical_kl(p_d, r_d)
                total_ent += _categorical_entropy(p_d)
                total_decisions += 1
        denom = max(1, total_decisions)
        return {
            "mean_exact_kl_nats": round(total_kl / denom, 6),
            "mean_entropy_nats": round(total_ent / denom, 6),
        }


class AdamWOptimizer:
    """Pure-stdlib AdamW optimizer supporting ascent (maximize) or descent."""

    def __init__(
        self,
        policy: StructuredCoTPolicy,
        lr: float = 0.03,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        maximize: bool = True,
        grad_clip_norm: float = 5.0,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.grad_clip_norm = grad_clip_norm
        self.step_count = 0
        self.m = {k: [0.0] * len(v) for k, v in policy.params.items()}
        self.v = {k: [0.0] * len(v) for k, v in policy.params.items()}

    def step(self, policy: StructuredCoTPolicy, grads: dict[str, list[float]]) -> float:
        self.step_count += 1
        # Global L2 gradient norm clipping
        sq_norm = sum(g * g for vec in grads.values() for g in vec)
        grad_norm = math.sqrt(sq_norm)
        scale = (
            self.grad_clip_norm / grad_norm
            if self.grad_clip_norm > 0.0 and grad_norm > self.grad_clip_norm
            else 1.0
        )

        bc1 = 1.0 - (self.beta1**self.step_count)
        bc2 = 1.0 - (self.beta2**self.step_count)
        sign = 1.0 if self.maximize else -1.0

        for name, param in policy.params.items():
            g_vec = grads[name]
            m_vec = self.m[name]
            v_vec = self.v[name]
            apply_wd = not name.startswith("b_")
            for i in range(len(param)):
                g = g_vec[i] * scale
                m_i = self.beta1 * m_vec[i] + (1.0 - self.beta1) * g
                v_i = self.beta2 * v_vec[i] + (1.0 - self.beta2) * (g * g)
                m_vec[i] = m_i
                v_vec[i] = v_i
                m_hat = m_i / bc1
                v_hat = v_i / bc2
                if apply_wd and self.weight_decay > 0.0:
                    param[i] *= 1.0 - self.lr * self.weight_decay
                param[i] += sign * self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return grad_norm


def compute_group_advantages(
    rewards: list[float],
    eps: float = 1e-8,
) -> tuple[list[float], bool]:
    """Compute GRPO group-relative normalized advantages.

    Returns ``(advantages, is_zero_variance)``.
    """
    n = len(rewards)
    if n == 0:
        return [], True
    mean_r = sum(rewards) / float(n)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / float(n)
    std_r = math.sqrt(var_r)
    if std_r < 1e-12:
        return [0.0] * n, True
    return [(r - mean_r) / (std_r + eps) for r in rewards], False


def grpo_objective_and_grads(
    policy: StructuredCoTPolicy,
    ref_policy: StructuredCoTPolicy,
    problem: ReasoningProblem,
    group_trajs: list[Trajectory],
    advantages: list[float],
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[float, dict[str, list[float]], dict[str, float]]:
    """Compute exact GRPO objective J_GRPO(theta) and exact analytical gradient.

    Objective (maximized):
      J(theta) = (1/G) sum_{i=1}^G (1/|o_i|) sum_{t=1}^{|o_i|} [
          min(rho_{i,t} * A_i, clip(rho_{i,t}, 1 - eps, 1 + eps) * A_i)
          - beta * k3_KL(pi_theta || pi_ref)
      ]

    Returns ``(objective_value, grads, telemetry)``.
    """
    grads = policy.zero_grads()
    g_size = len(group_trajs)
    total_obj = 0.0
    total_surr = 0.0
    total_k3_kl = 0.0
    total_clipped = 0
    total_decisions = 0

    for traj, adv in zip(group_trajs, advantages, strict=True):
        pi_lps, _, _ = policy.evaluate_trajectory_log_probs(problem, traj)
        ref_lps, _, _ = ref_policy.evaluate_trajectory_log_probs(problem, traj)
        seq_len = traj.num_decisions
        inv_weight = 1.0 / float(g_size * seq_len)

        d_lps = [0.0] * seq_len
        for t in range(seq_len):
            log_pi = pi_lps[t]
            log_old = traj.old_log_probs[t]
            log_ref = ref_lps[t]

            log_ratio = max(-20.0, min(20.0, log_pi - log_old))
            rho = math.exp(log_ratio)
            clipped_rho = min(1.0 + clip_eps, max(1.0 - clip_eps, rho))

            unclipped = rho * adv
            clipped = clipped_rho * adv
            if unclipped <= clipped:
                surr = unclipped
                d_surr_d_log_pi = rho * adv
            else:
                surr = clipped
                d_surr_d_log_pi = 0.0
                total_clipped += 1

            k3 = schulman_k3_kl(log_pi, log_ref)
            d_k3_d_log_pi = schulman_k3_kl_grad_wrt_log_pi(log_pi, log_ref)

            total_surr += surr * inv_weight
            total_k3_kl += k3 * inv_weight
            total_obj += (surr - kl_beta * k3) * inv_weight
            d_lps[t] = inv_weight * (d_surr_d_log_pi - kl_beta * d_k3_d_log_pi)
            total_decisions += 1

        policy.backprop_selected_log_probs(problem, traj, d_lps, grads)

    telemetry = {
        "objective": total_obj,
        "surrogate": total_surr,
        "k3_kl": total_k3_kl,
        "clip_fraction": total_clipped / float(max(1, total_decisions)),
    }
    return total_obj, grads, telemetry


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for GRPO / REINFORCE / SFT training runs."""

    regime_name: str
    group_size: int = 8
    num_iterations: int = 2  # PPO/GRPO inner epochs (mu) per rollout batch
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    learning_rate: float = 0.035
    weight_decay: float = 1e-4
    batch_prompts: int = 12
    total_rollout_budget: int = 2880  # Matched across all RL/SFT regimes!
    use_naive_proxy_reward: bool = False
    seed: int = 1350


def train_grpo(
    base_policy: StructuredCoTPolicy,
    train_problems: tuple[ReasoningProblem, ...],
    config: GRPOConfig,
    reward_fn: Callable[[ReasoningProblem, str], float] | None = None,
) -> tuple[StructuredCoTPolicy, dict[str, object]]:
    """Train a policy with exact GRPO under a matched rollout budget."""
    policy = base_policy.clone()
    ref_policy = base_policy.clone()
    optimizer = AdamWOptimizer(
        policy,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        maximize=True,
    )
    rng = random.Random(config.seed)
    audit = RewardAuditTrail(f"{config.regime_name}:train")

    rollouts_used = 0
    total_groups = 0
    zero_var_groups = 0
    adv_var_sum = 0.0
    update_steps = 0
    clip_frac_sum = 0.0
    k3_kl_sum = 0.0
    proxy_reward_sum = 0.0
    prompt_pool = list(train_problems)

    while rollouts_used + config.batch_prompts * config.group_size <= config.total_rollout_budget:
        rng.shuffle(prompt_pool)
        batch = prompt_pool[: config.batch_prompts]

        # 1. Sample G rollouts per prompt from pi_old (current policy snapshot)
        batch_groups: list[tuple[ReasoningProblem, list[Trajectory], list[float]]] = []
        for prob in batch:
            trajs: list[Trajectory] = []
            rewards: list[float] = []
            for _ in range(config.group_size):
                traj = policy.sample_trajectory(prob, rng, greedy=False)
                ev = evaluate_completion(prob, traj.completion_text)
                audit.record(ev)
                if reward_fn is not None:
                    r = reward_fn(prob, traj.completion_text)
                elif config.use_naive_proxy_reward:
                    r = naive_proxy_reward(prob, traj.completion_text)
                else:
                    r = ev.total_reward
                proxy_reward_sum += r
                trajs.append(traj)
                rewards.append(r)
                rollouts_used += 1

            advs, is_zv = compute_group_advantages(rewards)
            total_groups += 1
            if is_zv:
                zero_var_groups += 1
            else:
                adv_var_sum += sum(a * a for a in advs) / float(len(advs))
            batch_groups.append((prob, trajs, advs))

        # 2. Run mu GRPO inner optimization iterations over the collected batch
        for _ in range(config.num_iterations):
            batch_grads = policy.zero_grads()
            batch_k3 = 0.0
            batch_clip = 0.0
            inv_b = 1.0 / float(len(batch_groups))
            for prob, trajs, advs in batch_groups:
                _, g_i, tel = grpo_objective_and_grads(
                    policy=policy,
                    ref_policy=ref_policy,
                    problem=prob,
                    group_trajs=trajs,
                    advantages=advs,
                    clip_eps=config.clip_eps,
                    kl_beta=config.kl_beta,
                )
                for k, vec in g_i.items():
                    dst = batch_grads[k]
                    for idx, val in enumerate(vec):
                        dst[idx] += val * inv_b
                batch_k3 += tel["k3_kl"] * inv_b
                batch_clip += tel["clip_fraction"] * inv_b

            optimizer.step(policy, batch_grads)
            update_steps += 1
            k3_kl_sum += batch_k3
            clip_frac_sum += batch_clip

    stats = {
        "regime_name": config.regime_name,
        "group_size": config.group_size,
        "kl_beta": config.kl_beta,
        "clip_eps": config.clip_eps,
        "rollouts_used": rollouts_used,
        "optimizer_steps": update_steps,
        "total_groups": total_groups,
        "zero_variance_group_rate": round(zero_var_groups / float(max(1, total_groups)), 6),
        "mean_nonzero_adv_sq": round(
            adv_var_sum / float(max(1, total_groups - zero_var_groups)), 6
        ),
        "mean_train_k3_kl": round(k3_kl_sum / float(max(1, update_steps)), 6),
        "mean_clip_fraction": round(clip_frac_sum / float(max(1, update_steps)), 6),
        "mean_optimized_reward": round(proxy_reward_sum / float(max(1, rollouts_used)), 6),
        "train_audit_summary": audit.summary(),
        "final_weight_sha256": policy.weight_sha256(),
    }
    return policy, stats


def train_reinforce_baseline(
    base_policy: StructuredCoTPolicy,
    train_problems: tuple[ReasoningProblem, ...],
    kl_beta: float = 0.04,
    learning_rate: float = 0.035,
    batch_prompts: int = 12,
    total_rollout_budget: int = 2880,
    seed: int = 1350,
) -> tuple[StructuredCoTPolicy, dict[str, object]]:
    """Vanilla REINFORCE with running global baseline (NO per-prompt group norm)."""
    policy = base_policy.clone()
    ref_policy = base_policy.clone()
    optimizer = AdamWOptimizer(
        policy,
        lr=learning_rate,
        weight_decay=1e-4,
        maximize=True,
    )
    rng = random.Random(seed)
    audit = RewardAuditTrail("reinforce_no_group_norm:train")

    rollouts_used = 0
    update_steps = 0
    running_baseline = 0.35
    k3_kl_sum = 0.0
    reward_sum = 0.0
    prompt_pool = list(train_problems)

    while rollouts_used + batch_prompts <= total_rollout_budget:
        rng.shuffle(prompt_pool)
        batch = prompt_pool[:batch_prompts]
        batch_grads = policy.zero_grads()
        batch_rewards: list[float] = []
        batch_k3 = 0.0
        inv_b = 1.0 / float(len(batch))

        for prob in batch:
            traj = policy.sample_trajectory(prob, rng, greedy=False)
            ev = evaluate_completion(prob, traj.completion_text)
            audit.record(ev)
            r = ev.total_reward
            batch_rewards.append(r)
            reward_sum += r
            rollouts_used += 1

            adv = r - running_baseline
            pi_lps, _, _ = policy.evaluate_trajectory_log_probs(prob, traj)
            ref_lps, _, _ = ref_policy.evaluate_trajectory_log_probs(prob, traj)
            seq_len = traj.num_decisions
            d_lps = [0.0] * seq_len
            for t in range(seq_len):
                k3 = schulman_k3_kl(pi_lps[t], ref_lps[t])
                d_k3 = schulman_k3_kl_grad_wrt_log_pi(pi_lps[t], ref_lps[t])
                batch_k3 += k3 * inv_b / float(seq_len)
                d_lps[t] = (inv_b / float(seq_len)) * (adv - kl_beta * d_k3)
            policy.backprop_selected_log_probs(prob, traj, d_lps, batch_grads)

        batch_mean = sum(batch_rewards) / float(len(batch_rewards))
        running_baseline = 0.9 * running_baseline + 0.1 * batch_mean
        optimizer.step(policy, batch_grads)
        update_steps += 1
        k3_kl_sum += batch_k3

    stats = {
        "regime_name": "reinforce_no_group_norm",
        "group_size": 1,
        "kl_beta": kl_beta,
        "rollouts_used": rollouts_used,
        "optimizer_steps": update_steps,
        "mean_train_k3_kl": round(k3_kl_sum / float(max(1, update_steps)), 6),
        "mean_optimized_reward": round(reward_sum / float(max(1, rollouts_used)), 6),
        "train_audit_summary": audit.summary(),
        "final_weight_sha256": policy.weight_sha256(),
    }
    return policy, stats


def train_sft_baseline(
    base_policy: StructuredCoTPolicy,
    train_problems: tuple[ReasoningProblem, ...],
    learning_rate: float = 0.025,
    batch_prompts: int = 12,
    total_trace_budget: int = 2880,
    teacher_noise_rate: float = 0.24,
    seed: int = 1350,
) -> tuple[StructuredCoTPolicy, dict[str, object]]:
    """Matched-budget Supervised Fine-Tuning (SFT) baseline on offline traces.

    Simulates open-source teacher distillation traces (e.g., mixed-quality
    offline reasoning traces where ~76% are exact and ~24% exhibit occasional
    teacher step slips or shortcut drift without online verifier rejection).
    """
    policy = base_policy.clone()
    optimizer = AdamWOptimizer(
        policy,
        lr=learning_rate,
        weight_decay=5e-4,
        maximize=True,  # maximize log-likelihood = minimize NLL
    )
    rng = random.Random(seed)
    audit = RewardAuditTrail("sft_only_baseline:train")

    # Pre-construct deterministic offline SFT corpus (one trace per train prompt)
    sft_corpus: list[tuple[ReasoningProblem, Trajectory]] = []
    for prob in train_problems:
        a_fmt = FMT_STRICT_XML if rng.random() > 0.06 else FMT_PLAIN_TEXT
        a_strat = (
            STRAT_FULL_DECOMP
            if rng.random() > teacher_noise_rate * 0.5
            else STRAT_SHALLOW_SHORTCUT
        )
        a_steps = tuple(
            spec.required_op_class
            if rng.random() > teacher_noise_rate
            else (spec.required_op_class + rng.choice((1, 2))) % 3
            for spec in prob.steps
        )
        a_ans = ANS_EXTRACT_FINAL if rng.random() > 0.05 else ANS_EXTRACT_EARLY
        text = render_trajectory(prob, a_fmt, a_strat, a_steps, a_ans)
        traj = Trajectory(
            a_fmt=a_fmt,
            a_strat=a_strat,
            a_steps=a_steps,
            a_ans=a_ans,
            old_log_probs=(0.0,) * (3 + prob.depth),
            completion_text=text,
        )
        ev = evaluate_completion(prob, text)
        audit.record(ev)
        sft_corpus.append((prob, traj))

    traces_seen = 0
    update_steps = 0
    nll_sum = 0.0

    while traces_seen + batch_prompts <= total_trace_budget:
        rng.shuffle(sft_corpus)
        batch = sft_corpus[:batch_prompts]
        batch_grads = policy.zero_grads()
        inv_b = 1.0 / float(len(batch))
        batch_nll = 0.0

        for prob, traj in batch:
            lps, _, _ = policy.evaluate_trajectory_log_probs(prob, traj)
            seq_len = traj.num_decisions
            batch_nll += -sum(lps) / float(seq_len) * inv_b
            d_lps = [inv_b / float(seq_len)] * seq_len
            policy.backprop_selected_log_probs(prob, traj, d_lps, batch_grads)
            traces_seen += 1

        optimizer.step(policy, batch_grads)
        update_steps += 1
        nll_sum += batch_nll

    stats = {
        "regime_name": "sft_only_baseline",
        "traces_seen": traces_seen,
        "teacher_noise_rate": teacher_noise_rate,
        "optimizer_steps": update_steps,
        "mean_train_nll": round(nll_sum / float(max(1, update_steps)), 6),
        "offline_corpus_audit": audit.summary(),
        "final_weight_sha256": policy.weight_sha256(),
    }
    return policy, stats


def serialize_policy_summary(policy: StructuredCoTPolicy) -> dict[str, object]:
    """Return compact deterministic policy fingerprint & norm summary."""
    l2_sq = sum(x * x for vec in policy.params.values() for x in vec)
    return {
        "num_parameters": policy.num_parameters,
        "weight_sha256": policy.weight_sha256(),
        "weight_l2_norm": round(math.sqrt(l2_sq), 6),
    }
