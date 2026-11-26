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
"""Comprehensive offline test suite for HIP-1350 Open Reasoning GRPO pipeline."""

from __future__ import annotations

import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from examples.grpo_reasoning.dataset import (
    DOMAINS,
    PROBLEM_FEATURE_DIM,
    STEP_FEATURE_DIM,
    build_reasoning_dataset,
    load_license_audit,
)
from examples.grpo_reasoning.grpo import (
    ANS_EXTRACT_FINAL,
    FMT_ANSWER_STUFFED,
    FMT_LENGTH_PADDED,
    FMT_STRICT_XML,
    STRAT_FULL_DECOMP,
    STRAT_UNSUPPORTED_GUESS,
    StructuredCoTPolicy,
    compute_group_advantages,
    grpo_objective_and_grads,
    render_trajectory,
    schulman_k3_kl,
    schulman_k3_kl_grad_wrt_log_pi,
)
from examples.grpo_reasoning.recipes.validator import (
    parse_simple_yaml,
    validate_all_recipes,
    validate_recipe_file,
)
from examples.grpo_reasoning.rewards import (
    RewardAuditTrail,
    correctness_reward,
    evaluate_completion,
    format_reward,
    naive_proxy_reward,
    reasoning_coherence_reward,
    trl_correctness_reward_func,
    trl_format_reward_func,
    verl_compute_score,
)
from examples.grpo_reasoning.run_benchmark import (
    DEFAULT_OUTPUT_PATH,
    _unbiased_pass_at_k,
    paired_bootstrap_ci,
    run_benchmark,
    verify_evaluation_receipt,
)


class TestDatasetAndLeakage(unittest.TestCase):
    """Tests for verifiable reasoning dataset generation and split disjointness."""

    def test_dataset_deterministic_and_sha256_pinned(self) -> None:
        ds1 = build_reasoning_dataset(seed=1350)
        ds2 = build_reasoning_dataset(seed=1350)
        self.assertEqual(ds1.dataset_manifest_sha256, ds2.dataset_manifest_sha256)
        self.assertEqual(ds1.split_sha256, ds2.split_sha256)
        self.assertEqual(len(ds1.train), 180)
        self.assertEqual(len(ds1.val), 60)
        self.assertEqual(len(ds1.held_out_test), 120)

    def test_strict_zero_leakage_across_all_splits(self) -> None:
        ds = build_reasoning_dataset(seed=1350)
        report = ds.verify_disjointness()
        self.assertTrue(report["zero_leakage"], msg=f"Leakage detected: {report}")
        for key, val in report.items():
            if key != "zero_leakage":
                self.assertEqual(val, 0, msg=f"Non-zero overlap in {key}: {val}")

    def test_domains_feature_dims_and_ood_depths(self) -> None:
        ds = build_reasoning_dataset(seed=1350)
        seen_domains = {p.domain for p in ds.held_out_test}
        self.assertEqual(seen_domains, set(DOMAINS))
        for p in (*ds.train, *ds.val, *ds.held_out_test):
            self.assertEqual(len(p.features), PROBLEM_FEATURE_DIM)
            self.assertEqual(len(p.step_features), p.depth)
            for sf in p.step_features:
                self.assertEqual(len(sf), STEP_FEATURE_DIM)
            if p.is_ood:
                self.assertIn(p.depth, (4, 5))
                self.assertEqual(p.split, "held_out_test")
            else:
                self.assertIn(p.depth, (2, 3))

    def test_ground_truth_exact_rendering_all_domains(self) -> None:
        ds = build_reasoning_dataset(seed=1350)
        by_domain = {}
        for p in ds.held_out_test:
            by_domain.setdefault(p.domain, p)
        self.assertEqual(set(by_domain.keys()), set(DOMAINS))
        for domain, prob in by_domain.items():
            exact_steps = tuple(s.required_op_class for s in prob.steps)
            text = render_trajectory(
                prob, FMT_STRICT_XML, STRAT_FULL_DECOMP, exact_steps, ANS_EXTRACT_FINAL
            )
            ev = evaluate_completion(prob, text)
            self.assertTrue(ev.is_exact_correct, msg=f"Domain {domain} failed: {ev}")
            self.assertTrue(ev.is_format_valid)
            self.assertAlmostEqual(ev.total_reward, 1.0, places=6)
            self.assertAlmostEqual(ev.verified_step_fraction, 1.0, places=6)
            self.assertEqual(ev.hacking_flags, ())


class TestAuditableRewardsAndAntiHacking(unittest.TestCase):
    """Tests for deterministic verifiers, anti-hacking guards, and audit trails."""

    def setUp(self) -> None:
        ds = build_reasoning_dataset(seed=1350)
        self.math_prob = next(
            p for p in ds.held_out_test if p.domain == "math_reasoning"
        )
        self.logic_prob = next(
            p for p in ds.held_out_test if p.domain == "symbolic_logic"
        )
        self.code_prob = next(p for p in ds.held_out_test if p.domain == "code_trace")

    def test_duplicate_answer_tag_rejected(self) -> None:
        exact_steps = tuple(s.required_op_class for s in self.math_prob.steps)
        stuffed = render_trajectory(
            self.math_prob,
            FMT_ANSWER_STUFFED,
            STRAT_FULL_DECOMP,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        ev = evaluate_completion(self.math_prob, stuffed)
        self.assertFalse(ev.is_exact_correct)
        self.assertEqual(ev.correctness_reward, 0.0)
        self.assertIn("DUPLICATE_TAG", ev.hacking_flags)
        self.assertIn("ANSWER_STUFFING", ev.hacking_flags)

    def test_trailing_garbage_rejected(self) -> None:
        exact_steps = tuple(s.required_op_class for s in self.math_prob.steps)
        clean = render_trajectory(
            self.math_prob,
            FMT_STRICT_XML,
            STRAT_FULL_DECOMP,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        dirty = clean + " trailing garbage text"
        ev = evaluate_completion(self.math_prob, dirty)
        self.assertFalse(ev.is_format_valid)
        self.assertFalse(ev.is_exact_correct)
        self.assertEqual(ev.correctness_reward, 0.0)
        self.assertIn("TRAILING_GARBAGE", ev.hacking_flags)

    def test_length_padding_hack_caught_but_tricks_naive_proxy(self) -> None:
        exact_steps = tuple(s.required_op_class for s in self.code_prob.steps)
        padded = render_trajectory(
            self.code_prob,
            FMT_LENGTH_PADDED,
            STRAT_FULL_DECOMP,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        ev = evaluate_completion(self.code_prob, padded)
        self.assertFalse(ev.is_exact_correct)
        self.assertEqual(ev.total_reward, 0.0)
        self.assertIn("LENGTH_PADDING_HACK", ev.hacking_flags)
        # Contrast: naive proxy is fooled by the padded XML completion
        proxy = naive_proxy_reward(self.code_prob, padded)
        self.assertGreater(proxy, 0.80)

    def test_unsupported_guess_penalized_and_not_exact_pass(self) -> None:
        guess = f"<think>lucky intuition</think><answer>{self.math_prob.canonical_answer}</answer>"
        ev = evaluate_completion(self.math_prob, guess)
        self.assertFalse(ev.is_exact_correct)
        self.assertIn("UNSUPPORTED_GUESS", ev.hacking_flags)
        self.assertLess(ev.total_reward, 0.30)

    def test_step_mismatch_between_think_and_answer_flagged(self) -> None:
        # Derive wrong final step in <think> while copying true answer into <answer>
        steps_text = "; ".join(
            f"step_{s.step_index}: {s.expression} => "
            f"{s.expected_state if s.step_index < self.math_prob.depth else s.alt_slip_state}"
            for s in self.math_prob.steps
        )
        mismatched = f"<think>{steps_text}</think><answer>{self.math_prob.canonical_answer}</answer>"
        ev = evaluate_completion(self.math_prob, mismatched)
        self.assertFalse(ev.is_exact_correct)
        self.assertIn("STEP_MISMATCH", ev.hacking_flags)

    def test_audit_trail_chain_determinism_and_tamper_sensitivity(self) -> None:
        exact_steps = tuple(s.required_op_class for s in self.logic_prob.steps)
        good = render_trajectory(
            self.logic_prob,
            FMT_STRICT_XML,
            STRAT_FULL_DECOMP,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        bad = render_trajectory(
            self.logic_prob,
            FMT_STRICT_XML,
            STRAT_UNSUPPORTED_GUESS,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        t1 = RewardAuditTrail("test_regime")
        t2 = RewardAuditTrail("test_regime")
        t3 = RewardAuditTrail("test_regime")
        ev_good = evaluate_completion(self.logic_prob, good)
        ev_bad = evaluate_completion(self.logic_prob, bad)
        total, breakdown, ahash = ev_good.as_tuple()
        self.assertEqual(total, ev_good.total_reward)
        self.assertEqual(ahash, ev_good.audit_hash)
        self.assertIn("verifier_version", breakdown)

        t1.record(ev_good)
        t1.record(ev_bad)
        t2.record(ev_good)
        t2.record(ev_bad)
        t3.record(ev_bad)
        t3.record(ev_good)

        self.assertEqual(t1.chain_sha256, t2.chain_sha256)
        self.assertNotEqual(t1.chain_sha256, t3.chain_sha256)

    def test_trl_and_verl_reward_hooks(self) -> None:
        exact_steps = tuple(s.required_op_class for s in self.math_prob.steps)
        good = render_trajectory(
            self.math_prob,
            FMT_STRICT_XML,
            STRAT_FULL_DECOMP,
            exact_steps,
            ANS_EXTRACT_FINAL,
        )
        bad = "plain text 0"
        fmt_scores = trl_format_reward_func([good, bad], domain="math_reasoning")
        self.assertEqual(fmt_scores, [0.25, 0.0])
        corr_scores = trl_correctness_reward_func(
            [good, bad],
            ground_truth=[self.math_prob.canonical_answer, self.math_prob.canonical_answer],
            domain="math_reasoning",
        )
        self.assertEqual(corr_scores, [0.60, 0.0])
        verl_good = verl_compute_score(
            "openthoughts",
            good,
            self.math_prob.canonical_answer,
            {"domain": "math_reasoning"},
        )
        verl_bad = verl_compute_score(
            "openthoughts",
            bad,
            self.math_prob.canonical_answer,
            {"domain": "math_reasoning"},
        )
        self.assertAlmostEqual(verl_good, 0.85, places=6)
        self.assertAlmostEqual(verl_bad, 0.0, places=6)

        # Direct component APIs return clean tuples
        f_s, f_ok, f_fl = format_reward(good, domain="math_reasoning")
        c_s, c_ok, c_ans, c_fl = correctness_reward(self.math_prob, good)
        r_s, r_frac, r_fl = reasoning_coherence_reward(self.math_prob, good)
        self.assertTrue(f_ok and c_ok)
        self.assertEqual(c_ans, self.math_prob.canonical_answer)
        self.assertAlmostEqual(f_s + c_s + r_s, 1.0, places=6)
        self.assertEqual((f_fl, c_fl, r_fl), ((), (), ()))
        self.assertAlmostEqual(r_frac, 1.0, places=6)


class TestGRPOMathAndGradients(unittest.TestCase):
    """Exact mathematical tests for GRPO objective, k3 KL, and finite-diff gradients."""

    def test_schulman_k3_kl_nonnegative_and_exact_derivative(self) -> None:
        self.assertAlmostEqual(schulman_k3_kl(-1.2, -1.2), 0.0, places=12)
        h = 1e-6
        for lp, lr in [(-0.4, -1.1), (-2.3, -0.5), (-0.9, -0.95)]:
            val = schulman_k3_kl(lp, lr)
            self.assertGreaterEqual(val, 0.0)
            ana = schulman_k3_kl_grad_wrt_log_pi(lp, lr)
            num = (schulman_k3_kl(lp + h, lr) - schulman_k3_kl(lp - h, lr)) / (2.0 * h)
            rel_err = abs(ana - num) / max(1e-8, abs(ana), abs(num))
            self.assertLess(rel_err, 1e-7)

    def test_group_advantage_normalization_and_zero_variance(self) -> None:
        advs, zv = compute_group_advantages([0.0, 0.25, 0.85, 1.0])
        self.assertFalse(zv)
        mean_a = sum(advs) / len(advs)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in advs) / len(advs))
        self.assertAlmostEqual(mean_a, 0.0, places=7)
        self.assertAlmostEqual(std_a, 1.0, places=6)

        advs_z, zv_z = compute_group_advantages([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(zv_z)
        self.assertEqual(advs_z, [0.0, 0.0, 0.0, 0.0])

    def test_pinned_base_policy_sha256_and_distributions(self) -> None:
        p1 = StructuredCoTPolicy.create_pinned_base(seed=1350)
        p2 = StructuredCoTPolicy.create_pinned_base(seed=1350)
        p3 = StructuredCoTPolicy.create_pinned_base(seed=1351)
        self.assertEqual(p1.weight_sha256(), p2.weight_sha256())
        self.assertNotEqual(p1.weight_sha256(), p3.weight_sha256())

        ds = build_reasoning_dataset(seed=1350)
        prob = ds.held_out_test[0]
        traj = p1.sample_trajectory(prob, random.Random(42))
        _, dists, _ = p1.evaluate_trajectory_log_probs(prob, traj)
        for dist in dists:
            self.assertAlmostEqual(sum(math.exp(lp) for lp in dist), 1.0, places=12)

    def test_analytical_grpo_gradients_match_central_finite_differences(self) -> None:
        """Verify analytical GRPO + k3 KL gradients across ALL 491 parameters (< 1e-6)."""
        ds = build_reasoning_dataset(seed=1350)
        prob = ds.train[0]
        ref_policy = StructuredCoTPolicy.create_pinned_base(seed=1350)
        policy = ref_policy.clone()

        # Perturb policy slightly away from ref so importance ratio rho != 1 and k3 != 0
        rng = random.Random(777)
        for vec in policy.params.values():
            for i in range(len(vec)):
                vec[i] += rng.gauss(0.0, 0.05)

        trajs = [ref_policy.sample_trajectory(prob, rng) for _ in range(4)]
        rewards = [evaluate_completion(prob, t.completion_text).total_reward for t in trajs]
        if len(set(rewards)) == 1:
            rewards = [0.1, 0.4, 0.7, 1.0]
        advs, _ = compute_group_advantages(rewards)

        _, ana_grads, _ = grpo_objective_and_grads(
            policy=policy,
            ref_policy=ref_policy,
            problem=prob,
            group_trajs=trajs,
            advantages=advs,
            clip_eps=0.2,
            kl_beta=0.04,
        )

        eps_fd = 1e-5
        max_rel_err = 0.0
        checked = 0
        for name, vec in policy.params.items():
            for idx in range(len(vec)):
                orig = vec[idx]
                vec[idx] = orig + eps_fd
                j_plus, _, _ = grpo_objective_and_grads(
                    policy, ref_policy, prob, trajs, advs, clip_eps=0.2, kl_beta=0.04
                )
                vec[idx] = orig - eps_fd
                j_minus, _, _ = grpo_objective_and_grads(
                    policy, ref_policy, prob, trajs, advs, clip_eps=0.2, kl_beta=0.04
                )
                vec[idx] = orig
                num_g = (j_plus - j_minus) / (2.0 * eps_fd)
                ana_g = ana_grads[name][idx]
                denom = max(1e-6, abs(ana_g), abs(num_g))
                rel_err = abs(ana_g - num_g) / denom
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                checked += 1

        self.assertEqual(checked, policy.num_parameters)
        self.assertLess(
            max_rel_err,
            1e-6,
            msg=f"GRPO gradient mismatch: max_rel_err={max_rel_err:.3e}",
        )


class TestRecipesAndLicenseAudit(unittest.TestCase):
    """Tests for TRL & verl recipes, zero Open-R1 usage, and license audit."""

    def test_all_shipped_recipes_validate_cleanly(self) -> None:
        reports = validate_all_recipes()
        self.assertEqual(len(reports), 2)
        frameworks = {r["trainer_framework"] for r in reports}
        self.assertEqual(frameworks, {"huggingface/trl", "volcengine/verl"})
        for r in reports:
            self.assertTrue(r["valid"], msg=f"Recipe {r['recipe_file']} errors: {r['errors']}")

    def test_validator_rejects_open_r1_and_zero_kl(self) -> None:
        bad_yaml = """
recipe_metadata:
  hip: HIP-1350
  trainer_framework: huggingface/open-r1
  recipe_license: Apache-2.0
  separate_license_review_required: true
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_revision: pinned
rl: grpo
trl:
  beta: 0.0
  num_generations: 2
  scale_rewards: false
  reward_funcs:
    - only_one
independent_held_out_eval:
  verifier_module: examples.grpo_reasoning.rewards
  require_zero_leakage_check: false
"""
        parsed = parse_simple_yaml(bad_yaml)
        self.assertEqual(parsed["rl"], "grpo")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad_recipe.yaml"
            p.write_text(bad_yaml, encoding="utf-8")
            report = validate_recipe_file(p)
            self.assertFalse(report["valid"])
            joined = " ".join(report["errors"]).lower()
            self.assertIn("open-r1", joined)

    def test_license_audit_separates_source_and_external_artifacts(self) -> None:
        audit = load_license_audit()
        self.assertEqual(audit["policy"]["source_code_license"], "Apache-2.0")
        by_id = {a["id"]: a for a in audit["artifacts"]}
        self.assertEqual(by_id["zoo-gym-grpo-reasoning-src"]["license"], "Apache-2.0")
        self.assertFalse(by_id["zoo-gym-grpo-reasoning-src"]["separate_license_review_required"])
        self.assertTrue(by_id["openthoughts-114k"]["separate_license_review_required"])
        self.assertTrue(by_id["qwen2.5-1.5b-instruct"]["separate_license_review_required"])
        self.assertFalse(by_id["huggingface-open-r1"]["maintained"])
        self.assertEqual(
            by_id["huggingface-open-r1"]["status"],
            "HISTORICAL_REFERENCE_ONLY_DO_NOT_USE",
        )


class TestEndToEndBenchmarkAndReceipt(unittest.TestCase):
    """End-to-end verification of benchmark invariants, negative regimes, and receipt."""

    def test_unbiased_pass_at_k_and_bootstrap_ci(self) -> None:
        self.assertAlmostEqual(_unbiased_pass_at_k(8, 0, 1), 0.0)
        self.assertAlmostEqual(_unbiased_pass_at_k(8, 4, 1), 0.5)
        self.assertAlmostEqual(_unbiased_pass_at_k(8, 5, 4), 1.0)
        self.assertAlmostEqual(_unbiased_pass_at_k(8, 1, 4), 0.5)
        ci = paired_bootstrap_ci([1.0] * 20, [0.2] * 20, seed=1350, num_resamples=200)
        self.assertAlmostEqual(ci["delta_mean"], 0.8)
        self.assertGreater(ci["ci95_low"], 0.7)

    def test_benchmark_receipt_and_all_scientific_invariants(self) -> None:
        # Use committed seed-1350 results if present, else compute fresh
        if DEFAULT_OUTPUT_PATH.exists():
            results = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
        else:
            results = run_benchmark(seed=1350)

        self.assertTrue(verify_evaluation_receipt(results))
        # Tampering with any receipt metric must fail verification
        tampered = copy.deepcopy(results)
        tampered["independent_evaluation_receipt"]["regime_fingerprints"][
            "grpo_kl_regularized"
        ]["pass_at_1"] = 0.999999
        self.assertFalse(verify_evaluation_receipt(tampered))

        regimes = results["regimes"]
        base_m = regimes["pinned_base_model"]["held_out_metrics"]
        sft_m = regimes["sft_only_baseline"]["held_out_metrics"]
        rein_m = regimes["reinforce_no_group_norm"]["held_out_metrics"]
        grpo_m = regimes["grpo_kl_regularized"]["held_out_metrics"]
        hack_m = regimes["neg_format_only_reward_hacking"]["held_out_metrics"]
        zkl_m = regimes["neg_grpo_zero_kl_collapse"]["held_out_metrics"]
        g2_m = regimes["neg_grpo_group_size_2"]["held_out_metrics"]

        # Primary positive claim: GRPO beats pinned base, SFT, and REINFORCE on held-out pass@1
        self.assertGreater(grpo_m["pass_at_1"], base_m["pass_at_1"] + 0.40)
        self.assertGreater(grpo_m["pass_at_1"], sft_m["pass_at_1"] + 0.15)
        self.assertGreater(grpo_m["pass_at_1"], rein_m["pass_at_1"] + 0.05)
        self.assertGreater(grpo_m["format_valid_rate"], 0.90)
        self.assertLess(grpo_m["mean_exact_kl_nats"], 0.60)

        # Negative Regime 1: Reward hacking achieves high proxy reward while collapsing true pass@1
        self.assertGreater(hack_m["mean_naive_proxy_reward"], 0.80)
        self.assertLess(hack_m["pass_at_1"], base_m["pass_at_1"])
        self.assertGreater(hack_m["flagged_hacking_rate"], 0.80)

        # Negative Regime 2: Zero-KL collapse increases KL, shrinks entropy, degrades OOD pass@1 & pass@4
        self.assertGreater(zkl_m["mean_exact_kl_nats"], grpo_m["mean_exact_kl_nats"])
        self.assertLess(zkl_m["mean_entropy_nats"], grpo_m["mean_entropy_nats"])
        self.assertLess(zkl_m["ood_pass_at_1"], grpo_m["ood_pass_at_1"] - 0.05)
        self.assertLess(zkl_m["pass_at_4"], grpo_m["pass_at_4"] - 0.05)

        # Negative Regime 3: Tiny group size G=2 suffers more zero-variance waste and lower pass@1
        self.assertGreater(
            regimes["neg_grpo_group_size_2"]["train_summary"]["zero_variance_group_rate"],
            regimes["grpo_kl_regularized"]["train_summary"]["zero_variance_group_rate"],
        )
        self.assertLess(g2_m["pass_at_1"], grpo_m["pass_at_1"] - 0.10)

    def test_small_budget_byte_reproducibility(self) -> None:
        r1 = run_benchmark(seed=1350, total_rollout_budget=288)
        r2 = run_benchmark(seed=1350, total_rollout_budget=288)
        s1 = json.dumps(r1, sort_keys=True)
        s2 = json.dumps(r2, sort_keys=True)
        self.assertEqual(s1, s2)
        self.assertTrue(verify_evaluation_receipt(r1))


if __name__ == "__main__":
    unittest.main()
