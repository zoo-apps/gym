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
"""Pure-stdlib YAML parser and schema validator for TRL & verl GRPO recipes.

Ensures zero external runtime dependencies (no PyYAML required) while strictly
validating:
1. Required GRPO training fields for both TRL (Zoo Gym) and ``verl``.
2. Positive KL regularization (``beta > 0`` / ``kl_loss_coef > 0``) and
   group size ``G >= 4`` for production recipes.
3. Explicit presence of independent held-out evaluation hooks.
4. Zero active references to deprecated/unmaintained ``open-r1`` / ``open_r1``.
5. Consistency with ``LICENSE_AUDIT.json`` (separate review flags preserved).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..dataset import load_license_audit


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None
    if re.match(r"^-?\d+$", val):
        return int(val)
    if re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", val):
        return float(val)
    return val


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse indented YAML mappings and sequences using pure Python stdlib.

    Supports nested dictionaries, scalar lists, and lists of mappings as used
    in Zoo Gym TRL and ``verl`` YAML configs.
    """
    raw_lines: list[tuple[int, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        raw_lines.append((indent, stripped))

    def _parse_block(idx: int, min_indent: int) -> tuple[Any, int]:
        if idx >= len(raw_lines):
            return {}, idx
        first_indent, first_content = raw_lines[idx]
        if first_content.startswith("- "):
            items: list[Any] = []
            while idx < len(raw_lines):
                indent, content = raw_lines[idx]
                if indent < min_indent or not content.startswith("- "):
                    break
                if indent != first_indent:
                    break
                item_body = content[2:].strip()
                idx += 1
                if ":" in item_body and not item_body.startswith(("http://", "https://")):
                    k, v = item_body.split(":", 1)
                    entry: dict[str, Any] = {k.strip(): _parse_scalar(v) if v.strip() else None}
                    while idx < len(raw_lines) and raw_lines[idx][0] > first_indent:
                        sub_indent, sub_content = raw_lines[idx]
                        if ":" in sub_content:
                            sk, sv = sub_content.split(":", 1)
                            if sv.strip():
                                entry[sk.strip()] = _parse_scalar(sv)
                                idx += 1
                            else:
                                child, idx = _parse_block(idx + 1, sub_indent + 1)
                                entry[sk.strip()] = child
                        else:
                            idx += 1
                    items.append(entry)
                else:
                    items.append(_parse_scalar(item_body))
            return items, idx

        mapping: dict[str, Any] = {}
        while idx < len(raw_lines):
            indent, content = raw_lines[idx]
            if indent < min_indent:
                break
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"Malformed YAML line: {content!r}")
            key, rest = content.split(":", 1)
            key = key.strip()
            rest_s = rest.strip()
            idx += 1
            if rest_s:
                mapping[key] = _parse_scalar(rest_s)
            else:
                if idx < len(raw_lines) and raw_lines[idx][0] > indent:
                    child, idx = _parse_block(idx, indent + 1)
                    mapping[key] = child
                else:
                    mapping[key] = None
        return mapping, idx

    parsed, _ = _parse_block(0, 0)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML document must be a mapping.")
    return parsed


def _walk_values(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.append(str(k))
            found.extend(_walk_values(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_values(item))
    elif obj is not None:
        found.append(str(obj))
    return found


def validate_recipe_file(recipe_path: Path) -> dict[str, Any]:
    """Validate a TRL or verl GRPO recipe YAML against HIP-1350 invariants."""
    text = recipe_path.read_text(encoding="utf-8")
    cfg = parse_simple_yaml(text)
    errors: list[str] = []

    # 1. Forbid any active reference to deprecated/unmaintained Open-R1 in config keys/values
    all_tokens = [tok.lower() for tok in _walk_values(cfg)]
    for tok in all_tokens:
        if "open-r1" in tok or "open_r1" in tok:
            errors.append(
                f"Forbidden reference to unmaintained Open-R1 in config value: {tok!r}"
            )

    # 2. Validate recipe metadata & license audit alignment
    meta = cfg.get("recipe_metadata")
    if not isinstance(meta, dict):
        errors.append("Missing required top-level 'recipe_metadata' section.")
        meta = {}
    else:
        if meta.get("hip") != "HIP-1350":
            errors.append(f"Expected hip=='HIP-1350', got {meta.get('hip')!r}")
        if meta.get("recipe_license") != "Apache-2.0":
            errors.append("Recipe license must be Apache-2.0.")
        if meta.get("separate_license_review_required") is not True:
            errors.append(
                "Recipes referencing external model weights/datasets must set "
                "'separate_license_review_required: true'."
            )

    # 3. Validate independent held-out evaluation hook
    eval_cfg = cfg.get("independent_held_out_eval")
    if not isinstance(eval_cfg, dict):
        errors.append("Missing required 'independent_held_out_eval' section.")
    else:
        if eval_cfg.get("require_zero_leakage_check") is not True:
            errors.append("independent_held_out_eval.require_zero_leakage_check must be true.")
        if not eval_cfg.get("verifier_module"):
            errors.append("independent_held_out_eval.verifier_module must be set.")

    # 4. Cross-check trainer framework against LICENSE_AUDIT.json
    audit = load_license_audit()
    approved_trainers = {
        a["name"].split(" ")[0]
        for a in audit.get("artifacts", [])
        if a.get("status") == "APPROVED_EXTERNAL_TRAINER" and a.get("maintained") is True
    }
    framework = str(meta.get("trainer_framework", ""))
    if framework not in approved_trainers:
        errors.append(
            f"Trainer framework {framework!r} is not in maintained approved trainers {sorted(approved_trainers)}."
        )

    # 5. Framework-specific schema checks
    if framework == "huggingface/trl":
        if cfg.get("rl") != "grpo":
            errors.append("TRL recipe must set 'rl: grpo'.")
        if not cfg.get("base_model") or not cfg.get("model_revision"):
            errors.append("TRL recipe must pin both 'base_model' and 'model_revision'.")
        trl_block = cfg.get("trl")
        if not isinstance(trl_block, dict):
            errors.append("TRL recipe missing 'trl' config mapping.")
        else:
            beta = trl_block.get("beta")
            if not isinstance(beta, (int, float)) or beta <= 0.0:
                errors.append(f"trl.beta must be > 0 to prevent KL collapse, got {beta!r}")
            num_gen = trl_block.get("num_generations")
            if not isinstance(num_gen, int) or num_gen < 4:
                errors.append(f"trl.num_generations must be >= 4, got {num_gen!r}")
            rfuncs = trl_block.get("reward_funcs")
            if not isinstance(rfuncs, list) or len(rfuncs) < 2:
                errors.append("trl.reward_funcs must list at least 2 auditable reward functions.")
            if trl_block.get("scale_rewards") is not True:
                errors.append("trl.scale_rewards must be true for GRPO group normalization.")
    elif framework == "volcengine/verl":
        algo = cfg.get("algorithm")
        if not isinstance(algo, dict) or algo.get("adv_estimator") != "grpo":
            errors.append("verl recipe must set algorithm.adv_estimator: grpo.")
        arr = cfg.get("actor_rollout_ref")
        if not isinstance(arr, dict):
            errors.append("verl recipe missing actor_rollout_ref section.")
        else:
            model_cfg = arr.get("model", {})
            if not model_cfg.get("path") or not model_cfg.get("revision"):
                errors.append("verl recipe must pin actor_rollout_ref.model.path and revision.")
            actor_cfg = arr.get("actor", {})
            if actor_cfg.get("use_kl_loss") is not True:
                errors.append("verl recipe must enable actor.use_kl_loss: true.")
            kl_coef = actor_cfg.get("kl_loss_coef")
            if not isinstance(kl_coef, (int, float)) or kl_coef <= 0.0:
                errors.append(f"verl actor.kl_loss_coef must be > 0, got {kl_coef!r}")
            if actor_cfg.get("kl_loss_type") != "low_var_kl":
                errors.append("verl actor.kl_loss_type must be 'low_var_kl' (Schulman k3).")
            rollout_cfg = arr.get("rollout", {})
            if not isinstance(rollout_cfg.get("n"), int) or rollout_cfg["n"] < 4:
                errors.append("verl rollout.n (group size G) must be >= 4.")
        crf = cfg.get("custom_reward_function")
        if not isinstance(crf, dict) or not crf.get("path") or not crf.get("name"):
            errors.append("verl recipe must specify custom_reward_function.path and name.")

    return {
        "recipe_file": recipe_path.name,
        "trainer_framework": framework,
        "valid": len(errors) == 0,
        "errors": errors,
        "parsed_summary": {
            "base_model": cfg.get("base_model")
            or cfg.get("actor_rollout_ref", {}).get("model", {}).get("path"),
            "kl_beta": cfg.get("trl", {}).get("beta")
            or cfg.get("actor_rollout_ref", {}).get("actor", {}).get("kl_loss_coef"),
            "group_size": cfg.get("trl", {}).get("num_generations")
            or cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n"),
        },
    }


def validate_all_recipes(recipes_dir: Path | None = None) -> list[dict[str, Any]]:
    """Validate all YAML recipes in ``examples/grpo_reasoning/recipes/``."""
    base = recipes_dir or Path(__file__).resolve().parent
    results = []
    for yaml_path in sorted(base.glob("*.yaml")):
        results.append(validate_recipe_file(yaml_path))
    return results
