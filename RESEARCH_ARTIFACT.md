# Research Artifact: Open Reasoning RL via GRPO

## 1. Methodology
This experiment reproduces a reasoning capability improvement in a small-scale model (`Qwen2.5-0.5B`) using **Group Relative Policy Optimization (GRPO)**. Unlike PPO, GRPO estimates the baseline from the mean reward of a group of outputs for the same prompt, significantly reducing computational overhead by removing the need for a separate Value Function (Critic) model.

### Data Strategy
- **Source**: `open-thoughts/OpenThoughts-110k`.
- **License**: Openly licensed (CC-BY-4.0).
- **Format**: Converted to ChatML format with a focus on Chain-of-Thought (CoT) extraction.

### Reward Modeling
We utilize three auditable reward functions:
1.  **Correctness Reward**: Binary/Scalar check against ground truth answers.
2.  **Format Reward**: Enforces the `<thought>...</thought><answer>...</answer>` structure.
3.  **Reasoning Reward**: A heuristic reward for thought traces exceeding a minimum token threshold to prevent "reasoning collapse."

## 2. Reproducible Benchmark
To validate the implementation, run:
```bash
python recipes/grpo_reasoning.py
```

### Baseline vs. Post-Training (Expected)
| Metric | Base Model (Qwen2.5-0.5B-Instruct) | GRPO Fine-tuned |
| :--- | :--- | :--- |
| Format Adherence | ~40% | >90% |
| Accuracy (Subset) | Baseline | Baseline + 2-5% |
| Avg. Thought Length | Variable | Consistent (>50 tokens) |

## 3. Versioned Specification
- **Framework**: `trl` (pinned version via `requirements.txt`)
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (Revision: `pinned`)
- **Hardware Requirement**: 1x A100 (40GB) or 1x L40S for optimal group generation speed.

## 4. Statement of Unproven Claims
- **Scalability**: While GRPO works for 0.5B models, the "DeepSeek-style" emergent reasoning (Aha! moments) typically requires larger parameter counts (7B+) or more extensive training steps than this micro-bounty allows.
- **Generalization**: The model is evaluated on a held-out set of OpenThoughts, but its performance on OOD (Out-of-Distribution) math or coding benchmarks remains unverified.
- **Reward Hacking**: Long-term training might lead to the model generating "filler" reasoning tokens to maximize the reasoning length reward without improving logic.

## 5. Governance & Licensing
- **Source Code**: Apache-2.0.
- **Artifacts**: This research artifact is provided under CC-BY-4.0.
- **Decision Making**: Implementation status is subject to Zoo-apps maintainer validation via PR merge.
