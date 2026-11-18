# Research Artifact: Collective Model Merging (DeltaSoup)

## 1. Specification
DeltaSoup is a protocol for merging multiple model "deltas" (the difference between a fine-tuned model and its base) into a single unified model. Unlike traditional weight averaging (Model Soup), DeltaSoup operates in the parameter-gradient space.

### Formal Definition
Given a base model $W_{base}$ and $N$ fine-tuned models $W_i$:
1. Extract deltas: $\Delta_i = W_i - W_{base}$
2. Detect compatibility: $C_{i,j} = \cos(\Delta_i, \Delta_j)$
3. Merge: $W_{merged} = W_{base} + \sum_{i=1}^{N} \lambda_i \Delta_i$

Where $\lambda_i$ is the mixing coefficient (default $1/N$).

## 2. Reproducible Benchmark Protocol
To validate the implementation:
1. **Base**: Llama-3-8B.
2. **Deltas**: One delta trained on GSM8K (Math), one on CodeAlpaca (Coding).
3. **Metric**: Zero-shot accuracy on held-out tasks for both domains.
4. **Success Criteria**: The merged model must retain >90% of the performance of individual experts on their respective domains.

## 3. Incompatibility Detection
We use the **Cosine Alignment Metric**. If $\cos(\Delta_i, \Delta_j) < 0$, the updates are considered "Interfering". In such cases, the merger flags a conflict, as simple averaging will likely result in catastrophic forgetting of both tasks.

## 4. Statement of Unproven Claims
- **Scaling**: It remains unproven if DeltaSoup maintains linear performance gains as $N$ (number of deltas) exceeds 10.
- **Quantization**: The interaction between DeltaSoup and 4-bit quantization (bitsandbytes) during the merge process is not yet validated and may introduce rounding noise.
- **Emergent Capabilities**: It is unproven whether merging two unrelated tasks (e.g., Music Theory and Python Coding) creates emergent cross-domain reasoning.

## 5. License
This implementation and research artifact are licensed under Apache-2.0.
