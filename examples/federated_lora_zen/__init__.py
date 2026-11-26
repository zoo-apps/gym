# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Federated LoRA and QLoRA for Zen across independently administered workers.

Reference implementation and reproducible research benchmark for zoo-apps/gym#3
(HIP-1350): evaluating 3-worker federated LoRA and 4-bit NF4 QLoRA adaptation,
naive FedAvg(A, B) vs exact effective-delta SVD aggregation, mid-run node-loss
cold recovery, and matched comparison against centralized LoRA/QLoRA baselines.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
