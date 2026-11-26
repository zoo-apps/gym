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
"""Open reasoning RL / GRPO training pipeline reference implementation (HIP-1350).

Clean-room Apache-2.0 pure-stdlib Python implementation of Group Relative Policy
Optimization (GRPO) with auditable deterministic verifiers, SHA-256 pinned
disjoint train/val/held-out reasoning datasets, matched baselines, honest
negative regimes, and schema-validated TRL & verl production recipes.
"""

__version__ = "0.1.0"
__hip__ = "HIP-1350"
