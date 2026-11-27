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
"""Adversarial Federated-Learning Benchmark Suite (HIP-1350 / zoo-apps/gym#5).

Pure-stdlib reference implementation reproducing poisoning, targeted backdoor,
stale-update / straggler lag, and coordinated collusion cartel scenarios across
six defenses with explicit false-rejection, attack-success, and quality-loss
metrics, plus a drop-in Flower strategy adapter shim.
"""

from __future__ import annotations

__version__ = "0.1.0"
__hip__ = "HIP-1350"
__issue__ = "zoo-apps/gym#5"
