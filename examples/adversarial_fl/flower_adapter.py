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
"""Drop-in Flower (``flwr.server.strategy.Strategy``) adapter & stdlib shim.

When Flower (``flwr``) is installed in the host environment,
``AdversarialDefenseStrategy`` subclasses ``flwr.server.strategy.Strategy``
directly and converts between Flower ``Parameters`` and flat float64 parameter
vectors. When ``flwr`` is absent (pure-stdlib offline benchmark mode), the
module exposes protocol-compatible dataclasses (``StdlibParameters``,
``StdlibFitRes``, ``StdlibClientProxy``) so the exact same strategy contract
and serialization path execute with zero third-party dependencies.
"""

from __future__ import annotations

import importlib.util
import struct
from dataclasses import dataclass, field
from typing import Any, Sequence

from .attacks import ClientSubmission
from .corpus import Example
from .defenses import AggregationResult, DefenseConfig, aggregate_submissions
from .model import NUM_PARAMS, evaluate_clean

FLOWER_AVAILABLE: bool = importlib.util.find_spec("flwr") is not None

if FLOWER_AVAILABLE:  # pragma: no cover - optional external dependency
    import flwr.common as _flwr_common
    from flwr.server.strategy import Strategy as _BaseStrategy
else:
    _flwr_common = None

    class _BaseStrategy:  # type: ignore[no-redef]
        """Pure-stdlib protocol mirror of ``flwr.server.strategy.Strategy``."""

        def initialize_parameters(self, client_manager: Any) -> Any:
            raise NotImplementedError

        def configure_fit(
            self, server_round: int, parameters: Any, client_manager: Any
        ) -> list[tuple[Any, Any]]:
            raise NotImplementedError

        def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[Any, Any]],
            failures: list[Any],
        ) -> tuple[Any | None, dict[str, Any]]:
            raise NotImplementedError

        def configure_evaluate(
            self, server_round: int, parameters: Any, client_manager: Any
        ) -> list[tuple[Any, Any]]:
            raise NotImplementedError

        def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[Any, Any]],
            failures: list[Any],
        ) -> tuple[float | None, dict[str, Any]]:
            raise NotImplementedError

        def evaluate(
            self, server_round: int, parameters: Any
        ) -> tuple[float, dict[str, Any]] | None:
            raise NotImplementedError


@dataclass(frozen=True)
class StdlibParameters:
    """Protocol-compatible stand-in for ``flwr.common.Parameters``."""

    tensors: list[bytes]
    tensor_type: str = "float64.flat"


@dataclass(frozen=True)
class StdlibClientProxy:
    """Protocol-compatible stand-in for ``flwr.server.client_proxy.ClientProxy``."""

    cid: str


@dataclass(frozen=True)
class StdlibFitRes:
    """Protocol-compatible stand-in for ``flwr.common.FitRes``."""

    parameters: StdlibParameters
    num_examples: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StdlibEvaluateRes:
    """Protocol-compatible stand-in for ``flwr.common.EvaluateRes``."""

    loss: float
    num_examples: int
    metrics: dict[str, Any] = field(default_factory=dict)


def encode_flat_params(values: Sequence[float]) -> StdlibParameters:
    """Pack a flat float64 vector into a deterministic wire ``Parameters`` object."""
    packed = struct.pack(f"<{len(values)}d", *values)
    if FLOWER_AVAILABLE and _flwr_common is not None:  # pragma: no cover
        return _flwr_common.Parameters(tensors=[packed], tensor_type="float64.flat")
    return StdlibParameters(tensors=[packed], tensor_type="float64.flat")


def decode_flat_params(parameters: Any) -> list[float]:
    """Unpack wire ``Parameters`` back into a flat float64 vector."""
    if getattr(parameters, "tensor_type", "") == "float64.flat":
        raw = parameters.tensors[0]
        count = len(raw) // 8
        return list(struct.unpack(f"<{count}d", raw))
    if FLOWER_AVAILABLE and _flwr_common is not None:  # pragma: no cover
        ndarrays = _flwr_common.parameters_to_ndarrays(parameters)
        flat: list[float] = []
        for arr in ndarrays:
            flat.extend(float(x) for x in arr.reshape(-1))
        return flat
    raise ValueError(f"Unsupported tensor_type: {getattr(parameters, 'tensor_type', None)!r}")


class AdversarialDefenseStrategy(_BaseStrategy):
    """Flower-compatible server strategy wrapping HIP-1350 defense aggregators.

    Accepts client parameter updates (encoded as updated weights ``theta_i`` or
    deltas when ``updates_are_deltas=True``), invokes ``aggregate_submissions``
    with the configured defense (including ``dso_hybrid_eval_cosine``), updates
    the server parameter state, and emits explicit acceptance/rejection audit
    metrics on every round.
    """

    def __init__(
        self,
        initial_params: Sequence[float],
        defense_config: DefenseConfig,
        eval_val: Sequence[Example] | None = None,
        eval_clean: Sequence[Example] | None = None,
        updates_are_deltas: bool = False,
    ) -> None:
        if len(initial_params) != NUM_PARAMS:
            raise ValueError(
                f"Expected {NUM_PARAMS} parameters, got {len(initial_params)}"
            )
        self.current_params: list[float] = list(initial_params)
        self.defense_config = defense_config
        self.eval_val = tuple(eval_val) if eval_val is not None else ()
        self.eval_clean = tuple(eval_clean) if eval_clean is not None else ()
        self.updates_are_deltas = updates_are_deltas
        self.last_aggregation: AggregationResult | None = None
        self.round_history: list[dict[str, Any]] = []

    def initialize_parameters(self, client_manager: Any = None) -> Any:
        """Return serialized initial global model parameters."""
        return encode_flat_params(self.current_params)

    def configure_fit(
        self,
        server_round: int,
        parameters: Any,
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Dispatch current global parameters and round metadata to clients."""
        clients = (
            client_manager.all().values()
            if hasattr(client_manager, "all")
            else list(client_manager)
        )
        fit_cfg = {
            "server_round": server_round,
            "defense": self.defense_config.name,
        }
        return [(client, {"parameters": parameters, "config": fit_cfg}) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[Any | None, dict[str, Any]]:
        """Aggregate client ``FitRes`` updates through the configured defense."""
        if not results:
            return None, {"accepted_count": 0, "rejected_count": 0, "failures": len(failures)}

        submissions: list[ClientSubmission] = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            vec = decode_flat_params(fit_res.parameters)
            if self.updates_are_deltas:
                delta = tuple(vec)
            else:
                delta = tuple(
                    vec[i] - self.current_params[i] for i in range(NUM_PARAMS)
                )
            metrics = dict(getattr(fit_res, "metrics", {}) or {})
            cid_raw = getattr(client_proxy, "cid", str(idx))
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                cid = idx

            base_round = int(metrics.get("base_round", server_round))
            staleness_lag = int(
                metrics.get("staleness_lag", max(0, server_round - base_round))
            )
            submissions.append(
                ClientSubmission(
                    client_id=cid,
                    round_idx=server_round,
                    base_round=base_round,
                    staleness_lag=staleness_lag,
                    delta=delta,
                    num_samples=int(getattr(fit_res, "num_examples", 1)),
                    is_malicious=bool(metrics.get("is_malicious", False)),
                    is_honest_straggler=bool(metrics.get("is_honest_straggler", False)),
                    is_minority_specialist=bool(
                        metrics.get("is_minority_specialist", False)
                    ),
                    attack_tag=str(metrics.get("attack_tag", "unknown")),
                )
            )

        agg_res = aggregate_submissions(
            submissions=submissions,
            config=self.defense_config,
            global_params=self.current_params,
            eval_val=self.eval_val if self.eval_val else None,
        )
        self.last_aggregation = agg_res

        if not agg_res.quorum_stalled:
            for i in range(NUM_PARAMS):
                self.current_params[i] += agg_res.aggregated_delta[i]

        honest_decs = [d for d in agg_res.decisions if not d.is_malicious]
        honest_rej = sum(1 for d in honest_decs if not d.accepted)
        round_frr = (100.0 * honest_rej / len(honest_decs)) if honest_decs else 0.0

        audit_metrics: dict[str, Any] = {
            "server_round": server_round,
            "defense": self.defense_config.name,
            "flower_runtime": "flwr" if FLOWER_AVAILABLE else "stdlib_shim",
            "accepted_count": agg_res.accepted_count,
            "rejected_count": agg_res.rejected_count,
            "quorum_stalled": agg_res.quorum_stalled,
            "round_frr": round(round_frr, 4),
            "failures": len(failures),
        }
        self.round_history.append(audit_metrics)
        return encode_flat_params(self.current_params), audit_metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Any,
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Dispatch evaluation config to participating clients."""
        clients = (
            client_manager.all().values()
            if hasattr(client_manager, "all")
            else list(client_manager)
        )
        return [(client, {"parameters": parameters, "server_round": server_round}) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """Weighted average of client evaluation losses."""
        if not results:
            return None, {"failures": len(failures)}
        total_examples = sum(int(r.num_examples) for _c, r in results)
        if total_examples <= 0:
            return None, {"failures": len(failures)}
        weighted_loss = (
            sum(float(r.loss) * int(r.num_examples) for _c, r in results)
            / total_examples
        )
        return weighted_loss, {
            "server_round": server_round,
            "total_examples": total_examples,
            "failures": len(failures),
        }

    def evaluate(
        self,
        server_round: int,
        parameters: Any,
    ) -> tuple[float, dict[str, Any]] | None:
        """Centralized held-out evaluation hook compatible with Flower servers."""
        if not self.eval_clean:
            return None
        params = decode_flat_params(parameters)
        report = evaluate_clean(params, self.eval_clean)
        return report.loss, {
            "server_round": server_round,
            "clean_accuracy": round(report.accuracy, 4),
            "per_domain_accuracy": {
                k: round(v, 4) for k, v in report.per_domain_accuracy.items()
            },
        }


def submissions_to_flower_results(
    submissions: Sequence[ClientSubmission],
    global_params: Sequence[float],
    updates_are_deltas: bool = False,
) -> list[tuple[StdlibClientProxy, StdlibFitRes]]:
    """Convert benchmark ``ClientSubmission`` list into Flower ``(Proxy, FitRes)`` pairs."""
    pairs: list[tuple[StdlibClientProxy, StdlibFitRes]] = []
    for sub in submissions:
        payload = (
            list(sub.delta)
            if updates_are_deltas
            else [global_params[i] + sub.delta[i] for i in range(NUM_PARAMS)]
        )
        proxy = StdlibClientProxy(cid=str(sub.client_id))
        fit_res = StdlibFitRes(
            parameters=encode_flat_params(payload),
            num_examples=sub.num_samples,
            metrics={
                "base_round": sub.base_round,
                "staleness_lag": sub.staleness_lag,
                "is_malicious": sub.is_malicious,
                "is_honest_straggler": sub.is_honest_straggler,
                "is_minority_specialist": sub.is_minority_specialist,
                "attack_tag": sub.attack_tag,
            },
        )
        pairs.append((proxy, fit_res))
    return pairs
