"""V2 metrics adapter that preserves the repository's existing logger style."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from train.logger import (
    Color,
    UniversalPricingTrainingLogger,
    fmt_num,
)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class HierarchicalPricingEpisodeMetrics:
    """Accumulate v2 economics, policies, and controller updates."""

    rewards: list[float] = field(default_factory=list)
    gross_agent_profits: list[float] = field(default_factory=list)
    net_agent_profits: list[float] = field(default_factory=list)
    agent_costs: list[float] = field(default_factory=list)
    gross_opponent_profits: list[float] = field(default_factory=list)
    net_opponent_profits: list[float] = field(default_factory=list)
    opponent_costs: list[float] = field(default_factory=list)
    regimes: list[float] = field(default_factory=list)
    opponent_regimes: list[float] = field(default_factory=list)
    market_shares: list[float] = field(default_factory=list)
    retention_rates: list[float] = field(default_factory=list)
    uniform_prices: list[float] = field(default_factory=list)
    bbp_new_prices: list[float] = field(default_factory=list)
    bbp_old_prices: list[float] = field(default_factory=list)
    strategy_bbp_probabilities: list[float] = field(default_factory=list)
    regime_changes: list[float] = field(default_factory=list)
    update_values: dict[str, list[float]] = field(default_factory=dict)

    def record_step(
        self,
        *,
        reward: float,
        info: Mapping[str, Any],
        policy_diagnostics: Mapping[str, float],
    ) -> None:
        self.rewards.append(float(reward))
        self.gross_agent_profits.append(
            float(info["gross_agent_profit"])
        )
        self.net_agent_profits.append(float(info["net_agent_profit"]))
        self.agent_costs.append(
            float(info["agent_bbp_operating_cost"])
        )
        self.gross_opponent_profits.append(
            float(info["gross_opponent_profit"])
        )
        self.net_opponent_profits.append(
            float(info["net_opponent_profit"])
        )
        self.opponent_costs.append(
            float(info["opponent_bbp_operating_cost"])
        )
        self.regimes.append(float(info["agent_regime"]))
        self.opponent_regimes.append(float(info["opponent_regime"]))
        self.regime_changes.append(float(bool(info["regime_changed"])))
        self.market_shares.append(float(info["market_share"]))
        self.retention_rates.append(float(info["retention_rate"]))
        self.uniform_prices.append(float(info["agent_uniform_price"]))
        self.bbp_new_prices.append(float(info["agent_bbp_new_price"]))
        self.bbp_old_prices.append(float(info["agent_bbp_old_price"]))
        if "bbp_regime_probability" in policy_diagnostics:
            self.strategy_bbp_probabilities.append(
                float(policy_diagnostics["bbp_regime_probability"])
            )

    def record_update(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"Non-finite update metric: {name}")
            self.update_values.setdefault(name, []).append(numeric)

    def summary(self) -> dict[str, float]:
        net_agent = float(np.sum(self.net_agent_profits))
        net_opponent = float(np.sum(self.net_opponent_profits))
        result = {
            "normalized_net_reward_total": float(np.sum(self.rewards)),
            "normalized_net_reward_mean": _mean(self.rewards),
            "gross_agent_profit_total": float(
                np.sum(self.gross_agent_profits)
            ),
            "net_agent_profit_total": net_agent,
            "agent_bbp_operating_cost_total": float(
                np.sum(self.agent_costs)
            ),
            "gross_opponent_profit_total": float(
                np.sum(self.gross_opponent_profits)
            ),
            "net_opponent_profit_total": net_opponent,
            "opponent_bbp_operating_cost_total": float(
                np.sum(self.opponent_costs)
            ),
            "net_profit_advantage_total": net_agent - net_opponent,
            "agent_bbp_period_fraction": _mean(self.regimes),
            "opponent_bbp_period_fraction": _mean(
                self.opponent_regimes
            ),
            "regime_change_count": float(np.sum(self.regime_changes)),
            "mean_market_share": _mean(self.market_shares),
            "mean_retention_rate": _mean(self.retention_rates),
            "mean_agent_uniform_price": _mean(self.uniform_prices),
            "mean_agent_bbp_new_price": _mean(self.bbp_new_prices),
            "mean_agent_bbp_old_price": _mean(self.bbp_old_prices),
            "mean_agent_bbp_price_spread": _mean(
                [
                    old - new
                    for new, old in zip(
                        self.bbp_new_prices, self.bbp_old_prices
                    )
                ]
            ),
            "std_agent_uniform_price": float(
                np.std(self.uniform_prices)
            )
            if self.uniform_prices
            else 0.0,
            "std_agent_bbp_new_price": float(
                np.std(self.bbp_new_prices)
            )
            if self.bbp_new_prices
            else 0.0,
            "std_agent_bbp_old_price": float(
                np.std(self.bbp_old_prices)
            )
            if self.bbp_old_prices
            else 0.0,
        }
        regime_arrays = {
            "uniform": np.asarray(self.regimes) == 0.0,
            "bbp": np.asarray(self.regimes) == 1.0,
        }
        for label, mask in regime_arrays.items():
            indices = np.flatnonzero(mask)

            def selected(values: Sequence[float]) -> list[float]:
                return [values[int(index)] for index in indices]

            gross = selected(self.gross_agent_profits)
            costs = selected(self.agent_costs)
            net = selected(self.net_agent_profits)
            advantages = [
                self.net_agent_profits[int(index)]
                - self.net_opponent_profits[int(index)]
                for index in indices
            ]
            result.update(
                {
                    f"{label}_period_count": float(len(indices)),
                    f"{label}_gross_agent_profit_total": float(
                        np.sum(gross)
                    ),
                    f"{label}_bbp_operating_cost_total": float(
                        np.sum(costs)
                    ),
                    f"{label}_net_agent_profit_total": float(np.sum(net)),
                    f"{label}_net_profit_advantage_total": float(
                        np.sum(advantages)
                    ),
                    f"{label}_mean_net_agent_profit": _mean(net),
                }
            )
        if self.strategy_bbp_probabilities:
            result["mean_bbp_regime_probability"] = _mean(
                self.strategy_bbp_probabilities
            )
        result.update(
            {
                f"mean_{name}": _mean(values)
                for name, values in self.update_values.items()
            }
        )
        return result


class HierarchicalPricingMetricsAdapter:
    """Compose the existing logger; do not modify its legacy/v1 implementation."""

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        verbose: bool = True,
    ) -> None:
        self.backend = UniversalPricingTrainingLogger(
            metrics_path, verbose=verbose
        )
        self.verbose = verbose

    def write_metric_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> None:
        self.backend.write_metric_records(records)

    def log_run_start(self, **values: Any) -> None:
        self.backend.log_run_start(**values)

    @staticmethod
    def _mean_loss(
        record: Mapping[str, Any], suffix: str
    ) -> float | None:
        values = [
            float(value)
            for name, value in record.items()
            if name.endswith(suffix) and name.startswith("mean_")
        ]
        return _mean(values) if values else None

    def log_episode(
        self,
        record: Mapping[str, Any],
        *,
        budget_steps: int,
    ) -> None:
        compatibility = dict(record)
        compatibility.update(
            {
                "raw_agent_profit_total": record[
                    "net_agent_profit_total"
                ],
                "profit_advantage_total": record[
                    "net_profit_advantage_total"
                ],
                "replay_size": record.get(
                    "active_replay_size", 0
                ),
                "replay_unit": record.get(
                    "active_replay_unit", "transitions"
                ),
                "replay_bbp_fraction": record.get(
                    "active_replay_bbp_fraction",
                    record["agent_bbp_period_fraction"],
                ),
            }
        )
        critic = self._mean_loss(record, "_critic_loss")
        actor = self._mean_loss(record, "_actor_loss")
        critic_gradient = self._mean_loss(
            record, "_critic_gradient_norm"
        )
        actor_gradient = self._mean_loss(
            record, "_actor_gradient_norm"
        )
        if all(
            value is not None
            for value in (
                critic,
                actor,
                critic_gradient,
                actor_gradient,
            )
        ):
            compatibility.update(
                {
                    "mean_critic_loss": critic,
                    "mean_actor_loss": actor,
                    "mean_critic_gradient_norm": critic_gradient,
                    "mean_actor_gradient_norm": actor_gradient,
                }
            )
        self.backend.log_episode(
            compatibility, budget_steps=budget_steps
        )
        if self.verbose:
            print(
                f"{Color.MAGENTA}[v2]{Color.END} "
                f"phase:{record['curriculum_phase']} "
                f"stage:{record['curriculum_stage_name']} "
                f"net:{float(record['net_agent_profit_total']):.2f} "
                f"gross:{float(record['gross_agent_profit_total']):.2f} "
                f"bbp-cost:{float(record['agent_bbp_operating_cost_total']):.2f}",
                flush=True,
            )

    def log_mastery(
        self,
        *,
        phase: str,
        stage: str,
        score: float,
        passed: bool,
        consecutive_passes: int,
    ) -> None:
        if self.verbose:
            color = Color.GREEN if passed else Color.YELLOW
            print(
                f"{color}[mastery]{Color.END} phase:{phase} stage:{stage} "
                f"score:{score:.3f} consecutive:{consecutive_passes}",
                flush=True,
            )

    def log_checkpoint(self, path: str | Path, steps: int) -> None:
        self.backend.log_checkpoint(path, steps)

    def log_validation(self, record: Mapping[str, Any]) -> None:
        compatibility = dict(record)
        compatibility.setdefault(
            "mean_raw_agent_profit_total",
            record.get("mean_net_agent_profit_total", 0.0),
        )
        compatibility.setdefault(
            "mean_profit_advantage_total",
            record.get("mean_net_profit_advantage_total", 0.0),
        )
        compatibility.setdefault(
            "mean_normalized_reward_total",
            record.get("mean_normalized_net_reward_total", 0.0),
        )
        self.backend.log_validation(compatibility)

    def log_terminal(
        self,
        status: str,
        *,
        environment_steps: int,
        message: str | None = None,
    ) -> None:
        self.backend.log_terminal(
            status,
            environment_steps=environment_steps,
            message=message,
        )
