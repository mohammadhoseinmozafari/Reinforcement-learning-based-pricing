from dataclasses import dataclass , field
from typing import (
    Any,
    List,
    Dict,
    Mapping,
)
import numpy as np


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass
class UniversalPricingEpisodeMetrics:
    """Accumulate auditable economic and policy signals for one episode."""

    rewards: List[float] = field(default_factory=list)
    agent_profits: List[float] = field(default_factory=list)
    opponent_profits: List[float] = field(default_factory=list)
    agent_regimes: List[float] = field(default_factory=list)
    opponent_regimes: List[float] = field(default_factory=list)
    regime_changes: List[float] = field(default_factory=list)
    regime_decisions: List[float] = field(default_factory=list)
    agent_uniform_prices: List[float] = field(default_factory=list)
    agent_new_prices: List[float] = field(default_factory=list)
    agent_old_prices: List[float] = field(default_factory=list)
    opponent_uniform_prices: List[float] = field(default_factory=list)
    opponent_new_prices: List[float] = field(default_factory=list)
    opponent_old_prices: List[float] = field(default_factory=list)
    market_shares: List[float] = field(default_factory=list)
    retention_rates: List[float] = field(default_factory=list)
    uniform_regime_probabilities: List[float] = field(default_factory=list)
    bbp_regime_probabilities: List[float] = field(default_factory=list)
    policy_diagnostic_values: Dict[str, List[float]] = field(
        default_factory=dict
    )

    def record_step(
        self,
        *,
        reward: float,
        info: Mapping[str, Any],
        agent_firm: Any,
        opponent_firm: Any,
        policy_diagnostics: Mapping[str, float] | None = None,
    ) -> None:
        """Record one applied market transition using effective regimes/prices."""

        self.rewards.append(float(reward))
        self.agent_profits.append(float(info["raw_agent_profit"]))
        self.opponent_profits.append(float(info["raw_opponent_profit"]))
        self.agent_regimes.append(float(info["agent_regime"]))
        self.opponent_regimes.append(float(info["opponent_regime"]))
        self.regime_changes.append(float(bool(info["regime_changed"])))
        self.regime_decisions.append(float(info["regime_decision_mask"]))
        self.agent_uniform_prices.append(float(agent_firm.uniform_price))
        self.agent_new_prices.append(float(agent_firm.price_new))
        self.agent_old_prices.append(float(agent_firm.price_old))
        self.opponent_uniform_prices.append(float(opponent_firm.uniform_price))
        self.opponent_new_prices.append(float(opponent_firm.price_new))
        self.opponent_old_prices.append(float(opponent_firm.price_old))
        self.market_shares.append(float(agent_firm.market_share))
        self.retention_rates.append(float(agent_firm.retention_rate))
        if policy_diagnostics:
            if "uniform_regime_probability" in policy_diagnostics:
                self.uniform_regime_probabilities.append(
                    float(policy_diagnostics["uniform_regime_probability"])
                )
            if "bbp_regime_probability" in policy_diagnostics:
                self.bbp_regime_probabilities.append(
                    float(policy_diagnostics["bbp_regime_probability"])
                )
            for name in (
                "uniform_mean",
                "uniform_log_std",
                "bbp_new_mean",
                "bbp_premium_mean",
                "regime_temperature",
                "uniform_price_temperature",
                "bbp_price_temperature",
                "actor_hidden_norm",
            ):
                if name in policy_diagnostics:
                    self.policy_diagnostic_values.setdefault(
                        name, []
                    ).append(float(policy_diagnostics[name]))

    def summary(self) -> Dict[str, float]:
        """Return stable flat metrics suitable for JSONL and terminal logging."""

        agent_profit = float(np.sum(self.agent_profits))
        opponent_profit = float(np.sum(self.opponent_profits))
        result = {
            "normalized_reward_total": float(np.sum(self.rewards)),
            "normalized_reward_mean": _mean(self.rewards),
            "raw_agent_profit_total": agent_profit,
            "raw_agent_profit_mean": _mean(self.agent_profits),
            "raw_opponent_profit_total": opponent_profit,
            "raw_opponent_profit_mean": _mean(self.opponent_profits),
            "profit_advantage_total": agent_profit - opponent_profit,
            "agent_bbp_period_fraction": _mean(self.agent_regimes),
            "opponent_bbp_period_fraction": _mean(self.opponent_regimes),
            "regime_change_count": float(np.sum(self.regime_changes)),
            "regime_decision_count": float(np.sum(self.regime_decisions)),
            "mean_agent_uniform_price": _mean(self.agent_uniform_prices),
            "mean_agent_bbp_new_price": _mean(self.agent_new_prices),
            "mean_agent_bbp_old_price": _mean(self.agent_old_prices),
            "mean_agent_bbp_price_spread": _mean(
                [
                    old - new
                    for new, old in zip(
                        self.agent_new_prices,
                        self.agent_old_prices,
                    )
                ]
            ),
            "mean_opponent_uniform_price": _mean(
                self.opponent_uniform_prices
            ),
            "mean_opponent_bbp_new_price": _mean(self.opponent_new_prices),
            "mean_opponent_bbp_old_price": _mean(self.opponent_old_prices),
            "mean_market_share": _mean(self.market_shares),
            "mean_retention_rate": _mean(self.retention_rates),
        }
        if self.uniform_regime_probabilities:
            result["mean_uniform_regime_probability"] = _mean(
                self.uniform_regime_probabilities
            )
        if self.bbp_regime_probabilities:
            result["mean_bbp_regime_probability"] = _mean(
                self.bbp_regime_probabilities
            )
        for name, values in self.policy_diagnostic_values.items():
            result[f"mean_policy_{name}"] = _mean(values)
        return result


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode_rewards: List[float] = field(default_factory=list)

    episode_profits: List[float] = field(default_factory=list)
    episode_opp_profits: List[float] = field(default_factory=list)

    episode_uniform_prices: List[float] = field(default_factory=list)
    episode_new_prices: List[float] = field(default_factory=list)
    episode_old_prices: List[float] = field(default_factory=list)

    episode_opp_uniform_prices: List[float] = field(default_factory=list)
    episode_opp_new_prices: List[float] = field(default_factory=list)
    episode_opp_old_prices: List[float] = field(default_factory=list)

    episode_market_shares: List[float] = field(default_factory=list)
    
    episode_regimes : List[float] = field(default_factory=list)
    episode_opp_regimes: List[float] =field(default_factory=list)

    eval_rewards: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    
    # Per-step tracking (for current episode)
    step_profits: List[float] = field(default_factory=list)
    step_uniform_prices: List[float] = field(default_factory=list)
    step_new_prices: List[float] = field(default_factory=list)
    step_old_prices: List[float] = field(default_factory=list)
    step_market_shares: List[float] = field(default_factory=list)
    
    step_opp_profits: List[float] = field(default_factory=list)
    step_opp_uniform_prices: List[float] = field(default_factory=list)
    step_opp_new_prices: List[float] = field(default_factory=list)
    step_opp_old_prices: List[float] = field(default_factory=list)

    step_regimes : List[float] = field(default_factory=list)
    step_opp_regimes : List[float] = field(default_factory=list)

    def reset_episode(self):
        """Reset per-episode tracking."""
        self.step_profits = []
        
        self.step_uniform_prices = []
        self.step_new_prices = []
        self.step_old_prices = []        
        
        self.step_market_shares = []
        

        self.step_opp_profits=[]
    
        self.step_opp_uniform_prices = []
        self.step_opp_new_prices = []
        self.step_opp_old_prices = []

        self.step_regimes = []
        self.step_opp_regimes = []

    def record_step(self, info: Dict):
        """Record metrics from a step."""

        self.step_profits.append(info.get("profit", 0.0))
        self.step_opp_profits.append(info.get("opponent_profit", 0.0))

        # PricingEnv action order: uniform, BBP-new, BBP-old.
        self.step_uniform_prices.append(info.get("uniform_price", 0.0))
        self.step_new_prices.append(info.get("bbp_price_new", 0.0))
        self.step_old_prices.append(info.get("bbp_price_old", 0.0))

        self.step_opp_uniform_prices.append(info.get("opponent_price_uniform", 0.0))
        self.step_opp_new_prices.append(info.get("opponent_price_new", 0.0))
        self.step_opp_old_prices.append(info.get("opponent_price_old", 0.0))

        self.step_market_shares.append(info.get("market_share", 0.0))

        self.step_regimes.append(info.get("regime", 10))
        self.step_opp_regimes.append(info.get("opponent_regime", 10))
        

        

    def end_episode(self, total_reward: float):
        """Finalize episode metrics."""
        self.episode_rewards.append(total_reward)
        self.episode_profits.append(sum(self.step_profits))

        self.episode_uniform_prices.append(float(np.mean(self.step_uniform_prices)) if self.step_uniform_prices else 0.0)
        self.episode_new_prices.append(float(np.mean(self.step_new_prices)) if self.step_new_prices else 0.0)
        self.episode_old_prices.append(float(np.mean(self.step_old_prices)) if self.step_old_prices else 0.0)

        self.episode_market_shares.append(float(np.mean(self.step_market_shares)) if self.step_market_shares else 0.0)
        

        self.episode_opp_profits.append(sum(self.step_opp_profits))
        
        self.episode_opp_uniform_prices.append(float(np.mean(self.step_opp_uniform_prices)) if self.step_opp_uniform_prices else 0.0)
        self.episode_opp_new_prices.append(float(np.mean(self.step_opp_new_prices)) if self.step_opp_new_prices else 0.0)
        self.episode_opp_old_prices.append(float(np.mean(self.step_opp_old_prices)) if self.step_opp_old_prices else 0.0)

        self.episode_opp_regimes.append(float(np.mean(self.step_opp_regimes)))
        self.episode_regimes.append(float(np.mean(self.step_regimes)))
