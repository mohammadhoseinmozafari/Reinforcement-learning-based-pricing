from dataclasses import dataclass , field
from typing import (
    List,
    Dict,
)
import numpy as np
@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode_rewards: List[float] = field(default_factory=list)

    episode_profits: List[float] = field(default_factory=list)
    episode_opp_profits: List[float] = field(default_factory=list)

    episode_prices: List[float] = field(default_factory=list)
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
    step_prices: List[float] = field(default_factory=list)
    step_market_shares: List[float] = field(default_factory=list)
    step_regimes : List[float] = field(default_factory=list)
    
    step_opp_profits: List[float] = field(default_factory=list)
    step_opp_uniform_prices: List[float] = field(default_factory=list)
    step_opp_new_prices: List[float] = field(default_factory=list)
    step_opp_old_prices: List[float] = field(default_factory=list)

    step_opp_regimes : List[float] = field(default_factory=list)

    def reset_episode(self):
        """Reset per-episode tracking."""
        self.step_profits = []
        self.step_prices = []
        self.step_market_shares = []

        self.step_opp_profits=[]
    
        self.step_opp_uniform_prices = []
        self.step_opp_new_prices = []
        self.step_opp_old_prices = []

    def record_step(self, info: Dict):
        """Record metrics from a step."""
        self.step_profits.append(info.get("profit", 0.0))
        self.step_prices.append(info.get("price", 0.0))
        self.step_market_shares.append(info.get("market_share", 0.0))
    
        self.step_opp_profits.append(info.get("opponent_profit", 0.0))

        self.step_opp_uniform_prices.append(info.get("opponent_price_uniform", 0.0))
        self.step_opp_new_prices.append(info.get("opponent_price_new", 0.0))
        self.step_opp_old_prices.append(info.get("opponent_price_old", 0.0))

    def end_episode(self, total_reward: float):
        """Finalize episode metrics."""
        self.episode_rewards.append(total_reward)
        self.episode_profits.append(sum(self.step_profits))
        self.episode_prices.append(float(np.mean(self.step_prices)) if self.step_prices else 0.0)
        self.episode_market_shares.append(float(np.mean(self.step_market_shares)) if self.step_market_shares else 0.0)
        self.episode_opp_regimes.append(float(np.mean(self.step_opp_regimes)))
        self.episode_regimes.append(float(np.mean(self.step_regimes)))

        self.episode_opp_profits.append(sum(self.step_opp_profits))
        
        self.episode_opp_uniform_prices.append(float(np.mean(self.step_opp_uniform_prices)) if self.step_opp_uniform_prices else 0.0)
        self.episode_opp_new_prices.append(float(np.mean(self.step_opp_new_prices)) if self.step_opp_new_prices else 0.0)
        self.episode_opp_old_prices.append(float(np.mean(self.step_opp_old_prices)) if self.step_opp_old_prices else 0.0)