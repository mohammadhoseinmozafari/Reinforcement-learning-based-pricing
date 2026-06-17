from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from evaluation.results import EvaluationResult
@dataclass 
class EvaluationMetrics:
    episode_rewards: List[float] = field(default_factory=list)
    episode_profits: List[float] = field(default_factory=list)
    episode_prices: List[float] = field(default_factory=list)
    episode_market_shares: List[float] = field(default_factory=list)

    step_profits: List[float] = field(default_factory=list)
    step_prices: List[float] = field(default_factory=list)
    step_market_shares: List[float] = field(default_factory=list)


    def reset_episode (self) -> None:
        
        self.step_profits = []
        self.step_prices = []
        self.step_market_shares  = []

    def record_step (self, info : Dict) -> None:
        self.step_profits.append(info.get("profit", 0.0))
        self.step_prices.append(info.get("price", 0.0))
        self.step_market_shares.append(info.get("market_share", 0.0))

    def end_episode(self, total_reward: float):

        self.episode_rewards.append(total_reward)
        self.episode_profits.append(sum(self.step_profits))
        self.episode_prices.append(float(np.mean(self.step_prices)) if self.step_prices else 0.0)
        self.episode_market_shares.append(float(np.mean(self.step_market_shares)) if self.step_market_shares else 0.0)

        
    def to_results(self) -> EvaluationResult:
        return EvaluationResult (
            avg_reward =  float(np.mean(self.episode_rewards)),
            std_reward= float(np.std(self.episode_rewards)),
            avg_profit= float(np.mean(self.episode_profits)),
            total_profit= float(np.sum(self.episode_profits)),
            avg_price= float(np.mean(self.episode_prices)),
            avg_market_share = float(np.mean(self.episode_market_shares)),

        )
            

    