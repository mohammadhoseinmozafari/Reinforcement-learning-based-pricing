from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from evaluation.results import EvaluationResult
@dataclass
class EvaluationMetrics:
    """Metrics tracked during evaluation."""
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

    # Per-step tracking (for current episode)
    step_profits: List[float] = field(default_factory=list)
    step_opp_profits: List[float] = field(default_factory=list)

    step_uniform_prices: List[float] = field(default_factory=list)
    step_new_prices: List[float] = field(default_factory=list)
    step_old_prices: List[float] = field(default_factory=list)

    step_opp_uniform_prices: List[float] = field(default_factory=list)
    step_opp_new_prices: List[float] = field(default_factory=list)
    step_opp_old_prices: List[float] = field(default_factory=list)

    step_market_shares: List[float] = field(default_factory=list)
    
    

    step_regimes : List[float] = field(default_factory=list)
    step_opp_regimes : List[float] = field(default_factory=list)

    def reset_episode(self):
        """Reset per-episode tracking."""
        self.step_profits = []
        self.step_opp_profits=[]

        self.step_uniform_prices = []
        self.step_new_prices = []
        self.step_old_prices = []
        
        self.step_opp_uniform_prices = []
        self.step_opp_new_prices = []
        self.step_opp_old_prices = []

        self.step_market_shares = []
    

        self.step_regimes = []
        self.step_opp_regimes = []

    def record_step(self, info: Dict):
        """Record metrics from a step."""

        self.step_profits.append(info.get("profit", 0.0))
        self.step_opp_profits.append(info.get("opponent_profit", 0.0))

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
        self.episode_opp_profits.append(sum(self.step_opp_profits))

        self.episode_uniform_prices.append(float(np.mean(self.step_uniform_prices)) if self.step_uniform_prices else 0.0)
        self.episode_new_prices.append(float(np.mean(self.step_new_prices)) if self.step_new_prices else 0.0)
        self.episode_old_prices.append(float(np.mean(self.step_old_prices)) if self.step_old_prices else 0.0)

        self.episode_market_shares.append(float(np.mean(self.step_market_shares)) if self.step_market_shares else 0.0)
        

        
        self.episode_opp_uniform_prices.append(float(np.mean(self.step_opp_uniform_prices)) if self.step_opp_uniform_prices else 0.0)
        self.episode_opp_new_prices.append(float(np.mean(self.step_opp_new_prices)) if self.step_opp_new_prices else 0.0)
        self.episode_opp_old_prices.append(float(np.mean(self.step_opp_old_prices)) if self.step_opp_old_prices else 0.0)

        self.episode_opp_regimes.append(float(np.mean(self.step_opp_regimes)))
        self.episode_regimes.append(float(np.mean(self.step_regimes)))
    
    def to_results(self) -> EvaluationResult:
        return EvaluationResult (
            episode_rewards =  self.episode_rewards,
            std_reward= float(np.std(self.episode_rewards)),

            episode_profits= self.episode_profits,
            total_profit= float(np.sum(self.episode_profits)),

            episode_opp_profits = self.episode_opp_profits,
            total_opp_profit =float(np.sum(self.episode_opp_profits)),
            
            episode_uniform_prices= self.episode_uniform_prices,
            episode_new_prices = self.episode_new_prices,
            episode_old_prices = self.episode_old_prices,

            episode_opp_uniform_prices=self.episode_opp_uniform_prices,
            episode_opp_new_prices=self.episode_opp_new_prices,
            episode_opp_old_prices=self.episode_opp_old_prices,


            episode_market_shares = self.episode_market_shares,

        )
    
    def collect_steps(self)-> Dict:
        """Return the current episode's raw step series.

        Prefer :meth:`to_results` for a complete multi-episode evaluation.
        """
        return {
            "step_profits" : self.step_profits,

            "step_uniform_prices"   :   self.step_uniform_prices,
            "step_new_prices"       : self.step_new_prices,
            "step_old_prices"   : self.step_old_prices,
            
            "step_market_shares": self.step_market_shares,
            
            "step_opp_profits" : self.step_opp_profits,
            "step_opp_uniform_prices": self.step_opp_uniform_prices,
            "step_opp_new_prices": self.step_opp_new_prices,
            "step_opp_old_prices": self.step_opp_old_prices,

        }
