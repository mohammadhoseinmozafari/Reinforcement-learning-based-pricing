from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationResult :
    avg_reward : float
    std_reward: float
    avg_profit : float
    total_profit : float
    avg_price: float
    avg_market_share : float

    def to_dict(self) -> Dict[str, float] :
        return {
            "avg_reward" : self.avg_reward,
            "std_reward" : self.std_reward,
            "avg_profit" : self.avg_profit,
            "total_profit" : self.total_profit,
            "avg_price" : self.avg_price,
            "avg_market_share" : self.avg_market_share
        }