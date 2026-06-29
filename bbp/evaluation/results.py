from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class EvaluationResult :
    episode_rewards : List[float]
    std_reward: float
    episode_profits: List[float]
    total_profit : float
    episode_opp_profits : List[float]
    total_opp_profit : float
    episode_opp_uniform_prices : List[float]
    episode_opp_new_prices  : List[float]
    episode_opp_old_prices  :   List[float]
    episode_prices: List[float]
    episode_market_shares : List[float]

    def to_dict(self) -> Dict[str, Any] :
        return asdict(self)

       
       
       
       
       
       