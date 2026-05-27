from typing import Any, Dict

import numpy as np
from config.constants import EPISODE_LENGTH
from env.uniform_pricing_env import make_uniform_pricing_env
from train.uniform_training.uniform_training import TrainingConfig
price_bounds = {
    "aggressive_uniform": {
        "min" : 0.5,
        "max": 2.5
    }
    ,
    "passive_uniform": {
        "min": 1.5,
        "max": 3.5

    },
    "premium_uniform": {
        "min": 3.0,
        "max": 5.0
    }
}




def grid_search(opponent_type: str) -> Dict[str, Any]:
    p_bounds= price_bounds.get(opponent_type)
    assert p_bounds
    min_price , max_price = p_bounds['min'],p_bounds['max']
    grid = np.arange(min_price, max_price, 0.01)
    prices = []
    profits = []
    market_shares = []
    demands = []

    config = TrainingConfig(opponent_type=opponent_type)

    for price in grid:
        env = make_uniform_pricing_env(
            opponent=config.opponent_type,
            num_consumers=config.num_consumers,
            episode_length=config.episode_length,
            seed=config.seed,
        )
        env.reset()
        
        normalized_price = env._price_to_normalized(price)
        action = np.array([normalized_price], dtype=np.float32)
        profit = 0.0
        demand = 0.0
        market_share = 0.0
        for i in range(EPISODE_LENGTH):
        
        
        
        
    
        
            next_state, reward, terminated, truncated, info = env.step(action)
            profit+=info.get('profit', 0.0)
            demand+= info.get('demand', 0)
            market_share+= info.get('market_share', 0.0)

        avg_market_share = market_share/ EPISODE_LENGTH
        avg_demand = demand / EPISODE_LENGTH
        prices.append(price)
        profits.append(profit)
        market_shares.append(avg_market_share)
        demands.append(avg_demand)
        
        env.close()

    # Convert to numpy arrays
    prices = np.array(prices)
    profits = np.array(profits)
    optimal_idx  = int(np.argmax(profits))

    market_shares = np.array(market_shares)
    demands = np.array(demands)
    return {
        "prices": prices,
        "profits": profits,
        "optimal_idx": optimal_idx ,
        "market_shares": market_shares,
        "demands": demands
    }


