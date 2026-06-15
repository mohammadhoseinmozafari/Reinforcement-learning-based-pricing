"""
Uniform Price Learning Environment

Simplified single-agent environment for learning optimal uniform pricing.
- Agent action: scalar price p_uniform (continuous)  
- Regime switching is disabled (always uniform pricing)
- Opponent can choose BBP of Uniform pricing strategies

"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


from env.models import HotellingMarket
from env.opponent_policies import (
    OpponentPolicy,
    OpponentObservation,
    ConstantOpponentPolicy,
    create_preset_opponent,
)
from config.constants import (
    NUM_CONSUMERS,
    EPISODE_LENGTH,
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_OLD_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_UNIFORM_MIN,
    PRICE_UNIFORM_MAX,
    RANDOM_SEED,
)


class UniformPricingEnv(gym.Env):
    """
    Single-agent environment for uniform pricing only.
    
    Simplifications from full hierarchical environment:
    - Agent  always use uniform pricing (regime = 0)
    - Agent action is a single scalar: uniform price
    - Observation is a flat vector of market features
    - No strategy controller.
    
    This focused environment is ideal for 'Uniform pricing experiments'
    where we want to learn optimal uniform pricing against
    a stationary opponent.

    """
    
    metadata = {
        "render_modes": [],
        "name": "uniform_pricing_v1"
    }
    
    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
        seed: Optional[int] = RANDOM_SEED,
    ):
        """
        Initialize uniform pricing environment.
        
        Args:
            opponent_policy: Policy for opponent (uses ConstantOpponentPolicy if None)
            num_consumers: Number of consumers in market
            episode_length: Total steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.seed_value = seed
        
        self.market = HotellingMarket(num_consumers=num_consumers, seed=seed)
        

        if opponent_policy is None:
            self.opponent_policy = ConstantOpponentPolicy(
                uniform_price=3.5,
                regime=0  # Always uniform
            )
        else:
            self.opponent_policy = opponent_policy

        
        # =====================================
        # ACTION SPACE: Single uniform price
        # =====================================
        # Action is normalized to [-1, 1], scaled to price range internally
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # =====================================
        # OBSERVATION SPACE
        # =====================================
        # Flat observation vector with key market features
        # Features:
        #   - own_market_share 
        #   - own_price 
        #   - opponent_price_uniform
        #   - opponent_price_new
        #   - opponent_price_old
        #   - last_demand_ratio 
        #   - opponent regime


        obs_dim = 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self._timestep = 0
        self._last_action = 0.5  
        
        self._episode_profits = []
        self._episode_prices = []
        self._episode_market_shares = []
    
    def _action_to_price(self, action: np.ndarray) -> float:
        """Convert normalized action [-1, 1] to actual price."""

        normalized = (float(action[0]) + 1.0) / 2.0 
        normalized = float(np.clip(normalized, 0.0, 1.0))
        price = PRICE_UNIFORM_MIN + normalized * (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        return price
    
    def _price_to_normalized(self, price: float) -> float:
        """Convert actual price to normalized [-1, 1] value."""
        normalized =  (price - PRICE_UNIFORM_MIN) / (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        normalized = 2.0 * normalized - 1.0
        return normalized
    
    def _bbp_price_to_normalized (self , price_new: float, price_old : float) -> Tuple[float, float]:
        price_new_normalized = (price_new- PRICE_BBP_NEW_MIN) / (PRICE_BBP_NEW_MAX - PRICE_BBP_NEW_MIN)
        price_old_normalized = (price_old- PRICE_BBP_OLD_MIN) / (PRICE_BBP_OLD_MAX - PRICE_BBP_OLD_MIN)
        price_new_normalized , price_old_normalized = price_new_normalized* 2.0 -1 , price_old_normalized* 2.0 - 1
        return price_new_normalized , price_old_normalized

    def _normalize_market_share (self, market_share: float) -> float:
        """Convert market share to normalized [-1, 1]"""
        normalized = market_share*2.0 -1
        return normalized
    
    def _normalize_demand_ratio (self, demand_ratio : float) -> float:
        """Convert demand ratio to normalized [-1, 1]"""
        normalized = demand_ratio* 2.0 -1
        return normalized
    
    def _normalize_regime(self, regime : int) -> float:
        """Conver regime to normalized [-1, 1]"""
        normalized = float(regime) * 2.0 - 1.0
        return normalized

    
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from current market state.
        
        Returns:
            Flat observation array
        """
        firm = self.market.firms[0]  # Learning agent is firm 0
        opponent = self.market.firms[1]
        
        # Market share
        market_share_norm = self._normalize_market_share(firm.market_share)
        
        # Prices (normalized)
        own_uniform_price_norm = self._price_to_normalized(firm.uniform_price)
        opp_uniform_price_norm = self._price_to_normalized(opponent.uniform_price)
        opp_price_new_norm , opp_price_old_norm = self._bbp_price_to_normalized(opponent.price_new, opponent.price_old)
        # Demand ratio
        demand_ratio = firm.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.0
        demand_ratio_norm = self._normalize_demand_ratio (demand_ratio)
        
        opponent_regime_norm = self._normalize_regime(opponent.pricing_regime)
        # Profit trend
        # profit_trend = firm.get_profit_trend()
        
        # Popularity change
        # pop_change = firm.get_popularity_change()
        
        obs = np.array([
            market_share_norm,
            own_uniform_price_norm,
            opp_uniform_price_norm,
            opp_price_new_norm,
            opp_price_old_norm,
            demand_ratio_norm,
            opponent_regime_norm
        ], dtype=np.float32)
        
        
        return obs
    
    def _get_opponent_observation(self) -> OpponentObservation:
        """Create opponent observation from current market state."""
        opponent = self.market.firms[1]
        firm = self.market.firms[0]
        opp_market_share = opponent.market_share 
        competitor_market_share = firm.market_share
        own_uniform_price = opponent.uniform_price
        own_price_new = opponent.price_new
        own_price_old = opponent.price_old
        competitor_uniform_price = firm.uniform_price
        
        return OpponentObservation(
            market_share=opp_market_share,
            competitor_market_share= competitor_market_share,
            own_uniform_price=own_uniform_price,
            own_price_new = own_price_new,
            own_price_old = own_price_old,
            competitor_uniform_price=competitor_uniform_price,
            last_demand_ratio=opponent.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.5,
            own_regime=opponent.pricing_regime,  # Always uniform
            competitor_regime=0,  # Always uniform
            timestep=self._timestep,
            episode_length=self.episode_length,
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation, info
        """
        if seed is not None:
            self.seed_value = seed
        
        # Reset market
        self.market.reset(seed=self.seed_value)
        
        # Force agent to uniform regime
        self.market.set_regimes(0, self.opponent_policy.regime)
        
        # Reset opponent policy
        self.opponent_policy.reset(seed=seed)
        
        # Reset state tracking
        self._timestep = 0
        self._last_action = 0.5
        self._episode_profits = []
        self._episode_prices = []
        self._episode_market_shares = []
        
        # Set initial prices
        initial_price = 5
        self.market.firms[0].set_prices(uniform_price=initial_price)
        
        obs = self._get_observation()
        info = {
            "timestep": 0,
            "market_share": self.market.firms[0].market_share,
            "price": initial_price,
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Normalized price action [0, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to price
        agent_price = self._action_to_price(action)
        self._last_action = float(action[0])
        
        # Get opponent's price from policy
        opp_obs = self._get_opponent_observation()
        opp_prices = self.opponent_policy.get_prices(opp_obs)

        
        # Set prices in market
        agent_prices = {
            "uniform_price": agent_price,
            "price_new": agent_price,  # Not used but required
            "price_old": agent_price,  # Not used but required
        }
        opp_prices_dict = {
            "uniform_price": opp_prices.get("uniform_price", opp_prices.get("price_new", 5.0)),
            "price_new": opp_prices.get("price_new", opp_prices.get("uniform_price", 5.0)),
            "price_old": opp_prices.get("price_old", opp_prices.get("price_new", 5.0)),
        }
        
        # Ensure uniform regime for agent only
        self.market.set_regimes(0, self.opponent_policy.regime)
        
        # Execute market step
        demand_0, demand_1 = self.market.step(agent_prices, opp_prices_dict)
        
        # Increment timestep
        self._timestep += 1
        
        # Get reward (profit)
        reward = float(self.market.firms[0].last_period_profit)
        
        # Track metrics
        self._episode_profits.append(reward)
        self._episode_prices.append(agent_price)
        self._episode_market_shares.append(self.market.firms[0].market_share)
        
        # Check termination
        terminated = False  # No early termination
        truncated = self._timestep >= self.episode_length
        
        # Build observation
        obs = self._get_observation()
        
        # Build info dict
        info = {
            "timestep": self._timestep,
            "profit": reward,
            "price": agent_price,
            "market_share": self.market.firms[0].market_share,
            "demand": demand_0,
            "opponent_price_uniform": opp_prices_dict['uniform_price'],
            "opponent_price_new": opp_prices_dict["price_new"],
            "opponent_price_old": opp_prices_dict["price_old"],
            "opponent_profit": float(self.market.firms[1].last_period_profit),
            "opponent_demand": demand_1,
            "opponent_regime" : self.market.firms[1].pricing_regime,
            "regime" : self.market.firms[0].pricing_regime
        }
        
        # Add episode summary on termination
        if truncated or terminated:
            info["episode_summary"] = {
                "total_profit": sum(self._episode_profits),
                "mean_profit": np.mean(self._episode_profits),
                "mean_price": np.mean(self._episode_prices),
                "mean_market_share": np.mean(self._episode_market_shares),
                "final_market_share": self._episode_market_shares[-1] if self._episode_market_shares else 0.0,
            }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Close environment."""
        pass


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def make_uniform_pricing_env(
    opponent: Union[str, OpponentPolicy] = "passive_uniform",
    **env_kwargs
) -> UniformPricingEnv:
    """
    Factory function to create uniform pricing environment.
    
    Args:
        opponent: Opponent preset name or OpponentPolicy instance
        **env_kwargs: Additional environment arguments
        
    Returns:
        UniformPricingEnv instance
    """
    if isinstance(opponent, str):
        opponent_policy = create_preset_opponent(opponent)
        
    else:
        opponent_policy = opponent

    
    print(f'Training is initialized with opponent police: {opponent_policy}')
    return UniformPricingEnv(
        opponent_policy=opponent_policy,
        **env_kwargs
    )



