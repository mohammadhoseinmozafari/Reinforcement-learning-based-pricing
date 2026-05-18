"""
Phase 2.1 — Uniform Price Learning Environment

Simplified single-agent environment for learning optimal uniform pricing.
- Agent action: scalar price p_uniform (continuous)
- Opponent uses fixed opponent policy  
- Regime switching is disabled (always uniform pricing)
- No BBP logic involved

This is a streamlined wrapper specifically for Phase 2.1 experiments.
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
    PRICE_UNIFORM_MIN,
    PRICE_UNIFORM_MAX,
    RANDOM_SEED,
)


class UniformPricingEnv(gym.Env):
    """
    Single-agent environment for uniform pricing only.
    
    Simplifications from full hierarchical environment:
    - Both firms always use uniform pricing (regime = 0)
    - Agent action is a single scalar: uniform price
    - Observation is a flat vector of market features
    - No strategy controller, no BBP
    
    This focused environment is ideal for Phase 2.1 experiments
    where we want to learn optimal uniform pricing against
    a stationary opponent.
    """
    
    metadata = {
        "render_modes": [],
        "name": "uniform_pricing_v0"
    }
    
    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
        seed: Optional[int] = RANDOM_SEED,
        normalize_obs: bool = True,
    ):
        """
        Initialize uniform pricing environment.
        
        Args:
            opponent_policy: Policy for opponent (uses ConstantOpponentPolicy if None)
            num_consumers: Number of consumers in market
            episode_length: Total steps per episode
            seed: Random seed for reproducibility
            normalize_obs: If True, normalize observations to roughly [-1, 1]
        """
        super().__init__()
        
        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.seed_value = seed
        self.normalize_obs = normalize_obs
        
        # Create market simulator
        self.market = HotellingMarket(num_consumers=num_consumers, seed=seed)
        
        # Set up opponent policy (default: constant uniform pricing)
        if opponent_policy is None:
            self.opponent_policy = ConstantOpponentPolicy(
                uniform_price=3.5,
                regime=0  # Always uniform
            )
        else:
            self.opponent_policy = opponent_policy
            # Force opponent to uniform regime
            self.opponent_policy.regime = 0
        
        # =====================================
        # ACTION SPACE: Single uniform price
        # =====================================
        # Action is normalized to [0, 1], scaled to price range internally
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # =====================================
        # OBSERVATION SPACE
        # =====================================
        # Flat observation vector with key market features
        # Features:
        #   - own_market_share (1)
        #   - own_price (1)  
        #   - opponent_price (1)
        #   - last_demand_ratio (1)
        #   - time_progress (1)
        #   - profit_trend (1)
        #   - popularity_change (1)
        obs_dim = 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # =====================================
        # STATE TRACKING
        # =====================================
        self._timestep = 0
        self._last_action = 0.5  # Normalized action
        
        # Metrics tracking for logging
        self._episode_profits = []
        self._episode_prices = []
        self._episode_market_shares = []
    
    def _action_to_price(self, action: np.ndarray) -> float:
        """Convert normalized action [0, 1] to actual price."""
        normalized = float(np.clip(action[0], 0.0, 1.0))
        price = PRICE_UNIFORM_MIN + normalized * (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        return price
    
    def _price_to_normalized(self, price: float) -> float:
        """Convert actual price to normalized [0, 1] value."""
        return (price - PRICE_UNIFORM_MIN) / (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from current market state.
        
        Returns:
            Flat observation array
        """
        firm = self.market.firms[0]  # Learning agent is firm 0
        opponent = self.market.firms[1]
        
        # Market share
        market_share = firm.market_share
        
        # Prices (normalized)
        own_price_norm = self._price_to_normalized(firm.uniform_price)
        opp_price_norm = self._price_to_normalized(opponent.uniform_price)
        
        # Demand ratio
        demand_ratio = firm.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.0
        
        # Time progress
        time_progress = self._timestep / self.episode_length if self.episode_length > 0 else 0.0
        
        # Profit trend
        profit_trend = firm.get_profit_trend()
        
        # Popularity change
        pop_change = firm.get_popularity_change()
        
        obs = np.array([
            market_share,
            own_price_norm,
            opp_price_norm,
            demand_ratio,
            time_progress,
            profit_trend,
            pop_change,
        ], dtype=np.float32)
        
        if self.normalize_obs:
            # Rough normalization to [-1, 1] range
            # market_share, demand_ratio already in [0, 1]
            # prices normalized to [0, 1]
            # time_progress in [0, 1]
            # profit_trend in [-1, 1]
            # pop_change roughly in [-1, 1]
            obs = obs * 2.0 - 1.0  # Map [0, 1] to [-1, 1]
        
        return obs
    
    def _get_opponent_observation(self) -> OpponentObservation:
        """Create opponent observation from current market state."""
        opponent = self.market.firms[1]
        firm = self.market.firms[0]
        
        return OpponentObservation(
            market_share=opponent.market_share,
            competitor_market_share=firm.market_share,
            own_uniform_price=opponent.uniform_price,
            competitor_uniform_price=firm.uniform_price,
            last_demand_ratio=opponent.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.5,
            own_regime=0,  # Always uniform
            competitor_regime=0,  # Always uniform
            timestep=self._timestep,
            episode_length=self.episode_length,
        )
    
    def reset(
        self,
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
        
        # Force both firms to uniform regime
        self.market.set_regimes(0, 0)
        
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
        opp_price = opp_prices["uniform_price"]
        
        # Set prices in market
        agent_prices = {
            "uniform_price": agent_price,
            "price_new": agent_price,  # Not used but required
            "price_old": agent_price,  # Not used but required
        }
        opp_prices_dict = {
            "uniform_price": opp_price,
            "price_new": opp_price,
            "price_old": opp_price,
        }
        
        # Ensure uniform regime
        self.market.set_regimes(0, 0)
        
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
            "opponent_price": opp_price,
            "opponent_profit": float(self.market.firms[1].last_period_profit),
            "opponent_demand": demand_1,
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
        opponent_policy.regime = 0  # Force uniform
    else:
        opponent_policy = opponent
        opponent_policy.regime = 0
    
    return UniformPricingEnv(
        opponent_policy=opponent_policy,
        **env_kwargs
    )



