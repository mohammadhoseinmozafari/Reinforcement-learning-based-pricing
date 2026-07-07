"""
Uniform Price Learning Environment

Simplified single-agent environment for learning optimal uniform pricing.
- Agent action: scalar price p_uniform (continuous)  
- Regime switching is disabled (always uniform pricing)
- Opponent can choose BBP of Uniform pricing strategies

"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

from log.internal_logger import setup_internal_logger
from .type import EnvironmentType
import gymnasium as gym
from gymnasium import spaces


from env.models import HotellingMarket
from env.opponent_policies import (
    OpponentPolicy,
    OpponentObservation,
    PreviousMarketState,
    PriceVector,
    
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


class PricingEnv(gym.Env):
    """
    Single-agent environment for pricing only.
    
    Simplifications from full hierarchical environment:
    - Observation is a flat vector of market features
    - No strategy controller.
    
    This focused environment is ideal for 'Pricing experiments'
    where we want to learn optimal pricing strategy against
    an opponent.

    """
    
    metadata = {
        "render_modes": [],
        "name": "pricing_v1"
    }
    
    def __init__(
        self,
        environment_type : EnvironmentType,
        opponent_policy: OpponentPolicy,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
        seed: Optional[int] = RANDOM_SEED,
    ):
        """
        Initialize pricing environment.
        
        Args:
            environment_type : type of the env (uniform pricing or bbp) 
            opponent_policy: Policy for opponent (uses ConstantOpponentPolicy if None)
            num_consumers: Number of consumers in market
            episode_length: Total steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.seed_value = seed
        self.environment_type = environment_type
        
        self.market = HotellingMarket(num_consumers=num_consumers, seed=seed)
        


        self.opponent_policy = opponent_policy

        self.internal_loggr = setup_internal_logger(
            name = f"environment.{self.environment_type.value}_{self.opponent_policy.__class__.__name__}",
            log_dir="log/logs/environment_logs/",
            filename="environment_internal.log"

        )        
        # =====================================
        # ACTION SPACE: Flattened price (uniform_price , price_new , price_old)
        # =====================================
        # Action is normalized to [-1, 1], scaled to price range internally
        action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

        
        # =====================================
        # OBSERVATION SPACE
        # =====================================
        # Flat observation vector with key market features
        # Features:
        #   - own market share 
        #   - opponent market share

        #   - own_uniform_price
        #   - own_price_new
        #   - own_price_old
        #
        #   - opponent_price_uniform
        #   - opponent_price_new
        #   - opponent_price_old

        #   - demand_ratio

        #   - own regime
        #   - opponent regime
        #   - profit trend
        #   - popularity change

        obs_dim = 13
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self._timestep = 0
        self._last_action = 0.5  
        
        self._episode_profits = []
        self._episode_prices = []
        self._episode_market_shares = []
    
    def _action_to_price(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to actual price.
            Input tensor : (uniform_price, bbp_price_new, bbp_price_old)
        """

        normalized = (action + 1.0) / 2.0 
        normalized = np.clip(normalized, 0.0, 1.0)        
        uniform_price = PRICE_UNIFORM_MIN + normalized[0] * (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        bbp_price_new = PRICE_BBP_NEW_MIN + normalized[1] * (PRICE_BBP_NEW_MAX - PRICE_BBP_NEW_MIN)
        bbp_price_old = PRICE_BBP_OLD_MIN + normalized[2] * (PRICE_BBP_OLD_MAX - PRICE_BBP_OLD_MIN)
        return np.array([uniform_price, bbp_price_new, bbp_price_old])
    
    def _prices_to_normalized (self, prices : np.ndarray)-> np.ndarray:
        uniform_price = float(prices[0])
        bbp_price_new = float(prices[1])
        bbp_price_old = float(prices[2])

        uniform_normalized = self._uniform_price_to_normalized(uniform_price)
        bbp_new_normalized , bbp_old_normalized = self._bbp_price_to_normalized(bbp_price_new, bbp_price_old)

        return np.array([uniform_normalized, bbp_new_normalized, bbp_old_normalized])



    def _uniform_price_to_normalized(self, price: float) -> float:
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
        firm = self.market.firms[0] 
        firm_prices = np.array([firm.uniform_price, firm.price_new, firm.price_old]) # Learning agent is firm 0
        opponent = self.market.firms[1]
        opponent_prices = np.array([opponent.uniform_price, opponent.price_new, opponent.price_old])
        
        # Market share
        own_market_share_norm = self._normalize_market_share(firm.market_share)
        opp_market_share_norm = self._normalize_market_share(opponent.market_share)

        
        
        # Prices (normalized)
        own_uniform_price_norm, own_bbp_price_new_norm, own_bbp_price_old_norm  = self._prices_to_normalized(firm_prices)
        opp_uniform_price_norm, opp_bbp_price_new_norm, opp_bbp_price_old_norm  = self._prices_to_normalized(opponent_prices)
        # Demand ratio
        demand_ratio = firm.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.0
        demand_ratio_norm = self._normalize_demand_ratio (demand_ratio)
        
        own_regime_norm = self._normalize_regime(firm.pricing_regime)
        opponent_regime_norm = self._normalize_regime(opponent.pricing_regime)
        # Profit trend
        profit_trend = firm.get_profit_trend()
        
        # Popularity change
        pop_change = firm.get_popularity_change()
        
        obs = np.array([
            own_market_share_norm,
            opp_market_share_norm,

            own_uniform_price_norm,
            own_bbp_price_new_norm,
            own_bbp_price_old_norm,

            opp_uniform_price_norm,
            opp_bbp_price_new_norm,
            opp_bbp_price_old_norm,

            demand_ratio_norm,

            own_regime_norm,
            opponent_regime_norm,
            profit_trend,
            pop_change

        ], dtype=np.float32)
        
        
        return obs
    
    def _get_opponent_observation(
        self,
        competitor_prices: Optional[Dict[str, float]] = None,
    ) -> OpponentObservation:
        """Create the opponent view using current submitted agent prices.

        ``competitor_prices`` is supplied during ``step`` because the market's
        stored firm prices still describe the previous period until
        :meth:`HotellingMarket.step` executes.
        """
        opponent = self.market.firms[1]
        firm = self.market.firms[0]
        
        opp_market_share = opponent.market_share 
        competitor_market_share = firm.market_share
     
        own_uniform_price = opponent.uniform_price
        own_price_new = opponent.price_new
        own_price_old = opponent.price_old
     

        competitor_uniform_price = firm.uniform_price
        competitor_bbp_price_new = firm.price_new
        competitor_bbp_price_old = firm.price_old
        
        own_regime = opponent.pricing_regime
        competitor_regime = firm.pricing_regime
        previous = PreviousMarketState(
            own_market_share=opp_market_share,
            competitor_market_share=competitor_market_share,
            own_prices=PriceVector(
                uniform=own_uniform_price,
                new=own_price_new,
                old=own_price_old,
            ),
            competitor_prices=PriceVector(
                uniform=competitor_uniform_price,
                new=competitor_bbp_price_new,
                old=competitor_bbp_price_old,
            ),
            own_demand_ratio=(
                opponent.last_period_quantity / self.num_consumers
                if self.num_consumers > 0 else 0.5
            ),
            own_new_customer_ratio=opponent.get_new_old_ratio(),
        )
        submission = (
            PriceVector(
                uniform=float(competitor_prices["uniform_price"]),
                new=float(competitor_prices["price_new"]),
                old=float(competitor_prices["price_old"]),
            )
            if competitor_prices is not None else None
        )
        return OpponentObservation(
            previous=previous,
            competitor_submission=submission,
            competitor_established_share=self.market.get_established_share(0),
            own_regime=own_regime,
            competitor_regime=competitor_regime,
            decision_period=self._timestep,
            state_period=self._timestep - 1,
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
        
        own_regime = 0 if self.environment_type == "uniform_pricing" else 1

        self.market.set_regimes(own_regime, self.opponent_policy.regime)
        
        # Reset opponent policy
        self.opponent_policy.reset(seed=seed)
        
        # Reset state tracking
        self._timestep = 0
        self._last_action = 0.5
        self._episode_profits = []
        self._episode_prices = []
        self._episode_market_shares = []
        
        # Set initial prices
        initial_uniform_price = 5
        initial_bbp_price_new = 4.5
        initial_bbp_price_old = 5.5
        
        self.market.firms[0].set_prices(
            uniform_price=initial_uniform_price, 
            price_new= initial_bbp_price_new, 
            price_old=initial_bbp_price_old
            )
        
        obs = self._get_observation()
        info = {
            "timestep": 0,
            "market_share": self.market.firms[0].market_share,
            "uniform_price": initial_uniform_price,
            "bbp_price_new": initial_bbp_price_new,
            "bbp_price_old" : initial_bbp_price_old
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Normalized price action [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to price
        agent_price = self._action_to_price(action)
        self._last_action = action

        agent_prices = {
            "uniform_price": agent_price[0],
            "price_new": agent_price[1],
            "price_old": agent_price[2],
        }

        own_regime = 0 if self.environment_type == "uniform_pricing" else 1
        self.market.set_regimes(own_regime, self.opponent_policy.regime)

        # The myopic opponent is a within-period follower and therefore sees
        # the agent's current submitted prices rather than last period's prices.
        opp_obs = self._get_opponent_observation(agent_prices)
        opp_prices = self.opponent_policy.get_prices(opp_obs)

        opp_prices_dict = {
            "uniform_price": opp_prices.get("uniform_price", opp_prices.get("price_new", 5.0)),
            "price_new": opp_prices.get("price_new", opp_prices.get("uniform_price", 5.0)),
            "price_old": opp_prices.get("price_old", opp_prices.get("price_new", 5.0)),
        }

        # Execute market step
        demand_0, demand_1 = self.market.step(agent_prices, opp_prices_dict)
        
        # Increment timestep
        self._timestep += 1
        
        # Get reward (profit)
        own_profit = float(self.market.firms[0].last_period_profit)
        opponent_profit = float(self.market.firms[1].last_period_profit)
        reward = own_profit + 0.1 * (own_profit - opponent_profit)

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
            
            "uniform_price": float(agent_price[0]),
            "bbp_price_new" : float(agent_price[1]),
            "bbp_price_old": float(agent_price[2]),

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

def make_pricing_env(
    environment_type : EnvironmentType ,
    opponent: Union[str, OpponentPolicy] = "passive_uniform",
    **env_kwargs
) -> PricingEnv:
    """
    Factory function to create pricing environment.
    
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
    return PricingEnv(
        environment_type,
        opponent_policy=opponent_policy,
        **env_kwargs
    )



