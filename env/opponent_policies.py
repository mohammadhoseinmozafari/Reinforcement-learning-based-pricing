"""
Opponent Pricing Policies for Phase 2 Single-Agent Training.

This module provides stationary, deterministic opponent policies for training
a single learning firm against fixed opponent behavior.

Key Design Principles:
- Opponent does NOT learn or adapt using RL
- Opponent provides stable, reproducible pricing behavior
- Policies are easily swappable for experimentation
- Clean interface for environment integration

Classes:
    OpponentPolicy: Abstract base class
    ConstantOpponentPolicy: Fixed prices regardless of state
    Phase1EmpiricalOpponentPolicy: Uses Phase 1 experiment results
    RuleBasedOpponentPolicy: Simple rule-based price adjustments
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from config.constants import (
    PRICE_UNIFORM_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_BBP_OLD_MAX,
)


# =============================================================================
# DATA CLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class PriceBounds:
    """Price bounds for enforcing constraints."""
    uniform_min: float = PRICE_UNIFORM_MIN
    uniform_max: float = PRICE_UNIFORM_MAX
    bbp_new_min: float = PRICE_BBP_NEW_MIN
    bbp_new_max: float = PRICE_BBP_NEW_MAX
    bbp_old_min: float = PRICE_BBP_OLD_MIN
    bbp_old_max: float = PRICE_BBP_OLD_MAX
    
    def clip_uniform(self, price: float) -> float:
        """Clip uniform price to valid bounds."""
        return np.clip(price, self.uniform_min, self.uniform_max)
    
    def clip_bbp_new(self, price: float) -> float:
        """Clip BBP new customer price to valid bounds."""
        return np.clip(price, self.bbp_new_min, self.bbp_new_max)
    
    def clip_bbp_old(self, price: float) -> float:
        """Clip BBP old customer price to valid bounds."""
        return np.clip(price, self.bbp_old_min, self.bbp_old_max)


@dataclass
class OpponentObservation:
    """
    Structured observation for opponent policy decision-making.
    
    This provides a clean interface for policies to access market state
    without depending on the raw observation format.
    """
    # Market state
    market_share: float = 0.5
    competitor_market_share: float = 0.5
    
    # Price information
    own_uniform_price: float = 2.0
    own_price_new: float = 1.5
    own_price_old: float = 2.5
    competitor_uniform_price: float = 2.0
    competitor_price_new: float = 1.5
    competitor_price_old: float = 2.5
    
    # Demand information
    last_demand_ratio: float = 0.5  # demand / num_consumers
    new_old_ratio: float = 0.5
    
    # Regime information
    own_regime: int = 0  # 0 = Uniform, 1 = BBP
    competitor_regime: int = 0
    
    # Time information
    timestep: int = 0
    episode_length: int = 200
    
    @property
    def time_progress(self) -> float:
        """Episode progress [0, 1]."""
        return self.timestep / self.episode_length if self.episode_length > 0 else 0.0
    
    @classmethod
    def from_raw_observation(
        cls,
        raw_obs: Dict[str, Any],
        timestep: int = 0,
        episode_length: int = 200
    ) -> 'OpponentObservation':
        """
        Create OpponentObservation from raw environment observation.
        
        Args:
            raw_obs: Raw observation dict from environment
            timestep: Current timestep
            episode_length: Total episode length
            
        Returns:
            Structured OpponentObservation
        """
        pc_obs = raw_obs.get("pricing_controller", {})
        
        return cls(
            market_share=float(pc_obs.get("market_share", [0.5])[0]),
            competitor_market_share=1.0 - float(pc_obs.get("market_share", [0.5])[0]),
            own_uniform_price=float(pc_obs.get("own_prices", [2.0, 1.5, 2.5])[0]),
            own_price_new=float(pc_obs.get("own_prices", [2.0, 1.5, 2.5])[1]),
            own_price_old=float(pc_obs.get("own_prices", [2.0, 1.5, 2.5])[2]),
            competitor_uniform_price=float(pc_obs.get("comp_prices", [2.0, 1.5, 2.5])[0]),
            competitor_price_new=float(pc_obs.get("comp_prices", [2.0, 1.5, 2.5])[1]),
            competitor_price_old=float(pc_obs.get("comp_prices", [2.0, 1.5, 2.5])[2]),
            last_demand_ratio=float(pc_obs.get("last_demand", [0.5])[0]),
            new_old_ratio=float(pc_obs.get("new_old_ratio", [0.5])[0]),
            own_regime=int(pc_obs.get("regime", [0])[0]),
            competitor_regime=int(pc_obs.get("competitor_regime", [0])[0]),
            timestep=timestep,
            episode_length=episode_length,
        )


# =============================================================================
# BASE CLASS
# =============================================================================

class OpponentPolicy(ABC):
    """
    Abstract base class for opponent pricing policies.
    
    Opponent policies must be:
    - Deterministic (given same state, return same price)
    - Stationary (do not learn or adapt over time)
    - Bounded (respect price limits)
    
    Subclasses must implement:
        get_uniform_price(observation) -> float
        get_bbp_prices(observation) -> Tuple[float, float]
    """
    
    def __init__(
        self,
        regime: int = 0,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize opponent policy.
        
        Args:
            regime: Pricing regime (0 = Uniform, 1 = BBP)
            bounds: Price bounds (uses defaults if None)
            seed: Random seed for reproducibility (if policy uses randomness)
        """
        self.regime = regime
        self.bounds = bounds or PriceBounds()
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    
    @abstractmethod
    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """
        Get uniform price for current state.
        
        Args:
            observation: Current market observation
            
        Returns:
            Uniform price (scalar)
        """
        pass
    
    @abstractmethod
    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """
        Get BBP prices for current state.
        
        Args:
            observation: Current market observation
            
        Returns:
            Tuple of (price_new, price_old) for BBP regime
        """
        pass
    
    def get_strategy(self, observation: OpponentObservation) -> int:
        """
        Get pricing regime/strategy.
        
        Default implementation returns fixed regime.
        Override for dynamic regime selection.
        
        Args:
            observation: Current market observation
            
        Returns:
            Regime (0 = Uniform, 1 = BBP)
        """
        return self.regime
    
    def get_prices(self, observation: OpponentObservation) -> Dict[str, float]:
        """
        Get all prices based on current regime.
        
        This is the main interface for environment integration.
        
        Args:
            observation: Current market observation
            
        Returns:
            Dict with uniform_price, price_new, price_old
        """
        uniform_price = self.get_uniform_price(observation)
        price_new, price_old = self.get_bbp_prices(observation)
        
        # Enforce bounds
        uniform_price = self.bounds.clip_uniform(uniform_price)
        price_new = self.bounds.clip_bbp_new(price_new)
        price_old = self.bounds.clip_bbp_old(price_old)
        
        # Enforce BBP constraint: price_old >= price_new
        price_old = max(price_old, price_new)
        
        return {
            "uniform_price": float(uniform_price),
            "price_new": float(price_new),
            "price_old": float(price_old),
        }
    
    def get_action(self, observation: OpponentObservation) -> Dict[str, Any]:
        """
        Get full action (strategy + pricing) for environment.
        
        Args:
            observation: Current market observation
            
        Returns:
            Action dict compatible with environment step()
        """
        strategy = self.get_strategy(observation)
        prices = self.get_prices(observation)
        
        # Convert prices to normalized action space [0, 1]
        uniform_action = (prices["uniform_price"] - PRICE_UNIFORM_MIN) / (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        new_action = (prices["price_new"] - PRICE_BBP_NEW_MIN) / (PRICE_BBP_NEW_MAX - PRICE_BBP_NEW_MIN)
        old_action = (prices["price_old"] - PRICE_BBP_OLD_MIN) / (PRICE_BBP_OLD_MAX - PRICE_BBP_OLD_MIN)
        
        return {
            "strategy": strategy,
            "pricing": np.array([uniform_action, new_action, old_action], dtype=np.float32),
        }
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset policy state (for episode boundaries).
        
        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regime={self.regime})"


# =============================================================================
# CONSTANT OPPONENT POLICY
# =============================================================================

class ConstantOpponentPolicy(OpponentPolicy):
    """
    Opponent that always returns fixed prices.
    
    Simple baseline opponent that ignores market state entirely.
    Useful for:
    - Testing basic learning agent behavior
    - Establishing performance baselines
    - Debugging environment mechanics
    """
    
    def __init__(
        self,
        uniform_price: float = 2.5,
        price_new: float = 2.0,
        price_old: float = 3.0,
        regime: int = 0,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize constant opponent.
        
        Args:
            uniform_price: Fixed uniform price
            price_new: Fixed BBP price for new customers
            price_old: Fixed BBP price for established customers
            regime: Pricing regime (0 = Uniform, 1 = BBP)
            bounds: Price bounds
            seed: Random seed (unused but kept for interface consistency)
        """
        super().__init__(regime=regime, bounds=bounds, seed=seed)
        
        # Store fixed prices (clipped to bounds)
        self._uniform_price = self.bounds.clip_uniform(uniform_price)
        self._price_new = self.bounds.clip_bbp_new(price_new)
        self._price_old = self.bounds.clip_bbp_old(max(price_old, price_new))
    
    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Return fixed uniform price."""
        return self._uniform_price
    
    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Return fixed BBP prices."""
        return self._price_new, self._price_old
    
    def __repr__(self) -> str:
        return (f"ConstantOpponentPolicy(uniform={self._uniform_price:.2f}, "
                f"new={self._price_new:.2f}, old={self._price_old:.2f}, regime={self.regime})")




# =============================================================================
# RULE-BASED OPPONENT POLICY
# =============================================================================

class RuleBasedOpponentPolicy(OpponentPolicy):
    """
    Opponent that adjusts prices based on simple market rules.
    
    Implements deterministic price adjustments based on:
    - Market share (if losing share, lower prices)
    - Competitor prices (price matching with offset)
    
    All adjustments are bounded and use small step sizes to maintain
    stability for the learning agent.
    
    Rules are intentionally simple and DO NOT involve any learning.
    """
    
    def __init__(
        self,
        base_uniform_price: float = 2.5,
        base_price_new: float = 2.0,
        base_price_old: float = 3.0,
        regime: int = 0,
        # Adjustment parameters
        market_share_sensitivity: float = 0.5,
        price_step: float = 0.5,
        target_market_share: float = 0.5,
        # Price matching parameters
        enable_price_matching: bool = False,
        price_matching_offset: float = -0.1,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize rule-based opponent.
        
        Args:
            base_uniform_price: Starting uniform price
            base_price_new: Starting BBP new price
            base_price_old: Starting BBP old price
            regime: Pricing regime (0 = Uniform, 1 = BBP)
            market_share_sensitivity: How much to adjust based on market share [0, 1]
            price_step: Maximum price adjustment per step
            target_market_share: Target market share (adjusts prices to achieve)
            enable_price_matching: Whether to match competitor prices
            price_matching_offset: Offset from competitor price (negative = undercut)
            bounds: Price bounds
            seed: Random seed
        """
        super().__init__(regime=regime, bounds=bounds, seed=seed)
        
        # Base prices
        self._base_uniform = self.bounds.clip_uniform(base_uniform_price)
        self._base_new = self.bounds.clip_bbp_new(base_price_new)
        self._base_old = self.bounds.clip_bbp_old(base_price_old)
        
        # Adjustment parameters
        self.market_share_sensitivity = np.clip(market_share_sensitivity, 0.0, 1.0)
        self.price_step = abs(price_step)
        print(self.price_step)
        self.target_market_share = np.clip(target_market_share, 0.0, 1.0)
        
        # Price matching
        self.enable_price_matching = enable_price_matching
        self.price_matching_offset = price_matching_offset
    
    def _compute_market_share_adjustment(self, observation: OpponentObservation) -> float:
        """
        Compute price adjustment based on market share.
        
        If below target: lower prices (negative adjustment)
        If above target: raise prices (positive adjustment)
        
        Args:
            observation: Current market observation
            
        Returns:
            Price adjustment factor
        """
        share_diff = observation.market_share - self.target_market_share
        
        # Scale adjustment by sensitivity and step size
        adjustment = share_diff * self.market_share_sensitivity * self.price_step
        
        # Bound adjustment to prevent extreme swings
        return np.clip(adjustment, -self.price_step, self.price_step)
    
    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """
        Get uniform price with rule-based adjustments.
        
        Args:
            observation: Current market observation
            
        Returns:
            Adjusted uniform price
        """
        price = self._base_uniform
        
        # Apply market share adjustment
        adjustment = self._compute_market_share_adjustment(observation)
        price += adjustment
        
        # Optional: price matching
        if self.enable_price_matching:
            competitor_price = observation.competitor_uniform_price
            matched_price = competitor_price + self.price_matching_offset
            # Blend base price with matched price
            price = 0.7 * price + 0.3 * matched_price
        
        return self.bounds.clip_uniform(price)
    
    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """
        Get BBP prices with rule-based adjustments.
        
        Args:
            observation: Current market observation
            
        Returns:
            Tuple of (price_new, price_old)
        """
        adjustment = self._compute_market_share_adjustment(observation)
        
        price_new = self._base_new + adjustment
        price_old = self._base_old + adjustment
        
        # Optional: price matching for BBP
        if self.enable_price_matching:
            matched_new = observation.competitor_price_new + self.price_matching_offset
            matched_old = observation.competitor_price_old + self.price_matching_offset
            price_new = 0.7 * price_new + 0.3 * matched_new
            price_old = 0.7 * price_old + 0.3 * matched_old
        
        price_new = self.bounds.clip_bbp_new(price_new)
        price_old = self.bounds.clip_bbp_old(price_old)
        price_old = max(price_old, price_new)
        
        return price_new, price_old

class RandomizedRuleBasedOpponentPolicy(RuleBasedOpponentPolicy):
    """
    A reactive opponent whose base prices are randomized each episode.
    
    On each reset(), a new base_uniform_price is sampled from a given range.
    Optionally, other parameters like sensitivity and price_step can also
    be randomized for even broader domain randomization.
    """
    def __init__(
        self,
        base_uniform_range=(0.5, 5.0),
        base_price_new_factor=0.8,          # new price = factor * base_uniform
        base_price_old_factor=1.2,          # old price = factor * base_uniform
        regime=0,
        market_share_sensitivity=1.0,
        price_step=1.0,
        # If you want to randomize sensitivity/step as well, provide ranges:
        sensitivity_range=None,             # e.g., (0.5, 1.5)
        step_range=None,                    # e.g., (0.5, 1.5)
        enable_price_matching=False,
        price_matching_offset=-0.1,
        bounds=None,
        seed=None,
    ):
        """
        Args:
            base_uniform_range: (min, max) for uniform base price.
            base_price_new_factor: multiplier for new price relative to base.
            base_price_old_factor: multiplier for old price relative to base.
            sensitivity_range: optional (min, max) to randomize sensitivity.
            step_range: optional (min, max) to randomize price_step.
            (Other args same as RuleBasedOpponentPolicy)
        """
        self.base_uniform_range = base_uniform_range
        self.base_price_new_factor = base_price_new_factor
        self.base_price_old_factor = base_price_old_factor
        self.sensitivity_range = sensitivity_range
        self.step_range = step_range

        # We'll start with a dummy base price, reset will override it.
        # Call parent with dummy values; reset will reinitialize internal fields.
        super().__init__(
            base_uniform_price=np.mean(base_uniform_range),
            base_price_new=np.mean(base_uniform_range)*base_price_new_factor,
            base_price_old=np.mean(base_uniform_range)*base_price_old_factor,
            regime=regime,
            market_share_sensitivity=market_share_sensitivity,
            price_step=price_step,
            enable_price_matching=enable_price_matching,
            price_matching_offset=price_matching_offset,
            bounds=bounds,
            seed=seed,
        )

    def reset(self, seed=None):
        """Randomize parameters for a new episode, then call parent reset."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Randomize base prices
        base_uniform = self.rng.uniform(*self.base_uniform_range)
        self._base_uniform = self.bounds.clip_uniform(base_uniform)
        self._base_new = self.bounds.clip_bbp_new(base_uniform * self.base_price_new_factor)
        self._base_old = self.bounds.clip_bbp_old(max(
            base_uniform * self.base_price_old_factor,
            self._base_new
        ))

        # Optional: randomize sensitivity
        if self.sensitivity_range is not None:
            self.market_share_sensitivity = self.rng.uniform(*self.sensitivity_range)

        # Optional: randomize price step
        if self.step_range is not None:
            self.price_step = self.rng.uniform(*self.step_range)

        # Call parent reset to reseed the RNG if needed (but we already handled seed)
        super().reset(seed=seed)
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(regime={self.regime})"
                f"(uniform range= {self.base_uniform_range})"
                f"(market share sensitivity= {self.market_share_sensitivity})"
                f"(price step = {self.price_step})")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_opponent_policy(
    policy_type: str,
    **kwargs
) -> OpponentPolicy:
    """
    Factory function to create opponent policies.
    
    Args:
        policy_type: One of "constant", "phase1", "rule_based"
        **kwargs: Arguments passed to policy constructor
        
    Returns:
        OpponentPolicy instance
        
    Raises:
        ValueError: If policy_type is unknown
    """
    policies = {
        "constant": ConstantOpponentPolicy,
        
        "rule_based": RuleBasedOpponentPolicy,
        "randomized_rule_based" : RandomizedRuleBasedOpponentPolicy,
        "rule": RuleBasedOpponentPolicy,
    }
    
    policy_type = policy_type.lower()
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}. "
                        f"Available: {list(policies.keys())}")
    
    return policies[policy_type](**kwargs)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Common opponent configurations for experiments
OPPONENT_PRESETS = {
     "premium_uniform": {
        "policy_type": "constant",
        "uniform_price": 3.5,
        "price_new": 3.0,
        "price_old": 4.0,
        "regime": 0,
    },
    "premium_passive_uniform": {
        "policy_type": "constant",
        "uniform_price": 3.0,
        "price_new": 2.5,
        "price_old": 3.5,
        "regime": 0,
    },
    "passive_uniform": {
        "policy_type": "constant",
        "uniform_price": 2.5,
        "price_new": 2.0,
        "price_old": 3.0,
        "regime": 0,
    },

    "passive_aggressive_uniform": {
        "policy_type": "constant",
        "uniform_price": 2.0,
        "price_new": 1.5,
        "price_old": 2.5,
        "regime": 0,
    },

    "aggressive_uniform": {
        "policy_type": "constant",
        "uniform_price": 1.5,
        "price_new": 1.2,
        "price_old": 2.0,
        "regime": 0,
    },



    "premium_bbp": {
        "policy_type": "constant",
        "uniform_price": 3.5,
        "price_new": 3.0,
        "price_old": 4.0,
        "regime": 1,
    },

    "premium_passive_bbp": {
        "policy_type": "constant",
        "uniform_price": 3.0,
        "price_new": 2.5,
        "price_old": 3.5,
        "regime": 1,
    },
    "passive_bbp": {
        "policy_type": "constant",
        "uniform_price": 2.5,
        "price_new": 2.0,
        "price_old": 3.0,
        "regime": 1,
    },

    "passive_aggressive_bbp": {
        "policy_type": "constant",
        "uniform_price": 2.0,
        "price_new": 1.5,
        "price_old": 2.5,
        "regime": 1,
    },

    "aggressive_bbp": {
        "policy_type": "constant",
        "uniform_price": 1.5,
        "price_new": 1.0,
        "price_old": 2.0,
        "regime": 1,
    },
    


    
    "premium_random_reactive_uniform":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(3.5, 5.0),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":0,
        "sensitivity_range" :(0.0, 0.3),
        "step_range" :(0,0.5),
    },

    "passive_random_reactive_uniform":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(2.0, 3.5),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":0,
        "sensitivity_range" :(0.3, 0.6),
        "step_range" :(0,0.5),
    },

    
    "aggressive_random_reactive_uniform":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(0.5, 2.0),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":0,
        "sensitivity_range" :(0.6 ,1.0),
        "step_range" :(0,0.5),
    },

     "premium_random_reactive_bbp":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(3.5, 5.0),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":1,
        "sensitivity_range" :(0.0, 0.3),
        "step_range" :(0,0.5),
    },

    "passive_random_reactive_bbp":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(2.0, 3.5),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":1,
        "sensitivity_range" :(0.3, 0.6),
        "step_range" :(0,0.5),
    },

    
    "aggressive_random_reactive_bbp":{
        "policy_type" : "randomized_rule_based",
        "base_uniform_range" :(0.5, 2.0),
        "base_price_new_factor": 0.8,          # new price = factor * base_uniform
        "base_price_old_factor" :1.2,          # old price = factor * base_uniform
        "regime":1,
        "sensitivity_range" :(0.6 ,1.0),
        "step_range" :(0,0.5),
    },

    

}


def create_preset_opponent(preset_name: str, **override_kwargs) -> OpponentPolicy:
    """
    Create opponent from preset configuration.
    
    Args:
        preset_name: Name of preset from OPPONENT_PRESETS
        **override_kwargs: Arguments to override preset values
        
    Returns:
        OpponentPolicy instance
        
    Raises:
        ValueError: If preset_name is unknown
    """
    if preset_name not in OPPONENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(OPPONENT_PRESETS.keys())}")
    
    config = OPPONENT_PRESETS[preset_name].copy()
    config.update(override_kwargs)
    
    policy_type = config.pop("policy_type")
    return create_opponent_policy(policy_type, **config)
