"""
Opponent Pricing Policies for Phase 2 Single-Agent Training.

This module provides stationary, deterministic opponent policies for training
a single learning firm against fixed opponent behavior.

Key Design Principles:
- Opponent does NOT learn or adapt using RL
- Opponent provides episode-stable, reproducible pricing behavior
- Policies are easily swappable for experimentation
- Clean interface for environment integration

Classes:
    OpponentPolicy: Abstract base class
    FixedUniformOpponentPolicy: One randomized price held for an episode
    UniformRandomOpponentPolicy: Episode base price with per-step Gaussian noise
    UniformMyopicOpponent: Analytical one-period uniform best response
    UniformUndercutterOpponent: Delayed response below the agent's prior price
    UniformTitForTatOpponent: Cooperative pricing with finite retaliation
    BBPFixedDiscriminatorOpponent: Episode-fixed randomized BBP spread
    BBPAcquisitionPredatorOpponent: Aggressive new-customer acquisition
    BBPLoyaltyHarvesterOpponent: High established-customer pricing
    BBPMyopicSegmentOptimizerOpponent: Segmented one-period grid optimization
    ConstantOpponentPolicy: Fixed prices regardless of state
    Phase1EmpiricalOpponentPolicy: Uses Phase 1 experiment results
    RuleBasedOpponentPolicy: Simple rule-based price adjustments
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Sequence
from dataclasses import dataclass, field
import numpy as np

from config.constants import (
    PRICE_UNIFORM_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_BBP_OLD_MAX,
    MARGINAL_COST,
    TRANSPORTATION_COST,
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


def _bounded_bbp_prices(
    bounds: PriceBounds,
    price_new: float,
    price_old: float,
    min_spread: float = 1e-6,
) -> Tuple[float, float]:
    """Clip a BBP pair while preserving its required old-customer premium."""
    bounded_new = float(bounds.clip_bbp_new(price_new))
    bounded_old = float(bounds.clip_bbp_old(price_old))
    bounded_old = max(bounded_old, bounded_new + min_spread)
    bounded_old = float(bounds.clip_bbp_old(bounded_old))
    if bounded_old < bounded_new + min_spread:
        bounded_new = float(bounds.clip_bbp_new(bounded_old - min_spread))
    if bounded_old < bounded_new + min_spread:
        raise ValueError("BBP bounds cannot satisfy the required price spread")
    return bounded_new, bounded_old


@dataclass(frozen=True)
class PriceVector:
    """Uniform and segmented prices submitted by one firm."""

    uniform: float = 2.0
    new: float = 1.5
    old: float = 2.5


@dataclass(frozen=True)
class PreviousMarketState:
    """Market information produced by the most recently cleared period."""

    own_market_share: float = 0.5
    competitor_market_share: float = 0.5
    own_prices: PriceVector = field(default_factory=PriceVector)
    competitor_prices: PriceVector = field(default_factory=PriceVector)
    own_demand_ratio: float = 0.5
    own_new_customer_ratio: float = 0.5


@dataclass(frozen=True)
class OpponentObservation:
    """Information available when the opponent chooses its period-t prices.

    ``previous`` contains outcomes and posted prices from the completed period
    ``t-1``. ``competitor_submission`` contains the learning agent's submitted
    period-``t`` prices when the opponent is allowed to observe them. Current
    demand, profit, and market share are intentionally absent because they do
    not exist until the market clears.
    """

    previous: PreviousMarketState = field(default_factory=PreviousMarketState)
    competitor_submission: Optional[PriceVector] = None
    competitor_established_share: float = 0.0
    own_regime: int = 0  # 0 = Uniform, 1 = BBP
    competitor_regime: int = 0
    decision_period: int = 0
    state_period: int = -1
    episode_length: int = 200
    
    @property
    def time_progress(self) -> float:
        """Episode progress [0, 1]."""
        return self.decision_period / self.episode_length if self.episode_length > 0 else 0.0
    
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
        
        own_prices = pc_obs.get("own_prices", [2.0, 1.5, 2.5])
        competitor_prices = pc_obs.get("comp_prices", [2.0, 1.5, 2.5])
        own_market_share = float(pc_obs.get("market_share", [0.5])[0])
        return cls(
            previous=PreviousMarketState(
                own_market_share=own_market_share,
                competitor_market_share=1.0 - own_market_share,
                own_prices=PriceVector(*map(float, own_prices)),
                competitor_prices=PriceVector(*map(float, competitor_prices)),
                own_demand_ratio=float(pc_obs.get("last_demand", [0.5])[0]),
                own_new_customer_ratio=float(
                    pc_obs.get("new_old_ratio", [0.5])[0]
                ),
            ),
            competitor_established_share=float(
                pc_obs.get("competitor_established_share", [0.0])[0]
            ),
            own_regime=int(pc_obs.get("regime", [0])[0]),
            competitor_regime=int(pc_obs.get("competitor_regime", [0])[0]),
            decision_period=timestep,
            state_period=timestep - 1,
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


class FixedUniformOpponentPolicy(OpponentPolicy):
    """Sample one uniform price per episode and hold it fixed.

    At every reset, the opponent samples
    ``fixed_price ~ Uniform(p_min + margin, p_max - margin)``. Every action in
    that episode uses the sampled price. This exposes an agent to varied but
    stationary competitors, making it suitable for introductory curriculum
    stages focused on learning demand and profit responses.
    """

    def __init__(
        self,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        margin: float = 0.25,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the episode-randomized uniform opponent.

        Args:
            p_min: Lower source bound before applying ``margin``.
            p_max: Upper source bound before applying ``margin``.
            margin: Distance kept from both ends of the source interval.
            bounds: Market price bounds. Defaults to :class:`PriceBounds`.
            seed: Optional seed controlling episode price samples.

        Raises:
            ValueError: If bounds or margin do not define a valid interval.
        """
        super().__init__(regime=0, bounds=bounds, seed=seed)
        self.p_min = self.bounds.uniform_min if p_min is None else float(p_min)
        self.p_max = self.bounds.uniform_max if p_max is None else float(p_max)
        self.margin = float(margin)
        self._validate_sampling_interval()
        self._fixed_price = self._sample_fixed_price()

    def _sample_fixed_price(self) -> float:
        lower = self.p_min + self.margin
        upper = self.p_max - self.margin
        return float(self.rng.uniform(lower, upper))

    @property
    def fixed_price(self) -> float:
        """Price posted throughout the current episode."""
        return self._fixed_price

    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new episode and sample exactly one new fixed price."""
        super().reset(seed=seed)
        self._fixed_price = self._sample_fixed_price()

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Return the current episode's fixed uniform price."""
        return self._fixed_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Return bounded placeholders; this policy always uses uniform regime."""
        price_new = float(self.bounds.clip_bbp_new(self._fixed_price))
        price_old = float(
            self.bounds.clip_bbp_old(max(self._fixed_price, price_new))
        )
        return price_new, price_old


    def _validate_sampling_interval(self) -> None:
        values = (self.p_min, self.p_max, self.margin)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("p_min, p_max, and margin must be finite")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if (
            self.p_min < self.bounds.uniform_min
            or self.p_max > self.bounds.uniform_max
        ):
            raise ValueError("p_min and p_max must lie within uniform price bounds")
        if self.p_min + self.margin >= self.p_max - self.margin:
            raise ValueError("margin leaves no non-empty price sampling interval")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fixed_price={self._fixed_price:.2f}, "
            f"range=({self.p_min + self.margin:.2f}, "
            f"{self.p_max - self.margin:.2f}), regime=0)"
        )


class UniformRandomOpponentPolicy(OpponentPolicy):
    """Post a noisy uniform price that changes at every market step.

    A base price is sampled once per episode from ``Uniform(p_min, p_max)``.
    Each call to :meth:`get_uniform_price` then samples independent Gaussian
    noise and returns ``clip(p_base + noise, p_min, p_max)``. This provides a
    stochastic curriculum opponent for learning robustness to price noise.
    """

    def __init__(
        self,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        sigma: float = 0.25,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the per-step randomized uniform opponent.

        Args:
            p_min: Minimum base and posted price.
            p_max: Maximum base and posted price.
            sigma: Standard deviation of the per-step Gaussian noise.
            bounds: Market price bounds. Defaults to :class:`PriceBounds`.
            seed: Optional seed controlling base prices and step noise.

        Raises:
            ValueError: If the price interval or noise scale is invalid.
        """
        super().__init__(regime=0, bounds=bounds, seed=seed)
        self.p_min = self.bounds.uniform_min if p_min is None else float(p_min)
        self.p_max = self.bounds.uniform_max if p_max is None else float(p_max)
        self.sigma = float(sigma)
        self._validate_parameters()
        self._base_price = self._sample_base_price()
        self._current_price = self._base_price


    def _sample_base_price(self) -> float:
        return float(self.rng.uniform(self.p_min, self.p_max))

    @property
    def base_price(self) -> float:
        """Base price sampled for the current episode."""
        return self._base_price

    @property
    def current_price(self) -> float:
        """Most recently posted noisy price."""
        return self._current_price

    def reset(self, seed: Optional[int] = None) -> None:
        """Sample a new episode base price and clear the previous step price."""
        super().reset(seed=seed)
        self._base_price = self._sample_base_price()
        self._current_price = self._base_price

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Sample and return this step's bounded noisy price."""
        noise = float(self.rng.normal(loc=0.0, scale=self.sigma))
        self._current_price = float(
            np.clip(self._base_price + noise, self.p_min, self.p_max)
        )
        return self._current_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Reuse the current step price without drawing additional noise."""
        price_new = float(self.bounds.clip_bbp_new(self._current_price))
        price_old = float(
            self.bounds.clip_bbp_old(max(self._current_price, price_new))
        )
        return price_new, price_old

    def _validate_parameters(self) -> None:
        values = (self.p_min, self.p_max, self.sigma)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("p_min, p_max, and sigma must be finite")
        if (
            self.p_min < self.bounds.uniform_min
            or self.p_max > self.bounds.uniform_max
        ):
            raise ValueError("p_min and p_max must lie within uniform price bounds")
        if self.p_min >= self.p_max:
            raise ValueError("p_min must be less than p_max")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base_price={self._base_price:.2f}, "
            f"range=({self.p_min:.2f}, {self.p_max:.2f}), "
            f"sigma={self.sigma:.2f}, regime=0)"
        )


class UniformMyopicOpponent(OpponentPolicy):
    """Choose an analytical one-period best-response uniform price.

    Against a uniform competitor, the response uses its current uniform price.
    Against a BBP competitor, it uses the established-share-weighted average of
    the current new- and old-customer prices. The environment supplies current
    prices, making this policy a Stackelberg follower within each period.
    """

    def __init__(
        self,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        transport_cost: float = TRANSPORTATION_COST,
        best_response_offset: float = 0.5,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the analytical short-term rational opponent.

        Args:
            p_min: Minimum permitted response price.
            p_max: Maximum permitted response price.
            transport_cost: Hotelling transportation-cost coefficient.
            best_response_offset: Constant term in the derived response.
            bounds: Market price bounds. Defaults to :class:`PriceBounds`.
            seed: Retained for the common policy interface; response is deterministic.

        Raises:
            ValueError: If bounds or economic parameters are invalid.
        """
        super().__init__(regime=0, bounds=bounds, seed=seed)
        self.p_min = self.bounds.uniform_min if p_min is None else float(p_min)
        self.p_max = self.bounds.uniform_max if p_max is None else float(p_max)
        self.transport_cost = float(transport_cost)
        self.best_response_offset = float(best_response_offset)
        self._validate_parameters()
        self._current_price = self.p_min
        self._effective_competitor_price = 0.0
        self._last_established_share = 0.0
        self._last_unclipped_price = self.p_min

    def _validate_parameters(self) -> None:
        numeric_values = (
            self.p_min,
            self.p_max,
            self.transport_cost,
            self.best_response_offset,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("myopic policy parameters must be finite")
        if (
            self.p_min < self.bounds.uniform_min
            or self.p_max > self.bounds.uniform_max
        ):
            raise ValueError("p_min and p_max must lie within uniform price bounds")
        if self.p_min >= self.p_max:
            raise ValueError("p_min must be less than p_max")
        if self.transport_cost <= 0:
            raise ValueError("transport_cost must be positive")

    @property
    def current_price(self) -> float:
        """Most recently selected best-response price."""
        return self._current_price

    @property
    def effective_competitor_price(self) -> float:
        """Competitor price used by the most recent analytical response."""
        return self._effective_competitor_price

    @property
    def last_established_share(self) -> float:
        """Established-customer share used by the most recent response."""
        return self._last_established_share

    @property
    def last_unclipped_price(self) -> float:
        """Most recent analytical response before applying price bounds."""
        return self._last_unclipped_price

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Calculate and return the bounded analytical best response."""
        competitor_regime = float(observation.competitor_regime)
        if competitor_regime not in (0.0, 1.0):
            raise ValueError("competitor_regime must be 0 (uniform) or 1 (BBP)")

        if competitor_regime == 0.0:
            established_share = 0.0
            prices = (
                observation.competitor_submission
                or observation.previous.competitor_prices
            )
            effective_price = float(prices.uniform)
        else:
            established_share = float(np.clip(
                observation.competitor_established_share, 0.0, 1.0
            ))
            prices = (
                observation.competitor_submission
                or observation.previous.competitor_prices
            )
            price_new = float(prices.new)
            price_old = float(prices.old)
            effective_price = (
                (1.0 - established_share) * price_new
                + established_share * price_old
            )
        if not np.isfinite(effective_price):
            raise ValueError("competitor prices must be finite")

        response = (
            self.transport_cost
            + effective_price
            + self.best_response_offset
        ) / 2.0
        self._effective_competitor_price = float(effective_price)
        self._last_established_share = established_share
        self._last_unclipped_price = float(response)
        self._current_price = float(np.clip(response, self.p_min, self.p_max))
        return self._current_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Reuse the selected uniform price for inactive BBP placeholders."""
        price_new = float(self.bounds.clip_bbp_new(self._current_price))
        price_old = float(
            self.bounds.clip_bbp_old(max(self._current_price, price_new))
        )
        return price_new, price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(range=({self.p_min:.2f}, "
            f"{self.p_max:.2f}), transport_cost={self.transport_cost:.2f}, "
            f"best_response_offset={self.best_response_offset:.2f}, regime=0)"
        )


class UniformUndercutterOpponent(OpponentPolicy):
    """Undercut an observed agent price after an episode-fixed delay.

    At reset, the policy samples an undercut amount from ``Uniform(0.1, 1.0)``
    and a reaction delay from ``[1, 2]``. Each step uses the selected delayed
    agent price minus that amount, clipped to the configured price interval.
    """

    def __init__(
        self,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        delta_min: float = 0.1,
        delta_max: float = 1.0,
        reaction_delays: Sequence[int] = (1, 2),
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the episode-randomized undercutting opponent.

        Args:
            p_min: Minimum posted uniform price.
            p_max: Maximum posted uniform price.
            delta_min: Minimum episode undercut amount.
            delta_max: Maximum episode undercut amount.
            reaction_delays: Candidate delays measured in observed agent prices.
            bounds: Market price bounds. Defaults to :class:`PriceBounds`.
            seed: Optional seed controlling episode parameter samples.

        Raises:
            ValueError: If price, delta, or delay parameters are invalid.
        """
        super().__init__(regime=0, bounds=bounds, seed=seed)
        self.p_min = self.bounds.uniform_min if p_min is None else float(p_min)
        self.p_max = self.bounds.uniform_max if p_max is None else float(p_max)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.reaction_delays = tuple(reaction_delays)
        self._validate_parameters()
        self._agent_price_history: list[float] = []
        self._delta = self._sample_delta()
        self._reaction_delay = self._sample_reaction_delay()
        self._current_price = self.p_max

    def _validate_parameters(self) -> None:
        numeric_values = (
            self.p_min,
            self.p_max,
            self.delta_min,
            self.delta_max,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("undercutter price and delta parameters must be finite")
        if (
            self.p_min < self.bounds.uniform_min
            or self.p_max > self.bounds.uniform_max
        ):
            raise ValueError("p_min and p_max must lie within uniform price bounds")
        if self.p_min >= self.p_max:
            raise ValueError("p_min must be less than p_max")
        if self.delta_min < 0 or self.delta_min >= self.delta_max:
            raise ValueError("delta bounds must satisfy 0 <= delta_min < delta_max")
        if not self.reaction_delays:
            raise ValueError("reaction_delays must not be empty")
        if any(
            not isinstance(delay, int)
            or isinstance(delay, bool)
            or delay < 1
            for delay in self.reaction_delays
        ):
            raise ValueError("reaction delays must be positive integers")

    def _sample_delta(self) -> float:
        return float(self.rng.uniform(self.delta_min, self.delta_max))

    def _sample_reaction_delay(self) -> int:
        return int(self.rng.choice(self.reaction_delays))

    @property
    def delta(self) -> float:
        """Undercut amount sampled for the current episode."""
        return self._delta

    @property
    def reaction_delay(self) -> int:
        """Agent-price observation delay sampled for the current episode."""
        return self._reaction_delay

    @property
    def current_price(self) -> float:
        """Most recently posted undercut price."""
        return self._current_price

    def reset(self, seed: Optional[int] = None) -> None:
        """Sample episode response parameters and clear agent-price history."""
        super().reset(seed=seed)
        self._delta = self._sample_delta()
        self._reaction_delay = self._sample_reaction_delay()
        self._agent_price_history = []
        self._current_price = self.p_max

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Undercut the delayed observed agent price and apply price bounds."""
        observed_agent_price = float(
            observation.previous.competitor_prices.uniform
        )
        self._agent_price_history.append(observed_agent_price)
        history_index = max(
            len(self._agent_price_history) - self._reaction_delay,
            0,
        )
        delayed_agent_price = self._agent_price_history[history_index]
        self._current_price = float(
            np.clip(delayed_agent_price - self._delta, self.p_min, self.p_max)
        )
        return self._current_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Reuse the current undercut price for inactive BBP placeholders."""
        price_new = float(self.bounds.clip_bbp_new(self._current_price))
        price_old = float(
            self.bounds.clip_bbp_old(max(self._current_price, price_new))
        )
        return price_new, price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(delta={self._delta:.2f}, "
            f"reaction_delay={self._reaction_delay}, "
            f"range=({self.p_min:.2f}, {self.p_max:.2f}), regime=0)"
        )


class UniformTitForTatOpponent(OpponentPolicy):
    """Cooperate at a reference price and retaliate against aggressive cuts.

    The policy posts an episode-specific reference price until the observed
    agent price falls below ``reference_price - threshold``. It then undercuts
    the observed agent price for exactly ``punishment_length`` steps before
    returning to cooperation. Episode parameters are sampled at reset.
    """

    def __init__(
        self,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
        threshold_min: float = 0.3,
        threshold_max: float = 1.0,
        punishment_lengths: Sequence[int] = (3, 5, 8),
        delta_min: float = 0.2,
        delta_max: float = 0.8,
        mid_price: Optional[float] = None,
        high_price: Optional[float] = None,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the repeated-game uniform opponent.

        Args:
            p_min: Minimum posted price during punishment.
            p_max: Maximum allowed posted price.
            threshold_min: Minimum aggressive-cut threshold.
            threshold_max: Maximum aggressive-cut threshold.
            punishment_lengths: Candidate punishment durations in steps.
            delta_min: Minimum punishment undercut amount.
            delta_max: Maximum punishment undercut amount.
            mid_price: Lower bound for the cooperative reference price.
            high_price: Upper bound for the cooperative reference price.
            bounds: Market price bounds. Defaults to :class:`PriceBounds`.
            seed: Optional seed controlling episode parameter samples.

        Raises:
            ValueError: If price or episode-randomization parameters are invalid.
        """
        super().__init__(regime=0, bounds=bounds, seed=seed)
        self.p_min = self.bounds.uniform_min if p_min is None else float(p_min)
        self.p_max = self.bounds.uniform_max if p_max is None else float(p_max)
        default_midpoint = (self.p_min + self.p_max) / 2.0
        self.mid_price = default_midpoint if mid_price is None else float(mid_price)
        self.high_price = self.p_max if high_price is None else float(high_price)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.punishment_lengths = tuple(punishment_lengths)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self._validate_parameters()
        self._punishment_remaining = 0
        self._sample_episode_parameters()
        self._current_price = self._reference_price

    def _validate_parameters(self) -> None:
        numeric_values = (
            self.p_min,
            self.p_max,
            self.mid_price,
            self.high_price,
            self.threshold_min,
            self.threshold_max,
            self.delta_min,
            self.delta_max,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("tit-for-tat parameters must be finite")
        if (
            self.p_min < self.bounds.uniform_min
            or self.p_max > self.bounds.uniform_max
            or self.p_min >= self.p_max
        ):
            raise ValueError("p_min and p_max must define a valid bounded interval")
        if not self.p_min <= self.mid_price < self.high_price <= self.p_max:
            raise ValueError(
                "reference bounds must satisfy p_min <= mid_price < "
                "high_price <= p_max"
            )
        if self.threshold_min < 0 or self.threshold_min >= self.threshold_max:
            raise ValueError(
                "threshold bounds must satisfy 0 <= threshold_min < threshold_max"
            )
        if self.delta_min < 0 or self.delta_min >= self.delta_max:
            raise ValueError("delta bounds must satisfy 0 <= delta_min < delta_max")
        if not self.punishment_lengths:
            raise ValueError("punishment_lengths must not be empty")
        if any(
            not isinstance(length, int)
            or isinstance(length, bool)
            or length < 1
            for length in self.punishment_lengths
        ):
            raise ValueError("punishment lengths must be positive integers")

    def _sample_episode_parameters(self) -> None:
        self._threshold = float(
            self.rng.uniform(self.threshold_min, self.threshold_max)
        )
        self._punishment_length = int(self.rng.choice(self.punishment_lengths))
        self._delta = float(self.rng.uniform(self.delta_min, self.delta_max))
        self._reference_price = float(
            self.rng.uniform(self.mid_price, self.high_price)
        )

    @property
    def threshold(self) -> float:
        """Aggressive-cut threshold sampled for the current episode."""
        return self._threshold

    @property
    def punishment_length(self) -> int:
        """Punishment duration sampled for the current episode."""
        return self._punishment_length

    @property
    def delta(self) -> float:
        """Punishment undercut amount sampled for the current episode."""
        return self._delta

    @property
    def reference_price(self) -> float:
        """Cooperative price sampled for the current episode."""
        return self._reference_price

    @property
    def punishment_remaining(self) -> int:
        """Number of punishment actions remaining after the latest action."""
        return self._punishment_remaining

    @property
    def is_punishing(self) -> bool:
        """Whether the next action remains in the punishment phase."""
        return self._punishment_remaining > 0

    @property
    def current_price(self) -> float:
        """Most recently posted price."""
        return self._current_price

    def reset(self, seed: Optional[int] = None) -> None:
        """Sample episode parameters and return to cooperative state."""
        super().reset(seed=seed)
        self._sample_episode_parameters()
        self._punishment_remaining = 0
        self._current_price = self._reference_price

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        """Return a cooperative or punishment price for the current step."""
        agent_price = float(observation.previous.competitor_prices.uniform)
        aggressive_cut = agent_price < self._reference_price - self._threshold
        if self._punishment_remaining == 0 and aggressive_cut:
            self._punishment_remaining = self._punishment_length

        if self._punishment_remaining > 0:
            self._current_price = float(
                np.clip(agent_price - self._delta, self.p_min, self.p_max)
            )
            self._punishment_remaining -= 1
        else:
            self._current_price = self._reference_price
        return self._current_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        """Reuse the current uniform price for inactive BBP placeholders."""
        price_new = float(self.bounds.clip_bbp_new(self._current_price))
        price_old = float(
            self.bounds.clip_bbp_old(max(self._current_price, price_new))
        )
        return price_new, price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(reference_price={self._reference_price:.2f}, "
            f"threshold={self._threshold:.2f}, delta={self._delta:.2f}, "
            f"punishment_length={self._punishment_length}, regime=0)"
        )


class BBPFixedDiscriminatorOpponent(OpponentPolicy):
    """Use one randomized but episode-fixed BBP price spread."""

    def __init__(
        self,
        mid_price: float = 2.75,
        high_price: float = 4.0,
        discount_min: float = 0.5,
        discount_max: float = 2.0,
        markup_min: float = 0.5,
        markup_max: float = 2.0,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the non-adaptive introductory BBP opponent."""
        super().__init__(regime=1, bounds=bounds, seed=seed)
        self.mid_price = float(mid_price)
        self.high_price = float(high_price)
        self.discount_min = float(discount_min)
        self.discount_max = float(discount_max)
        self.markup_min = float(markup_min)
        self.markup_max = float(markup_max)
        self._validate_parameters()
        self._sample_episode_prices()

    def _validate_parameters(self) -> None:
        values = (
            self.mid_price, self.high_price,
            self.discount_min, self.discount_max,
            self.markup_min, self.markup_max,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("fixed discriminator parameters must be finite")
        if self.mid_price >= self.high_price:
            raise ValueError("mid_price must be less than high_price")
        if self.discount_min <= 0 or self.discount_min >= self.discount_max:
            raise ValueError("discount bounds must satisfy 0 < min < max")
        if self.markup_min <= 0 or self.markup_min >= self.markup_max:
            raise ValueError("markup bounds must satisfy 0 < min < max")

    def _sample_episode_prices(self) -> None:
        self._base_price = float(self.rng.uniform(self.mid_price, self.high_price))
        self._discount = float(
            self.rng.uniform(self.discount_min, self.discount_max)
        )
        self._markup = float(self.rng.uniform(self.markup_min, self.markup_max))
        self._price_new, self._price_old = _bounded_bbp_prices(
            self.bounds,
            self._base_price - self._discount,
            self._base_price + self._markup,
        )

    @property
    def base_price(self) -> float:
        return self._base_price

    @property
    def discount(self) -> float:
        return self._discount

    @property
    def markup(self) -> float:
        return self._markup

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self._sample_episode_prices()

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        return float(self.bounds.clip_uniform(self._base_price))

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        return self._price_new, self._price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(new={self._price_new:.2f}, "
            f"old={self._price_old:.2f}, base={self._base_price:.2f}, regime=1)"
        )


class BBPAcquisitionPredatorOpponent(OpponentPolicy):
    """Discount acquisition aggressively while charging established buyers more."""

    def __init__(
        self,
        epsilon_min: float = 0.0,
        epsilon_max: float = 1.0,
        spread_min: float = 1.0,
        spread_max: float = 4.0,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the episode-randomized customer-acquisition predator."""
        super().__init__(regime=1, bounds=bounds, seed=seed)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)
        self.spread_min = float(spread_min)
        self.spread_max = float(spread_max)
        self._validate_parameters()
        self._sample_episode_prices()

    def _validate_parameters(self) -> None:
        values = (
            self.epsilon_min, self.epsilon_max,
            self.spread_min, self.spread_max,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("acquisition predator parameters must be finite")
        if self.epsilon_min < 0 or self.epsilon_min >= self.epsilon_max:
            raise ValueError("epsilon bounds must satisfy 0 <= min < max")
        if self.epsilon_max > self.bounds.bbp_new_max - self.bounds.bbp_new_min:
            raise ValueError("epsilon_max exceeds the new-customer price range")
        if self.spread_min <= 0 or self.spread_min >= self.spread_max:
            raise ValueError("spread bounds must satisfy 0 < min < max")

    def _sample_episode_prices(self) -> None:
        self._epsilon = float(
            self.rng.uniform(self.epsilon_min, self.epsilon_max)
        )
        self._spread = float(self.rng.uniform(self.spread_min, self.spread_max))
        raw_new = self.bounds.bbp_new_min + self._epsilon
        self._price_new, self._price_old = _bounded_bbp_prices(
            self.bounds,
            raw_new,
            raw_new + self._spread,
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def spread(self) -> float:
        return self._spread

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self._sample_episode_prices()

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        return float(self.bounds.clip_uniform(self._price_new))

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        return self._price_new, self._price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(new={self._price_new:.2f}, "
            f"old={self._price_old:.2f}, spread={self._spread:.2f}, regime=1)"
        )


class BBPLoyaltyHarvesterOpponent(OpponentPolicy):
    """Offer moderate acquisition prices and harvest established customers."""

    def __init__(
        self,
        mid_low: float = 2.0,
        mid_price: float = 3.0,
        mid_high: float = 3.5,
        p_max: Optional[float] = None,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the episode-randomized loyalty-harvesting opponent."""
        super().__init__(regime=1, bounds=bounds, seed=seed)
        self.mid_low = float(mid_low)
        self.mid_price = float(mid_price)
        self.mid_high = float(mid_high)
        self.p_max = self.bounds.bbp_old_max if p_max is None else float(p_max)
        self._validate_parameters()
        self._sample_episode_prices()

    def _validate_parameters(self) -> None:
        values = (self.mid_low, self.mid_price, self.mid_high, self.p_max)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("loyalty harvester parameters must be finite")
        if not (
            self.bounds.bbp_new_min <= self.mid_low < self.mid_price
            <= self.bounds.bbp_new_max
        ):
            raise ValueError("new-customer interval lies outside BBP new bounds")
        if not (
            self.bounds.bbp_old_min <= self.mid_high < self.p_max
            <= self.bounds.bbp_old_max
        ):
            raise ValueError("old-customer interval lies outside BBP old bounds")
        if self.mid_high <= self.mid_price:
            raise ValueError("mid_high must exceed the maximum new-customer price")

    def _sample_episode_prices(self) -> None:
        raw_new = float(self.rng.uniform(self.mid_low, self.mid_price))
        raw_old = float(self.rng.uniform(self.mid_high, self.p_max))
        self._price_new, self._price_old = _bounded_bbp_prices(
            self.bounds, raw_new, raw_old
        )

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self._sample_episode_prices()

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        midpoint = (self._price_new + self._price_old) / 2.0
        return float(self.bounds.clip_uniform(midpoint))

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        return self._price_new, self._price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(new={self._price_new:.2f}, "
            f"old={self._price_old:.2f}, regime=1)"
        )


class BBPMyopicSegmentOptimizerOpponent(OpponentPolicy):
    """Grid-search the one-period optimum for new and established segments."""

    def __init__(
        self,
        grid_size: int = 25,
        min_spread: float = 0.1,
        transport_cost: float = TRANSPORTATION_COST,
        marginal_cost: float = MARGINAL_COST,
        market_state_weight: float = 0.25,
        bounds: Optional[PriceBounds] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the deterministic segmented one-period optimizer."""
        super().__init__(regime=1, bounds=bounds, seed=seed)
        self.grid_size = grid_size
        self.min_spread = float(min_spread)
        self.transport_cost = float(transport_cost)
        self.marginal_cost = float(marginal_cost)
        self.market_state_weight = float(market_state_weight)
        self._validate_parameters()
        self.new_candidate_prices = np.linspace(
            self.bounds.bbp_new_min,
            self.bounds.bbp_new_max,
            self.grid_size,
            dtype=np.float64,
        )
        self.old_candidate_prices = np.linspace(
            self.bounds.bbp_old_min,
            self.bounds.bbp_old_max,
            self.grid_size,
            dtype=np.float64,
        )
        self._price_new = float(self.new_candidate_prices[0])
        self._price_old = float(self.old_candidate_prices[-1])
        self._last_expected_profit = 0.0

    def _validate_parameters(self) -> None:
        if not isinstance(self.grid_size, int) or isinstance(self.grid_size, bool):
            raise ValueError("grid_size must be an integer")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        values = (
            self.min_spread,
            self.transport_cost,
            self.marginal_cost,
            self.market_state_weight,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("segment optimizer parameters must be finite")
        if self.min_spread < 0:
            raise ValueError("min_spread must be non-negative")
        if self.transport_cost <= 0:
            raise ValueError("transport_cost must be positive")
        if not 0.0 <= self.market_state_weight <= 1.0:
            raise ValueError("market_state_weight must be in [0, 1]")
        if self.bounds.bbp_old_max < self.bounds.bbp_new_min + self.min_spread:
            raise ValueError("BBP bounds cannot satisfy min_spread")

    def _expected_share(
        self,
        candidate_prices: np.ndarray,
        competitor_price: float,
        observed_share: float,
    ) -> np.ndarray:
        hotelling_share = 0.5 + (
            competitor_price - candidate_prices
        ) / (2.0 * self.transport_cost)
        blended_share = (
            (1.0 - self.market_state_weight) * hotelling_share
            + self.market_state_weight * observed_share
        )
        return np.clip(blended_share, 0.0, 1.0)

    def expected_profits(self, observation: OpponentObservation) -> np.ndarray:
        """Return expected profit for every new/old candidate combination."""
        new_prices = self.new_candidate_prices[:, None]
        old_prices = self.old_candidate_prices[None, :]
        observed_share = float(np.clip(
            observation.previous.own_market_share, 0.0, 1.0
        ))
        competitor_prices = (
            observation.competitor_submission
            or observation.previous.competitor_prices
        )
        new_share = self._expected_share(
            new_prices,
            float(competitor_prices.new),
            observed_share,
        )
        old_share = self._expected_share(
            old_prices,
            float(competitor_prices.old),
            observed_share,
        )
        new_weight = float(np.clip(
            observation.previous.own_new_customer_ratio, 0.0, 1.0
        ))
        if observation.previous.own_demand_ratio <= 0:
            new_weight = 0.5
        old_weight = 1.0 - new_weight
        profits = (
            new_weight * (new_prices - self.marginal_cost) * new_share
            + old_weight * (old_prices - self.marginal_cost) * old_share
        )
        valid = old_prices >= new_prices + self.min_spread
        return np.where(valid, profits, -np.inf)

    @property
    def last_expected_profit(self) -> float:
        return self._last_expected_profit

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        midpoint = (self._price_new + self._price_old) / 2.0
        return float(self.bounds.clip_uniform(midpoint))

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        profits = self.expected_profits(observation)
        best_flat_index = int(np.argmax(profits))
        best_new_index, best_old_index = np.unravel_index(
            best_flat_index, profits.shape
        )
        self._price_new = float(self.new_candidate_prices[best_new_index])
        self._price_old = float(self.old_candidate_prices[best_old_index])
        self._last_expected_profit = float(profits[best_new_index, best_old_index])
        return self._price_new, self._price_old

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(grid_size={self.grid_size}, "
            f"min_spread={self.min_spread:.2f}, regime=1)"
        )



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
        policy_type: Registered policy type such as ``constant``,
            ``fixed_uniform``, or ``rule_based``.
        **kwargs: Arguments passed to policy constructor
        
    Returns:
        OpponentPolicy instance
        
    Raises:
        ValueError: If policy_type is unknown
    """
    policies = {
        "fixed_uniform": FixedUniformOpponentPolicy,
        "uniform_random": UniformRandomOpponentPolicy,
        "uniform_myopic": UniformMyopicOpponent,
        "uniform_undercutter": UniformUndercutterOpponent,
        "uniform_tit_for_tat": UniformTitForTatOpponent,
        "bbp_fixed_discriminator": BBPFixedDiscriminatorOpponent,
        "bbp_acquisition_predator": BBPAcquisitionPredatorOpponent,
        "bbp_loyalty_harvester": BBPLoyaltyHarvesterOpponent,
        "bbp_myopic_segment_optimizer": BBPMyopicSegmentOptimizerOpponent,
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
    "bbp_myopic_segment_optimizer": {
        "policy_type": "bbp_myopic_segment_optimizer",
        "grid_size": 25,
        "min_spread": 0.1,
        "transport_cost": TRANSPORTATION_COST,
        "marginal_cost": MARGINAL_COST,
        "market_state_weight": 0.25,
    },
    "bbp_loyalty_harvester": {
        "policy_type": "bbp_loyalty_harvester",
        "mid_low": 2.0,
        "mid_price": 3.0,
        "mid_high": 3.5,
        "p_max": PRICE_BBP_OLD_MAX,
    },
    "bbp_acquisition_predator": {
        "policy_type": "bbp_acquisition_predator",
        "epsilon_min": 0.0,
        "epsilon_max": 1.0,
        "spread_min": 1.0,
        "spread_max": 4.0,
    },
    "bbp_fixed_discriminator": {
        "policy_type": "bbp_fixed_discriminator",
        "mid_price": 2.75,
        "high_price": 4.0,
        "discount_min": 0.5,
        "discount_max": 2.0,
        "markup_min": 0.5,
        "markup_max": 2.0,
    },
    "uniform_tit_for_tat": {
        "policy_type": "uniform_tit_for_tat",
        "p_min": PRICE_UNIFORM_MIN,
        "p_max": PRICE_UNIFORM_MAX,
        "threshold_min": 0.3,
        "threshold_max": 1.0,
        "punishment_lengths": (3, 5, 8),
        "delta_min": 0.2,
        "delta_max": 0.8,
        "mid_price": (PRICE_UNIFORM_MIN + PRICE_UNIFORM_MAX) / 2.0,
        "high_price": PRICE_UNIFORM_MAX,
    },
    "uniform_undercutter": {
        "policy_type": "uniform_undercutter",
        "p_min": PRICE_UNIFORM_MIN,
        "p_max": PRICE_UNIFORM_MAX,
        "delta_min": 0.1,
        "delta_max": 1.0,
        "reaction_delays": (1, 2),
    },
    "uniform_myopic": {
        "policy_type": "uniform_myopic",
        "p_min": PRICE_UNIFORM_MIN,
        "p_max": PRICE_UNIFORM_MAX,
        "transport_cost": TRANSPORTATION_COST,
        "best_response_offset": 0.5,
    },
    "uniform_random": {
        "policy_type": "uniform_random",
        "p_min": PRICE_UNIFORM_MIN,
        "p_max": PRICE_UNIFORM_MAX,
        "sigma": 0.25,
    },
    "uniform_fixed": {
        "policy_type": "fixed_uniform",
        "p_min": PRICE_UNIFORM_MIN,
        "p_max": PRICE_UNIFORM_MAX,
        "margin": 0.25,
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
