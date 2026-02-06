"""
Pure economic simulator for Hotelling duopoly with behavior-based pricing.

No RL or PettingZoo imports. Pure economics only.
Handles consumers, firms, demand, market shares, and profits.

Key Features:
- Consumers make utility-maximizing decisions considering present and future
- Firms can choose uniform pricing or BBP (Behavior-Based Pricing)
- BBP allows price discrimination between new and established consumers
"""

from typing import Tuple, List, Dict, Optional

import numpy as np


from .modules import Firm , Consumer

from config.constants import (
    NUM_CONSUMERS,
    HOTELLING_LEFT,
    HOTELLING_RIGHT,
    FIRM_A_LOCATION,
    FIRM_B_LOCATION,
    ALPHA_MIN,
    ALPHA_MAX,
    BETA_MIN,
    BETA_MAX,
    DEBUG_MODE,
)



# =============================================================================
# MARKET CLASS
# =============================================================================

class HotellingMarket:
    """
    Pure economic simulator for Hotelling duopoly.
    
    Manages consumers, firms, and market clearing each period.
    This class contains NO RL logic - it's purely economic simulation.
    """

    def __init__(
        self,
        num_consumers: int = NUM_CONSUMERS,
        seed: Optional[int] = 42,
        exclusivity_mean: Optional[float] = None,
        strategicness_mean: Optional[float] = None,
    ):
        """
        Initialize market.
        
        Args:
            num_consumers: Number of consumers
            seed: Random seed for reproducibility
        """
        self.num_consumers = num_consumers
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.exclusivity_mean = exclusivity_mean
        self.strategicness_mean = strategicness_mean
        
        # Create agents
        self.consumers: List[Consumer] = self._create_consumers()
        self.firms: List[Firm] = [
            Firm(0, FIRM_A_LOCATION),
            Firm(1, FIRM_B_LOCATION),
        ]
        
        # Track time
        self.current_period = 0

    def _create_consumers(self) -> List[Consumer]:
        """Create consumer population with random attributes."""
        consumers = []
        for i in range(self.num_consumers):
            consumer = Consumer(
                consumer_id=i,
                location=self.rng.uniform(HOTELLING_LEFT, HOTELLING_RIGHT),
                exclusivity_preference=self.rng.uniform(
                    ALPHA_MIN,
                    ALPHA_MAX
                ) if self.exclusivity_mean is None else np.clip(
                    self.rng.normal(self.exclusivity_mean, 0.1),
                    ALPHA_MIN,
                    ALPHA_MAX
                ),
                strategic_foresight=self.rng.uniform(
                    BETA_MIN,
                    BETA_MAX
                ) if self.strategicness_mean is None else np.clip(
                    self.rng.normal(self.strategicness_mean, 0.1),
                    BETA_MIN,
                    BETA_MAX
                ),
            )
            consumers.append(consumer)
        return consumers

    def set_prices(
        self,
        firm_0_prices: Dict[str, float],
        firm_1_prices: Dict[str, float],
    ):
        """
        Set prices for both firms.
        
        Args:
            firm_0_prices: Dict with uniform_price, price_new, price_old
            firm_1_prices: Dict with uniform_price, price_new, price_old
        """
        self.firms[0].set_prices(
            uniform_price=firm_0_prices.get("uniform_price"),
            price_new=firm_0_prices.get("price_new"),
            price_old=firm_0_prices.get("price_old"),
        )
        self.firms[1].set_prices(
            uniform_price=firm_1_prices.get("uniform_price"),
            price_new=firm_1_prices.get("price_new"),
            price_old=firm_1_prices.get("price_old"),
        )

    def set_regimes(self, regime_0: int, regime_1: int):
        """
        Set pricing regimes for both firms.
        
        Args:
            regime_0: Firm 0 regime (0=Uniform, 1=BBP)
            regime_1: Firm 1 regime (0=Uniform, 1=BBP)
        """
        self.firms[0].set_regime(regime_0)
        self.firms[1].set_regime(regime_1)

    def step(
        self,
        firm_0_prices: Dict[str, float],
        firm_1_prices: Dict[str, float],
    ) -> Tuple[int, int]:
        """
        Execute one market period.
        
        Args:
            firm_0_prices: Prices for firm 0
            firm_1_prices: Prices for firm 1
            
        Returns:
            (demand_firm_0, demand_firm_1)
        """
        self.current_period += 1
        
        # Set prices
        self.set_prices(firm_0_prices, firm_1_prices)
        
        # Reset period counters
        for firm in self.firms:
            firm.start_period()
        
        # Consumer choice phase
        for consumer in self.consumers:
            firm_choice = consumer.choose_firm(self.firms[0], self.firms[1])
            chosen_firm = self.firms[firm_choice]
            
            # Get price consumer pays
            price_paid = chosen_firm.get_price_for_consumer(consumer)
            
            # Record purchase
            chosen_firm.record_purchase(consumer, price_paid)
            consumer.update_purchase(firm_choice, price_paid)
            
            if DEBUG_MODE:
                print(f"Consumer {consumer.id} (loc={consumer.location:.2f}) "
                      f"chose Firm {firm_choice} at price {price_paid:.2f}")
        
        # End period calculations
        for firm in self.firms:
            competitor = self.firms[1 - firm.firm_id]
            firm.end_period(self.num_consumers, competitor)
        
        if DEBUG_MODE:
            for firm in self.firms:
                print(f"Firm {firm.firm_id}: demand={firm.period_demand}, "
                      f"share={firm.market_share:.2f}, profit={firm.last_period_profit:.2f}")
        
        return self.firms[0].period_demand, self.firms[1].period_demand

    def get_firm_state(self, firm_id: int) -> Dict[str, float]:
        """
        Get detailed state for a firm (for observations).
        
        Args:
            firm_id: Which firm (0 or 1)
            
        Returns:
            Dict with state variables
        """
        firm = self.firms[firm_id]
        competitor = self.firms[1 - firm_id]
        
        return {
            # Strategy controller observations
            "market_share": firm.market_share,
            "popularity_change": firm.get_popularity_change(),
            "retention_rate": firm.retention_rate,
            "profit_trend": firm.get_profit_trend(),
            "relative_popularity": min(firm.relative_popularity, 10.0),  # Cap for stability
            "competitor_regime": float(competitor.pricing_regime),
            
            # Pricing controller observations  
            "new_old_ratio": firm.get_new_old_ratio(),
            "own_uniform_price": firm.uniform_price,
            "own_price_new": firm.price_new,
            "own_price_old": firm.price_old,
            "comp_uniform_price": competitor.uniform_price,
            "comp_price_new": competitor.price_new,
            "comp_price_old": competitor.price_old,
            "last_demand": firm.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0,
            "regime": float(firm.pricing_regime),
            
            # Additional useful metrics
            "period_profit": firm.last_period_profit,
            "cumulative_profit": firm.cumulative_profit,
            "demand_new": firm.period_demand_new,
            "demand_old": firm.period_demand_old,
        }

    def get_market_concentration(self) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI).
        
        Returns:
            HHI value in [0, 1] (0.5 = equal split, 1 = monopoly)
        """
        s0 = self.firms[0].market_share
        s1 = self.firms[1].market_share
        return s0**2 + s1**2

    def reset(self, seed: Optional[int] = None):
        """
        Reset market for new episode.
        
        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        
        # Recreate consumers with fresh random attributes
        self.consumers = self._create_consumers()
        
        # Reset firms
        for firm in self.firms:
            firm.reset()
        
        self.current_period = 0

    def get_consumer_stats(self) -> Dict[str, float]:
        """Get aggregate consumer statistics."""
        alphas = [c.alpha for c in self.consumers]
        betas = [c.beta for c in self.consumers]
        locations = [c.location for c in self.consumers]
        
        return {
            "mean_alpha": float(np.mean(alphas)),
            "std_alpha": float(np.std(alphas)),
            "mean_beta": float(np.mean(betas)),
            "std_beta": float(np.std(betas)),
            "mean_location": float(np.mean(locations)),
            "std_location": float(np.std(locations)),
        }
