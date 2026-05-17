from typing import Dict, List, Optional
from env.models import Consumer
import numpy as np
from config.constants import (
    MAX_HISTORY_LENGTH,
    MARGINAL_COST,
    MAX_HISTORY_LENGTH
)

class Firm:
    """
    Single firm in the Hotelling market.
    
    Firms can use two pricing strategies:
    - Uniform: Single price for all consumers
    - BBP: Different prices for new vs established consumers
    
    Tracks market performance including demand, market share, profits.
    """

    def __init__(self, firm_id: int, location: float):
        """
        Initialize firm.
        
        Args:
            firm_id: 0 or 1
            location: Position on Hotelling line [0, 1]
        """
        self.firm_id = firm_id
        self.location = location
        
        # ======================
        # PRICING STATE
        # ======================
        self.pricing_regime = 0  # 0 = Uniform, 1 = BBP
        
        # Current prices
        self.uniform_price = 2.0
        self.price_new = 1.5      # Price for new consumers (BBP)
        self.price_old = 2.5      # Price for established consumers (BBP)
        
        # Price history for trends (list of [uniform, new, old])
        self.price_history: List[List[float]] = []
        
        # ======================
        # MARKET STATE
        # ======================
        # Current period
        self.period_demand = 0
        self.period_demand_new = 0      # Demand from new consumers
        self.period_demand_old = 0      # Demand from established consumers
        
        # History
        self.demand_history: List[int] = []
        self.market_share_history: List[float] = []
        self.profit_history: List[float] = []
        self.retention_history: List[float] = []  # Retention rate history
        
        # Current values (computed each period)
        self.market_share = 0.0
        self.retention_rate = 0.0
        self.relative_popularity = 0.0
        
        # ======================
        # PERFORMANCE TRACKING
        # ======================
        self.last_period_profit = 0.0
        self.cumulative_profit = 0.0
        self.last_period_quantity = 0


    # ========================
    # PRICING METHODS
    # ========================
    
    def set_regime(self, regime: int):
        """
        Set pricing regime.
        
        Args:
            regime: 0 for Uniform, 1 for BBP
        """
        self.pricing_regime = int(regime)

    def set_prices(
        self,
        uniform_price: Optional[float] = None,
        price_new: Optional[float] = None,
        price_old: Optional[float] = None,
    ):
        """
        Set prices for the firm.
        
        Args:
            uniform_price: Single price for uniform regime
            price_new: Price for new consumers in BBP
            price_old: Price for established consumers in BBP
        """
        if uniform_price is not None:
            self.uniform_price = np.clip(uniform_price, 0.0, 10.0)
        
        if price_new is not None:
            self.price_new = np.clip(price_new, 0.0, 10.0)
        
        if price_old is not None:
            self.price_old = np.clip(price_old, 0.0, 10.0)
            # Enforce constraint: price_old >= price_new
            self.price_old = max(self.price_old, self.price_new)

    def get_price_for_consumer(self, consumer: Optional[Consumer] = None) -> float:
        """
        Get the price shown to a specific consumer.
        
        Args:
            consumer: Consumer object (optional)
            
        Returns:
            Price this consumer sees
        """
        if self.pricing_regime == 0:  # Uniform
            return self.uniform_price
        else:  # BBP
            if consumer is not None and consumer.is_established_with(self.firm_id):
                return self.price_old
            return self.price_new

    def get_current_prices(self) -> Dict[str, float]:
        """Get all current prices as dict."""
        return {
            "uniform_price": self.uniform_price,
            "price_new": self.price_new,
            "price_old": self.price_old,
        }

    # ========================
    # PERIOD MANAGEMENT
    # ========================
    
    def start_period(self):
        """Reset period-specific counters."""
        self.period_demand = 0
        self.period_demand_new = 0
        self.period_demand_old = 0

    def record_purchase(self, consumer: Consumer, price: float):
        """
        Record a purchase from a consumer.
        
        Args:
            consumer: Consumer who purchased
            price: Price paid
        """
        self.period_demand += 1
        
        if consumer.is_established_with(self.firm_id):
            self.period_demand_old += 1
        else:
            self.period_demand_new += 1

    def end_period(self, total_consumers: int, competitor: 'Firm'):
        """
        Finalize period calculations.
        
        Args:
            total_consumers: Total number of consumers in market
            competitor: The other firm
        """
        # Record demand history
        self.demand_history.append(self.period_demand)
        if len(self.demand_history) > MAX_HISTORY_LENGTH:
            self.demand_history.pop(0)
        
        # Calculate market share for this period
        self.market_share = (
            self.period_demand / total_consumers 
            if total_consumers > 0 else 0.0
        )
        self.market_share_history.append(self.market_share)
        if len(self.market_share_history) > MAX_HISTORY_LENGTH:
            self.market_share_history.pop(0)
        
        # Calculate relative popularity (clamped to avoid inf values that break neural networks)
        if competitor.market_share > 0:
            self.relative_popularity = self.market_share / competitor.market_share
        else:
            # Use a large but finite value instead of inf
            self.relative_popularity = 10.0 if self.market_share > 0 else 1.0
        # Clamp to reasonable range for RL stability
        self.relative_popularity = min(self.relative_popularity, 10.0)
        
        # Calculate retention rate
        self._calculate_retention_rate()
        
        # Calculate profit
        profit = self._calculate_profit()
        self.last_period_profit = profit
        self.cumulative_profit += profit
        
        self.profit_history.append(profit)
        if len(self.profit_history) > MAX_HISTORY_LENGTH:
            self.profit_history.pop(0)
        
        # Record price history
        self.price_history.append([
            self.uniform_price, self.price_new, self.price_old
        ])
        if len(self.price_history) > MAX_HISTORY_LENGTH:
            self.price_history.pop(0)
        
        # Store for observation
        self.last_period_quantity = self.period_demand

    def _calculate_profit(self) -> float:
        """Calculate profit for current period."""
        if self.pricing_regime == 0:  # Uniform pricing
            profit = self.period_demand * (self.uniform_price - MARGINAL_COST)
        else:  # BBP
            profit = (
                self.period_demand_new * (self.price_new - MARGINAL_COST) +
                self.period_demand_old * (self.price_old - MARGINAL_COST)
            )
        return profit

    def _calculate_retention_rate(self):
        """
        Calculate retention rate.
        
        Retention = (old customers this period) / (total customers last period)
        """
        if len(self.demand_history) < 2:
            self.retention_rate = 0.0
        else:
            prev_demand = self.demand_history[-2] if len(self.demand_history) >= 2 else 0
            if prev_demand > 0:
                self.retention_rate = self.period_demand_old / prev_demand
            else:
                self.retention_rate = 0.0
        
        self.retention_history.append(self.retention_rate)
        if len(self.retention_history) > MAX_HISTORY_LENGTH:
            self.retention_history.pop(0)

    # ========================
    # OBSERVATION HELPERS
    # ========================
    
    def get_popularity_change(self) -> float:
        """Get change in market share from previous period."""
        if len(self.market_share_history) < 2:
            return 0.0
        return self.market_share_history[-1] - self.market_share_history[-2]

    def get_profit_trend(self) -> float:
        """
        Get profit trend (recent vs earlier).
        
        Returns value in [-1, 1] indicating profit direction.
        """
        if len(self.profit_history) < 4:
            return 0.0
        
        recent = np.mean(self.profit_history[-2:])
        earlier = np.mean(self.profit_history[-4:-2])
        
        if earlier == 0:
            return 0.0 if recent == 0 else 1.0
        
        # Normalized trend
        trend = (recent - earlier) / (abs(earlier) + 1e-6)
        return float(np.clip(trend, -1.0, 1.0))

    def get_new_old_ratio(self) -> float:
        """Get ratio of new customers to total demand."""
        if self.period_demand == 0:
            return 0.0
        return self.period_demand_new / self.period_demand

    def reset(self):
        """Reset firm for new episode."""
        self.pricing_regime = 0
        self.uniform_price = 2.0
        self.price_new = 1.5
        self.price_old = 2.5
        
        self.period_demand = 0
        self.period_demand_new = 0
        self.period_demand_old = 0
        
        self.demand_history = []
        self.market_share_history = []
        self.profit_history = []
        self.retention_history = []
        self.price_history = []
        
        self.market_share = 0.0
        self.retention_rate = 0.0
        self.relative_popularity = 0.0
        
        self.last_period_profit = 0.0
        self.cumulative_profit = 0.0
        self.last_period_quantity = 0

