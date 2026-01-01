
from typing import  List, Dict, Optional
import numpy as np
from config.constants import (
    TRANSPORTATION_COST,
    BASE_VALUE,
    BBP_RETENTION_PERIODS,
    CONSUMER_FORESIGHT_HORIZON,
    MAX_HISTORY_LENGTH,
    MARGINAL_COST,
    MAX_HISTORY_LENGTH,
    DEBUG_MODE
)
# =============================================================================
# CONSUMER CLASS
# =============================================================================

class Consumer:
    """
    Single consumer on the Hotelling line.
    
    Consumers make purchasing decisions based on:
    - Instant utility: V - price - transport_cost * distance - alpha * popularity
    - Expected future utility: discounted sum over H periods
    
    Parameters:
        alpha (exclusivity_pref): How much consumer dislikes popular firms [0, 1]
        beta (strategicness): Discount factor for future utility [0, 1]
    """

    def __init__(
        self,
        consumer_id: int,
        location: float,
        exclusivity_preference: float,
        strategic_foresight: float,
    ):
        """
        Initialize consumer.
        
        Args:
            consumer_id: Unique identifier
            location: Position on Hotelling line [0, 1]
            exclusivity_preference: Alpha - preference for exclusive (less popular) firms
            strategic_foresight: Beta - discount factor for future utility
        """
        self.id = consumer_id
        self.location = location
        self.alpha = exclusivity_preference  # Exclusivity seeking
        self.beta = strategic_foresight       # Strategic foresight / discount factor
        
        # Purchase history: list of {firm_id, price} dicts, most recent first
        self.purchase_history: List[Dict] = []
        
        # Track which firm consumer bought from last period (None if no history)
        self.last_firm_choice: Optional[int] = None
        
    
    @property
    def exclusivity_pref(self) -> float:
        """Alias for alpha."""
        return self.alpha
    
    @property
    def strategicness(self) -> float:
        """Alias for beta."""
        return self.beta

    def get_info(self) -> Dict:
        """Return consumer information as dict."""
        return {
            "id": self.id,
            "location": self.location,
            "exclusivity seekness": self.alpha,
            "strategicness": self.beta,
        }

    def update_purchase(self, firm_id: int, price: float):
        """
        Record a purchase.
        
        Args:
            firm_id: Which firm (0 or 1)
            price: Price paid
        """
        self.last_firm_choice = firm_id
        self.purchase_history.insert(0, {"firm_id": firm_id, "price": price})
        
        # Keep history bounded
        if len(self.purchase_history) > MAX_HISTORY_LENGTH:
            self.purchase_history.pop()

    def is_established_with(self, firm_id: int) -> bool:
        """
        Check if consumer is established with a firm.
        
        A consumer is "established" if they purchased from the same firm
        for the last BBP_RETENTION_PERIODS consecutive periods.
        
        Args:
            firm_id: Firm to check
            
        Returns:
            True if established with this firm
        """
        if len(self.purchase_history) < BBP_RETENTION_PERIODS:
            return False
        
        # Check last N purchases
        for i in range(BBP_RETENTION_PERIODS):
            if self.purchase_history[i]["firm_id"] != firm_id:
                return False
        return True

    def instant_utility(self, firm: 'Firm') -> float:
        """
        Calculate instant utility from purchasing from a firm.
        
        U_instant = V - price - t * |location - firm_location| - alpha * popularity
        
        Args:
            firm: Firm object
            
        Returns:
            Utility value
        """
        price = firm.get_price_for_consumer(self)
        distance = abs(self.location - firm.location)
        
        utility = (
            BASE_VALUE 
            - price 
            - TRANSPORTATION_COST * distance 
            - self.alpha * firm.market_share
        )
        if DEBUG_MODE:
            print(f"Consumer {self.id} instant utility from Firm {firm.firm_id}: "
                  f"V={BASE_VALUE}, price={price:.2f}, distance={distance:.2f}, "
                  f"popularity={firm.market_share:.2f} => U={utility:.2f}")
        return utility

    def expected_price(self, firm: 'Firm') -> float:
        """
        Estimate expected future price from a firm.
        
        Uses exponential moving average of past prices paid to this firm.
        If no history, uses current posted price.
        
        Args:
            firm: Firm object
            
        Returns:
            Expected price
        """
        # Get prices paid to this firm
        prices = [
            record["price"] 
            for record in self.purchase_history 
            if record["firm_id"] == firm.firm_id
        ]
        
        if not prices:
            # No history - use current price
            return firm.get_price_for_consumer(self)
        
        # Exponential moving average (more recent prices weighted higher)
        weights = np.array([0.9 ** i for i in range(len(prices))])
        weights = weights / weights.sum()
        expected_price = np.dot(prices, weights)
        if DEBUG_MODE:
            print(f"Consumer {self.id} expected price from Firm {firm.firm_id}: "
                  f"prices={prices}, expected_price={expected_price:.2f}")
        return expected_price

    def expected_popularity(self, firm: 'Firm') -> float:
        """
        Estimate expected future popularity of a firm.
        
        Uses simple moving average of recent market shares.
        
        Args:
            firm: Firm object
            
        Returns:
            Expected market share
        """
        history = firm.market_share_history
        if not history:
            return firm.market_share
        
        # Use recent history
        recent = history[-min(5, len(history)):]
        expected_popularity = np.mean(recent)
        if DEBUG_MODE:
            print(f"Consumer {self.id} expected popularity from Firm {firm.firm_id}: "
                  f"recent={recent}, expected_popularity={expected_popularity:.2f}")
        return expected_popularity

    def expected_utility(self, firm: 'Firm') -> float:
        """
        Calculate expected utility for one future period.
        
        U_expected = V - E[price] - t * distance - alpha * E[popularity]
        
        Args:
            firm: Firm object
            
        Returns:
            Expected utility
        """
        exp_price = self.expected_price(firm)
        exp_pop = self.expected_popularity(firm)
        distance = abs(self.location - firm.location)
        
        utility = (
            BASE_VALUE 
            - exp_price 
            - TRANSPORTATION_COST * distance 
            - self.alpha * exp_pop
        )
        if DEBUG_MODE:
            print(f"Consumer {self.id} expected utility from Firm {firm.firm_id}: "
                  f"V={BASE_VALUE}, E[price]={exp_price:.2f}, distance={distance:.2f}, "
                  f"E[popularity]={exp_pop:.2f} => U={utility:.2f}")
        return utility

    def total_utility(self, firm: 'Firm') -> float:
        """
        Calculate total utility including future expectations.
        
        U_total = U_instant + sum_{h=1}^{H} beta^h * U_expected
        
        The expected utility is simplified by assuming similar expected utility
        each period (could be extended for more sophisticated forecasting).
        
        Args:
            firm: Firm object
            
        Returns:
            Total discounted utility
        """
        instant = self.instant_utility(firm)
        
        if self.beta == 0:
            return instant
        
        # Calculate discounted future utility over H periods
        expected = self.expected_utility(firm)
        future_sum = 0.0
        
        for h in range(1, CONSUMER_FORESIGHT_HORIZON + 1):
            future_sum += (self.beta ** h) * expected
        
        return instant + future_sum

    def choose_firm(self, firm_a: 'Firm', firm_b: 'Firm') -> int:
        """
        Choose which firm to purchase from.
        
        Consumer computes total utility for each firm and chooses higher.
        
        Args:
            firm_a: First firm (id=0)
            firm_b: Second firm (id=1)
            
        Returns:
            0 for firm_a, 1 for firm_b
        """
        u_a = self.total_utility(firm_a)
        u_b = self.total_utility(firm_b)
        
        if u_a >= u_b:
            return 0
        return 1

    def reset(self):
        """Reset consumer state for new episode."""
        self.purchase_history = []
        self.last_firm_choice = None


# =============================================================================
# FIRM CLASS  
# =============================================================================



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
        
        # Calculate relative popularity
        if competitor.market_share > 0:
            self.relative_popularity = self.market_share / competitor.market_share
        else:
            self.relative_popularity = float('inf') if self.market_share > 0 else 1.0
        
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
        return np.clip(trend, -1.0, 1.0)

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

