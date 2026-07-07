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
    TRANSPORTATION_COST,
    BASE_VALUE,
    BBP_RETENTION_PERIODS,
    CONSUMER_FORESIGHT_HORIZON,
    MAX_HISTORY_LENGTH,
    MARGINAL_COST,
    MAX_HISTORY_LENGTH,
    
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
        expected_popularity = float(np.mean(recent))
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
# Firm CLASS
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
        self.uniform_price = 3.0
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
        if len(self.profit_history) < 10:
            return 0.0
        
        recent = np.mean(self.profit_history[-5:])
        earlier = np.mean(self.profit_history[-10:-5])
        
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

    def get_established_share(self, firm_id: int) -> float:
        """Return the population share established with the selected firm.

        The denominator is the full consumer population, including consumers
        currently buying from the competitor. This matches the analytical BBP
        best-response weight used by uniform myopic opponents.

        Args:
            firm_id: Firm identifier, either 0 or 1.

        Returns:
            Fraction of consumers established with ``firm_id`` in ``[0, 1]``.

        Raises:
            ValueError: If ``firm_id`` does not identify a market firm.
        """
        if firm_id not in (0, 1):
            raise ValueError("firm_id must be 0 or 1")
        if not self.consumers:
            return 0.0
        established_count = sum(
            consumer.is_established_with(firm_id)
            for consumer in self.consumers
        )
        return float(established_count / len(self.consumers))

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
