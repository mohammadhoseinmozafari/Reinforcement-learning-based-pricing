
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
from env.models import Firm
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