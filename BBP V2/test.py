"""
Simple test suite for the Hotelling market simulator.

Tests basic functionality of Consumer, Firm, and Market classes.
"""

import numpy as np
from env.target_system import HotellingMarket
from env.modules import Firm , Consumer
from config.constants import NUM_CONSUMERS, DEBUG_MODE


def test_consumer_creation(num_consumers: int = 3):
    """Test consumer initialization."""
    print("=" * 60)
    print("TEST 1: Consumer Creation")
    print("=" * 60)
    
    market = HotellingMarket(num_consumers=num_consumers, seed=42)
    
    for i, consumer in enumerate(market.consumers):
        info = consumer.get_info()
        print(f"Consumer {i}: loc={info['location']:.2f}, exclusivity seekness={info['exclusivity seekness']:.2f}, strategicness={info['strategicness']:.2f}")
    
    print("✓ Consumer creation test passed\n")


def test_firm_pricing():
    """Test firm pricing mechanisms."""
    print("=" * 60)
    print("TEST 2: Firm Pricing")
    print("=" * 60)
    
    firm = Firm(firm_id=0, location=0.0)
    
    # Test uniform pricing
    firm.set_regime(0)
    firm.set_prices(uniform_price=2.5)
    print(f"Uniform regime - price: {firm.get_price_for_consumer():.2f}")
    
    # Test BBP pricing
    firm.set_regime(1)
    firm.set_prices(price_new=1.5, price_old=2.5)
    print(f"BBP regime - new consumer price: {firm.get_price_for_consumer():.2f}")
    
    # Create consumer and test established status
    consumer = Consumer(0, location=0.5, exclusivity_preference=0.3, strategic_foresight=0.5)
    print(f"Consumer is established with firm? {consumer.is_established_with(0)}")
    
    # Simulate purchases to make consumer established
    consumer.update_purchase(0, 1.5)
    consumer.update_purchase(0, 1.6)
    consumer.update_purchase(0, 1.7)
    print(f"After 3 purchases from firm 0: {consumer.is_established_with(0)}")
    print(f"BBP regime - established customer price: {firm.get_price_for_consumer(consumer):.2f}")
    
    print("✓ Firm pricing test passed\n")


def test_market_step():
    """Test single market period."""
    print("=" * 60)
    print("TEST 3: Market Step Execution")
    print("=" * 60)
    
    market = HotellingMarket(num_consumers=3, seed=42)
    for consumer in market.consumers:
        print(consumer.get_info())
    # Set regimes
    market.set_regimes(regime_0=0, regime_1=0)  # Both uniform
    
    # Set prices
    firm_a_prices = {
        "uniform_price": 2.0,
        "price_new": 1.5,
        "price_old": 2.5,
    }
    firm_b_prices = {
        "uniform_price": 2.2,
        "price_new": 1.7,
        "price_old": 2.7,
    }
    
    # Execute step
    demand_0, demand_1 = market.step(firm_a_prices, firm_b_prices)
    
    print(f"Firm 0: demand={demand_0}, market_share={market.firms[0].market_share:.2f}")
    print(f"Firm 1: demand={demand_1}, market_share={market.firms[1].market_share:.2f}")
    print(f"Firm 0 profit: {market.firms[0].last_period_profit:.2f}")
    print(f"Firm 1 profit: {market.firms[1].last_period_profit:.2f}")
    
    print("✓ Market step test passed\n")


def test_bbp_strategy():
    """Test BBP pricing strategy."""
    print("=" * 60)
    print("TEST 4: BBP Strategy (Multiple Periods)")
    print("=" * 60)
    
    market = HotellingMarket(num_consumers=30, seed=42)
    
    # Period 1: Uniform pricing
    market.set_regimes(regime_0=0, regime_1=0)
    prices_a = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
    prices_b = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
    
    demand_a, demand_b = market.step(prices_a, prices_b)
    print(f"Period 1 (Uniform): Firm 0 demand={demand_a}, Firm 1 demand={demand_b}")
    
    # Period 2: Firm 0 switches to BBP
    market.set_regimes(regime_0=1, regime_1=0)
    prices_a = {"uniform_price": 2.0, "price_new": 1.2, "price_old": 2.8}
    prices_b = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
    
    demand_a, demand_b = market.step(prices_a, prices_b)
    print(f"Period 2 (Firm 0 BBP): Firm 0 demand={demand_a}, Firm 1 demand={demand_b}")
    print(f"  Firm 0: new_customers={market.firms[0].period_demand_new}, old_customers={market.firms[0].period_demand_old}")
    print(f"  Firm 0 profit: {market.firms[0].last_period_profit:.2f}")
    
    # Period 3: Continue
    prices_a = {"uniform_price": 2.0, "price_new": 1.2, "price_old": 2.8}
    prices_b = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
    
    demand_a, demand_b = market.step(prices_a, prices_b)
    print(f"Period 3 (Firm 0 BBP): Firm 0 demand={demand_a}, Firm 1 demand={demand_b}")
    print(f"  Firm 0 retention_rate: {market.firms[0].retention_rate:.2f}")
    
    print("✓ BBP strategy test passed\n")


def test_observations():
    """Test observation generation."""
    print("=" * 60)
    print("TEST 5: Observation Generation")
    print("=" * 60)
    
    market = HotellingMarket(num_consumers=25, seed=42)
    
    # Run a few steps
    for t in range(3):
        prices_a = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
        prices_b = {"uniform_price": 2.1, "price_new": 1.6, "price_old": 2.6}
        market.step(prices_a, prices_b)
    
    # Get observations
    state_0 = market.get_firm_state(firm_id=0)
    state_1 = market.get_firm_state(firm_id=1)
    
    print("Firm 0 State:")
    for key, val in state_0.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("\nFirm 1 State:")
    for key, val in state_1.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("✓ Observation generation test passed\n")


def test_episode_simulation():
    """Test full episode simulation."""
    print("=" * 60)
    print("TEST 6: Full Episode Simulation (100 steps)")
    print("=" * 60)
    
    market = HotellingMarket(num_consumers=40, seed=42)
    
    total_profit_0 = 0.0
    total_profit_1 = 0.0
    
    # Simulate episode
    for t in range(100):
        # Simple strategy: Firm 0 uses BBP, Firm 1 uses uniform
        regime_0 = 1 if t % 20 < 10 else 0
        regime_1 = 0
        
        market.set_regimes(regime_0, regime_1)
        
        # Random prices
        prices_a = {
            "uniform_price": np.random.uniform(1.5, 2.5),
            "price_new": np.random.uniform(1.0, 1.8),
            "price_old": np.random.uniform(2.0, 3.0),
        }
        prices_b = {
            "uniform_price": np.random.uniform(1.5, 2.5),
            "price_new": np.random.uniform(1.0, 1.8),
            "price_old": np.random.uniform(2.0, 3.0),
        }
        
        market.step(prices_a, prices_b)
        
        total_profit_0 += market.firms[0].last_period_profit
        total_profit_1 += market.firms[1].last_period_profit
    
    print(f"Episode Summary (100 steps):")
    print(f"Firm 0: total_profit={total_profit_0:.2f}, avg_profit={total_profit_0/100:.2f}")
    print(f"Firm 1: total_profit={total_profit_1:.2f}, avg_profit={total_profit_1/100:.2f}")
    print(f"Market concentration (HHI): {market.get_market_concentration():.3f}")
    
    print("✓ Episode simulation test passed\n")


if __name__ == "__main__":
    # test_consumer_creation()
    # test_firm_pricing()
    test_market_step()
    # test_bbp_strategy()
    # test_observations()
    # test_episode_simulation()
    
    # print("=" * 60)
    # print("✓ ALL TESTS PASSED")
    # print("=" * 60)