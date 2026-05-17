# Quick Start Guide

## Installation

```bash
cd "c:\Users\ASUS\Desktop\Article\Dynamic Pricing\BBP V2"

# Install dependencies
pip install pettingzoo gymnasium numpy

# Optional: for Stable-Baselines3 training
pip install stable-baselines3 sb3-contrib tensorboard
```

## Running Tests

```bash
# Run all tests
python test.py

# Run environment examples
python train_example.py
```

## Basic Usage

### 1. Simple Random Episode

```python
from env.hotelling_env import HotellingDuopolyEnv
import numpy as np

# Create environment
env = HotellingDuopolyEnv(num_consumers=50, episode_length=200, strategy_cycle=10)

# Reset for new episode
obs, info = env.reset(seed=42)

# Run episode
for t in range(200):
    # Random actions
    actions = {
        "firm_0": {
            "strategy": np.random.randint(0, 2),
            "pricing": np.random.rand(3)
        },
        "firm_1": {
            "strategy": np.random.randint(0, 2),
            "pricing": np.random.rand(3)
        }
    }
    
    obs, rewards, terms, truncs, info = env.step(actions)
    
    print(f"Step {t+1}: Firm 0 profit = {rewards['firm_0']:.2f}")
    
    if any(truncs.values()):
        break

env.close()
```

### 2. Access Hierarchical Observations

```python
# After stepping the environment
obs = obs["firm_0"]  # Get firm_0's observation

# Strategy controller sees:
sc_obs = obs["strategy_controller"]
print(f"Market share: {sc_obs['market_share']}")
print(f"Retention rate: {sc_obs['retention_rate']}")
print(f"Competitor regime: {sc_obs['competitor_regime']}")

# Pricing controller sees:
pc_obs = obs["pricing_controller"]
print(f"Own prices: {pc_obs['own_prices']}")
print(f"Competitor prices: {pc_obs['comp_prices']}")
print(f"New/Old ratio: {pc_obs['new_old_ratio']}")
```

### 3. Understanding the Economy

```python
from env.target_system import HotellingMarket, Consumer, Firm

# Create market
market = HotellingMarket(num_consumers=30, seed=42)

# Set pricing regimes
market.set_regimes(regime_0=0, regime_1=1)  # Firm 0: Uniform, Firm 1: BBP

# Set prices
prices_0 = {"uniform_price": 2.0, "price_new": 1.5, "price_old": 2.5}
prices_1 = {"uniform_price": 2.1, "price_new": 1.6, "price_old": 2.6}

# Execute market step
demand_0, demand_1 = market.step(prices_0, prices_1)

print(f"Firm 0: demand={demand_0}, profit={market.firms[0].last_period_profit:.2f}")
print(f"Firm 1: demand={demand_1}, profit={market.firms[1].last_period_profit:.2f}")

# Get detailed state for observations
state_0 = market.get_firm_state(firm_id=0)
print(f"Firm 0 market share: {state_0['market_share']:.2f}")
print(f"Firm 0 retention rate: {state_0['retention_rate']:.2f}")
```

### 4. Training with Stable-Baselines3 (Example)

```python
from stable_baselines3 import PPO
from env.hotelling_env import HotellingDuopolyEnv
import numpy as np

# For single-agent training, wrap or train one agent at a time
class SingleAgentWrapper:
    """Wrapper to train single agent while other plays random"""
    def __init__(self, env, agent_id="firm_0"):
        self.env = env
        self.agent_id = agent_id
        self.other_agent = "firm_1" if agent_id == "firm_0" else "firm_0"
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs[self.agent_id]
    
    def step(self, action):
        # Apply action to training agent
        actions = {
            self.agent_id: action,
            self.other_agent: {
                "strategy": np.random.randint(0, 2),
                "pricing": np.random.rand(3)
            }
        }
        obs, rewards, terms, truncs, _ = self.env.step(actions)
        return obs[self.agent_id], rewards[self.agent_id], terms[self.agent_id], truncs[self.agent_id], {}

# Create and train
env = HotellingDuopolyEnv(num_consumers=50, episode_length=200)
wrapped_env = SingleAgentWrapper(env, agent_id="firm_0")

# Note: Requires environment flattening for SB3
# This is simplified for illustration
```

## Key Concepts

### Regimes
- **Uniform (0)**: Firm posts single price to all consumers
- **BBP (1)**: Firm posts two prices:
  - `price_new`: For consumers without purchase history
  - `price_old`: For established customers (bought last N periods)

### Consumer Behavior
- Consumers are rational and forward-looking
- They calculate utility including future expectations
- Utility = instant + discounted sum of expected future utilities
- Choose firm with highest total utility

### Market Share
- Calculated as: demand / total_consumers
- Each consumer buys from exactly one firm per period
- Firms compete for market share through prices and regimes

### Profit
- Uniform regime: profit = demand × (price - marginal_cost)
- BBP regime: profit = new_demand × price_new + old_demand × price_old - costs

## Customization

### Change Market Parameters

Edit `config/constants.py`:
```python
NUM_CONSUMERS = 100              # More consumers
EPISODE_LENGTH = 500             # Longer episodes
STRATEGY_CYCLE_LENGTH = 20       # Strategy decides more often
CONSUMER_FORESIGHT_HORIZON = 5   # Consumers look further ahead
BBP_RETENTION_PERIODS = 3        # Need more purchases to be "established"
```

### Change Price Ranges

```python
PRICE_UNIFORM_MIN = 1.0
PRICE_UNIFORM_MAX = 6.0

PRICE_BBP_NEW_MIN = 0.5
PRICE_BBP_NEW_MAX = 3.0

PRICE_BBP_OLD_MIN = 2.0
PRICE_BBP_OLD_MAX = 6.0
```

## Debugging

### Enable Debug Output

```python
from config.constants import DEBUG_MODE
# Set DEBUG_MODE = True in constants.py

# Or manually:
from env.target_system import HotellingMarket

market = HotellingMarket(num_consumers=10, seed=42)
# Set prices and run
demand_0, demand_1 = market.step(prices_0, prices_1)
# Will print detailed information about consumer choices
```

### Monitor Training Metrics

```python
from env.hotelling_env import HotellingDuopolyEnv

env = HotellingDuopolyEnv(num_consumers=50, episode_length=200)
obs, _ = env.reset()

total_profit_0 = 0
total_profit_1 = 0

for _ in range(200):
    actions = {
        "firm_0": {"strategy": 0, "pricing": [0.5, 0.4, 0.6]},
        "firm_1": {"strategy": 0, "pricing": [0.5, 0.4, 0.6]}
    }
    obs, rewards, terms, truncs, info = env.step(actions)
    
    total_profit_0 += rewards["firm_0"]
    total_profit_1 += rewards["firm_1"]

print(f"Avg profit firm 0: {total_profit_0 / 200:.2f}")
print(f"Avg profit firm 1: {total_profit_1 / 200:.2f}")
```

## Troubleshooting

### Issue: "market_share is negative"
- Check: Are demand values correct?
- Fix: Ensure consumers are being assigned properly

### Issue: "Price out of range"
- Check: Action normalization is correct
- Fix: Verify action_to_prices scaling

### Issue: "Episode doesn't terminate"
- Expected: Episodes run exactly EPISODE_LENGTH steps
- Check: Using `truncations` not `terminations` for early stopping

## Next Steps

1. **Understand the economy**: Run `test.py` and study outputs
2. **Explore environment**: Run `train_example.py` examples
3. **Implement RL training**: Start with single-agent training
4. **Analyze results**: Study what strategies emerge
5. **Extend model**: Add new features/parameters

## Research Questions to Explore

1. When is BBP profitable vs. uniform pricing?
2. What consumer characteristics favor BBP?
3. How does strategy cycle length (K) affect outcomes?
4. What happens with asymmetric information?
5. Can firms learn cooperative strategies?
6. What's the consumer welfare impact of BBP?

---

**Status**: ✅ Environment is fully functional and tested. Ready for RL training!
