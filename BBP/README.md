# Dynamic Pricing Model (BBP)

A simulation of dynamic pricing behavior in a duopoly market with strategic consumers and firms.

## What It Does

- **Firms**: Set prices strategically based on market observations
- **Consumers**: Choose between firms based on price, location, brand popularity, and expectations about future prices

## Files

- `firm.py` - Firm agents that set prices
- `consumer.py` - Consumer agents that choose firms
- `test.ipynb` - Simulation and analysis

## Key Concepts

### Market Setup
- Linear market: positions range from 0 to 1
- Two competing firms with different locations
- Multiple consumers scattered across the market

### Consumer Decision Making
Consumers weigh:
1. **Instant utility**: Current price, location mismatch, brand popularity
2. **Future expectations**: Predicted next-period price and popularity
3. **Strategicness**: How much they care about future vs immediate utility

Consumer types:
- **Myopic** (strategicness < 0.3): Focus on current prices only
- **Balanced** (0.3 ≤ strategicness < 0.7): Mix of current and future
- **Strategic** (strategicness ≥ 0.7): Heavy weight on future expectations

### Firm Parameters
- `location`: Position on market (0 to 1)
- `discount_factor`: Weights future profits vs current profits
- `popularity`: Brand strength affecting consumer perception
- `price_bounds`: Min/max prices the firm can set
