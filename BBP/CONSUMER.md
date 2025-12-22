# Consumer Class

Agent that chooses between firms.

## Constructor

```python
Consumer(location, exclusivity_sensitivity, strategicness, T, V)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `location` | float | [0, 1] | Position on linear market |
| `exclusivity_sensitivity` | float | [0, 1] | How much consumer values brand exclusivity |
| `strategicness` | float | [0, 1] | Weight on future prices (see types below) |
| `T` | float | ≥ 0 | Disutility from location mismatch per unit distance |
| `V` | float | | Product value/willingness to pay |

## Consumer Types

Determined by `strategicness` value:

| Type | Range | Behavior |
|------|-------|----------|
| Myopic | < 0.3 | Only cares about current prices |
| Balanced | 0.3-0.7 | Mixes current and expected future prices |
| Strategic | ≥ 0.7 | Heavily weights expected future prices |

Access via `consumer.strategicness_type`

## Utility Calculation

### Instant Utility (Current Period)
```
U_instant = V - price - (T × distance) - (exclusivity_sensitivity × popularity)
```

Where:
- `V` = product value
- `price` = firm's current price
- `distance` = |consumer_location - firm_location|
- `popularity` = firm's brand strength

### Expected Future Utility (Next Period)
Uses 3-period moving average of firm's price and popularity history to predict next period:
```
U_future = V - expected_price - (T × distance) + (exclusivity_sensitivity × expected_popularity)
```

Note: Sign on popularity is reversed (expected popularity adds utility, not subtracts).

## Decision Making

### `choose_firm(firms)`
Given a list of two firms, consumer:
1. Calculates instant utility for each firm
2. Calculates expected future utility for each firm
3. Weights both based on `strategicness`
4. Chooses firm with higher total utility

## Methods

### `get_mismatch_cost(firm)`
Returns distance between consumer and firm locations.

### `get_instant_utility(firm)`
Calculates utility if purchasing from firm right now.

### `get_expected_future_price(firm)`
Returns 3-period moving average of firm's price history.

### `get_expected_future_popularity(firm)`
Returns 3-period moving average of firm's popularity history.

### `get_expected_future_utility(firm)`
Calculates expected utility if firm is purchased from in next period.

## Attributes Tracked

- `purchase_history` - Record of all purchases made
