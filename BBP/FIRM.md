# Firm Class

Agent that sets prices in the market.

## Constructor

```python
Firm(id, location, discount_factor, price_bounds, popularity=0.5)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `id` | int | - | Unique identifier |
| `location` | float | [0, 1] | Position on linear market |
| `discount_factor` | float | [0, 1] | Weight on future profits (0=myopic, 1=patient) |
| `price_bounds` | tuple | - | (min_price, max_price) allowed prices |
| `popularity` | float | [0, 1] | Brand strength (default 0.5) |

## Methods

### `get_price()`
Returns the most recent price set by the firm, or `None` if no price set yet.

### `get_popularity()`
Returns current brand popularity.

### `update_popularity(change)`
Adjusts popularity by the given amount.

### `choose_price(observation)`
Selects a price based on market observation. Stores price in history and returns it.

**Currently**: Randomly samples within price bounds.

### `get_price_trend()`
Returns list of all historical prices.

### `get_popularity_trend()`
Returns list of all historical popularity values.

## Attributes Tracked

- `price_history` - All prices ever set
- `profit_history` - All profits earned
- `popularity_history` - Popularity evolution
