import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.constants import EPISODE_LENGTH
from env.uniform_pricing_env import make_uniform_pricing_env
from train.uniform_training.uniform_training import TrainingConfig
MIN_PRICE = 1.5
MAX_PRICE = 3.5

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# =============================================================================
# GRID SEARCH (your code, slightly improved)
# =============================================================================

grid = np.arange(MIN_PRICE, MAX_PRICE, 0.01)
prices = []
profits = []
market_shares = []
demands = []

config = TrainingConfig(opponent_type='passive_uniform')

for price in grid:
    env = make_uniform_pricing_env(
        opponent=config.opponent_type,
        num_consumers=config.num_consumers,
        episode_length=config.episode_length,
        seed=config.seed,
    )
    env.reset()
    
    normalized_price = env._price_to_normalized(price)
    action = np.array([normalized_price], dtype=np.float32)
    profit = 0.0
    demand = 0.0
    market_share = 0.0
    for i in range(EPISODE_LENGTH):
    
    
    
    
   
    
        next_state, reward, terminated, truncated, info = env.step(action)
        profit+=info.get('profit', 0.0)
        demand+= info.get('demand', 0)
        market_share+= info.get('market_share', 0.0)

    avg_market_share = market_share/ EPISODE_LENGTH
    avg_demand = demand / EPISODE_LENGTH
    prices.append(price)
    profits.append(profit)
    market_shares.append(avg_market_share)
    demands.append(avg_demand)
    
    env.close()

# Convert to numpy arrays
prices = np.array(prices)
profits = np.array(profits)
market_shares = np.array(market_shares)
demands = np.array(demands)

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Market Response to Uniform Pricing', fontsize=16, fontweight='bold')

# ---- 1. Profit vs Price ----
ax = axes[0, 0]
ax.plot(prices, profits, 'b-', lw=2, label='Profit')
ax.fill_between(prices, 0, profits, alpha=0.15, color='blue')

# Mark optimal price
optimal_idx = np.argmax(profits)
optimal_price = prices[optimal_idx]
optimal_profit = profits[optimal_idx]

ax.axvline(x=optimal_price, color='red', linestyle='--', lw=1.5, alpha=0.7)
ax.scatter([optimal_price], [optimal_profit], color='red', s=100, zorder=5,
           label=f'Optimal: p*={optimal_price:.2f}, π*={optimal_profit:.2f}')

ax.set_xlabel('Price', fontsize=12)
ax.set_ylabel('Profit per Period', fontsize=12)
ax.set_title('Profit vs Price', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# ---- 2. Market Share vs Price ----
ax = axes[0, 1]
ax.plot(prices, market_shares, 'g-', lw=2, label='Market Share')
ax.fill_between(prices, 0, market_shares, alpha=0.15, color='green')

# Mark share at optimal price
optimal_share = market_shares[optimal_idx]
ax.scatter([optimal_price], [optimal_share], color='red', s=100, zorder=5,
           label=f'At p*: share={optimal_share:.2f}')

# Reference lines
ax.axhline(y=0.5, color='gray', linestyle=':', lw=1, alpha=0.5, label='Equal split (0.5)')
ax.axhline(y=0.7, color='orange', linestyle=':', lw=1, alpha=0.5, label='Dominant (0.7)')

ax.set_xlabel('Price', fontsize=12)
ax.set_ylabel('Market Share', fontsize=12)
ax.set_title('Market Share vs Price', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# ---- 3. Profit & Share Together (Dual Axis) ----
ax1 = axes[1, 0]
color_profit = 'tab:blue'
color_share = 'tab:green'

ax1.plot(prices, profits, color=color_profit, lw=2, label='Profit')
ax1.set_xlabel('Price', fontsize=12)
ax1.set_ylabel('Profit', fontsize=12, color=color_profit)
ax1.tick_params(axis='y', labelcolor=color_profit)

ax2 = ax1.twinx()
ax2.plot(prices, market_shares, color=color_share, lw=2, linestyle='--', label='Market Share')
ax2.set_ylabel('Market Share', fontsize=12, color=color_share)
ax2.tick_params(axis='y', labelcolor=color_share)

# Mark optimal
ax1.axvline(x=optimal_price, color='red', linestyle='--', lw=1.5, alpha=0.7)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title('Profit and Market Share vs Price', fontsize=13)
ax1.grid(True, alpha=0.3)

# ---- 4. Revenue Decomposition (Price × Demand) ----
ax = axes[1, 1]
revenue = prices * demands
ax.plot(prices, revenue, 'purple', lw=2, label='Revenue (P × Q)')
ax.plot(prices, profits, 'blue', lw=2, alpha=0.5, label='Profit')

# Theoretical max revenue
max_rev_idx = np.argmax(revenue)
ax.scatter([prices[max_rev_idx]], [revenue[max_rev_idx]], color='purple', s=80, zorder=5,
           label=f'Max Revenue: p={prices[max_rev_idx]:.2f}')

ax.set_xlabel('Price', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Revenue & Profit Decomposition', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('passive uniform')
# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)
print(f"Price range: [{MIN_PRICE:.2f}, {MAX_PRICE:.2f}]")
print(f"Step size: 0.01 ({len(grid)} points)")
print()
print(f"Optimal price:       p* = {optimal_price:.2f}")
print(f"Maximum profit:      π* = {optimal_profit:.2f}")
print(f"Market share at p*:  s* = {optimal_share:.3f}")
print(f"Demand at p*:        D* = {demands[optimal_idx]:.0f} consumers")
print()
print(f"Revenue-maximizing price: p_rev = {prices[max_rev_idx]:.2f}")
print(f"Maximum revenue:          R_max = {revenue[max_rev_idx]:.2f}")