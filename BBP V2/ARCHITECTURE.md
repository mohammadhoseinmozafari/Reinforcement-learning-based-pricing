# Hierarchical Hotelling Duopoly Environment - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│         PettingZoo ParallelEnv (hotelling_env.py)       │
├─────────────────────────────────────────────────────────┤
│  • Hierarchical action/observation spaces               │
│  • Multi-agent coordination (2 firms)                   │
│  • Episode management and reward calculation            │
│                                                         │
│  Action Flow:                                           │
│  ├─ Every K steps: Strategy Controller chooses regime   │
│  └─ Every step: Pricing Controller optimizes prices     │
│                                                         │
│  Observation: Hierarchical dict with SC and PC info     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│    Economic Simulator (target_system.py)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ HotellingMarket                                  │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  • Manages consumers and firms                   │   │
│  │  • Executes market clearing each period         │   │
│  │  • Calculates profits and market shares         │   │
│  │  • Generates observations                       │   │
│  └──────────┬──────────────────────────────────────┘   │
│             │                                            │
│    ┌────────┴────────┐                                  │
│    ▼                 ▼                                   │
│  ┌────────┐      ┌──────────────┐                      │
│  │Consumer│      │ Firm         │                      │
│  ├────────┤      ├──────────────┤                      │
│  │location│      │regime        │                      │
│  │α (excl)│      │prices        │                      │
│  │β (disc)│      │market_share  │                      │
│  │utility │      │profit        │                      │
│  │history │      │retention     │                      │
│  └────────┘      └──────────────┘                      │
│                                                         │
│  Market Mechanics:                                      │
│  1. Set prices for each firm                           │
│  2. Each consumer chooses firm based on utility        │
│  3. Calculate market shares and profits               │
│  4. Update firm and consumer histories                 │
│  5. Generate observations                              │
│                                                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│      Configuration (config/constants.py)                │
├─────────────────────────────────────────────────────────┤
│  • Consumer parameters (α, β ranges)                    │
│  • Price ranges (uniform, new, old)                     │
│  • Hierarchical control parameters (K)                  │
│  • Observation dimensions                              │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Component Architecture

### 1. Consumer Class

**Attributes:**
```python
Consumer:
  ├─ Location (0-1 on Hotelling line)
  ├─ Alpha (exclusivity preference: 0-1)
  ├─ Beta (strategic foresight: 0-1)
  ├─ Purchase History (list of {firm_id, price})
  └─ Last Firm Choice (binary: 0 or 1)
```

**Behavior:**
```
For each period:
  1. Calculate instant utility for each firm
     U_instant(firm) = V - price - t*distance - α*popularity
  
  2. For each firm, estimate future utilities:
     Expected price (exponential moving average)
     Expected popularity (simple moving average)
     Expected utility = U_instant with expectations
  
  3. Calculate total discounted utility:
     U_total = U_instant + Σ(β^h * U_expected) for h=1..H
  
  4. Choose firm with max utility
```

---

### 2. Firm Class

**Attributes:**
```python
Firm:
  ├─ Location (0 or 1 on line)
  ├─ Regime (0: Uniform, 1: BBP)
  ├─ Prices:
  │  ├─ uniform_price
  │  ├─ price_new (BBP for new customers)
  │  └─ price_old (BBP for established customers)
  ├─ Period State:
  │  ├─ period_demand (total)
  │  ├─ period_demand_new
  │  ├─ period_demand_old
  ├─ Histories:
  │  ├─ demand_history
  │  ├─ market_share_history
  │  ├─ profit_history
  │  ├─ retention_history
  │  └─ price_history
  └─ Current Metrics:
     ├─ market_share
     ├─ retention_rate
     ├─ relative_popularity
     └─ last_period_profit
```

**Key Methods:**
- `get_price_for_consumer(consumer)`: Returns price consumer sees
- `record_purchase(consumer, price)`: Track demand type (new/old)
- `end_period()`: Finalize calculations
- `get_profit_trend()`: Recent vs earlier profit
- `get_popularity_change()`: Market share change

---

### 3. Market Class

**State:**
```python
HotellingMarket:
  ├─ consumers: List[Consumer]
  ├─ firms: List[Firm]
  ├─ current_period: int
  └─ rng: RandomState
```

**Execution Flow (each step):**
```
1. set_prices() / set_regimes()
   - Update firm pricing and regime choices
   
2. Consumer choice phase:
   for each consumer:
     firm_choice = consumer.choose_firm(firm_0, firm_1)
     price = firms[firm_choice].get_price_for_consumer(consumer)
     record_purchase(consumer, price)
   
3. Period finalization:
   for each firm:
     Calculate market share
     Calculate retention rate
     Calculate profit
     Update histories
```

---

### 4. Environment Class (PettingZoo Integration)

**Hierarchical Control Structure:**

```
Time    Step    Strategy    Pricing      Observation
────────────────────────────────────────────────────
0       1       ← SC Act    PC Act       ← SC obs
        2                   PC Act       PC obs only
        3                   PC Act       PC obs only
        ...     (K-1 steps total in cycle)
        10                  PC Act       PC obs only
        
K       11      ← SC Act    PC Act       ← SC obs (new decision)
        12                  PC Act       PC obs only
        ...
```

**Action Interpretation:**
```python
actions[agent] = {
    "strategy": int (0 or 1),          # Every K steps
    "pricing": np.array([a, b, c])     # Every step (normalized 0-1)
}

# Pricing action scaling:
uniform_price = PRICE_UNIFORM_MIN + a * (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
price_new = PRICE_BBP_NEW_MIN + b * (PRICE_BBP_NEW_MAX - PRICE_BBP_NEW_MIN)
price_old = PRICE_BBP_OLD_MIN + c * (PRICE_BBP_OLD_MAX - PRICE_BBP_OLD_MIN)

# Constraint: price_old >= price_new (enforced)
```

---

## Data Flow Diagrams

### Single Market Step

```
┌──────────────────┐
│ set_prices()     │
│ set_regimes()    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│ For each consumer:           │
│  • Calculate utilities       │
│  • Choose firm               │
│  • Record purchase           │
│  • Update history            │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ For each firm:               │
│  • Calculate market share    │
│  • Calculate retention       │
│  • Calculate profit          │
│  • Update histories          │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Return:                      │
│  • Demand for each firm      │
│  • Firm states for obs       │
└──────────────────────────────┘
```

### Observation Generation

```
For each agent:

Strategy Controller Observation:
├─ market_share: firm.market_share
├─ popularity_change: firm.get_popularity_change()
├─ retention_rate: firm.retention_rate
├─ profit_trend: firm.get_profit_trend()
├─ relative_popularity: firm.relative_popularity
├─ competitor_regime: competitor.pricing_regime
└─ time_progress: current_step / episode_length

Pricing Controller Observation:
├─ market_share: firm.market_share
├─ new_old_ratio: firm.period_demand_new / firm.period_demand
├─ own_prices: [uniform, new, old]
├─ comp_prices: [uniform, new, old]
├─ last_demand: firm.period_demand / num_consumers
├─ regime: firm.pricing_regime
├─ competitor_regime: competitor.pricing_regime
└─ market_concentration: HHI
```

---

## Consumer Utility Calculation (Core Economic Model)

### Instant Utility

```
U_instant = V - P - τ * |x_c - x_f| - α * S_f

Where:
  V = BASE_VALUE (10.0)
  P = price shown to consumer
  τ = TRANSPORTATION_COST (1.0)
  x_c = consumer location
  x_f = firm location
  α = consumer exclusivity preference
  S_f = firm market share
```

### Expected Future Utility

```
E[U_future] = V - E[P_f] - τ * |x_c - x_f| - α * E[S_f]

Where expectations are based on:
  E[P_f] = exponential moving average of past prices
  E[S_f] = simple moving average of recent market shares
```

### Total Discounted Utility

```
U_total = U_instant(t) + Σ(β^h * E[U_future(t+h)])
          for h = 1 to CONSUMER_FORESIGHT_HORIZON

Where β = consumer.beta (strategic foresight / discount factor)
```

### Consumer Choice Rule

```
Choose firm_i if U_total(firm_i) >= U_total(firm_j)
for all j ≠ i
```

---

## Established Customer Definition

A consumer is **established** with a firm if they have:
- Purchased from the **same firm** 
- For the **last BBP_RETENTION_PERIODS consecutive periods**
- No breaks in the chain

**Example (BBP_RETENTION_PERIODS = 2):**
```
Purchase history:  [Firm 0, Firm 0, Firm 1, Firm 0, Firm 0]
                    └─────┬─────┘ └─break─┘
                    older part of history

Is established with Firm 0?
  Last 2 purchases: [Firm 0, Firm 0] ✓ YES
  
Is established with Firm 1?
  Last 2 purchases: [Firm 0, Firm 0] ✗ NO
```

---

## Profit Calculation

### Uniform Regime

```
Profit = demand × (uniform_price - MARGINAL_COST)
       = demand × uniform_price  (assuming MARGINAL_COST = 0)
```

### BBP Regime

```
Profit = (demand_new × price_new) + (demand_old × price_old) - costs
       = (demand_new × price_new) + (demand_old × price_old)
       
Where:
  demand_new = consumers without established status
  demand_old = consumers with established status
```

---

## Episode Lifecycle

```
1. Initialize
   ├─ Create market with random consumers
   ├─ Reset firm states
   └─ Set timestep = 0

2. Reset (start of episode)
   ├─ Reset all consumers
   ├─ Reset all firms
   ├─ Generate initial observations
   └─ Return (obs, info)

3. Per-timestep Loop (for 0 to EPISODE_LENGTH-1)
   ├─ Receive actions
   ├─ Update regime every K steps (strategy controller)
   ├─ Execute market step
   ├─ Calculate rewards (profits)
   ├─ Generate observations
   ├─ Check truncation: timestep >= EPISODE_LENGTH
   └─ Return (obs, rewards, terms, truncs, info)

4. Termination
   └─ Episode ends when timestep >= EPISODE_LENGTH
      (No early termination, only time-based truncation)
```

---

## Space Definitions

### Action Space

```python
Dict({
    "strategy": Discrete(2),           # Regime choice
    "pricing": Box(0, 1, (3,), float32)  # Normalized prices
})
```

### Observation Space (Hierarchical)

```python
Dict({
    "strategy_controller": Dict({
        "market_share": Box(0, 1, (1,)),
        "popularity_change": Box(-1, 1, (1,)),
        "retention_rate": Box(0, 1, (1,)),
        "profit_trend": Box(-1, 1, (1,)),
        "relative_popularity": Box(0, 10, (1,)),
        "competitor_regime": Discrete(2),
        "time_progress": Box(0, 1, (1,))
    }),
    "pricing_controller": Dict({
        "market_share": Box(0, 1, (1,)),
        "new_old_ratio": Box(0, 1, (1,)),
        "own_prices": Box(0, 10, (3,)),
        "comp_prices": Box(0, 10, (3,)),
        "last_demand": Box(0, 1, (1,)),
        "regime": Discrete(2),
        "competitor_regime": Discrete(2),
        "market_concentration": Box(0, 1, (1,))
    })
})
```

---

## Key Parameters and Their Effects

| Parameter | Range | Effect |
|-----------|-------|--------|
| CONSUMER_FORESIGHT_HORIZON (H) | 1-5 | How far ahead consumers look |
| BBP_RETENTION_PERIODS | 1-3 | How many periods to be "established" |
| STRATEGY_CYCLE_LENGTH (K) | 5-20 | How often strategy changes |
| ALPHA (α) | 0-1 | Consumer exclusivity seeking |
| BETA (β) | 0-1 | Consumer discount factor |
| NUM_CONSUMERS | 20-100 | Market size |

---

## Extension Points

### Add to Consumer Utility
- Habit formation
- Brand loyalty
- Quality signals
- Advertising effects

### Add to Firm Strategy
- Capacity constraints
- Production costs
- Quality decisions
- Advertising spend

### Add to Market
- Entry/exit dynamics
- Multi-product firms
- Spatial distribution effects
- Information asymmetry

---

This architecture supports research into:
1. **When is BBP profitable?** (regime optimization)
2. **Price discrimination effects** (consumer surplus, welfare)
3. **Market dynamics** (stability, convergence)
4. **Strategic interaction** (competitive effects)
5. **Consumer heterogeneity** (α, β effects)

