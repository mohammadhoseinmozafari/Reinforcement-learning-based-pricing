"""
Constants and hyperparameters for Hotelling duopoly with BBP.

Unified configuration for economic model and RL environment.
"""

# =============================================================================
# ENVIRONMENT PARAMETERS
# =============================================================================

NUM_CONSUMERS = 50
EPISODE_LENGTH = 100

# Hotelling line
HOTELLING_LEFT = 0.0
HOTELLING_RIGHT = 1.0
FIRM_A_LOCATION = 0.0
FIRM_B_LOCATION = 1.0

# Economic parameters
TRANSPORTATION_COST = 1.0
BASE_VALUE = 10
MARGINAL_COST = 0.0

# =============================================================================
# CONSUMER BEHAVIOR
# =============================================================================

# Exclusivity seeking (alpha): how much consumer dislikes popular firms
ALPHA_MIN = 0.0
ALPHA_MAX = 1.0

# Strategic foresight (beta): discount factor for future utility
BETA_MIN = 0.0
BETA_MAX = 1.0

# Horizon for expected utility calculation
CONSUMER_FORESIGHT_HORIZON = 3  # H periods ahead consumers look

# =============================================================================
# PRICING PARAMETERS
# =============================================================================

PRICE_UNIFORM_MIN = 0.5
PRICE_UNIFORM_MAX = 5.0

PRICE_BBP_NEW_MIN = 0.5
PRICE_BBP_NEW_MAX = 4.0

PRICE_BBP_OLD_MIN = 1.0
PRICE_BBP_OLD_MAX = 5.0

# Consumer is "established" if bought from same firm for last N periods
BBP_RETENTION_PERIODS = 2

# =============================================================================
# HIERARCHY PARAMETERS
# =============================================================================

STRATEGY_CYCLE_LENGTH = 10  # K: strategy acts every K steps
FORCED_DEFAULT_CYCLE = 1
NUM_STRATEGY_ACTIONS = 2    # {0: Uniform, 1: BBP}
PRICING_ACTION_DIM = 3      # [uniform_price, price_new, price_old]

# =============================================================================
# OBSERVATION PARAMETERS - STRATEGY CONTROLLER
# =============================================================================
# S_SC(t) = (popularity, popularity_change, retention_rate, profit_trend, 
#            relative_popularity, competitor_regime, time_in_episode)

OBS_SC_POPULARITY = 1
OBS_SC_POPULARITY_CHANGE = 1
OBS_SC_RETENTION_RATE = 1
OBS_SC_PROFIT_TREND = 1
OBS_SC_RELATIVE_POPULARITY = 1
OBS_SC_COMPETITOR_REGIME = 1  # Binary: 0 or 1
OBS_SC_TIME_PROGRESS = 1      # Episode progress (0 to 1)

TOTAL_OBS_SC_DIM = (
    OBS_SC_POPULARITY + OBS_SC_POPULARITY_CHANGE + OBS_SC_RETENTION_RATE +
    OBS_SC_PROFIT_TREND + OBS_SC_RELATIVE_POPULARITY + OBS_SC_COMPETITOR_REGIME +
    OBS_SC_TIME_PROGRESS
)

# =============================================================================
# OBSERVATION PARAMETERS - PRICING CONTROLLER
# =============================================================================
# S_PC(t) = (popularity, new_old_ratio, own_prev_prices, competitor_prev_prices,
#            last_demand, regime, competitor_regime, market_concentration)

OBS_PC_POPULARITY = 1
OBS_PC_NEW_OLD_RATIO = 1
OBS_PC_OWN_PREV_PRICES = 3    # [uniform, new, old]
OBS_PC_COMP_PREV_PRICES = 3   # [uniform, new, old] 
OBS_PC_LAST_DEMAND = 1
OBS_PC_REGIME = 1             # Own regime (0 or 1)
OBS_PC_COMP_REGIME = 1        # Competitor regime
OBS_PC_MARKET_CONCENTRATION = 1  # HHI or similar

TOTAL_OBS_PC_DIM = (
    OBS_PC_POPULARITY + OBS_PC_NEW_OLD_RATIO + OBS_PC_OWN_PREV_PRICES +
    OBS_PC_COMP_PREV_PRICES + OBS_PC_LAST_DEMAND + OBS_PC_REGIME +
    OBS_PC_COMP_REGIME + OBS_PC_MARKET_CONCENTRATION
)

# =============================================================================
# HISTORY TRACKING
# =============================================================================

MAX_HISTORY_LENGTH = 100  # Max periods to track for trends

# =============================================================================
# AGENT IDs
# =============================================================================

AGENT_IDS = ["firm_0", "firm_1"]

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# DEBUG MODE
# =============================================================================

DEBUG_MODE = False  # Set to True for verbose output
