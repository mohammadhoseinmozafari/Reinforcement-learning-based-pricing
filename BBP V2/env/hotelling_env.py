"""
PettingZoo ParallelEnv for Hotelling duopoly with hierarchical BBP control.

Implements two-agent competitive environment compatible with
Stable-Baselines3 and CleanRL through gymnasium integration.

Architecture:
- Strategy Controller (SC): Decides pricing regime (Uniform vs BBP) every K steps
- Pricing Controller (PC): Optimizes prices every step
- Hierarchical action space with Dict structure for both controllers
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from gymnasium import spaces

from pettingzoo import ParallelEnv
from env.target_system import HotellingMarket
from config.constants import (
    AGENT_IDS,
    EPISODE_LENGTH,
    STRATEGY_CYCLE_LENGTH,
    NUM_STRATEGY_ACTIONS,
    PRICING_ACTION_DIM,
    PRICE_UNIFORM_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_BBP_OLD_MAX,
    NUM_CONSUMERS,
    RANDOM_SEED,
    CONSUMER_FORESIGHT_HORIZON,
    MAX_HISTORY_LENGTH,
    OBS_SC_POPULARITY,
    OBS_SC_POPULARITY_CHANGE,
    OBS_SC_RETENTION_RATE,
    OBS_SC_PROFIT_TREND,
    OBS_SC_RELATIVE_POPULARITY,
    OBS_SC_COMPETITOR_REGIME,
    OBS_SC_TIME_PROGRESS,
    OBS_PC_POPULARITY,
    OBS_PC_NEW_OLD_RATIO,
    OBS_PC_OWN_PREV_PRICES,
    OBS_PC_COMP_PREV_PRICES,
    OBS_PC_LAST_DEMAND,
    OBS_PC_REGIME,
    OBS_PC_COMP_REGIME,
    OBS_PC_MARKET_CONCENTRATION,
)


class HotellingDuopolyEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for Hotelling duopoly with hierarchical control.

    Hierarchical Architecture:
    ===========================
    
    Each firm has two levels of control:
    
    1. STRATEGY CONTROLLER (SC):
       - Acts every K timesteps
       - Decision: Choose pricing regime (0=Uniform, 1=BBP)
       - Commits to choice for K steps
       - Observation: Market share, trends, competitor info
    
    2. PRICING CONTROLLER (PC):
       - Acts every timestep
       - Decision: Set prices (uniform_price, price_new, price_old)
       - Observation: Current market conditions, demand, regime
    
    Hierarchical Action/Observation:
    ================================
    - Actions: Dict with "strategy" and "pricing" keys
    - Observations: Dict with state info for SC and PC
    - Rewards: Firm profit (max revenue - cost)
    """

    metadata = {
        "name": "hotelling_duopoly_hierarchical_v0",
        "render_modes": [],
    }

    def __init__(
        self,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
        strategy_cycle: int = STRATEGY_CYCLE_LENGTH,
        seed: Optional[int] = RANDOM_SEED,
    ):
        """
        Initialize hierarchical Hotelling environment.

        Args:
            num_consumers: Number of consumers in market
            episode_length: Horizon T (total steps per episode)
            strategy_cycle: Strategy controller frequency K (steps between decisions)
            seed: Random seed
        """
        super().__init__()

        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.strategy_cycle = strategy_cycle
        self.seed_value = seed

        # Economic simulator
        self.market = HotellingMarket(num_consumers=num_consumers, seed=seed)

        # =====================================
        # TIMING & HIERARCHICAL STATE
        # =====================================
        self.timestep = 0
        self.steps_in_cycle = 0

        # Track regime commitments
        self.regimes = {"firm_0": 0, "firm_1": 0}
        self.regime_commit_steps = {"firm_0": 0, "firm_1": 0}

        # =====================================
        # ACTION SPACES
        # =====================================
        # Hierarchical: strategy (discrete) + pricing (continuous)
        self.action_spaces = {
            agent: spaces.Dict(
                {
                    "strategy": spaces.Discrete(NUM_STRATEGY_ACTIONS),
                    "pricing": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(PRICING_ACTION_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in AGENT_IDS
        }

        # =====================================
        # OBSERVATION SPACES
        # =====================================
        # Strategy controller observations
        # NOTE: All observations use Box spaces for RL compatibility (neural networks expect arrays)
        sc_obs_space = spaces.Dict(
            {
                "market_share": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "popularity_change": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "retention_rate": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "profit_trend": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "relative_popularity": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32),
                "competitor_regime": spaces.Box(0.0, float(NUM_STRATEGY_ACTIONS - 1), shape=(1,), dtype=np.float32),
                "time_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Pricing controller observations
        # NOTE: All observations use Box spaces for RL compatibility (neural networks expect arrays)
        pc_obs_space = spaces.Dict(
            {
                "market_share": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "new_old_ratio": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "own_prices": spaces.Box(0.0, 10.0, shape=(3,), dtype=np.float32),
                "comp_prices": spaces.Box(0.0, 10.0, shape=(3,), dtype=np.float32),
                "last_demand": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "regime": spaces.Box(0.0, float(NUM_STRATEGY_ACTIONS - 1), shape=(1,), dtype=np.float32),
                "competitor_regime": spaces.Box(0.0, float(NUM_STRATEGY_ACTIONS - 1), shape=(1,), dtype=np.float32),
                "market_concentration": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Combined hierarchical observation
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "strategy_controller": sc_obs_space,
                    "pricing_controller": pc_obs_space,
                }
            )
            for agent in AGENT_IDS
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed (optional)
            options: Additional options (unused)

        Returns:
            observations, infos
        """
        if seed is not None:
            self.seed_value = seed
            self.market = HotellingMarket(num_consumers=self.num_consumers, seed=seed)

        # Reset market
        self.market.reset(seed=self.seed_value)

        # Reset timing
        self.timestep = 0
        self.steps_in_cycle = 0

        # Reset regimes
        self.regimes = {"firm_0": 0, "firm_1": 0}
        self.regime_commit_steps = {"firm_0": 0, "firm_1": 0}

        # Get observations
        observations = {agent: self._get_observation(agent) for agent in AGENT_IDS}
        infos = {agent: {} for agent in AGENT_IDS}

        return observations, infos

    def step(
        self, actions: Dict[str, Dict[str, Any]]
    ) -> Tuple[
        Dict[str, Dict],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """
        Execute one environment step.

        Hierarchical Step Logic:
        ========================
        1. Every K steps (steps_in_cycle == 0): Update regime from strategy controller
        2. Every step: Get pricing actions and convert to prices
        3. Execute market step
        4. Calculate rewards (firm profits)
        5. Check episode termination

        Args:
            actions: {agent_id: {"strategy": int, "pricing": np.ndarray}}

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.timestep += 1

        # ===============================================
        # STRATEGY CONTROLLER: Update regime at start of each cycle
        # ===============================================
        if self.steps_in_cycle == 0:
            # Strategy controller makes decisions at the START of each cycle
            for agent in AGENT_IDS:
                strategy_action = actions[agent]["strategy"]
                self.regimes[agent] = int(strategy_action)
                self.regime_commit_steps[agent] = 0

        # Increment counters AFTER checking for strategy decision
        self.steps_in_cycle += 1
        for agent in AGENT_IDS:
            self.regime_commit_steps[agent] += 1

        # Reset cycle if complete (will trigger strategy update next step)
        if self.steps_in_cycle >= self.strategy_cycle:
            self.steps_in_cycle = 0

        # ===============================================
        # PRICING CONTROLLER: Convert actions to prices
        # ===============================================
        prices_dict = {}
        for agent in AGENT_IDS:
            regime = self.regimes[agent]
            pricing_action = actions[agent]["pricing"]

            # Convert normalized action [0,1]^3 to prices
            prices_dict[agent] = self._action_to_prices(pricing_action, regime)

        # ===============================================
        # MARKET EXECUTION
        # ===============================================
        self.market.set_regimes(self.regimes["firm_0"], self.regimes["firm_1"])
        demand_0, demand_1 = self.market.step(prices_dict["firm_0"], prices_dict["firm_1"])

        # ===============================================
        # COMPUTE REWARDS
        # ===============================================
        rewards = {}
        for firm_idx, firm in enumerate(self.market.firms):
            agent_id = AGENT_IDS[firm_idx]
            # Reward = firm profit
            rewards[agent_id] = firm.last_period_profit

        # ===============================================
        # CHECK TERMINATION
        # ===============================================
        terminations = {agent: False for agent in AGENT_IDS}  # No early termination
        truncations = {agent: self.timestep >= self.episode_length for agent in AGENT_IDS}

        # ===============================================
        # GET OBSERVATIONS & INFOS
        # ===============================================
        observations = {agent: self._get_observation(agent) for agent in AGENT_IDS}
        
        infos = {
            agent: {
                "market_share": self.market.firms[idx].market_share,
                "regime": self.regimes[agent],
                "profit": rewards[agent],
            }
            for idx, agent in enumerate(AGENT_IDS)
        }

        return observations, rewards, terminations, truncations, infos

    def _action_to_prices(
        self, pricing_action: np.ndarray, regime: int
    ) -> Dict[str, float]:
        """
        Convert normalized pricing actions to actual prices.

        Action space: [0, 1]^3 -> [price_uniform, price_new, price_old]

        Args:
            pricing_action: Normalized action [0, 1]^3
            regime: Current regime (0=Uniform, 1=BBP)

        Returns:
            Dict with uniform_price, price_new, price_old
        """
        # Extract components
        uniform_action = pricing_action[0]
        new_action = pricing_action[1]
        old_action = pricing_action[2]

        # Scale to actual price ranges
        uniform_price = PRICE_UNIFORM_MIN + uniform_action * (PRICE_UNIFORM_MAX - PRICE_UNIFORM_MIN)
        price_new = PRICE_BBP_NEW_MIN + new_action * (PRICE_BBP_NEW_MAX - PRICE_BBP_NEW_MIN)
        price_old = PRICE_BBP_OLD_MIN + old_action * (PRICE_BBP_OLD_MAX - PRICE_BBP_OLD_MIN)

        # Ensure price_old >= price_new (BBP constraint)
        price_old = max(price_old, price_new)

        return {
            "uniform_price": float(uniform_price),
            "price_new": float(price_new),
            "price_old": float(price_old),
        }

    def _get_observation(self, agent: str) -> Dict[str, Dict]:
        """
        Get hierarchical observation for an agent.

        Returns observations for both strategy and pricing controllers.

        Args:
            agent: Agent ID (firm_0 or firm_1)

        Returns:
            Hierarchical observation dict
        """
        firm_id = 0 if agent == "firm_0" else 1
        firm = self.market.firms[firm_id]
        competitor = self.market.firms[1 - firm_id]

        # Time progress (for curriculum or any time-dependent features)
        time_progress = self.timestep / self.episode_length if self.episode_length > 0 else 0.0

        # ===================================
        # STRATEGY CONTROLLER OBSERVATION
        # ===================================
        # Clamp relative_popularity to avoid inf values that break neural networks
        clamped_relative_popularity = min(firm.relative_popularity, 10.0) if np.isfinite(firm.relative_popularity) else 10.0
        
        sc_observation = {
            "market_share": np.array([firm.market_share], dtype=np.float32),
            "popularity_change": np.array([firm.get_popularity_change()], dtype=np.float32),
            "retention_rate": np.array([firm.retention_rate], dtype=np.float32),
            "profit_trend": np.array([firm.get_profit_trend()], dtype=np.float32),
            "relative_popularity": np.array([clamped_relative_popularity], dtype=np.float32),
            "competitor_regime": np.array([float(competitor.pricing_regime)], dtype=np.float32),
            "time_progress": np.array([time_progress], dtype=np.float32),
        }

        # ===================================
        # PRICING CONTROLLER OBSERVATION
        # ===================================
        pc_observation = {
            "market_share": np.array([firm.market_share], dtype=np.float32),
            "new_old_ratio": np.array([firm.get_new_old_ratio()], dtype=np.float32),
            "own_prices": np.array(
                [firm.uniform_price, firm.price_new, firm.price_old],
                dtype=np.float32,
            ),
            "comp_prices": np.array(
                [competitor.uniform_price, competitor.price_new, competitor.price_old],
                dtype=np.float32,
            ),
            "last_demand": np.array(
                [firm.last_period_quantity / self.num_consumers if self.num_consumers > 0 else 0.0],
                dtype=np.float32,
            ),
            "regime": np.array([float(firm.pricing_regime)], dtype=np.float32),
            "competitor_regime": np.array([float(competitor.pricing_regime)], dtype=np.float32),
            "market_concentration": np.array(
                [self.market.get_market_concentration()], dtype=np.float32
            ),
        }

        # ===================================
        # COMBINE INTO HIERARCHICAL OBS
        # ===================================
        observation = {
            "strategy_controller": sc_observation,
            "pricing_controller": pc_observation,
        }

        return observation

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Close environment."""
        pass
