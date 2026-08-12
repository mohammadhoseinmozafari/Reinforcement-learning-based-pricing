"""Universal Gym environment for agent-controlled pricing regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.constants import (
    EPISODE_LENGTH,
    MARGINAL_COST,
    NUM_CONSUMERS,
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_OLD_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_UNIFORM_MIN,
)
from env.consumer_population import ConsumerPopulationGenerator
from env.models import HotellingMarket
from env.opponent_policies import (
    OpponentObservation,
    OpponentPolicy,
    PreviousMarketState,
    PriceVector,
    create_preset_opponent,
)
from env.pricing_contracts import (
    PricingAction,
    PricingActionCodec,
    PricingObservationCodec,
    PricingRegime,
)
from train.universal_pricing_protocol import (
    BalancedOpponentSchedule,
    ConsumerPopulationSpec,
    EpisodeSeedBundle,
    OpponentEpisodeAssignment,
    OpponentFamily,
    OpponentPoolConfig,
    ProtocolConfigError,
    RunSeedBundle,
    SeedDeriver,
)


@dataclass(frozen=True)
class UniversalPricingEpisodeContext:
    """Resolved seeds and scheduled opponent for one episode."""

    episode_index: int
    episode_seed_bundle: EpisodeSeedBundle
    opponent_assignment: OpponentEpisodeAssignment

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_index, int)
            or isinstance(self.episode_index, bool)
            or self.episode_index < 0
        ):
            raise ProtocolConfigError(
                "episode_index must be a nonnegative integer"
            )
        if self.opponent_assignment.episode_index != self.episode_index:
            raise ProtocolConfigError(
                "Opponent assignment episode index does not match context"
            )
        if (
            self.opponent_assignment.opponent_seed
            != self.episode_seed_bundle.opponent_seed
        ):
            raise ProtocolConfigError(
                "Opponent assignment and episode seed bundle disagree"
            )


class UniversalPricingEpisodeContextFactory:
    """Resolve deterministic seeds and opponents for explicit episode indices."""

    def __init__(
        self,
        run_seed_bundle: RunSeedBundle,
        opponent_pool: OpponentPoolConfig,
    ) -> None:
        self.run_seed_bundle = run_seed_bundle
        self.opponent_pool = opponent_pool
        self.opponent_schedule = BalancedOpponentSchedule(
            opponent_pool,
            run_seed_bundle.opponent_schedule_seed,
        )

    def create(self, episode_index: int) -> UniversalPricingEpisodeContext:
        episode_seed_bundle = SeedDeriver.derive_episode_bundle(
            self.run_seed_bundle,
            episode_index,
        )
        assignment = self.opponent_schedule.assignment(episode_index)
        return UniversalPricingEpisodeContext(
            episode_index=episode_index,
            episode_seed_bundle=episode_seed_bundle,
            opponent_assignment=assignment,
        )


class PricingPriceTransform:
    """Transform normalized pricing controls and bounded market prices."""

    def __init__(
        self,
        *,
        uniform_min: float = PRICE_UNIFORM_MIN,
        uniform_max: float = PRICE_UNIFORM_MAX,
        bbp_new_min: float = PRICE_BBP_NEW_MIN,
        bbp_new_max: float = PRICE_BBP_NEW_MAX,
        bbp_old_min: float = PRICE_BBP_OLD_MIN,
        bbp_old_max: float = PRICE_BBP_OLD_MAX,
    ) -> None:
        self.uniform_min = float(uniform_min)
        self.uniform_max = float(uniform_max)
        self.bbp_new_min = float(bbp_new_min)
        self.bbp_new_max = float(bbp_new_max)
        self.bbp_old_min = float(bbp_old_min)
        self.bbp_old_max = float(bbp_old_max)
        if not (
            self.uniform_min < self.uniform_max
            and self.bbp_new_min < self.bbp_new_max
            and self.bbp_old_min < self.bbp_old_max
            and self.bbp_new_max <= self.bbp_old_max
        ):
            raise ProtocolConfigError("Invalid universal pricing bounds")

    @staticmethod
    def _control_fraction(control: float) -> float:
        value = float(control)
        if not np.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("Normalized price control must be in [-1, 1]")
        return (value + 1.0) / 2.0

    @staticmethod
    def _normalize_bounded(
        value: float,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        value = float(value)
        if (
            not np.isfinite(value)
            or value < lower_bound - 1e-9
            or value > upper_bound + 1e-9
        ):
            raise ValueError(
                f"Price must be in [{lower_bound}, {upper_bound}]"
            )
        clipped = float(np.clip(value, lower_bound, upper_bound))
        return 2.0 * (
            (clipped - lower_bound) / (upper_bound - lower_bound)
        ) - 1.0

    def controls_to_prices(self, action: PricingAction) -> PriceVector:
        uniform_fraction = self._control_fraction(action.uniform_control)
        new_fraction = self._control_fraction(action.bbp_new_control)
        premium_fraction = self._control_fraction(
            action.bbp_premium_control
        )
        uniform_price = self.uniform_min + uniform_fraction * (
            self.uniform_max - self.uniform_min
        )
        new_price = self.bbp_new_min + new_fraction * (
            self.bbp_new_max - self.bbp_new_min
        )
        old_floor = max(self.bbp_old_min, new_price)
        old_price = old_floor + premium_fraction * (
            self.bbp_old_max - old_floor
        )
        return PriceVector(
            uniform=float(uniform_price),
            new=float(new_price),
            old=float(old_price),
        )

    def prices_to_controls(self, prices: PriceVector) -> np.ndarray:
        uniform_control = self._normalize_bounded(
            prices.uniform,
            self.uniform_min,
            self.uniform_max,
        )
        new_control = self._normalize_bounded(
            prices.new,
            self.bbp_new_min,
            self.bbp_new_max,
        )
        old_floor = max(self.bbp_old_min, float(prices.new))
        if (
            not np.isfinite(prices.old)
            or prices.old < old_floor - 1e-9
            or prices.old > self.bbp_old_max + 1e-9
        ):
            raise ValueError(
                "BBP old-customer price must be between its conditional "
                "floor and maximum"
            )
        premium_fraction = (
            (float(np.clip(prices.old, old_floor, self.bbp_old_max)) - old_floor)
            / (self.bbp_old_max - old_floor)
        )
        premium_control = 2.0 * premium_fraction - 1.0
        return np.asarray(
            [uniform_control, new_control, premium_control],
            dtype=np.float32,
        )

    def prices_to_observation_features(
        self,
        prices: PriceVector,
    ) -> tuple[float, float, float]:
        return (
            self._normalize_bounded(
                prices.uniform,
                self.uniform_min,
                self.uniform_max,
            ),
            self._normalize_bounded(
                prices.new,
                self.bbp_new_min,
                self.bbp_new_max,
            ),
            self._normalize_bounded(
                prices.old,
                self.bbp_old_min,
                self.bbp_old_max,
            ),
        )


@dataclass(frozen=True)
class RegimeDecisionResult:
    """Proposed and effective regime facts for one transition."""

    proposed_regime: PricingRegime
    effective_regime: PricingRegime
    regime_decision_allowed: bool
    regime_changed: bool

    @property
    def regime_decision_mask(self) -> float:
        return 1.0 if self.regime_decision_allowed else 0.0


class RegimeCommitmentController:
    """Enforce fixed-period commitments with an immediate first decision."""

    def __init__(self, commitment_length: int) -> None:
        if (
            not isinstance(commitment_length, int)
            or isinstance(commitment_length, bool)
            or commitment_length <= 0
        ):
            raise ProtocolConfigError(
                "commitment_length must be a positive integer"
            )
        self.commitment_length = commitment_length
        self.reset()

    def reset(self) -> None:
        self.effective_regime = PricingRegime.UNIFORM
        self._remaining_periods = 0

    @property
    def regime_decision_allowed(self) -> bool:
        return self._remaining_periods == 0

    @property
    def commitment_progress(self) -> float:
        if self.regime_decision_allowed:
            return 1.0
        elapsed_periods = self.commitment_length - self._remaining_periods
        return float(elapsed_periods / self.commitment_length)

    def decide(
        self,
        proposed_regime: PricingRegime,
    ) -> RegimeDecisionResult:
        proposed_regime = PricingRegime(proposed_regime)
        allowed = self.regime_decision_allowed
        previous_regime = self.effective_regime
        if allowed:
            self.effective_regime = proposed_regime
            self._remaining_periods = self.commitment_length
        return RegimeDecisionResult(
            proposed_regime=proposed_regime,
            effective_regime=self.effective_regime,
            regime_decision_allowed=allowed,
            regime_changed=allowed
            and self.effective_regime is not previous_regime,
        )

    def advance_period(self) -> None:
        if self._remaining_periods <= 0:
            raise RuntimeError(
                "Cannot advance regime commitment before a decision"
            )
        self._remaining_periods -= 1


class UniversalPricingObservationBuilder:
    """Build the frozen normalized 18-feature universal observation."""

    def __init__(
        self,
        price_transform: PricingPriceTransform,
        episode_length: int,
    ) -> None:
        if episode_length <= 0:
            raise ProtocolConfigError("episode_length must be positive")
        self.price_transform = price_transform
        self.episode_length = episode_length

    @staticmethod
    def _normalize_ratio(value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("Observation ratio must be finite")
        return float(2.0 * np.clip(value, 0.0, 1.0) - 1.0)

    @staticmethod
    def _normalize_regime(regime: int) -> float:
        return -1.0 if PricingRegime(regime) is PricingRegime.UNIFORM else 1.0

    def build(
        self,
        market: HotellingMarket,
        timestep: int,
        commitment_controller: RegimeCommitmentController,
    ) -> np.ndarray:
        agent = market.firms[0]
        opponent = market.firms[1]
        if timestep == 0:
            own_market_share = 0.5
            opponent_market_share = 0.5
        else:
            own_market_share = agent.market_share
            opponent_market_share = opponent.market_share

        own_price_features = self.price_transform.prices_to_observation_features(
            PriceVector(
                uniform=agent.uniform_price,
                new=agent.price_new,
                old=agent.price_old,
            )
        )
        opponent_price_features = (
            self.price_transform.prices_to_observation_features(
                PriceVector(
                    uniform=opponent.uniform_price,
                    new=opponent.price_new,
                    old=opponent.price_old,
                )
            )
        )
        demand_ratio = (
            agent.last_period_quantity / market.num_consumers
            if market.num_consumers > 0
            else 0.0
        )
        features = {
            "own_market_share": self._normalize_ratio(own_market_share),
            "opponent_market_share": self._normalize_ratio(
                opponent_market_share
            ),
            "own_uniform_price": own_price_features[0],
            "own_bbp_new_price": own_price_features[1],
            "own_bbp_old_price": own_price_features[2],
            "opponent_uniform_price": opponent_price_features[0],
            "opponent_bbp_new_price": opponent_price_features[1],
            "opponent_bbp_old_price": opponent_price_features[2],
            "own_demand_ratio": self._normalize_ratio(demand_ratio),
            "own_new_customer_ratio": self._normalize_ratio(
                agent.get_new_old_ratio()
            ),
            "own_retention_rate": self._normalize_ratio(
                agent.retention_rate
            ),
            "own_regime": self._normalize_regime(agent.pricing_regime),
            "opponent_regime": self._normalize_regime(
                opponent.pricing_regime
            ),
            "own_profit_trend": float(agent.get_profit_trend()),
            "own_popularity_change": float(agent.get_popularity_change()),
            "episode_progress": self._normalize_ratio(
                min(timestep / self.episode_length, 1.0)
            ),
            "regime_commitment_progress": self._normalize_ratio(
                commitment_controller.commitment_progress
            ),
            "regime_decision_allowed": (
                1.0
                if commitment_controller.regime_decision_allowed
                else -1.0
            ),
        }
        return PricingObservationCodec.encode(features)


class ProfitRewardNormalizer:
    """Normalize own raw economic profit against the fixed theoretical bound."""

    def __init__(
        self,
        num_consumers: int,
        maximum_price: float = PRICE_BBP_OLD_MAX,
        marginal_cost: float = MARGINAL_COST,
    ) -> None:
        self.num_consumers = int(num_consumers)
        self.maximum_price = float(maximum_price)
        self.marginal_cost = float(marginal_cost)
        self.maximum_step_profit = self.num_consumers * (
            self.maximum_price - self.marginal_cost
        )
        if self.num_consumers <= 0 or self.maximum_step_profit <= 0:
            raise ProtocolConfigError("Invalid reward normalization bound")

    def normalize(self, raw_profit: float) -> float:
        raw_profit = float(raw_profit)
        if not np.isfinite(raw_profit):
            raise ValueError("Raw profit must be finite")
        return float(raw_profit / self.maximum_step_profit)


class UniversalPricingEnv(gym.Env):
    """Single-agent universal pricing environment for the research protocol."""

    metadata = {
        "render_modes": [],
        "name": "universal_pricing_v1",
    }

    INITIAL_PRICES = PriceVector(uniform=2.75, new=2.25, old=3.0)

    def __init__(
        self,
        *,
        consumer_population_spec: ConsumerPopulationSpec,
        opponent_pool: OpponentPoolConfig,
        run_seed_bundle: RunSeedBundle,
        regime_commitment_length: int,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
        consumer_population_generator: ConsumerPopulationGenerator | None = None,
        price_transform: PricingPriceTransform | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(num_consumers, int)
            or isinstance(num_consumers, bool)
            or num_consumers <= 0
        ):
            raise ProtocolConfigError(
                "num_consumers must be a positive integer"
            )
        if (
            not isinstance(episode_length, int)
            or isinstance(episode_length, bool)
            or episode_length <= 0
        ):
            raise ProtocolConfigError(
                "episode_length must be a positive integer"
            )
        self.consumer_population_spec = consumer_population_spec
        self.opponent_pool = opponent_pool
        self.run_seed_bundle = run_seed_bundle
        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.consumer_population_generator = (
            consumer_population_generator or ConsumerPopulationGenerator()
        )
        self.price_transform = price_transform or PricingPriceTransform()
        self.episode_context_factory = UniversalPricingEpisodeContextFactory(
            run_seed_bundle,
            opponent_pool,
        )
        self.regime_commitment_controller = RegimeCommitmentController(
            regime_commitment_length
        )
        self.observation_builder = UniversalPricingObservationBuilder(
            self.price_transform,
            episode_length,
        )
        self.reward_normalizer = ProfitRewardNormalizer(num_consumers)
        self.market = HotellingMarket(
            num_consumers=num_consumers,
            seed=run_seed_bundle.consumer_population_seed,
        )

        self.action_space = spaces.Dict(
            {
                "regime": spaces.Discrete(2),
                "price_controls": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(PricingObservationCodec.FEATURE_COUNT,),
            dtype=np.float32,
        )

        self._episode_context: UniversalPricingEpisodeContext | None = None
        self._opponent_policy: OpponentPolicy | None = None
        self._timestep = 0
        self._raw_agent_profits: list[float] = []
        self._raw_opponent_profits: list[float] = []
        self._normalized_rewards: list[float] = []

    @property
    def episode_context(self) -> UniversalPricingEpisodeContext:
        if self._episode_context is None:
            raise RuntimeError("Environment must be reset before use")
        return self._episode_context

    @property
    def opponent_policy(self) -> OpponentPolicy:
        if self._opponent_policy is None:
            raise RuntimeError("Environment must be reset before use")
        return self._opponent_policy

    def _resolve_episode_context(
        self,
        seed: int | None,
        options: Mapping[str, Any] | None,
    ) -> tuple[UniversalPricingEpisodeContext, str]:
        options = {} if options is None else dict(options)
        unknown_options = set(options) - {"episode_index"}
        if unknown_options:
            raise ValueError(
                "Unknown reset option(s): "
                + ", ".join(sorted(unknown_options))
            )
        has_episode_index = "episode_index" in options
        if seed is not None and has_episode_index:
            raise ValueError(
                "Ad-hoc seed and protocol episode_index are mutually exclusive"
            )
        if seed is not None:
            if (
                not isinstance(seed, (int, np.integer))
                or isinstance(seed, (bool, np.bool_))
                or int(seed) < 0
            ):
                raise ValueError("seed must be a nonnegative integer")
            temporary_run_bundle = SeedDeriver.derive_run_bundle(int(seed))
            temporary_factory = UniversalPricingEpisodeContextFactory(
                temporary_run_bundle,
                self.opponent_pool,
            )
            return temporary_factory.create(0), "ad_hoc"
        episode_index = options.get("episode_index", 0)
        return self.episode_context_factory.create(episode_index), "protocol"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        context, seed_source = self._resolve_episode_context(seed, options)
        population = self.consumer_population_generator.generate(
            self.consumer_population_spec,
            self.num_consumers,
            context.episode_seed_bundle.consumer_seed,
        )
        self.market.install_consumer_population(population)
        self.market.set_prices(
            {
                "uniform_price": self.INITIAL_PRICES.uniform,
                "price_new": self.INITIAL_PRICES.new,
                "price_old": self.INITIAL_PRICES.old,
            },
            {
                "uniform_price": self.INITIAL_PRICES.uniform,
                "price_new": self.INITIAL_PRICES.new,
                "price_old": self.INITIAL_PRICES.old,
            },
        )

        assignment = context.opponent_assignment
        opponent_policy = create_preset_opponent(
            assignment.policy_name,
            seed=assignment.opponent_seed,
        )
        opponent_policy.reset(seed=assignment.opponent_seed)
        expected_regime = (
            PricingRegime.UNIFORM
            if assignment.opponent_family is OpponentFamily.UNIFORM
            else PricingRegime.BBP
        )
        if PricingRegime(opponent_policy.regime) is not expected_regime:
            raise ProtocolConfigError(
                "Scheduled opponent family does not match policy regime"
            )

        self._episode_context = context
        self._opponent_policy = opponent_policy
        self._timestep = 0
        self.regime_commitment_controller.reset()
        self.market.set_regimes(
            PricingRegime.UNIFORM,
            expected_regime,
        )
        self._raw_agent_profits = []
        self._raw_opponent_profits = []
        self._normalized_rewards = []

        observation = self.observation_builder.build(
            self.market,
            self._timestep,
            self.regime_commitment_controller,
        )
        info = {
            "episode_index": context.episode_index,
            "consumer_seed": context.episode_seed_bundle.consumer_seed,
            "opponent_seed": context.episode_seed_bundle.opponent_seed,
            "opponent_policy_name": assignment.policy_name,
            "opponent_family": assignment.opponent_family.value,
            "seed_source": seed_source,
        }
        return observation, info

    def _agent_submission(
        self,
        transformed_prices: PriceVector,
        effective_regime: PricingRegime,
    ) -> tuple[dict[str, float], PriceVector]:
        agent = self.market.firms[0]
        if effective_regime is PricingRegime.UNIFORM:
            market_prices = {
                "uniform_price": transformed_prices.uniform,
            }
            submission = PriceVector(
                uniform=transformed_prices.uniform,
                new=agent.price_new,
                old=agent.price_old,
            )
        else:
            market_prices = {
                "price_new": transformed_prices.new,
                "price_old": transformed_prices.old,
            }
            submission = PriceVector(
                uniform=agent.uniform_price,
                new=transformed_prices.new,
                old=transformed_prices.old,
            )
        return market_prices, submission

    def _build_opponent_observation(
        self,
        agent_submission: PriceVector,
    ) -> OpponentObservation:
        opponent = self.market.firms[1]
        agent = self.market.firms[0]
        if self._timestep == 0:
            opponent_market_share = 0.5
            agent_market_share = 0.5
        else:
            opponent_market_share = opponent.market_share
            agent_market_share = agent.market_share
        return OpponentObservation(
            previous=PreviousMarketState(
                own_market_share=opponent_market_share,
                competitor_market_share=agent_market_share,
                own_prices=PriceVector(
                    uniform=opponent.uniform_price,
                    new=opponent.price_new,
                    old=opponent.price_old,
                ),
                competitor_prices=PriceVector(
                    uniform=agent.uniform_price,
                    new=agent.price_new,
                    old=agent.price_old,
                ),
                own_demand_ratio=(
                    opponent.last_period_quantity / self.num_consumers
                    if self.num_consumers > 0
                    else 0.0
                ),
                own_new_customer_ratio=opponent.get_new_old_ratio(),
            ),
            competitor_submission=agent_submission,
            competitor_established_share=self.market.get_established_share(0),
            own_regime=opponent.pricing_regime,
            competitor_regime=agent.pricing_regime,
            decision_period=self._timestep,
            state_period=self._timestep - 1,
            episode_length=self.episode_length,
        )

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_context is None or self._opponent_policy is None:
            raise RuntimeError("Environment must be reset before step")
        structured_action = PricingActionCodec.from_gym(action)
        decision = self.regime_commitment_controller.decide(
            structured_action.regime
        )
        transformed_prices = self.price_transform.controls_to_prices(
            structured_action
        )
        agent_market_prices, agent_submission = self._agent_submission(
            transformed_prices,
            decision.effective_regime,
        )

        self.market.set_regimes(
            decision.effective_regime,
            self.opponent_policy.regime,
        )
        opponent_observation = self._build_opponent_observation(
            agent_submission
        )
        opponent_prices = self.opponent_policy.get_prices(
            opponent_observation
        )
        opponent_price_vector = PriceVector(
            uniform=float(opponent_prices["uniform_price"]),
            new=float(opponent_prices["price_new"]),
            old=float(opponent_prices["price_old"]),
        )
        self.market.step(agent_market_prices, opponent_prices)

        raw_agent_profit = float(self.market.firms[0].last_period_profit)
        raw_opponent_profit = float(self.market.firms[1].last_period_profit)
        normalized_reward = self.reward_normalizer.normalize(raw_agent_profit)
        self._raw_agent_profits.append(raw_agent_profit)
        self._raw_opponent_profits.append(raw_opponent_profit)
        self._normalized_rewards.append(normalized_reward)

        self._timestep += 1
        self.regime_commitment_controller.advance_period()
        terminated = False
        truncated = self._timestep >= self.episode_length
        observation = self.observation_builder.build(
            self.market,
            self._timestep,
            self.regime_commitment_controller,
        )

        effective_action = PricingAction(
            regime=decision.effective_regime,
            uniform_control=structured_action.uniform_control,
            bbp_new_control=structured_action.bbp_new_control,
            bbp_premium_control=structured_action.bbp_premium_control,
        )
        context = self.episode_context
        info: dict[str, Any] = {
            "raw_agent_profit": raw_agent_profit,
            "raw_opponent_profit": raw_opponent_profit,
            "profit_advantage": raw_agent_profit - raw_opponent_profit,
            "normalized_reward": normalized_reward,
            "agent_regime": int(decision.effective_regime),
            "opponent_regime": int(self.opponent_policy.regime),
            "regime_changed": decision.regime_changed,
            "regime_decision_allowed": decision.regime_decision_allowed,
            "opponent_policy_name": (
                context.opponent_assignment.policy_name
            ),
            "episode_index": context.episode_index,
            "consumer_seed": context.episode_seed_bundle.consumer_seed,
            "opponent_seed": context.episode_seed_bundle.opponent_seed,
            "regime_decision_mask": decision.regime_decision_mask,
            "effective_replay_action": (
                PricingActionCodec.to_replay_vector(effective_action)
            ),
            "opponent_price_controls": (
                self.price_transform.prices_to_controls(
                    opponent_price_vector
                )
            ),
        }
        if truncated:
            info["episode_summary"] = {
                "raw_agent_profit_total": float(
                    np.sum(self._raw_agent_profits)
                ),
                "raw_opponent_profit_total": float(
                    np.sum(self._raw_opponent_profits)
                ),
                "profit_advantage_total": float(
                    np.sum(self._raw_agent_profits)
                    - np.sum(self._raw_opponent_profits)
                ),
                "normalized_reward_total": float(
                    np.sum(self._normalized_rewards)
                ),
                "normalized_reward_mean": float(
                    np.mean(self._normalized_rewards)
                ),
            }
        return (
            observation,
            normalized_reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
