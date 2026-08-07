"""Hierarchical v2 Gym environment with symmetric net-profit accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.constants import EPISODE_LENGTH, NUM_CONSUMERS
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
from env.universal_pricing_env import (
    PricingPriceTransform,
    RegimeCommitmentController,
    RegimeDecisionResult,
    UniversalPricingObservationBuilder,
)
from train.universal_pricing_protocol import (
    BalancedOpponentSchedule,
    ConsumerPopulationSpec,
    OpponentEpisodeAssignment,
    OpponentFamily,
    OpponentPoolConfig,
    ProtocolConfigError,
)
from universal_pricing_v2.economics import (
    BBPOperatingCostConfig,
    BBPProfitAccounting,
)
from universal_pricing_v2.observations import (
    StrategyObservationBuilder,
    StrategyObservationCodec,
    StrategyObservationWindow,
    StrategyPeriodRecord,
)
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalSeedDeriver,
    HierarchicalTrainingPhase,
    MARKET_TIMING,
    StageEpisodeSeedBundle,
    UniversalPricingV2ProtocolConfig,
    V2ExperimentCoordinate,
)


@dataclass(frozen=True)
class V2EpisodeContext:
    """Explicit, reproducible context for one curriculum-local episode."""

    phase: HierarchicalTrainingPhase
    stage_index: int
    local_episode_index: int
    agent_regime_mode: AgentRegimeMode
    episode_seed_bundle: StageEpisodeSeedBundle
    opponent_assignment: OpponentEpisodeAssignment
    stage_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "phase", HierarchicalTrainingPhase(self.phase)
        )
        object.__setattr__(
            self, "agent_regime_mode", AgentRegimeMode(self.agent_regime_mode)
        )
        for name in ("stage_index", "local_episode_index"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ProtocolConfigError(f"{name} must be nonnegative")
        if (
            self.opponent_assignment.episode_index
            != self.local_episode_index
        ):
            raise ProtocolConfigError(
                "Opponent assignment local episode mismatch"
            )
        if (
            self.opponent_assignment.opponent_seed
            != self.episode_seed_bundle.opponent_seed
        ):
            raise ProtocolConfigError(
                "Opponent assignment seed mismatch"
            )
        if not self.stage_key:
            raise ProtocolConfigError("stage_key must be non-empty")


class V2EpisodeContextFactory:
    """Build explicit stage or balanced contexts from stable seed namespaces."""

    def __init__(
        self,
        *,
        run_seed: int,
        opponent_pool: OpponentPoolConfig,
    ) -> None:
        self.run_seed = int(run_seed)
        self.opponent_pool = opponent_pool

    @staticmethod
    def _family(policy_name: str) -> OpponentFamily:
        if policy_name.startswith("uniform_"):
            return OpponentFamily.UNIFORM
        if policy_name.startswith("bbp_"):
            return OpponentFamily.BBP
        raise ProtocolConfigError(
            f"Cannot infer opponent family for {policy_name}"
        )

    def create(
        self,
        *,
        phase: HierarchicalTrainingPhase,
        stage_index: int,
        local_episode_index: int,
        agent_regime_mode: AgentRegimeMode,
        opponent_policy_name: str | None,
        stage_key: str,
    ) -> V2EpisodeContext:
        phase = HierarchicalTrainingPhase(phase)
        seeds = HierarchicalSeedDeriver.episode_bundle(
            self.run_seed,
            phase,
            stage_index,
            local_episode_index,
        )
        if opponent_policy_name is None:
            scheduled = BalancedOpponentSchedule(
                self.opponent_pool, seeds.schedule_seed
            ).assignment(local_episode_index)
            assignment = OpponentEpisodeAssignment(
                episode_index=local_episode_index,
                opponent_family=scheduled.opponent_family,
                policy_name=scheduled.policy_name,
                opponent_seed=seeds.opponent_seed,
            )
        else:
            all_registered = (
                self.opponent_pool.uniform_policies
                + self.opponent_pool.bbp_policies
            )
            if opponent_policy_name not in all_registered:
                raise ProtocolConfigError(
                    f"Unregistered v2 opponent: {opponent_policy_name}"
                )
            assignment = OpponentEpisodeAssignment(
                episode_index=local_episode_index,
                opponent_family=self._family(opponent_policy_name),
                policy_name=opponent_policy_name,
                opponent_seed=seeds.opponent_seed,
            )
        return V2EpisodeContext(
            phase=phase,
            stage_index=stage_index,
            local_episode_index=local_episode_index,
            agent_regime_mode=agent_regime_mode,
            episode_seed_bundle=seeds,
            opponent_assignment=assignment,
            stage_key=stage_key,
        )


class HierarchicalPricingEnvironmentV2(gym.Env):
    """Agent-selected regimes after independent price-skill pretraining."""

    metadata = {"render_modes": [], "name": "universal_pricing_v2"}
    INITIAL_PRICES = PriceVector(uniform=2.75, new=2.25, old=3.0)

    def __init__(
        self,
        *,
        consumer_population_spec: ConsumerPopulationSpec,
        opponent_pool: OpponentPoolConfig,
        run_seed: int,
        regime_commitment_length: int = 10,
        bbp_operating_cost_rate: float = 0.01,
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
        self.run_seed = int(run_seed)
        self.num_consumers = num_consumers
        self.episode_length = episode_length
        self.regime_commitment_length = regime_commitment_length
        self.consumer_population_generator = (
            consumer_population_generator or ConsumerPopulationGenerator()
        )
        self.price_transform = price_transform or PricingPriceTransform()
        self.context_factory = V2EpisodeContextFactory(
            run_seed=self.run_seed,
            opponent_pool=opponent_pool,
        )
        self.regime_commitment_controller = RegimeCommitmentController(
            regime_commitment_length
        )
        self.pricing_observation_builder = UniversalPricingObservationBuilder(
            self.price_transform, episode_length
        )
        self.strategy_observation_builder = StrategyObservationBuilder(
            num_consumers=num_consumers,
            episode_length=episode_length,
        )
        self.strategy_window = StrategyObservationWindow(
            regime_commitment_length
        )
        self.profit_accounting = BBPProfitAccounting(
            num_consumers,
            BBPOperatingCostConfig(
                capacity_rate=bbp_operating_cost_rate
            ),
        )
        self.market = HotellingMarket(
            num_consumers=num_consumers,
            seed=self.run_seed,
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
        self.observation_space = spaces.Dict(
            {
                "pricing": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(PricingObservationCodec.FEATURE_COUNT,),
                    dtype=np.float32,
                ),
                "strategy": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(StrategyObservationCodec.FEATURE_COUNT,),
                    dtype=np.float32,
                ),
            }
        )
        self._context: V2EpisodeContext | None = None
        self._opponent_policy: OpponentPolicy | None = None
        self._timestep = 0
        self._gross_agent_profits: list[float] = []
        self._gross_opponent_profits: list[float] = []
        self._agent_costs: list[float] = []
        self._opponent_costs: list[float] = []
        self._net_agent_profits: list[float] = []
        self._net_opponent_profits: list[float] = []
        self._normalized_rewards: list[float] = []

    @property
    def episode_context(self) -> V2EpisodeContext:
        if self._context is None:
            raise RuntimeError("Environment must be reset before use")
        return self._context

    @property
    def opponent_policy(self) -> OpponentPolicy:
        if self._opponent_policy is None:
            raise RuntimeError("Environment must be reset before use")
        return self._opponent_policy

    def _default_context(self, seed: int | None) -> V2EpisodeContext:
        factory = (
            self.context_factory
            if seed is None
            else V2EpisodeContextFactory(
                run_seed=int(seed), opponent_pool=self.opponent_pool
            )
        )
        return factory.create(
            phase=HierarchicalTrainingPhase.JOINT_CONSOLIDATION,
            stage_index=0,
            local_episode_index=0,
            agent_regime_mode=AgentRegimeMode.LEARNED,
            opponent_policy_name=None,
            stage_key="ad_hoc_balanced" if seed is not None else "default_balanced",
        )

    def _resolve_context(
        self,
        seed: int | None,
        options: Mapping[str, Any] | None,
    ) -> tuple[V2EpisodeContext, str]:
        values = {} if options is None else dict(options)
        unknown = set(values) - {"episode_context"}
        if unknown:
            raise ValueError(
                "Unknown reset option(s): " + ", ".join(sorted(unknown))
            )
        if seed is not None and "episode_context" in values:
            raise ValueError(
                "Ad-hoc seed and explicit episode_context are mutually exclusive"
            )
        if "episode_context" in values:
            context = values["episode_context"]
            if not isinstance(context, V2EpisodeContext):
                raise TypeError(
                    "options['episode_context'] must be V2EpisodeContext"
                )
            return context, "protocol"
        if seed is not None and (
            not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
            or int(seed) < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        return self._default_context(seed), "ad_hoc" if seed is not None else "default"

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "pricing": self.pricing_observation_builder.build(
                self.market,
                self._timestep,
                self.regime_commitment_controller,
            ),
            "strategy": self.strategy_observation_builder.build(
                self.market,
                self._timestep,
                self.strategy_window,
            ),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        context, seed_source = self._resolve_context(seed, options)
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
        opponent = create_preset_opponent(
            assignment.policy_name, seed=assignment.opponent_seed
        )
        opponent.reset(seed=assignment.opponent_seed)
        expected_opponent_regime = (
            PricingRegime.UNIFORM
            if assignment.opponent_family is OpponentFamily.UNIFORM
            else PricingRegime.BBP
        )
        if PricingRegime(opponent.regime) is not expected_opponent_regime:
            raise ProtocolConfigError(
                "Opponent policy regime does not match scheduled family"
            )

        self._context = context
        self._opponent_policy = opponent
        self._timestep = 0
        self.strategy_window.reset()
        self.regime_commitment_controller.reset()
        initial_agent_regime = PricingRegime.UNIFORM
        if context.agent_regime_mode is AgentRegimeMode.FORCED_BBP:
            initial_agent_regime = PricingRegime.BBP
            self.regime_commitment_controller.decide(initial_agent_regime)
        elif context.agent_regime_mode is AgentRegimeMode.FORCED_UNIFORM:
            self.regime_commitment_controller.decide(initial_agent_regime)
        self.market.set_regimes(
            initial_agent_regime, expected_opponent_regime
        )

        self._gross_agent_profits = []
        self._gross_opponent_profits = []
        self._agent_costs = []
        self._opponent_costs = []
        self._net_agent_profits = []
        self._net_opponent_profits = []
        self._normalized_rewards = []
        observation = self._observation()
        if not self.observation_space.contains(observation):
            raise RuntimeError("V2 reset observation violates declared space")
        return observation, {
            "phase": context.phase.value,
            "stage_index": context.stage_index,
            "stage_key": context.stage_key,
            "local_episode_index": context.local_episode_index,
            "agent_regime_mode": context.agent_regime_mode.value,
            "consumer_seed": context.episode_seed_bundle.consumer_seed,
            "opponent_seed": context.episode_seed_bundle.opponent_seed,
            "opponent_family": assignment.opponent_family.value,
            "opponent_policy_name": assignment.policy_name,
            "seed_source": seed_source,
            "market_timing": MARKET_TIMING,
        }

    def _forced_regime(self) -> PricingRegime | None:
        mode = self.episode_context.agent_regime_mode
        if mode is AgentRegimeMode.FORCED_UNIFORM:
            return PricingRegime.UNIFORM
        if mode is AgentRegimeMode.FORCED_BBP:
            return PricingRegime.BBP
        return None

    def _resolve_regime(
        self, proposed_regime: PricingRegime
    ) -> RegimeDecisionResult:
        forced = self._forced_regime()
        if forced is None:
            return self.regime_commitment_controller.decide(proposed_regime)
        internal_allowed = (
            self.regime_commitment_controller.regime_decision_allowed
        )
        previous = self.regime_commitment_controller.effective_regime
        if internal_allowed:
            self.regime_commitment_controller.decide(forced)
        return RegimeDecisionResult(
            proposed_regime=proposed_regime,
            effective_regime=forced,
            regime_decision_allowed=False,
            regime_changed=previous is not forced,
        )

    def _agent_market_prices(
        self,
        prices: PriceVector,
        regime: PricingRegime,
    ) -> dict[str, float]:
        if regime is PricingRegime.UNIFORM:
            return {"uniform_price": prices.uniform}
        return {"price_new": prices.new, "price_old": prices.old}

    def _opponent_observation(
        self,
        previous_agent_regime: PricingRegime,
    ) -> OpponentObservation:
        opponent, agent = self.market.firms[1], self.market.firms[0]
        opponent_share = 0.5 if self._timestep == 0 else opponent.market_share
        agent_share = 0.5 if self._timestep == 0 else agent.market_share
        return OpponentObservation(
            previous=PreviousMarketState(
                own_market_share=float(opponent_share),
                competitor_market_share=float(agent_share),
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
                ),
                own_new_customer_ratio=opponent.get_new_old_ratio(),
            ),
            # V2 is a simultaneous-move game. Both firms decide period-t
            # prices from the completed period-(t-1) information set.
            competitor_submission=None,
            competitor_established_share=self.market.get_established_share(0),
            own_regime=opponent.pricing_regime,
            competitor_regime=previous_agent_regime,
            decision_period=self._timestep,
            state_period=self._timestep - 1,
            episode_length=self.episode_length,
        )

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._context is None or self._opponent_policy is None:
            raise RuntimeError("Environment must be reset before step")
        structured = PricingActionCodec.from_gym(action)
        boundary_before_action = (
            self.regime_commitment_controller.regime_decision_allowed
        )
        if boundary_before_action and self._timestep > 0:
            self.strategy_window.reset()
        previous_agent_regime = PricingRegime(
            self.market.firms[0].pricing_regime
        )
        decision = self._resolve_regime(structured.regime)
        transformed = self.price_transform.controls_to_prices(structured)
        agent_market_prices = self._agent_market_prices(
            transformed, decision.effective_regime
        )
        self.market.set_regimes(
            decision.effective_regime, self.opponent_policy.regime
        )
        opponent_observation = self._opponent_observation(
            previous_agent_regime
        )
        opponent_prices = self.opponent_policy.get_prices(
            opponent_observation
        )
        opponent_vector = PriceVector(
            uniform=float(opponent_prices["uniform_price"]),
            new=float(opponent_prices["price_new"]),
            old=float(opponent_prices["price_old"]),
        )
        self.market.step(agent_market_prices, opponent_prices)

        accounting = self.profit_accounting.calculate(
            gross_agent_profit=self.market.firms[0].last_period_profit,
            agent_regime=decision.effective_regime,
            gross_opponent_profit=self.market.firms[1].last_period_profit,
            opponent_regime=self.opponent_policy.regime,
        )
        self._gross_agent_profits.append(accounting.gross_agent_profit)
        self._gross_opponent_profits.append(
            accounting.gross_opponent_profit
        )
        self._agent_costs.append(accounting.agent_bbp_operating_cost)
        self._opponent_costs.append(
            accounting.opponent_bbp_operating_cost
        )
        self._net_agent_profits.append(accounting.net_agent_profit)
        self._net_opponent_profits.append(accounting.net_opponent_profit)
        self._normalized_rewards.append(accounting.normalized_net_reward)

        agent = self.market.firms[0]
        self.strategy_window.record(
            StrategyPeriodRecord(
                own_market_share=float(agent.market_share),
                demand_ratio=float(
                    agent.last_period_quantity / self.num_consumers
                ),
                new_customer_ratio=float(agent.get_new_old_ratio()),
                retention_rate=float(agent.retention_rate),
                normalized_net_profit=accounting.normalized_net_reward,
                normalized_net_profit_advantage=float(
                    accounting.net_profit_advantage
                    / self.profit_accounting.capacity
                ),
                own_active_price=(
                    self.strategy_observation_builder.active_price(agent)
                ),
                opponent_active_price=(
                    self.strategy_observation_builder.active_price(
                        self.market.firms[1]
                    )
                ),
            )
        )

        effective_action = PricingAction(
            regime=decision.effective_regime,
            uniform_control=structured.uniform_control,
            bbp_new_control=structured.bbp_new_control,
            bbp_premium_control=structured.bbp_premium_control,
        )
        completed_transition_decision_mask = (
            decision.regime_decision_mask
            if self._forced_regime() is None
            else 0.0
        )
        self._timestep += 1
        self.regime_commitment_controller.advance_period()
        terminated = False
        truncated = self._timestep >= self.episode_length
        observation = self._observation()
        if not self.observation_space.contains(observation):
            raise RuntimeError("V2 step observation violates declared space")

        context = self.episode_context
        info: dict[str, Any] = {
            **accounting.to_info(),
            "agent_regime": int(decision.effective_regime),
            "opponent_regime": int(self.opponent_policy.regime),
            "regime_changed": bool(decision.regime_changed),
            "regime_decision_allowed": bool(
                completed_transition_decision_mask
            ),
            "regime_decision_mask": completed_transition_decision_mask,
            "next_regime_decision_allowed": bool(
                self.regime_commitment_controller.regime_decision_allowed
            ),
            "effective_replay_action": (
                PricingActionCodec.to_replay_vector(effective_action)
            ),
            "opponent_price_controls": (
                self.price_transform.prices_to_controls(opponent_vector)
            ),
            "opponent_policy_name": (
                context.opponent_assignment.policy_name
            ),
            "opponent_family": (
                context.opponent_assignment.opponent_family.value
            ),
            "phase": context.phase.value,
            "stage_index": context.stage_index,
            "stage_key": context.stage_key,
            "local_episode_index": context.local_episode_index,
            "agent_regime_mode": context.agent_regime_mode.value,
            "market_timing": MARKET_TIMING,
            "opponent_observed_current_agent_submission": False,
            "opponent_observed_agent_regime": int(
                opponent_observation.competitor_regime
            ),
            "opponent_observed_agent_uniform_price": float(
                opponent_observation.previous.competitor_prices.uniform
            ),
            "opponent_observed_agent_bbp_new_price": float(
                opponent_observation.previous.competitor_prices.new
            ),
            "opponent_observed_agent_bbp_old_price": float(
                opponent_observation.previous.competitor_prices.old
            ),
            "consumer_seed": context.episode_seed_bundle.consumer_seed,
            "opponent_seed": context.episode_seed_bundle.opponent_seed,
            "agent_uniform_price": float(agent.uniform_price),
            "agent_bbp_new_price": float(agent.price_new),
            "agent_bbp_old_price": float(agent.price_old),
            "opponent_uniform_price": float(
                self.market.firms[1].uniform_price
            ),
            "opponent_bbp_new_price": float(
                self.market.firms[1].price_new
            ),
            "opponent_bbp_old_price": float(
                self.market.firms[1].price_old
            ),
            "market_share": float(agent.market_share),
            "retention_rate": float(agent.retention_rate),
        }
        if truncated:
            info["episode_summary"] = {
                "gross_agent_profit_total": float(
                    np.sum(self._gross_agent_profits)
                ),
                "agent_bbp_operating_cost_total": float(
                    np.sum(self._agent_costs)
                ),
                "net_agent_profit_total": float(
                    np.sum(self._net_agent_profits)
                ),
                "gross_opponent_profit_total": float(
                    np.sum(self._gross_opponent_profits)
                ),
                "opponent_bbp_operating_cost_total": float(
                    np.sum(self._opponent_costs)
                ),
                "net_opponent_profit_total": float(
                    np.sum(self._net_opponent_profits)
                ),
                "net_profit_advantage_total": float(
                    np.sum(self._net_agent_profits)
                    - np.sum(self._net_opponent_profits)
                ),
                "normalized_net_reward_total": float(
                    np.sum(self._normalized_rewards)
                ),
                "normalized_net_reward_mean": float(
                    np.mean(self._normalized_rewards)
                ),
            }
        return (
            observation,
            accounting.normalized_net_reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class HierarchicalPricingEnvironmentFactoryV2:
    """Resolve a v2 coordinate into an isolated environment and contexts."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        *,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
    ) -> None:
        self.protocol = protocol
        self.num_consumers = num_consumers
        self.episode_length = episode_length

    def create_environment(
        self, coordinate: V2ExperimentCoordinate
    ) -> HierarchicalPricingEnvironmentV2:
        bundle = self.protocol.run_seed_bundle(
            coordinate.training_seed_index
        )
        return self.create_environment_with_run_seed(
            coordinate, bundle.run_seed
        )

    def create_environment_with_run_seed(
        self,
        coordinate: V2ExperimentCoordinate,
        run_seed: int,
    ) -> HierarchicalPricingEnvironmentV2:
        if coordinate.agent_architecture not in self.protocol.agent_profiles:
            raise ProtocolConfigError("Coordinate architecture is invalid")
        return HierarchicalPricingEnvironmentV2(
            consumer_population_spec=self.protocol.population_spec(
                coordinate.distribution_combination
            ),
            opponent_pool=self.protocol.opponent_pool,
            run_seed=int(run_seed),
            regime_commitment_length=(
                self.protocol.regime_commitment_length
            ),
            bbp_operating_cost_rate=(
                self.protocol.bbp_operating_cost_rate
            ),
            num_consumers=self.num_consumers,
            episode_length=self.episode_length,
        )

    def context_factory(
        self, coordinate: V2ExperimentCoordinate
    ) -> V2EpisodeContextFactory:
        return V2EpisodeContextFactory(
            run_seed=self.protocol.run_seed_bundle(
                coordinate.training_seed_index
            ).run_seed,
            opponent_pool=self.protocol.opponent_pool,
        )
