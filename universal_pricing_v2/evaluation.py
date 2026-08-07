"""Paired forced-regime, learned-strategy, oracle, and transfer evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from env.pricing_contracts import (
    PricingAction,
    PricingActionCodec,
    PricingRegime,
)
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    DistributionCombination,
)
from universal_pricing_v2.agents import (
    BaseHierarchicalPricingAgent,
    HierarchicalPricingAgentFactory,
)
from universal_pricing_v2.environment import (
    HierarchicalPricingEnvironmentFactoryV2,
)
from universal_pricing_v2.observations import (
    StrategyObservationCodec,
    StrategyObservationFeature,
)
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalSeedDeriver,
    HierarchicalTrainingPhase,
    PricingSkill,
    UniversalPricingV2ProtocolConfig,
    V2ExperimentCoordinate,
    V2SeedNamespace,
)
from universal_pricing_v2.replay import PricingSkillTransition


class EvaluationRegimeMode(str, Enum):
    LEARNED = "learned"
    FORCED_UNIFORM = "forced_uniform"
    FORCED_BBP = "forced_bbp"
    RANDOM_REGIME = "random_regime"


@dataclass(frozen=True)
class StrategyMasteryScenario:
    """One distribution-hidden macro decision scenario."""

    name: str
    expected_regime: PricingRegime
    observation: Mapping[str, np.ndarray]


class StrategyMasteryScenarioFactory:
    """Create paired parity and established-customer evidence scenarios."""

    @staticmethod
    def _price(value: float) -> float:
        return float(2.0 * (value - 0.5) / 4.5 - 1.0)

    @classmethod
    def _observation(
        cls,
        *,
        own_share: float,
        own_price: float,
        opponent_price: float,
        new_customer_ratio: float,
        retention_rate: float,
        normalized_profit: float,
        normalized_advantage: float,
        has_history: bool,
    ) -> Mapping[str, np.ndarray]:
        ratio = lambda value: float(2.0 * value - 1.0)
        strategy = StrategyObservationCodec.encode(
            {
                StrategyObservationFeature.CURRENT_AGENT_REGIME: -1.0,
                StrategyObservationFeature.OPPONENT_REGIME: -1.0,
                StrategyObservationFeature.CURRENT_OWN_MARKET_SHARE: ratio(
                    own_share
                ),
                StrategyObservationFeature.CURRENT_OPPONENT_MARKET_SHARE: ratio(
                    1.0 - own_share
                ),
                StrategyObservationFeature.CURRENT_OWN_ACTIVE_PRICE_LEVEL: cls._price(
                    own_price
                ),
                StrategyObservationFeature.CURRENT_OPPONENT_ACTIVE_PRICE_LEVEL: cls._price(
                    opponent_price
                ),
                StrategyObservationFeature.WINDOW_MEAN_OWN_MARKET_SHARE: ratio(
                    own_share
                ),
                StrategyObservationFeature.WINDOW_MARKET_SHARE_CHANGE: 0.0,
                StrategyObservationFeature.WINDOW_MEAN_DEMAND_RATIO: ratio(
                    own_share
                ),
                StrategyObservationFeature.WINDOW_MEAN_NEW_CUSTOMER_RATIO: ratio(
                    new_customer_ratio
                ),
                StrategyObservationFeature.WINDOW_MEAN_RETENTION: ratio(
                    retention_rate
                ),
                StrategyObservationFeature.WINDOW_MEAN_NORMALIZED_NET_PROFIT: (
                    normalized_profit
                ),
                StrategyObservationFeature.WINDOW_MEAN_NORMALIZED_NET_PROFIT_ADVANTAGE: (
                    normalized_advantage
                ),
                StrategyObservationFeature.WINDOW_NET_PROFIT_TREND: 0.0,
                StrategyObservationFeature.WINDOW_POPULARITY_CHANGE: 0.0,
                StrategyObservationFeature.WINDOW_MEAN_OWN_ACTIVE_PRICE: cls._price(
                    own_price
                ),
                StrategyObservationFeature.WINDOW_MEAN_OPPONENT_ACTIVE_PRICE: cls._price(
                    opponent_price
                ),
                StrategyObservationFeature.EPISODE_PROGRESS: -0.5,
                StrategyObservationFeature.WINDOW_HISTORY_AVAILABLE: (
                    1.0 if has_history else -1.0
                ),
            }
        )
        pricing = np.zeros(18, dtype=np.float32)
        pricing[11] = -1.0
        pricing[12] = -1.0
        pricing[15] = -0.5
        pricing[16] = 1.0
        pricing[17] = 1.0
        return {"pricing": pricing, "strategy": strategy}

    @classmethod
    def paired(
        cls, environment_seed: int
    ) -> tuple[StrategyMasteryScenario, StrategyMasteryScenario]:
        generator = np.random.default_rng(
            HierarchicalSeedDeriver.derive(
                int(environment_seed),
                V2SeedNamespace.VALIDATION,
                local_index=901,
            )
        )
        parity_price = float(generator.uniform(2.6, 3.4))
        established_share = float(generator.uniform(0.55, 0.75))
        established_own_price = float(generator.uniform(3.0, 3.2))
        established_opponent_price = float(generator.uniform(3.8, 4.2))
        return (
            StrategyMasteryScenario(
                name="no_established_customer_parity",
                expected_regime=PricingRegime.UNIFORM,
                observation=cls._observation(
                    own_share=0.5,
                    own_price=parity_price,
                    opponent_price=parity_price,
                    new_customer_ratio=1.0,
                    retention_rate=0.0,
                    normalized_profit=0.0,
                    normalized_advantage=0.0,
                    has_history=False,
                ),
            ),
            StrategyMasteryScenario(
                name="established_customer_bbp_advantage",
                expected_regime=PricingRegime.BBP,
                observation=cls._observation(
                    own_share=established_share,
                    own_price=established_own_price,
                    opponent_price=established_opponent_price,
                    new_customer_ratio=0.5,
                    retention_rate=0.95,
                    normalized_profit=0.35,
                    normalized_advantage=0.20,
                    has_history=True,
                ),
            ),
        )


class StrategyMasteryScenarioEvaluator:
    """Measure categorical selections on the predeclared scenario panel."""

    def selection_accuracies(
        self,
        agent: BaseHierarchicalPricingAgent,
        environment_seeds: Sequence[int],
    ) -> tuple[float, float]:
        if not environment_seeds:
            raise ValueError("Strategy scenarios require evaluation seeds")
        saved_state = copy.deepcopy(agent.state_dict())
        uniform_correct: list[float] = []
        bbp_correct: list[float] = []
        try:
            for seed in environment_seeds:
                parity, established = StrategyMasteryScenarioFactory.paired(
                    int(seed)
                )
                for scenario, results in (
                    (parity, uniform_correct),
                    (established, bbp_correct),
                ):
                    agent.reset_recurrent_state()
                    selected = agent.select_action(
                        scenario.observation,
                        regime_mode=AgentRegimeMode.LEARNED,
                        deterministic=True,
                    ).regime
                    results.append(
                        float(selected is scenario.expected_regime)
                    )
        finally:
            agent.load_state_dict(saved_state)
        return float(np.mean(uniform_correct)), float(np.mean(bbp_correct))


def _agent_mode(mode: EvaluationRegimeMode) -> AgentRegimeMode:
    if mode is EvaluationRegimeMode.FORCED_UNIFORM:
        return AgentRegimeMode.FORCED_UNIFORM
    if mode is EvaluationRegimeMode.FORCED_BBP:
        return AgentRegimeMode.FORCED_BBP
    return AgentRegimeMode.LEARNED


class V2EvaluationRepository:
    """Write evaluation outputs atomically with short temporary names."""

    @staticmethod
    def _json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".e-", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    value,
                    stream,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def write(
        self,
        output_directory: str | Path,
        *,
        suite: str,
        episodes: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        episode_path = directory / f"{suite}_episodes.jsonl"
        descriptor, temporary = tempfile.mkstemp(
            prefix=".j-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(
                        json.dumps(
                            episode, sort_keys=True, allow_nan=False
                        )
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, episode_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        summary_path = directory / f"{suite}_summary.json"
        self._json(summary_path, dict(summary))
        return episode_path, summary_path


class UniversalPricingV2Evaluator:
    """Evaluate one trained hierarchy on paired seeds and opponent policies."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        coordinate: V2ExperimentCoordinate,
        *,
        device: str = "cpu",
        episode_length: int | None = None,
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.device = device
        self.episode_length = episode_length

    def _factory(
        self,
        coordinate: V2ExperimentCoordinate,
    ) -> HierarchicalPricingEnvironmentFactoryV2:
        keyword = (
            {} if self.episode_length is None else {
                "episode_length": self.episode_length
            }
        )
        return HierarchicalPricingEnvironmentFactoryV2(
            self.protocol, **keyword
        )

    def evaluate_agent(
        self,
        agent: BaseHierarchicalPricingAgent,
        environment_seeds: Sequence[int],
        *,
        regime_mode: EvaluationRegimeMode,
        opponent_policy_names: Sequence[str] | None = None,
        evaluation_coordinate: V2ExperimentCoordinate | None = None,
        suite: str = "validation",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate without changing weights, counters, RNG, or online state."""

        saved_state = copy.deepcopy(agent.state_dict())
        try:
            return self._evaluate_agent_unrestored(
                agent,
                environment_seeds,
                regime_mode=regime_mode,
                opponent_policy_names=opponent_policy_names,
                evaluation_coordinate=evaluation_coordinate,
                suite=suite,
            )
        finally:
            agent.load_state_dict(saved_state)

    def _evaluate_agent_unrestored(
        self,
        agent: BaseHierarchicalPricingAgent,
        environment_seeds: Sequence[int],
        *,
        regime_mode: EvaluationRegimeMode,
        opponent_policy_names: Sequence[str] | None = None,
        evaluation_coordinate: V2ExperimentCoordinate | None = None,
        suite: str = "validation",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mode = EvaluationRegimeMode(regime_mode)
        coordinate = evaluation_coordinate or self.coordinate
        opponents = tuple(
            opponent_policy_names
            or (
                self.protocol.opponent_pool.uniform_policies
                + self.protocol.opponent_pool.bbp_policies
            )
        )
        registered = set(
            self.protocol.opponent_pool.uniform_policies
            + self.protocol.opponent_pool.bbp_policies
        )
        if not opponents or set(opponents) - registered:
            raise ValueError("Evaluation opponent panel is invalid")
        episodes: list[dict[str, Any]] = []
        inference_seconds: list[float] = []
        for seed_index, environment_seed in enumerate(environment_seeds):
            factory = self._factory(coordinate)
            environment = factory.create_environment_with_run_seed(
                coordinate, int(environment_seed)
            )
            context_factory = environment.context_factory
            random_regime_generator = np.random.default_rng(
                HierarchicalSeedDeriver.derive(
                    int(environment_seed),
                    V2SeedNamespace.VALIDATION,
                    local_index=seed_index,
                )
            )
            for opponent_index, opponent_name in enumerate(opponents):
                environment_mode = _agent_mode(mode)
                context = context_factory.create(
                    phase=HierarchicalTrainingPhase.JOINT_CONSOLIDATION,
                    stage_index=opponent_index,
                    local_episode_index=seed_index,
                    agent_regime_mode=environment_mode,
                    opponent_policy_name=opponent_name,
                    stage_key=f"evaluation:{opponent_name}",
                )
                observation, reset_info = environment.reset(
                    options={"episode_context": context}
                )
                agent.reset_recurrent_state()
                totals = {
                    "normalized_net_reward_total": 0.0,
                    "gross_agent_profit_total": 0.0,
                    "agent_bbp_operating_cost_total": 0.0,
                    "net_agent_profit_total": 0.0,
                    "gross_opponent_profit_total": 0.0,
                    "opponent_bbp_operating_cost_total": 0.0,
                    "net_opponent_profit_total": 0.0,
                }
                regimes: list[int] = []
                shares: list[float] = []
                retention: list[float] = []
                uniform_prices: list[float] = []
                new_prices: list[float] = []
                old_prices: list[float] = []
                regime_changes = 0
                while True:
                    selected_mode = environment_mode
                    if mode is EvaluationRegimeMode.RANDOM_REGIME:
                        selected_mode = (
                            AgentRegimeMode.FORCED_BBP
                            if int(random_regime_generator.integers(0, 2))
                            else AgentRegimeMode.FORCED_UNIFORM
                        )
                    started = time.perf_counter()
                    action = agent.select_action(
                        observation,
                        regime_mode=selected_mode,
                        deterministic=True,
                    )
                    inference_seconds.append(
                        time.perf_counter() - started
                    )
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(PricingActionCodec.to_gym(action))
                    totals["normalized_net_reward_total"] += float(reward)
                    for name in (
                        "gross_agent_profit",
                        "agent_bbp_operating_cost",
                        "net_agent_profit",
                        "gross_opponent_profit",
                        "opponent_bbp_operating_cost",
                        "net_opponent_profit",
                    ):
                        totals[f"{name}_total"] += float(info[name])
                    regimes.append(int(info["agent_regime"]))
                    regime_changes += int(info["regime_changed"])
                    shares.append(float(info["market_share"]))
                    retention.append(float(info["retention_rate"]))
                    uniform_prices.append(
                        float(info["agent_uniform_price"])
                    )
                    new_prices.append(
                        float(info["agent_bbp_new_price"])
                    )
                    old_prices.append(
                        float(info["agent_bbp_old_price"])
                    )
                    # Evaluation updates only online recurrent context.
                    transition = PricingSkillTransition.from_environment_step(
                        pricing_skill=PricingSkill.UNIFORM,
                        observation=observation,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        stage_key=context.stage_key,
                    )
                    agent.observe_pricing_transition(transition)
                    observation = next_observation
                    if terminated or truncated:
                        break
                totals["net_profit_advantage_total"] = (
                    totals["net_agent_profit_total"]
                    - totals["net_opponent_profit_total"]
                )
                combination = coordinate.distribution_combination
                episodes.append(
                    {
                        "suite": suite,
                        "regime_mode": mode.value,
                        "evaluation_seed_index": seed_index,
                        "evaluation_seed": int(environment_seed),
                        "opponent_policy_name": opponent_name,
                        "opponent_family": reset_info["opponent_family"],
                        "market_timing": reset_info["market_timing"],
                        "location_distribution": combination.location.value,
                        "strategicness_distribution": (
                            combination.strategicness.value
                        ),
                        "exclusivity_distribution": (
                            combination.exclusivity.value
                        ),
                        **totals,
                        "bbp_period_fraction": float(np.mean(regimes)),
                        "regime_change_count": regime_changes,
                        "mean_market_share": float(np.mean(shares)),
                        "mean_retention_rate": float(np.mean(retention)),
                        "mean_uniform_price": float(
                            np.mean(uniform_prices)
                        ),
                        "mean_bbp_new_price": float(np.mean(new_prices)),
                        "mean_bbp_old_price": float(np.mean(old_prices)),
                        "mean_bbp_price_spread": float(
                            np.mean(
                                np.asarray(old_prices)
                                - np.asarray(new_prices)
                            )
                        ),
                    }
                )
            environment.close()
        return episodes, self._summary(
            episodes,
            suite=suite,
            regime_mode=mode,
            mean_inference_seconds=(
                float(np.mean(inference_seconds))
                if inference_seconds
                else 0.0
            ),
        )

    @staticmethod
    def _summary(
        episodes: Sequence[Mapping[str, Any]],
        *,
        suite: str,
        regime_mode: EvaluationRegimeMode,
        mean_inference_seconds: float,
    ) -> dict[str, Any]:
        numeric_names = (
            "normalized_net_reward_total",
            "gross_agent_profit_total",
            "agent_bbp_operating_cost_total",
            "net_agent_profit_total",
            "gross_opponent_profit_total",
            "opponent_bbp_operating_cost_total",
            "net_opponent_profit_total",
            "net_profit_advantage_total",
            "bbp_period_fraction",
            "regime_change_count",
            "mean_market_share",
            "mean_retention_rate",
            "mean_uniform_price",
            "mean_bbp_new_price",
            "mean_bbp_old_price",
            "mean_bbp_price_spread",
        )
        result: dict[str, Any] = {
            "suite": suite,
            "regime_mode": regime_mode.value,
            "episode_count": len(episodes),
            "mean_inference_seconds": mean_inference_seconds,
        }
        for name in numeric_names:
            result[f"mean_{name}"] = (
                float(np.mean([float(item[name]) for item in episodes]))
                if episodes
                else 0.0
            )
        result["by_opponent_family"] = {
            family: {
                "episode_count": len(selected),
                "mean_net_agent_profit_total": float(
                    np.mean(
                        [
                            float(item["net_agent_profit_total"])
                            for item in selected
                        ]
                    )
                )
                if selected
                else 0.0,
                "mean_net_profit_advantage_total": float(
                    np.mean(
                        [
                            float(item["net_profit_advantage_total"])
                            for item in selected
                        ]
                    )
                )
                if selected
                else 0.0,
                "mean_bbp_period_fraction": float(
                    np.mean(
                        [
                            float(item["bbp_period_fraction"])
                            for item in selected
                        ]
                    )
                )
                if selected
                else 0.0,
            }
            for family in ("uniform", "bbp")
            for selected in [
                [
                    item
                    for item in episodes
                    if item["opponent_family"] == family
                ]
            ]
        }
        return result

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        environment_seeds: Sequence[int],
        *,
        suite: str,
        output_directory: str | Path | None = None,
        include_counterfactuals: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = Path(checkpoint_path)
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        agent = HierarchicalPricingAgentFactory.create(
            self.protocol.agent_profiles[
                self.coordinate.agent_architecture
            ],
            self.protocol.run_seed_bundle(
                self.coordinate.training_seed_index
            ),
            device=self.device,
        )
        agent.load(path)
        modes = (
            tuple(EvaluationRegimeMode)
            if include_counterfactuals
            else (EvaluationRegimeMode.LEARNED,)
        )
        episodes: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        for mode in modes:
            mode_episodes, mode_summary = self.evaluate_agent(
                agent,
                environment_seeds,
                regime_mode=mode,
                suite=suite,
            )
            for episode in mode_episodes:
                episode["checkpoint_sha256"] = checkpoint_hash
            episodes.extend(mode_episodes)
            summaries[mode.value] = mode_summary
        summary = {
            "suite": suite,
            "architecture": self.coordinate.agent_architecture.value,
            "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_hash,
            "seed_count": len(environment_seeds),
            "episode_count": len(episodes),
            "counterfactuals": summaries,
        }
        if output_directory is not None:
            V2EvaluationRepository().write(
                output_directory,
                suite=suite,
                episodes=episodes,
                summary=summary,
            )
        return episodes, summary

    def transfer_matrix(
        self,
        agent: BaseHierarchicalPricingAgent,
        environment_seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Evaluate the trained policy on every one of the 27 populations."""

        results: list[dict[str, Any]] = []
        for location in ConsumerDistributionFamily:
            for strategicness in ConsumerDistributionFamily:
                for exclusivity in ConsumerDistributionFamily:
                    evaluation_coordinate = V2ExperimentCoordinate(
                        agent_architecture=(
                            self.coordinate.agent_architecture
                        ),
                        distribution_combination=DistributionCombination(
                            location=location,
                            strategicness=strategicness,
                            exclusivity=exclusivity,
                        ),
                        training_seed_index=(
                            self.coordinate.training_seed_index
                        ),
                    )
                    _, summary = self.evaluate_agent(
                        agent,
                        environment_seeds,
                        regime_mode=EvaluationRegimeMode.LEARNED,
                        evaluation_coordinate=evaluation_coordinate,
                        suite="transfer",
                    )
                    results.append(
                        {
                            "training_distribution": (
                                self.coordinate.distribution_combination.identifier
                            ),
                            "evaluation_distribution": (
                                evaluation_coordinate.distribution_combination.identifier
                            ),
                            **summary,
                        }
                    )
        return results


@dataclass(frozen=True)
class ConstantPriceOracleResult:
    regime: PricingRegime
    controls: tuple[float, ...]
    mean_net_profit: float
    episode_count: int


class ConstantPriceOracleEvaluator:
    """Deterministic coarse-to-fine constant-price baseline."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        coordinate: V2ExperimentCoordinate,
        *,
        episode_length: int | None = None,
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.episode_length = episode_length

    def _score(
        self,
        *,
        regime: PricingRegime,
        controls: tuple[float, ...],
        opponent_policy_name: str,
        environment_seeds: Sequence[int],
    ) -> float:
        profits: list[float] = []
        mode = (
            AgentRegimeMode.FORCED_UNIFORM
            if regime is PricingRegime.UNIFORM
            else AgentRegimeMode.FORCED_BBP
        )
        for seed_index, seed in enumerate(environment_seeds):
            keyword = (
                {} if self.episode_length is None else {
                    "episode_length": self.episode_length
                }
            )
            factory = HierarchicalPricingEnvironmentFactoryV2(
                self.protocol, **keyword
            )
            environment = factory.create_environment_with_run_seed(
                self.coordinate, int(seed)
            )
            context = environment.context_factory.create(
                phase=HierarchicalTrainingPhase.UNIFORM_PRICING,
                stage_index=0,
                local_episode_index=seed_index,
                agent_regime_mode=mode,
                opponent_policy_name=opponent_policy_name,
                stage_key="oracle",
            )
            _, _ = environment.reset(options={"episode_context": context})
            if regime is PricingRegime.UNIFORM:
                action = PricingAction(
                    regime=regime,
                    uniform_control=controls[0],
                    bbp_new_control=0.0,
                    bbp_premium_control=0.0,
                )
            else:
                action = PricingAction(
                    regime=regime,
                    uniform_control=0.0,
                    bbp_new_control=controls[0],
                    bbp_premium_control=controls[1],
                )
            total = 0.0
            while True:
                _, _, terminated, truncated, info = environment.step(
                    PricingActionCodec.to_gym(action)
                )
                total += float(info["net_agent_profit"])
                if terminated or truncated:
                    break
            profits.append(total)
            environment.close()
        return float(np.mean(profits))

    def optimize(
        self,
        *,
        regime: PricingRegime,
        opponent_policy_name: str,
        environment_seeds: Sequence[int],
    ) -> ConstantPriceOracleResult:
        regime = PricingRegime(regime)
        if regime is PricingRegime.UNIFORM:
            candidates = [(float(value),) for value in np.linspace(-1, 1, 91)]
        else:
            coarse = np.linspace(-1, 1, 15)
            candidates = [
                (float(new), float(premium))
                for new in coarse
                for premium in coarse
            ]
        scored = [
            (
                self._score(
                    regime=regime,
                    controls=controls,
                    opponent_policy_name=opponent_policy_name,
                    environment_seeds=environment_seeds,
                ),
                controls,
            )
            for controls in candidates
        ]
        best_profit, best_controls = max(scored, key=lambda item: item[0])
        if regime is PricingRegime.BBP:
            step = 2.0 / 14.0
            new_values = np.clip(
                np.linspace(
                    best_controls[0] - step,
                    best_controls[0] + step,
                    9,
                ),
                -1.0,
                1.0,
            )
            premium_values = np.clip(
                np.linspace(
                    best_controls[1] - step,
                    best_controls[1] + step,
                    9,
                ),
                -1.0,
                1.0,
            )
            refined = {
                (float(new), float(premium))
                for new in new_values
                for premium in premium_values
            }
            refined_scores = [
                (
                    self._score(
                        regime=regime,
                        controls=controls,
                        opponent_policy_name=opponent_policy_name,
                        environment_seeds=environment_seeds,
                    ),
                    controls,
                )
                for controls in refined
            ]
            best_profit, best_controls = max(
                [*scored, *refined_scores], key=lambda item: item[0]
            )
        return ConstantPriceOracleResult(
            regime=regime,
            controls=best_controls,
            mean_net_profit=float(best_profit),
            episode_count=len(environment_seeds),
        )


class RandomPriceBaselineEvaluator:
    """Paired private-RNG random pricing baseline for mastery scores."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        coordinate: V2ExperimentCoordinate,
        *,
        episode_length: int | None = None,
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.episode_length = episode_length

    def evaluate(
        self,
        *,
        regime: PricingRegime,
        opponent_policy_name: str,
        environment_seeds: Sequence[int],
    ) -> float:
        profits: list[float] = []
        agent_mode = (
            AgentRegimeMode.FORCED_UNIFORM
            if PricingRegime(regime) is PricingRegime.UNIFORM
            else AgentRegimeMode.FORCED_BBP
        )
        for seed_index, seed in enumerate(environment_seeds):
            keyword = (
                {} if self.episode_length is None else {
                    "episode_length": self.episode_length
                }
            )
            environment = HierarchicalPricingEnvironmentFactoryV2(
                self.protocol, **keyword
            ).create_environment_with_run_seed(self.coordinate, int(seed))
            context = environment.context_factory.create(
                phase=HierarchicalTrainingPhase.UNIFORM_PRICING,
                stage_index=0,
                local_episode_index=seed_index,
                agent_regime_mode=agent_mode,
                opponent_policy_name=opponent_policy_name,
                stage_key="random_baseline",
            )
            _, _ = environment.reset(options={"episode_context": context})
            generator = np.random.default_rng(
                HierarchicalSeedDeriver.derive(
                    int(seed),
                    V2SeedNamespace.VALIDATION,
                    stage_index=1,
                )
            )
            total = 0.0
            while True:
                controls = generator.uniform(-1.0, 1.0, size=3)
                action = PricingAction(
                    regime=regime,
                    uniform_control=float(controls[0]),
                    bbp_new_control=float(controls[1]),
                    bbp_premium_control=float(controls[2]),
                )
                _, _, terminated, truncated, info = environment.step(
                    PricingActionCodec.to_gym(action)
                )
                total += float(info["net_agent_profit"])
                if terminated or truncated:
                    break
            profits.append(total)
            environment.close()
        return float(np.mean(profits))
