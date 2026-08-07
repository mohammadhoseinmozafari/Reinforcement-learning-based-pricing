"""Environment, observation, economic, and reproducibility tests for v2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from env.consumer_population import ConsumerPopulationSnapshot
from env.models import HotellingMarket
from env.opponent_policies import PriceVector
from env.pricing_contracts import (
    PricingAction,
    PricingActionCodec,
    PricingRegime,
)
from env.universal_pricing_env import PricingPriceTransform
from universal_pricing_v2.economics import BBPProfitAccounting
from universal_pricing_v2.evaluation import (
    StrategyMasteryScenarioFactory,
)
from universal_pricing_v2.environment import (
    HierarchicalPricingEnvironmentFactoryV2,
)
from universal_pricing_v2.observations import StrategyObservationCodec
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalTrainingPhase,
    V2ExperimentMatrix,
    load_universal_pricing_v2_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v2.yaml"
)


def action(
    regime: PricingRegime,
    controls: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return PricingActionCodec.to_gym(PricingAction(regime, *controls))


class UniversalPricingV2EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_v2_protocol(PROTOCOL_PATH)
        cls.coordinate = V2ExperimentMatrix(cls.protocol).coordinates()[0]

    def make_environment(
        self, *, episode_length: int = 12, num_consumers: int = 50
    ):
        return HierarchicalPricingEnvironmentFactoryV2(
            self.protocol,
            episode_length=episode_length,
            num_consumers=num_consumers,
        ).create_environment(self.coordinate)

    def context(
        self,
        environment,
        mode: AgentRegimeMode,
        *,
        opponent: str = "uniform_fixed",
        episode: int = 0,
    ):
        return environment.context_factory.create(
            phase=HierarchicalTrainingPhase.UNIFORM_PRICING,
            stage_index=0,
            local_episode_index=episode,
            agent_regime_mode=mode,
            opponent_policy_name=opponent,
            stage_key="test-stage",
        )

    def test_dict_observation_and_frozen_strategy_order(self) -> None:
        environment = self.make_environment()
        observation, reset_info = environment.reset(
            options={
                "episode_context": self.context(
                    environment, AgentRegimeMode.LEARNED
                )
            }
        )
        self.assertEqual(set(observation), {"pricing", "strategy"})
        self.assertEqual(observation["pricing"].shape, (18,))
        self.assertEqual(observation["strategy"].shape, (19,))
        self.assertTrue(environment.observation_space.contains(observation))
        decoded = StrategyObservationCodec.decode(observation["strategy"])
        self.assertEqual(len(decoded), 19)
        self.assertNotIn("opponent_policy_name", decoded)
        self.assertNotIn("distribution", " ".join(decoded))
        self.assertEqual(reset_info["opponent_policy_name"], "uniform_fixed")

    def test_forced_regimes_ignore_strategy_proposals_and_apply_cost(self) -> None:
        uniform = self.make_environment(episode_length=1)
        bbp = self.make_environment(episode_length=1)
        uniform.reset(
            options={
                "episode_context": self.context(
                    uniform, AgentRegimeMode.FORCED_UNIFORM
                )
            }
        )
        bbp.reset(
            options={
                "episode_context": self.context(
                    bbp, AgentRegimeMode.FORCED_BBP
                )
            }
        )
        _, _, _, _, uniform_info = uniform.step(
            action(PricingRegime.BBP)
        )
        _, bbp_reward, _, _, bbp_info = bbp.step(
            action(PricingRegime.UNIFORM)
        )
        self.assertEqual(uniform_info["agent_regime"], PricingRegime.UNIFORM)
        self.assertEqual(bbp_info["agent_regime"], PricingRegime.BBP)
        self.assertEqual(uniform_info["regime_decision_mask"], 0.0)
        self.assertEqual(bbp_info["regime_decision_mask"], 0.0)
        self.assertEqual(uniform_info["agent_bbp_operating_cost"], 0.0)
        self.assertEqual(bbp_info["agent_bbp_operating_cost"], 2.5)
        self.assertAlmostEqual(
            bbp_reward, bbp_info["net_agent_profit"] / 250.0
        )

    def test_learned_regime_decision_reopens_after_ten_periods(self) -> None:
        environment = self.make_environment(episode_length=11)
        environment.reset(
            options={
                "episode_context": self.context(
                    environment, AgentRegimeMode.LEARNED
                )
            }
        )
        for period in range(11):
            proposal = (
                PricingRegime.BBP
                if period == 0
                else PricingRegime.UNIFORM
            )
            _, _, _, _, info = environment.step(action(proposal))
            self.assertEqual(
                bool(info["regime_decision_mask"]),
                period in (0, 10),
            )
            expected = (
                PricingRegime.UNIFORM
                if period == 10
                else PricingRegime.BBP
            )
            self.assertEqual(info["agent_regime"], expected)

    def test_equal_prices_have_equal_gross_profit_and_exact_cost_gap(self) -> None:
        transform = PricingPriceTransform()
        controls = transform.prices_to_controls(
            PriceVector(uniform=3.0, new=3.0, old=3.0)
        )
        uniform = self.make_environment(
            episode_length=1, num_consumers=100
        )
        bbp = self.make_environment(episode_length=1, num_consumers=100)
        uniform.reset(
            options={
                "episode_context": self.context(
                    uniform, AgentRegimeMode.FORCED_UNIFORM
                )
            }
        )
        bbp.reset(
            options={
                "episode_context": self.context(
                    bbp, AgentRegimeMode.FORCED_BBP
                )
            }
        )
        _, _, _, _, uniform_info = uniform.step(
            action(PricingRegime.UNIFORM, tuple(controls))
        )
        _, _, _, _, bbp_info = bbp.step(
            action(PricingRegime.BBP, tuple(controls))
        )
        self.assertAlmostEqual(
            uniform_info["gross_agent_profit"],
            bbp_info["gross_agent_profit"],
            places=6,
        )
        self.assertAlmostEqual(
            uniform_info["net_agent_profit"]
            - bbp_info["net_agent_profit"],
            5.0,
            places=6,
        )

    def test_established_customer_bbp_advantage_survives_v2_cost(self) -> None:
        snapshot = ConsumerPopulationSnapshot(
            locations=np.linspace(0.005, 0.995, 100),
            strategicness=np.zeros(100),
            exclusivity=np.zeros(100),
        )
        market = HotellingMarket(num_consumers=100, seed=1)
        market.install_consumer_population(snapshot)
        for consumer in market.consumers[:50]:
            consumer.update_purchase(0, 2.0)

        uniform_market = deepcopy(market)
        uniform_market.set_regimes(0, 0)
        uniform_market.step(
            {"uniform_price": 3.1, "price_new": 1.5, "price_old": 2.5},
            {"uniform_price": 4.0, "price_new": 4.0, "price_old": 4.0},
        )
        bbp_market = deepcopy(market)
        bbp_market.set_regimes(1, 0)
        bbp_market.step(
            {"uniform_price": 2.0, "price_new": 3.0, "price_old": 4.0},
            {"uniform_price": 4.0, "price_new": 4.0, "price_old": 4.0},
        )
        accounting = BBPProfitAccounting(100)
        uniform_net = accounting.calculate(
            gross_agent_profit=uniform_market.firms[0].last_period_profit,
            agent_regime=PricingRegime.UNIFORM,
            gross_opponent_profit=uniform_market.firms[1].last_period_profit,
            opponent_regime=PricingRegime.UNIFORM,
        ).net_agent_profit
        bbp_net = accounting.calculate(
            gross_agent_profit=bbp_market.firms[0].last_period_profit,
            agent_regime=PricingRegime.BBP,
            gross_opponent_profit=bbp_market.firms[1].last_period_profit,
            opponent_regime=PricingRegime.UNIFORM,
        ).net_agent_profit
        self.assertGreater(bbp_net, uniform_net)

    def test_repeated_contexts_produce_identical_trajectories(self) -> None:
        first = self.make_environment(episode_length=3)
        second = self.make_environment(episode_length=3)
        first_context = self.context(
            first, AgentRegimeMode.LEARNED, opponent="bbp_fixed_discriminator"
        )
        second_context = self.context(
            second, AgentRegimeMode.LEARNED, opponent="bbp_fixed_discriminator"
        )
        first_observation, _ = first.reset(
            options={"episode_context": first_context}
        )
        second_observation, _ = second.reset(
            options={"episode_context": second_context}
        )
        for name in first_observation:
            np.testing.assert_array_equal(
                first_observation[name], second_observation[name]
            )
        for _ in range(3):
            first_step = first.step(
                action(PricingRegime.BBP, (0.1, -0.2, 0.3))
            )
            second_step = second.step(
                action(PricingRegime.BBP, (0.1, -0.2, 0.3))
            )
            for name in first_step[0]:
                np.testing.assert_array_equal(
                    first_step[0][name], second_step[0][name]
                )
            self.assertEqual(first_step[1:4], second_step[1:4])
            for key in (
                "gross_agent_profit",
                "net_agent_profit",
                "gross_opponent_profit",
                "net_opponent_profit",
            ):
                self.assertEqual(first_step[4][key], second_step[4][key])

    def test_strategy_mastery_scenarios_hide_domains_and_encode_evidence(
        self,
    ) -> None:
        parity, established = StrategyMasteryScenarioFactory.paired(123)
        self.assertEqual(parity.expected_regime, PricingRegime.UNIFORM)
        self.assertEqual(established.expected_regime, PricingRegime.BBP)
        for scenario in (parity, established):
            self.assertEqual(
                set(scenario.observation), {"pricing", "strategy"}
            )
            self.assertEqual(scenario.observation["pricing"].shape, (18,))
            self.assertEqual(scenario.observation["strategy"].shape, (19,))
        self.assertEqual(parity.observation["strategy"][18], -1.0)
        self.assertEqual(established.observation["strategy"][18], 1.0)


if __name__ == "__main__":
    unittest.main()
