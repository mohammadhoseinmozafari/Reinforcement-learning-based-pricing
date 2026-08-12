"""Integration and economic checks for universal pricing environments."""

from copy import deepcopy
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from env.consumer_population import ConsumerPopulationSnapshot
from env.models import HotellingMarket
from env.pricing_contracts import PricingAction, PricingActionCodec, PricingRegime
from env.pricing_factory import UniversalPricingEnvironmentFactory
from train.universal_pricing_protocol import (
    AgentArchitecture,
    ExperimentMatrix,
    ExperimentCoordinate,
    load_universal_pricing_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


def action(regime: PricingRegime, controls=(0.0, 0.0, 0.0)):
    return PricingActionCodec.to_gym(
        PricingAction(regime, *controls)
    )


def population_arrays(environment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [consumer.location for consumer in environment.market.consumers]
        ),
        np.asarray(
            [consumer.strategicness for consumer in environment.market.consumers]
        ),
        np.asarray(
            [
                consumer.exclusivity_pref
                for consumer in environment.market.consumers
            ]
        ),
    )


class UniversalPricingReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.matrix = ExperimentMatrix(cls.protocol)
        cls.coordinate = cls.matrix.coordinates()[0]

    def make_environment(self, coordinate=None):
        return UniversalPricingEnvironmentFactory(
            self.protocol,
            num_consumers=40,
            episode_length=8,
        ).create_environment(coordinate or self.coordinate)

    def test_identical_episode_context_reproduces_full_trajectory(self) -> None:
        first = self.make_environment()
        second = self.make_environment()
        first_reset = first.reset(options={"episode_index": 7})
        second_reset = second.reset(options={"episode_index": 7})
        np.testing.assert_array_equal(first_reset[0], second_reset[0])
        self.assertEqual(first_reset[1], second_reset[1])
        for first_values, second_values in zip(
            population_arrays(first),
            population_arrays(second),
        ):
            np.testing.assert_array_equal(first_values, second_values)

        actions = [
            action(
                PricingRegime.BBP if timestep < 4 else PricingRegime.UNIFORM,
                (
                    -0.8 + timestep * 0.2,
                    0.5 - timestep * 0.1,
                    -0.5 + timestep * 0.1,
                ),
            )
            for timestep in range(8)
        ]
        for selected_action in actions:
            first_transition = first.step(selected_action)
            second_transition = second.step(selected_action)
            np.testing.assert_array_equal(
                first_transition[0],
                second_transition[0],
            )
            self.assertEqual(
                first_transition[1:4],
                second_transition[1:4],
            )
            for field_name in (
                "raw_agent_profit",
                "raw_opponent_profit",
                "profit_advantage",
                "agent_regime",
                "opponent_regime",
            ):
                self.assertEqual(
                    first_transition[4][field_name],
                    second_transition[4][field_name],
                )

    def test_global_rng_perturbation_does_not_change_episode(self) -> None:
        expected = self.make_environment()
        expected_observation, expected_info = expected.reset(
            options={"episode_index": 3}
        )
        expected_population = population_arrays(expected)

        np.random.seed(900)
        np.random.random(1000)
        random.seed(901)
        for _ in range(1000):
            random.random()
        torch.manual_seed(902)
        torch.rand(1000)

        repeated = self.make_environment()
        repeated_observation, repeated_info = repeated.reset(
            options={"episode_index": 3}
        )
        np.testing.assert_array_equal(
            expected_observation,
            repeated_observation,
        )
        self.assertEqual(expected_info, repeated_info)
        for first_values, second_values in zip(
            expected_population,
            population_arrays(repeated),
        ):
            np.testing.assert_array_equal(first_values, second_values)

    def test_different_episode_indices_change_streams_and_balance_families(self) -> None:
        environment = self.make_environment()
        populations = []
        families = []
        opponents = []
        for episode_index in range(4):
            _, info = environment.reset(
                options={"episode_index": episode_index}
            )
            populations.append(population_arrays(environment)[0])
            families.append(info["opponent_family"])
            opponents.append(info["opponent_seed"])
        self.assertFalse(np.array_equal(populations[0], populations[1]))
        self.assertEqual(len(set(opponents)), 4)
        self.assertEqual(set(families[0:2]), {"uniform", "bbp"})
        self.assertEqual(set(families[2:4]), {"uniform", "bbp"})

    def test_architectures_share_episode_population_and_opponent(self) -> None:
        base = self.coordinate
        coordinates = [
            ExperimentCoordinate(
                agent_architecture=architecture,
                distribution_combination=base.distribution_combination,
                curriculum_id=base.curriculum_id,
                training_seed_index=base.training_seed_index,
            )
            for architecture in AgentArchitecture
        ]
        environments = [
            self.make_environment(coordinate)
            for coordinate in coordinates
        ]
        reset_results = [
            environment.reset(options={"episode_index": 5})
            for environment in environments
        ]
        self.assertEqual(
            len({result[1]["opponent_seed"] for result in reset_results}),
            1,
        )
        self.assertEqual(
            len(
                {
                    result[1]["opponent_policy_name"]
                    for result in reset_results
                }
            ),
            1,
        )
        reference_population = population_arrays(environments[0])
        for environment in environments[1:]:
            for reference, actual in zip(
                reference_population,
                population_arrays(environment),
            ):
                np.testing.assert_array_equal(reference, actual)

    def test_plain_reset_and_ad_hoc_seed_are_reproducible(self) -> None:
        environment = self.make_environment()
        first_observation, first_info = environment.reset()
        first_population = population_arrays(environment)
        second_observation, second_info = environment.reset()
        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(first_info, second_info)
        for first_values, second_values in zip(
            first_population,
            population_arrays(environment),
        ):
            np.testing.assert_array_equal(first_values, second_values)

        ad_hoc_first, ad_hoc_info = environment.reset(seed=4321)
        ad_hoc_population = population_arrays(environment)
        ad_hoc_second, repeated_info = environment.reset(seed=4321)
        np.testing.assert_array_equal(ad_hoc_first, ad_hoc_second)
        self.assertEqual(ad_hoc_info, repeated_info)
        for first_values, second_values in zip(
            ad_hoc_population,
            population_arrays(environment),
        ):
            np.testing.assert_array_equal(first_values, second_values)

    def test_all_distribution_combinations_reset_for_both_families(self) -> None:
        for combination in self.matrix.distribution_combinations():
            coordinate = ExperimentCoordinate(
                agent_architecture=AgentArchitecture.SAC,
                distribution_combination=combination,
                curriculum_id=self.protocol.curriculum_id,
                training_seed_index=0,
            )
            environment = UniversalPricingEnvironmentFactory(
                self.protocol,
                num_consumers=5,
                episode_length=1,
            ).create_environment(coordinate)
            families = set()
            for episode_index in (0, 1):
                observation, info = environment.reset(
                    options={"episode_index": episode_index}
                )
                families.add(info["opponent_family"])
                self.assertTrue(
                    environment.observation_space.contains(observation)
                )
                _, _, _, truncated, _ = environment.step(
                    action(PricingRegime.UNIFORM)
                )
                self.assertTrue(truncated)
            self.assertEqual(families, {"uniform", "bbp"})


class UniversalPricingEconomicScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.coordinate = ExperimentMatrix(cls.protocol).coordinates()[0]

    def uniform_fixed_episode_index(self) -> int:
        environment = UniversalPricingEnvironmentFactory(
            self.protocol,
            num_consumers=100,
            episode_length=1,
        ).create_environment(self.coordinate)
        for episode_index in range(20):
            context = environment.episode_context_factory.create(episode_index)
            if context.opponent_assignment.policy_name == "uniform_fixed":
                return episode_index
        self.fail("Balanced schedule did not cover uniform_fixed")

    def test_equal_effective_prices_make_regimes_equivalent_without_history(
        self,
    ) -> None:
        episode_index = self.uniform_fixed_episode_index()
        factory = UniversalPricingEnvironmentFactory(
            self.protocol,
            num_consumers=100,
            episode_length=1,
        )
        uniform_environment = factory.create_environment(self.coordinate)
        bbp_environment = factory.create_environment(self.coordinate)
        uniform_environment.reset(options={"episode_index": episode_index})
        bbp_environment.reset(options={"episode_index": episode_index})

        uniform_control = 2.0 * (3.0 - 0.5) / (5.0 - 0.5) - 1.0
        new_control = 2.0 * (3.0 - 0.5) / (4.0 - 0.5) - 1.0
        uniform_transition = uniform_environment.step(
            action(
                PricingRegime.UNIFORM,
                (uniform_control, new_control, -1.0),
            )
        )
        bbp_transition = bbp_environment.step(
            action(
                PricingRegime.BBP,
                (uniform_control, new_control, -1.0),
            )
        )
        self.assertAlmostEqual(
            uniform_transition[4]["raw_agent_profit"],
            bbp_transition[4]["raw_agent_profit"],
            places=6,
        )
        self.assertAlmostEqual(
            uniform_transition[4]["raw_opponent_profit"],
            bbp_transition[4]["raw_opponent_profit"],
            places=6,
        )

    def test_established_customer_scenario_has_bbp_advantage(self) -> None:
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
            {
                "uniform_price": 3.1,
                "price_new": 1.5,
                "price_old": 2.5,
            },
            {
                "uniform_price": 4.0,
                "price_new": 4.0,
                "price_old": 4.0,
            },
        )

        bbp_market = deepcopy(market)
        bbp_market.set_regimes(1, 0)
        bbp_market.step(
            {
                "uniform_price": 2.0,
                "price_new": 3.0,
                "price_old": 4.0,
            },
            {
                "uniform_price": 4.0,
                "price_new": 4.0,
                "price_old": 4.0,
            },
        )
        self.assertGreater(
            bbp_market.firms[0].last_period_profit,
            uniform_market.firms[0].last_period_profit + 20.0,
        )


if __name__ == "__main__":
    unittest.main()
