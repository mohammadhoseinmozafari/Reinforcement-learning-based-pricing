"""Tests for the delayed uniform-price undercutting curriculum opponent."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import (
    OpponentObservation,
    PreviousMarketState,
    PriceVector,
    UniformUndercutterOpponent,
    create_opponent_policy,
    create_preset_opponent,
)


class UniformUndercutterOpponentTests(unittest.TestCase):
    """Verify episode parameters, delayed reactions, clipping, and registration."""

    @staticmethod
    def observation(agent_price: float) -> OpponentObservation:
        return OpponentObservation(
            previous=PreviousMarketState(
                competitor_prices=PriceVector(uniform=agent_price)
            )
        )

    def test_reset_samples_delta_and_delay_reproducibly(self):
        policy = UniformUndercutterOpponent()
        policy.reset(seed=14)

        expected_rng = np.random.RandomState(14)
        expected_delta = float(expected_rng.uniform(0.1, 1.0))
        expected_delay = int(expected_rng.choice((1, 2)))

        self.assertEqual(policy.delta, expected_delta)
        self.assertEqual(policy.reaction_delay, expected_delay)
        self.assertGreaterEqual(policy.delta, 0.1)
        self.assertLess(policy.delta, 1.0)
        self.assertIn(policy.reaction_delay, (1, 2))

    def test_one_step_delay_undercuts_latest_observed_agent_price(self):
        policy = UniformUndercutterOpponent(reaction_delays=(1,), seed=2)
        policy.reset(seed=8)

        first = policy.get_uniform_price(self.observation(3.0))
        second = policy.get_uniform_price(self.observation(4.0))

        self.assertAlmostEqual(first, 3.0 - policy.delta)
        self.assertAlmostEqual(second, 4.0 - policy.delta)

    def test_two_step_delay_uses_older_agent_price(self):
        policy = UniformUndercutterOpponent(reaction_delays=(2,), seed=2)
        policy.reset(seed=8)

        first = policy.get_uniform_price(self.observation(3.0))
        second = policy.get_uniform_price(self.observation(4.0))
        third = policy.get_uniform_price(self.observation(2.5))

        self.assertAlmostEqual(first, 3.0 - policy.delta)
        self.assertAlmostEqual(second, 3.0 - policy.delta)
        self.assertAlmostEqual(third, 4.0 - policy.delta)

    def test_undercut_price_is_clipped_to_configured_bounds(self):
        policy = UniformUndercutterOpponent(
            p_min=1.0,
            p_max=3.0,
            delta_min=0.9,
            delta_max=1.0,
            reaction_delays=(1,),
            seed=5,
        )
        low = policy.get_uniform_price(self.observation(1.1))
        high = policy.get_uniform_price(self.observation(10.0))

        self.assertEqual(low, 1.0)
        self.assertEqual(high, 3.0)

    def test_same_episode_seed_reproduces_parameters_and_price_sequence(self):
        policy = UniformUndercutterOpponent()
        agent_prices = (3.0, 4.0, 2.0, 3.5)
        policy.reset(seed=99)
        first = [
            policy.get_prices(self.observation(price))["uniform_price"]
            for price in agent_prices
        ]
        first_parameters = (policy.delta, policy.reaction_delay)

        policy.reset(seed=99)
        second = [
            policy.get_prices(self.observation(price))["uniform_price"]
            for price in agent_prices
        ]

        np.testing.assert_array_equal(first, second)
        self.assertEqual((policy.delta, policy.reaction_delay), first_parameters)

    def test_invalid_parameters_are_rejected(self):
        invalid_arguments = (
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 2.0, "p_max": 2.0},
            {"delta_min": -0.1},
            {"delta_min": 1.0, "delta_max": 1.0},
            {"reaction_delays": ()},
            {"reaction_delays": (0, 1)},
            {"reaction_delays": (1.5,)},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    UniformUndercutterOpponent(**arguments)

    def test_factory_and_curriculum_preset_create_policy(self):
        direct = create_opponent_policy("uniform_undercutter", seed=6)
        preset = create_preset_opponent("uniform_undercutter", seed=6)

        self.assertIsInstance(direct, UniformUndercutterOpponent)
        self.assertIsInstance(preset, UniformUndercutterOpponent)
        self.assertEqual(preset.reaction_delays, (1, 2))
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
