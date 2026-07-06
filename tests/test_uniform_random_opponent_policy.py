"""Tests for the noisy per-step uniform curriculum opponent."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import (
    OpponentObservation,
    UniformRandomOpponentPolicy,
    create_opponent_policy,
    create_preset_opponent,
)


class UniformRandomOpponentPolicyTests(unittest.TestCase):
    """Verify episode bases, step noise, clipping, and factory integration."""

    def setUp(self):
        self.observation = OpponentObservation()

    def test_reset_samples_base_and_each_step_samples_noise(self):
        policy = UniformRandomOpponentPolicy(
            p_min=1.0,
            p_max=4.0,
            sigma=0.2,
        )
        policy.reset(seed=21)

        expected_rng = np.random.RandomState(21)
        expected_base = float(expected_rng.uniform(1.0, 4.0))
        expected_prices = [
            float(np.clip(expected_base + expected_rng.normal(0.0, 0.2), 1.0, 4.0))
            for _ in range(3)
        ]
        actual_prices = [
            policy.get_uniform_price(self.observation) for _ in range(3)
        ]

        self.assertEqual(policy.base_price, expected_base)
        np.testing.assert_allclose(actual_prices, expected_prices)
        self.assertEqual(policy.current_price, actual_prices[-1])

    def test_posted_prices_are_clipped_to_configured_interval(self):
        policy = UniformRandomOpponentPolicy(
            p_min=1.0,
            p_max=2.0,
            sigma=100.0,
            seed=3,
        )
        for _ in range(50):
            price = policy.get_uniform_price(self.observation)
            self.assertGreaterEqual(price, 1.0)
            self.assertLessEqual(price, 2.0)

    def test_bbp_placeholder_access_does_not_draw_extra_noise(self):
        first = UniformRandomOpponentPolicy(seed=5)
        second = UniformRandomOpponentPolicy(seed=5)
        first.reset(seed=9)
        second.reset(seed=9)

        first.get_uniform_price(self.observation)
        first.get_bbp_prices(self.observation)
        second.get_uniform_price(self.observation)

        self.assertEqual(
            first.get_uniform_price(self.observation),
            second.get_uniform_price(self.observation),
        )

    def test_seed_reproduces_complete_episode_price_sequence(self):
        policy = UniformRandomOpponentPolicy(sigma=0.4)
        policy.reset(seed=77)
        first_sequence = [
            policy.get_prices(self.observation)["uniform_price"] for _ in range(8)
        ]
        policy.reset(seed=77)
        second_sequence = [
            policy.get_prices(self.observation)["uniform_price"] for _ in range(8)
        ]

        np.testing.assert_array_equal(first_sequence, second_sequence)

    def test_invalid_parameters_are_rejected(self):
        invalid_arguments = (
            {"sigma": -0.1},
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 2.0, "p_max": 2.0},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    UniformRandomOpponentPolicy(**arguments)

    def test_factory_and_curriculum_preset_create_policy(self):
        direct = create_opponent_policy("uniform_random", seed=4)
        preset = create_preset_opponent("uniform_random", seed=4)

        self.assertIsInstance(direct, UniformRandomOpponentPolicy)
        self.assertIsInstance(preset, UniformRandomOpponentPolicy)
        self.assertEqual(direct.regime, 0)
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
