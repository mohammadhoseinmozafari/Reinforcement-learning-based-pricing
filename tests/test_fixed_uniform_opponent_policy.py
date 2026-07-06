"""Tests for the episode-randomized fixed uniform curriculum opponent."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import (
    FixedUniformOpponentPolicy,
    OpponentObservation,
    create_opponent_policy,
    create_preset_opponent,
)


class FixedUniformOpponentPolicyTests(unittest.TestCase):
    """Verify sampling boundaries, episode stability, and reproducibility."""

    def setUp(self):
        self.observation = OpponentObservation()

    def test_price_is_sampled_once_and_remains_fixed_within_episode(self):
        policy = FixedUniformOpponentPolicy(
            p_min=1.0,
            p_max=4.0,
            margin=0.5,
            seed=7,
        )
        policy.reset(seed=123)
        episode_price = policy.fixed_price

        self.assertGreaterEqual(episode_price, 1.5)
        self.assertLess(episode_price, 3.5)
        for _ in range(10):
            self.assertEqual(
                policy.get_uniform_price(self.observation), episode_price
            )
            self.assertEqual(
                policy.get_prices(self.observation)["uniform_price"], episode_price
            )

    def test_reset_randomizes_between_episodes_reproducibly(self):
        policy = FixedUniformOpponentPolicy(margin=0.25)
        policy.reset(seed=11)
        first = policy.fixed_price
        policy.reset(seed=12)
        second = policy.fixed_price
        policy.reset(seed=11)

        self.assertNotEqual(first, second)
        self.assertEqual(policy.fixed_price, first)

    def test_reset_without_seed_advances_the_existing_rng(self):
        first_policy = FixedUniformOpponentPolicy(seed=9)
        second_policy = FixedUniformOpponentPolicy(seed=9)

        first_policy.reset()
        second_policy.reset()
        self.assertEqual(first_policy.fixed_price, second_policy.fixed_price)

        first_price = first_policy.fixed_price
        first_policy.reset()
        self.assertNotEqual(first_policy.fixed_price, first_price)

    def test_invalid_sampling_intervals_are_rejected(self):
        invalid_arguments = (
            {"margin": -0.1},
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 1.0, "p_max": 2.0, "margin": 0.5},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    FixedUniformOpponentPolicy(**arguments)

    def test_factory_and_curriculum_preset_create_the_policy(self):
        direct = create_opponent_policy(
            "fixed_uniform", p_min=1.0, p_max=3.0, margin=0.2, seed=5
        )
        preset = create_preset_opponent("uniform_fixed", seed=5)
        alias = create_preset_opponent("fixed_uniform", seed=5)

        self.assertIsInstance(direct, FixedUniformOpponentPolicy)
        self.assertIsInstance(preset, FixedUniformOpponentPolicy)
        self.assertIsInstance(alias, FixedUniformOpponentPolicy)
        self.assertEqual(direct.regime, 0)
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
