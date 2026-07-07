"""Tests for the cooperative uniform opponent with finite retaliation."""

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
    UniformTitForTatOpponent,
    create_opponent_policy,
    create_preset_opponent,
)


class UniformTitForTatOpponentTests(unittest.TestCase):
    """Verify cooperation, finite punishment, randomization, and registration."""

    @staticmethod
    def observation(agent_price: float) -> OpponentObservation:
        return OpponentObservation(
            previous=PreviousMarketState(
                competitor_prices=PriceVector(uniform=agent_price)
            )
        )

    def test_reset_samples_all_episode_parameters_reproducibly(self):
        policy = UniformTitForTatOpponent()
        policy.reset(seed=18)

        expected_rng = np.random.RandomState(18)
        expected_threshold = float(expected_rng.uniform(0.3, 1.0))
        expected_length = int(expected_rng.choice((3, 5, 8)))
        expected_delta = float(expected_rng.uniform(0.2, 0.8))
        expected_reference = float(expected_rng.uniform(2.75, 5.0))

        self.assertEqual(policy.threshold, expected_threshold)
        self.assertEqual(policy.punishment_length, expected_length)
        self.assertEqual(policy.delta, expected_delta)
        self.assertEqual(policy.reference_price, expected_reference)
        self.assertEqual(policy.punishment_remaining, 0)

    def test_cooperates_when_agent_does_not_cut_aggressively(self):
        policy = UniformTitForTatOpponent(seed=3)
        safe_agent_price = policy.reference_price - policy.threshold

        posted = policy.get_uniform_price(self.observation(safe_agent_price))

        self.assertEqual(posted, policy.reference_price)
        self.assertFalse(policy.is_punishing)

    def test_aggressive_cut_triggers_exactly_k_punishment_steps(self):
        policy = UniformTitForTatOpponent(
            punishment_lengths=(3,),
            seed=4,
        )
        aggressive_price = policy.reference_price - policy.threshold - 0.1
        safe_price = policy.reference_price

        punishment_prices = [
            policy.get_uniform_price(self.observation(aggressive_price))
        ]
        punishment_prices.extend(
            policy.get_uniform_price(self.observation(safe_price))
            for _ in range(2)
        )
        cooperative_price = policy.get_uniform_price(self.observation(safe_price))

        self.assertAlmostEqual(
            punishment_prices[0],
            max(policy.p_min, aggressive_price - policy.delta),
        )
        self.assertEqual(len(punishment_prices), policy.punishment_length)
        self.assertEqual(cooperative_price, policy.reference_price)
        self.assertFalse(policy.is_punishing)

    def test_new_cuts_during_punishment_do_not_extend_duration(self):
        policy = UniformTitForTatOpponent(
            punishment_lengths=(3,),
            seed=7,
        )
        aggressive_price = policy.reference_price - policy.threshold - 0.2
        for _ in range(3):
            policy.get_uniform_price(self.observation(aggressive_price))

        self.assertEqual(policy.punishment_remaining, 0)
        recovery = policy.get_uniform_price(
            self.observation(policy.reference_price)
        )
        self.assertEqual(recovery, policy.reference_price)

    def test_punishment_price_is_clipped_to_minimum(self):
        policy = UniformTitForTatOpponent(
            p_min=1.0,
            p_max=4.0,
            mid_price=2.0,
            high_price=4.0,
            punishment_lengths=(3,),
            delta_min=0.7,
            delta_max=0.8,
            seed=5,
        )
        posted = policy.get_uniform_price(self.observation(1.1))
        self.assertEqual(posted, 1.0)

    def test_same_seed_reproduces_parameters_and_behavior(self):
        policy = UniformTitForTatOpponent()
        agent_prices = (4.0, 1.0, 2.0, 3.0, 4.5, 4.5)
        policy.reset(seed=101)
        first = [
            policy.get_prices(self.observation(price))["uniform_price"]
            for price in agent_prices
        ]
        first_parameters = (
            policy.threshold,
            policy.punishment_length,
            policy.delta,
            policy.reference_price,
        )

        policy.reset(seed=101)
        second = [
            policy.get_prices(self.observation(price))["uniform_price"]
            for price in agent_prices
        ]

        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            (
                policy.threshold,
                policy.punishment_length,
                policy.delta,
                policy.reference_price,
            ),
            first_parameters,
        )

    def test_invalid_parameters_are_rejected(self):
        invalid_arguments = (
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 2.0, "p_max": 2.0},
            {"mid_price": 4.0, "high_price": 3.0},
            {"threshold_min": -0.1},
            {"threshold_min": 1.0, "threshold_max": 1.0},
            {"delta_min": -0.1},
            {"delta_min": 0.8, "delta_max": 0.8},
            {"punishment_lengths": ()},
            {"punishment_lengths": (0, 3)},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    UniformTitForTatOpponent(**arguments)

    def test_factory_and_curriculum_preset_create_policy(self):
        direct = create_opponent_policy("uniform_tit_for_tat", seed=2)
        preset = create_preset_opponent("uniform_tit_for_tat", seed=2)

        self.assertIsInstance(direct, UniformTitForTatOpponent)
        self.assertIsInstance(preset, UniformTitForTatOpponent)
        self.assertEqual(preset.punishment_lengths, (3, 5, 8))
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
