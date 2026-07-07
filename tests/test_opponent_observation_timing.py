"""Tests for the pricing environment's opponent decision-time contract."""

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import OpponentObservation, OpponentPolicy
from env.pricing_env import PricingEnv
from env.type import EnvironmentType


class RecordingOpponent(OpponentPolicy):
    """Record observations while posting a harmless fixed uniform price."""

    def __init__(self):
        super().__init__(regime=0)
        self.observations = []

    def get_uniform_price(self, observation):
        self.observations.append(observation)
        return 2.0

    def get_bbp_prices(self, observation):
        return 2.0, 2.0


def normalized_uniform(price):
    return 2.0 * (price - 0.5) / (5.0 - 0.5) - 1.0


class OpponentObservationTimingTests(unittest.TestCase):
    """Separate completed-period state from the current price submission."""

    def test_observation_carries_previous_state_and_current_submission(self):
        opponent = RecordingOpponent()
        env = PricingEnv(
            environment_type=EnvironmentType.UNIFORM_PRICING,
            opponent_policy=opponent,
            num_consumers=10,
            episode_length=3,
            seed=12,
        )
        env.reset(seed=12)

        first_price = 1.4
        first_action = np.array([
            normalized_uniform(first_price), 0.0, 0.0
        ], dtype=np.float32)
        _, _, _, _, first_info = env.step(first_action)

        first_observation = opponent.observations[0]
        self.assertEqual(first_observation.decision_period, 0)
        self.assertEqual(first_observation.state_period, -1)
        self.assertEqual(
            first_observation.previous.competitor_prices.uniform, 5.0
        )
        self.assertAlmostEqual(
            first_observation.competitor_submission.uniform,
            first_price,
            places=6,
        )

        second_price = 3.2
        second_action = np.array([
            normalized_uniform(second_price), 0.0, 0.0
        ], dtype=np.float32)
        env.step(second_action)

        second_observation = opponent.observations[1]
        self.assertEqual(second_observation.decision_period, 1)
        self.assertEqual(second_observation.state_period, 0)
        self.assertAlmostEqual(
            second_observation.previous.competitor_prices.uniform,
            first_price,
            places=6,
        )
        self.assertAlmostEqual(
            second_observation.competitor_submission.uniform,
            second_price,
            places=6,
        )
        self.assertAlmostEqual(
            second_observation.previous.competitor_market_share,
            first_info["market_share"],
        )

    def test_observation_and_previous_snapshot_are_immutable(self):
        observation = OpponentObservation()
        with self.assertRaises(FrozenInstanceError):
            observation.decision_period = 1
        with self.assertRaises(FrozenInstanceError):
            observation.previous.own_market_share = 0.9


if __name__ == "__main__":
    unittest.main()
