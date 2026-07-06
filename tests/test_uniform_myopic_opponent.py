"""Tests for the grid-search myopic uniform curriculum opponent."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import (
    OpponentObservation,
    UniformMyopicOpponent,
    create_opponent_policy,
    create_preset_opponent,
)


class UniformMyopicOpponentTests(unittest.TestCase):
    """Verify grid optimization, market-state response, and registration."""

    def test_selects_highest_profit_candidate_on_25_price_grid(self):
        policy = UniformMyopicOpponent(
            p_min=0.5,
            p_max=5.0,
            grid_size=25,
            market_state_weight=0.0,
        )
        observation = OpponentObservation(competitor_uniform_price=2.0)
        prices = np.linspace(0.5, 5.0, 25)
        demand = np.clip(0.5 + (2.0 - prices) / 2.0, 0.0, 1.0)
        profits = prices * demand
        expected_index = int(np.argmax(profits))

        selected = policy.get_uniform_price(observation)

        self.assertEqual(selected, float(prices[expected_index]))
        self.assertEqual(policy.current_price, selected)
        self.assertEqual(
            policy.last_expected_profit, float(profits[expected_index])
        )
        self.assertEqual(len(policy.candidate_prices), 25)

    def test_response_price_increases_against_more_expensive_competitor(self):
        policy = UniformMyopicOpponent(market_state_weight=0.0)
        low_competitor_price = policy.get_uniform_price(
            OpponentObservation(competitor_uniform_price=1.0)
        )
        high_competitor_price = policy.get_uniform_price(
            OpponentObservation(competitor_uniform_price=4.0)
        )

        self.assertGreater(high_competitor_price, low_competitor_price)

    def test_observed_market_share_influences_grid_optimum(self):
        policy = UniformMyopicOpponent(market_state_weight=0.5)
        low_share_price = policy.get_uniform_price(
            OpponentObservation(
                market_share=0.1,
                competitor_uniform_price=2.5,
            )
        )
        high_share_price = policy.get_uniform_price(
            OpponentObservation(
                market_share=0.9,
                competitor_uniform_price=2.5,
            )
        )

        self.assertGreater(high_share_price, low_share_price)

    def test_selected_prices_are_always_valid_grid_prices(self):
        policy = UniformMyopicOpponent(p_min=1.0, p_max=3.0, grid_size=25)
        for competitor_price in np.linspace(0.5, 5.0, 20):
            selected = policy.get_prices(
                OpponentObservation(
                    competitor_uniform_price=float(competitor_price)
                )
            )["uniform_price"]
            self.assertIn(selected, policy.candidate_prices)
            self.assertGreaterEqual(selected, 1.0)
            self.assertLessEqual(selected, 3.0)

    def test_search_is_deterministic_for_identical_market_state(self):
        policy = UniformMyopicOpponent(seed=10)
        observation = OpponentObservation(
            market_share=0.4,
            competitor_uniform_price=2.8,
        )
        prices = [policy.get_uniform_price(observation) for _ in range(5)]
        self.assertEqual(prices, [prices[0]] * 5)

    def test_invalid_parameters_are_rejected(self):
        invalid_arguments = (
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 2.0, "p_max": 2.0},
            {"grid_size": 1},
            {"grid_size": 2.5},
            {"transport_cost": 0.0},
            {"market_state_weight": -0.1},
            {"market_state_weight": 1.1},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    UniformMyopicOpponent(**arguments)

    def test_factory_and_curriculum_preset_create_policy(self):
        direct = create_opponent_policy("uniform_myopic")
        preset = create_preset_opponent("uniform_myopic")

        self.assertIsInstance(direct, UniformMyopicOpponent)
        self.assertIsInstance(preset, UniformMyopicOpponent)
        self.assertEqual(direct.grid_size, 25)
        self.assertEqual(preset.grid_size, 25)
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
