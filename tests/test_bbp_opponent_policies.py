"""Tests for curriculum opponents operating in the BBP pricing regime."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.opponent_policies import (
    BBPAcquisitionPredatorOpponent,
    BBPFixedDiscriminatorOpponent,
    BBPLoyaltyHarvesterOpponent,
    BBPMyopicSegmentOptimizerOpponent,
    OpponentObservation,
    create_opponent_policy,
    create_preset_opponent,
)


class RandomizedBBPOpponentTests(unittest.TestCase):
    """Verify episode randomization and valid segmented price spreads."""

    def setUp(self):
        self.observation = OpponentObservation()

    def test_fixed_discriminator_samples_one_episode_spread(self):
        policy = BBPFixedDiscriminatorOpponent()
        policy.reset(seed=12)
        rng = np.random.RandomState(12)
        expected_base = float(rng.uniform(2.75, 4.0))
        expected_discount = float(rng.uniform(0.5, 2.0))
        expected_markup = float(rng.uniform(0.5, 2.0))
        first = policy.get_bbp_prices(self.observation)

        self.assertEqual(policy.base_price, expected_base)
        self.assertEqual(policy.discount, expected_discount)
        self.assertEqual(policy.markup, expected_markup)
        self.assertEqual(first, policy.get_bbp_prices(self.observation))
        self.assertGreater(first[1], first[0])

    def test_acquisition_predator_keeps_new_price_near_floor(self):
        policy = BBPAcquisitionPredatorOpponent()
        policy.reset(seed=15)
        rng = np.random.RandomState(15)
        expected_epsilon = float(rng.uniform(0.0, 1.0))
        expected_spread = float(rng.uniform(1.0, 4.0))
        price_new, price_old = policy.get_bbp_prices(self.observation)

        self.assertEqual(policy.epsilon, expected_epsilon)
        self.assertEqual(policy.spread, expected_spread)
        self.assertAlmostEqual(price_new, 0.5 + expected_epsilon)
        self.assertGreater(price_old, price_new)

    def test_loyalty_harvester_separates_moderate_and_high_prices(self):
        policy = BBPLoyaltyHarvesterOpponent()
        policy.reset(seed=22)
        price_new, price_old = policy.get_bbp_prices(self.observation)

        self.assertGreaterEqual(price_new, 2.0)
        self.assertLess(price_new, 3.0)
        self.assertGreaterEqual(price_old, 3.5)
        self.assertLess(price_old, 5.0)
        self.assertGreater(price_old, price_new)

    def test_randomized_bbp_sequences_reproduce_from_episode_seed(self):
        policies = (
            BBPFixedDiscriminatorOpponent(),
            BBPAcquisitionPredatorOpponent(),
            BBPLoyaltyHarvesterOpponent(),
        )
        for policy in policies:
            with self.subTest(policy=type(policy).__name__):
                policy.reset(seed=91)
                first = policy.get_bbp_prices(self.observation)
                policy.reset(seed=91)
                second = policy.get_bbp_prices(self.observation)
                self.assertEqual(first, second)
                self.assertEqual(policy.regime, 1)

    def test_invalid_randomized_policy_parameters_are_rejected(self):
        invalid_cases = (
            (BBPFixedDiscriminatorOpponent, {"mid_price": 4.0, "high_price": 3.0}),
            (BBPFixedDiscriminatorOpponent, {"discount_min": 1.0, "discount_max": 1.0}),
            (BBPAcquisitionPredatorOpponent, {"epsilon_min": -0.1}),
            (BBPAcquisitionPredatorOpponent, {"spread_min": 1.0, "spread_max": 1.0}),
            (BBPLoyaltyHarvesterOpponent, {"mid_low": 3.0, "mid_price": 2.0}),
            (BBPLoyaltyHarvesterOpponent, {"mid_high": 2.5}),
        )
        for policy_class, arguments in invalid_cases:
            with self.subTest(policy=policy_class.__name__, arguments=arguments):
                with self.assertRaises(ValueError):
                    policy_class(**arguments)


class BBPMyopicSegmentOptimizerTests(unittest.TestCase):
    """Verify constrained segmented grid optimization and state response."""

    def setUp(self):
        self.policy = BBPMyopicSegmentOptimizerOpponent(
            grid_size=9,
            min_spread=0.25,
            market_state_weight=0.0,
        )

    def test_selects_global_best_valid_grid_pair(self):
        observation = OpponentObservation(
            competitor_price_new=2.0,
            competitor_price_old=4.0,
            new_old_ratio=0.4,
            last_demand_ratio=0.5,
        )
        profits = self.policy.expected_profits(observation)
        expected_indices = np.unravel_index(int(np.argmax(profits)), profits.shape)

        price_new, price_old = self.policy.get_bbp_prices(observation)

        self.assertEqual(
            price_new,
            float(self.policy.new_candidate_prices[expected_indices[0]]),
        )
        self.assertEqual(
            price_old,
            float(self.policy.old_candidate_prices[expected_indices[1]]),
        )
        self.assertGreaterEqual(price_old, price_new + self.policy.min_spread)
        self.assertEqual(
            self.policy.last_expected_profit,
            float(profits[expected_indices]),
        )

    def test_invalid_spread_pairs_never_win(self):
        observation = OpponentObservation(
            competitor_price_new=5.0,
            competitor_price_old=1.0,
        )
        profits = self.policy.expected_profits(observation)
        new_grid = self.policy.new_candidate_prices[:, None]
        old_grid = self.policy.old_candidate_prices[None, :]
        invalid = old_grid < new_grid + self.policy.min_spread
        self.assertTrue(np.all(np.isneginf(profits[invalid])))

    def test_segment_mix_changes_the_selected_prices(self):
        constrained_policy = BBPMyopicSegmentOptimizerOpponent(
            grid_size=9,
            min_spread=0.75,
            market_state_weight=0.0,
        )
        mostly_new = OpponentObservation(
            competitor_price_new=1.0,
            competitor_price_old=1.0,
            new_old_ratio=0.9,
            last_demand_ratio=0.5,
        )
        mostly_old = OpponentObservation(
            competitor_price_new=1.0,
            competitor_price_old=1.0,
            new_old_ratio=0.1,
            last_demand_ratio=0.5,
        )
        new_focused_pair = constrained_policy.get_bbp_prices(mostly_new)
        old_focused_pair = constrained_policy.get_bbp_prices(mostly_old)
        self.assertNotEqual(new_focused_pair, old_focused_pair)

    def test_optimizer_is_deterministic_and_bounded(self):
        observation = OpponentObservation(
            market_share=0.7,
            competitor_price_new=2.5,
            competitor_price_old=4.5,
            new_old_ratio=0.5,
        )
        pairs = [self.policy.get_bbp_prices(observation) for _ in range(4)]
        self.assertEqual(pairs, [pairs[0]] * 4)
        price_new, price_old = pairs[0]
        self.assertGreaterEqual(price_new, 0.5)
        self.assertLessEqual(price_new, 4.0)
        self.assertGreaterEqual(price_old, 1.0)
        self.assertLessEqual(price_old, 5.0)

    def test_invalid_optimizer_parameters_are_rejected(self):
        invalid_arguments = (
            {"grid_size": 1},
            {"grid_size": 2.5},
            {"min_spread": -0.1},
            {"transport_cost": 0.0},
            {"market_state_weight": -0.1},
            {"market_state_weight": 1.1},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    BBPMyopicSegmentOptimizerOpponent(**arguments)


class BBPOpponentFactoryTests(unittest.TestCase):
    """Verify all BBP curriculum names resolve through existing factories."""

    def test_factory_and_presets_create_expected_classes(self):
        expected = {
            "bbp_fixed_discriminator": BBPFixedDiscriminatorOpponent,
            "bbp_acquisition_predator": BBPAcquisitionPredatorOpponent,
            "bbp_loyalty_harvester": BBPLoyaltyHarvesterOpponent,
            "bbp_myopic_segment_optimizer": BBPMyopicSegmentOptimizerOpponent,
        }
        for name, expected_class in expected.items():
            with self.subTest(name=name):
                self.assertIsInstance(
                    create_opponent_policy(name, seed=3), expected_class
                )
                preset = create_preset_opponent(name, seed=3)
                self.assertIsInstance(preset, expected_class)
                self.assertEqual(preset.regime, 1)


if __name__ == "__main__":
    unittest.main()
