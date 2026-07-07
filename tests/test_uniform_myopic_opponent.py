"""Tests for the analytical uniform myopic curriculum opponent."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.models import HotellingMarket
from env.opponent_policies import (
    OpponentObservation,
    PreviousMarketState,
    PriceVector,
    UniformMyopicOpponent,
    create_opponent_policy,
    create_preset_opponent,
)
from env.pricing_env import PricingEnv
from env.type import EnvironmentType


def make_observation(
    *,
    regime=0,
    previous_uniform=2.0,
    previous_new=1.5,
    previous_old=2.5,
    submitted_uniform=None,
    submitted_new=None,
    submitted_old=None,
    established_share=0.0,
):
    """Build an explicitly timed opponent observation for policy tests."""
    submission = None
    if any(
        value is not None
        for value in (submitted_uniform, submitted_new, submitted_old)
    ):
        submission = PriceVector(
            uniform=previous_uniform if submitted_uniform is None else submitted_uniform,
            new=previous_new if submitted_new is None else submitted_new,
            old=previous_old if submitted_old is None else submitted_old,
        )
    return OpponentObservation(
        previous=PreviousMarketState(
            competitor_prices=PriceVector(
                previous_uniform, previous_new, previous_old
            )
        ),
        competitor_submission=submission,
        competitor_established_share=established_share,
        competitor_regime=regime,
    )


class UniformMyopicOpponentTests(unittest.TestCase):
    """Verify analytical formulas, bounds, diagnostics, and integration."""

    def test_uniform_competitor_uses_analytical_response(self):
        policy = UniformMyopicOpponent(
            transport_cost=1.2,
            best_response_offset=0.5,
        )
        observation = make_observation(regime=0, previous_uniform=2.8)

        selected = policy.get_uniform_price(observation)

        expected = (1.2 + 2.8 + 0.5) / 2.0
        self.assertAlmostEqual(selected, expected)
        self.assertAlmostEqual(policy.effective_competitor_price, 2.8)
        self.assertEqual(policy.last_established_share, 0.0)
        self.assertAlmostEqual(policy.last_unclipped_price, expected)

    def test_bbp_competitor_uses_established_share_weighted_price(self):
        policy = UniformMyopicOpponent()
        for established_share in (0.0, 0.5, 1.0):
            with self.subTest(established_share=established_share):
                observation = make_observation(
                    regime=1,
                    previous_new=1.5,
                    previous_old=4.5,
                    established_share=established_share,
                )
                effective_price = (
                    (1.0 - established_share) * 1.5
                    + established_share * 4.5
                )
                expected = (1.0 + effective_price + 0.5) / 2.0

                selected = policy.get_uniform_price(observation)

                self.assertAlmostEqual(selected, expected)
                self.assertAlmostEqual(
                    policy.effective_competitor_price, effective_price
                )
                self.assertAlmostEqual(
                    policy.last_established_share, established_share
                )

    def test_current_submitted_prices_take_precedence_over_previous_prices(self):
        policy = UniformMyopicOpponent()
        uniform = policy.get_uniform_price(make_observation(
            regime=0,
            previous_uniform=5.0,
            submitted_uniform=1.0,
        ))
        bbp = policy.get_uniform_price(make_observation(
            regime=1,
            previous_new=4.0,
            previous_old=5.0,
            submitted_new=1.0,
            submitted_old=3.0,
            established_share=0.25,
        ))

        self.assertAlmostEqual(uniform, (1.0 + 1.0 + 0.5) / 2.0)
        self.assertAlmostEqual(
            bbp, (1.0 + (0.75 * 1.0 + 0.25 * 3.0) + 0.5) / 2.0
        )

    def test_response_is_clipped_to_policy_bounds(self):
        policy = UniformMyopicOpponent(
            p_min=1.5,
            p_max=3.0,
            transport_cost=0.1,
            best_response_offset=-10.0,
        )
        low = policy.get_uniform_price(make_observation(
            regime=0, previous_uniform=0.5
        ))
        self.assertEqual(low, 1.5)
        self.assertLess(policy.last_unclipped_price, 1.5)

        high_policy = UniformMyopicOpponent(
            p_min=1.5,
            p_max=3.0,
            transport_cost=10.0,
        )
        high = high_policy.get_uniform_price(make_observation(
            regime=0, previous_uniform=5.0
        ))
        self.assertEqual(high, 3.0)
        self.assertGreater(high_policy.last_unclipped_price, 3.0)

    def test_market_reports_population_established_share(self):
        market = HotellingMarket(num_consumers=4, seed=4)
        for consumer in market.consumers[:3]:
            consumer.update_purchase(0, 2.0)
            consumer.update_purchase(0, 2.0)

        self.assertEqual(market.get_established_share(0), 0.75)
        self.assertEqual(market.get_established_share(1), 0.0)
        with self.assertRaises(ValueError):
            market.get_established_share(2)

    def test_pricing_environment_uses_current_agent_action(self):
        policy = UniformMyopicOpponent()
        env = PricingEnv(
            environment_type=EnvironmentType.UNIFORM_PRICING,
            opponent_policy=policy,
            num_consumers=4,
            episode_length=1,
            seed=8,
        )
        env.reset(seed=8)
        desired_uniform_price = 1.4
        normalized_uniform = (
            2.0 * (desired_uniform_price - 0.5) / (5.0 - 0.5) - 1.0
        )

        _, _, _, _, info = env.step(np.array([
            normalized_uniform, 0.0, 0.0
        ], dtype=np.float32))

        expected = (1.0 + desired_uniform_price + 0.5) / 2.0
        self.assertAlmostEqual(
            info["opponent_price_uniform"], expected, places=6
        )

    def test_bbp_environment_supplies_current_prices_and_established_share(self):
        policy = UniformMyopicOpponent()
        env = PricingEnv(
            environment_type=EnvironmentType.BBP_PRICING,
            opponent_policy=policy,
            num_consumers=4,
            episode_length=1,
            seed=9,
        )
        env.reset(seed=9)
        for consumer in env.market.consumers[:2]:
            consumer.update_purchase(0, 2.0)
            consumer.update_purchase(0, 2.0)

        desired_new = 1.5
        desired_old = 4.5
        normalized_new = 2.0 * (desired_new - 0.5) / (4.0 - 0.5) - 1.0
        normalized_old = 2.0 * (desired_old - 1.0) / (5.0 - 1.0) - 1.0
        _, _, _, _, info = env.step(np.array([
            0.0, normalized_new, normalized_old
        ], dtype=np.float32))

        established_share = 0.5
        effective_price = (
            (1.0 - established_share) * desired_new
            + established_share * desired_old
        )
        expected = (1.0 + effective_price + 0.5) / 2.0
        self.assertAlmostEqual(
            info["opponent_price_uniform"], expected, places=6
        )
        self.assertAlmostEqual(policy.last_established_share, 0.5)

    def test_invalid_parameters_and_regimes_are_rejected(self):
        invalid_arguments = (
            {"p_min": 0.0},
            {"p_max": 6.0},
            {"p_min": 2.0, "p_max": 2.0},
            {"transport_cost": 0.0},
            {"transport_cost": np.nan},
            {"best_response_offset": np.inf},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    UniformMyopicOpponent(**arguments)

        for invalid_regime in (0.5, 2):
            with self.subTest(invalid_regime=invalid_regime):
                with self.assertRaises(ValueError):
                    UniformMyopicOpponent().get_uniform_price(
                        make_observation(regime=invalid_regime)
                    )

    def test_factory_and_curriculum_preset_use_analytical_policy(self):
        direct = create_opponent_policy("uniform_myopic")
        preset = create_preset_opponent("uniform_myopic")

        self.assertIsInstance(direct, UniformMyopicOpponent)
        self.assertIsInstance(preset, UniformMyopicOpponent)
        self.assertEqual(direct.best_response_offset, 0.5)
        self.assertEqual(preset.transport_cost, 1.0)
        self.assertFalse(hasattr(preset, "candidate_prices"))
        self.assertEqual(preset.regime, 0)


if __name__ == "__main__":
    unittest.main()
