"""Tests for the frozen universal pricing action and observation contracts."""

import unittest

import numpy as np

from env.pricing_contracts import (
    AGENT_ARCHITECTURE_SPECS,
    AgentArchitecture,
    PricingAction,
    PricingActionCodec,
    PricingObservationCodec,
    PricingObservationFeature,
    PricingRegime,
)


class PricingActionCodecTests(unittest.TestCase):
    def test_round_trip_for_both_regimes(self) -> None:
        for regime in PricingRegime:
            with self.subTest(regime=regime):
                action = PricingAction(regime, -1.0, 0.25, 1.0)
                self.assertEqual(
                    PricingActionCodec.from_gym(
                        PricingActionCodec.to_gym(action)
                    ),
                    action,
                )
                self.assertEqual(
                    PricingActionCodec.from_replay_vector(
                        PricingActionCodec.to_replay_vector(action)
                    ),
                    action,
                )

    def test_replay_vector_has_frozen_order_and_length(self) -> None:
        action = PricingAction(PricingRegime.BBP, -0.5, 0.2, 0.7)
        vector = PricingActionCodec.to_replay_vector(action)
        self.assertEqual(vector.shape, (5,))
        np.testing.assert_allclose(vector, [0.0, 1.0, -0.5, 0.2, 0.7])

    def test_invalid_gym_actions_are_rejected(self) -> None:
        invalid_actions = (
            {"regime": 2, "price_controls": [0.0, 0.0, 0.0]},
            {"regime": True, "price_controls": [0.0, 0.0, 0.0]},
            {"regime": 0, "price_controls": [0.0, 0.0]},
            {"regime": 0, "price_controls": [0.0, np.nan, 0.0]},
            {"regime": 0, "price_controls": [0.0, np.inf, 0.0]},
            {"regime": 0, "price_controls": [0.0, 1.1, 0.0]},
        )
        for action in invalid_actions:
            with self.subTest(action=action), self.assertRaises(
                (TypeError, ValueError)
            ):
                PricingActionCodec.from_gym(action)

    def test_invalid_replay_one_hot_is_rejected(self) -> None:
        for prefix in ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                PricingActionCodec.from_replay_vector(
                    [*prefix, 0.0, 0.0, 0.0]
                )


class PricingObservationCodecTests(unittest.TestCase):
    def test_frozen_feature_order(self) -> None:
        self.assertEqual(
            PricingObservationCodec.FEATURE_NAMES,
            (
                "own_market_share",
                "opponent_market_share",
                "own_uniform_price",
                "own_bbp_new_price",
                "own_bbp_old_price",
                "opponent_uniform_price",
                "opponent_bbp_new_price",
                "opponent_bbp_old_price",
                "own_demand_ratio",
                "own_new_customer_ratio",
                "own_retention_rate",
                "own_regime",
                "opponent_regime",
                "own_profit_trend",
                "own_popularity_change",
                "episode_progress",
                "regime_commitment_progress",
                "regime_decision_allowed",
            ),
        )
        self.assertEqual(
            PricingObservationFeature.REGIME_DECISION_ALLOWED.index,
            17,
        )

    def test_named_features_encode_in_documented_order(self) -> None:
        expected = np.linspace(-1.0, 1.0, 18, dtype=np.float32)
        named = {
            feature: float(expected[index])
            for index, feature in enumerate(PricingObservationFeature)
        }
        vector = PricingObservationCodec.encode(named)
        np.testing.assert_allclose(vector, expected)
        self.assertEqual(
            PricingObservationCodec.decode(vector)["own_market_share"],
            -1.0,
        )

    def test_invalid_feature_count_bounds_and_finiteness_are_rejected(self) -> None:
        invalid_vectors = (
            np.zeros(17),
            np.zeros(19),
            np.asarray([0.0] * 17 + [1.01]),
            np.asarray([0.0] * 17 + [np.nan]),
            np.asarray([0.0] * 17 + [np.inf]),
        )
        for vector in invalid_vectors:
            with self.subTest(shape=vector.shape), self.assertRaises(ValueError):
                PricingObservationCodec.validate_vector(vector)


class AgentArchitectureContractTests(unittest.TestCase):
    def test_exact_architectures_and_reserved_class_names(self) -> None:
        self.assertEqual(
            {architecture.value for architecture in AgentArchitecture},
            {"sac", "rsac", "oe_rsac"},
        )
        self.assertEqual(
            {
                specification.implementation_class_name
                for specification in AGENT_ARCHITECTURE_SPECS.values()
            },
            {
                "SACPricingAgent",
                "RecurrentSACPricingAgent",
                "OpponentEmbeddingRecurrentSACPricingAgent",
            },
        )


if __name__ == "__main__":
    unittest.main()
