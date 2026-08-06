"""Contract tests for the universal pricing Gym environment."""

from pathlib import Path
import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np

from env.pricing_contracts import (
    PricingAction,
    PricingActionCodec,
    PricingObservationCodec,
    PricingRegime,
    REQUIRED_PRICING_STEP_INFO_FIELDS,
)
from env.universal_pricing_env import (
    PricingPriceTransform,
    RegimeCommitmentController,
)
from env.universal_pricing_factory import (
    UniversalPricingEnvironmentFactory,
)
from train.universal_pricing_protocol import (
    ExperimentMatrix,
    load_universal_pricing_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


def gym_action(
    regime: PricingRegime,
    controls: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return PricingActionCodec.to_gym(
        PricingAction(regime, *controls)
    )


class PricingPriceTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = PricingPriceTransform()

    def test_control_endpoints_and_conditional_premium(self) -> None:
        low = self.transform.controls_to_prices(
            PricingAction(PricingRegime.BBP, -1.0, -1.0, -1.0)
        )
        self.assertEqual(low.uniform, 0.5)
        self.assertEqual(low.new, 0.5)
        self.assertEqual(low.old, 1.0)

        high = self.transform.controls_to_prices(
            PricingAction(PricingRegime.BBP, 1.0, 1.0, 1.0)
        )
        self.assertEqual(high.uniform, 5.0)
        self.assertEqual(high.new, 4.0)
        self.assertEqual(high.old, 5.0)

    def test_transform_round_trip(self) -> None:
        for controls in (
            (-1.0, -1.0, -1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (-0.4, 0.25, 0.75),
        ):
            action = PricingAction(PricingRegime.BBP, *controls)
            prices = self.transform.controls_to_prices(action)
            recovered = self.transform.prices_to_controls(prices)
            np.testing.assert_allclose(recovered, controls, atol=1e-6)
            self.assertGreaterEqual(prices.old, prices.new)


class RegimeCommitmentControllerTests(unittest.TestCase):
    def test_first_decision_and_exact_ten_period_commitment(self) -> None:
        controller = RegimeCommitmentController(10)
        first = controller.decide(PricingRegime.BBP)
        self.assertTrue(first.regime_decision_allowed)
        self.assertTrue(first.regime_changed)
        self.assertEqual(first.effective_regime, PricingRegime.BBP)

        for period in range(1, 10):
            controller.advance_period()
            decision = controller.decide(PricingRegime.UNIFORM)
            self.assertFalse(
                decision.regime_decision_allowed,
                msg=f"proposal unexpectedly accepted at period {period}",
            )
            self.assertEqual(decision.effective_regime, PricingRegime.BBP)

        controller.advance_period()
        self.assertTrue(controller.regime_decision_allowed)
        next_decision = controller.decide(PricingRegime.UNIFORM)
        self.assertTrue(next_decision.regime_decision_allowed)
        self.assertTrue(next_decision.regime_changed)


class UniversalPricingEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.coordinate = ExperimentMatrix(cls.protocol).coordinates()[0]

    def make_environment(
        self,
        *,
        episode_length: int = 12,
        num_consumers: int = 50,
    ):
        return UniversalPricingEnvironmentFactory(
            self.protocol,
            num_consumers=num_consumers,
            episode_length=episode_length,
        ).create_environment(self.coordinate)

    def test_gym_checker_and_frozen_spaces(self) -> None:
        environment = self.make_environment(
            episode_length=5,
            num_consumers=10,
        )
        check_env(environment, skip_render_check=True)
        observation, _ = environment.reset(options={"episode_index": 0})
        self.assertTrue(environment.observation_space.contains(observation))
        self.assertEqual(observation.shape, (18,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(
            set(environment.action_space.spaces),
            {"regime", "price_controls"},
        )

    def test_reset_has_neutral_prices_and_immediate_regime_decision(self) -> None:
        environment = self.make_environment()
        observation, info = environment.reset(options={"episode_index": 0})
        decoded = PricingObservationCodec.decode(observation)
        self.assertEqual(decoded["own_market_share"], 0.0)
        self.assertEqual(decoded["opponent_market_share"], 0.0)
        self.assertEqual(decoded["own_uniform_price"], 0.0)
        self.assertEqual(decoded["own_bbp_new_price"], 0.0)
        self.assertEqual(decoded["own_bbp_old_price"], 0.0)
        self.assertEqual(decoded["own_regime"], -1.0)
        self.assertEqual(decoded["regime_commitment_progress"], 1.0)
        self.assertEqual(decoded["regime_decision_allowed"], 1.0)
        self.assertIn("opponent_policy_name", info)
        self.assertNotIn(
            "opponent_policy_name",
            PricingObservationCodec.FEATURE_NAMES,
        )

    def test_first_action_selects_bbp_and_lock_expires_at_period_ten(self) -> None:
        environment = self.make_environment(episode_length=12)
        environment.reset(options={"episode_index": 0})
        for timestep in range(11):
            proposed_regime = (
                PricingRegime.BBP
                if timestep == 0
                else PricingRegime.UNIFORM
            )
            observation, _, _, _, info = environment.step(
                gym_action(proposed_regime)
            )
            if timestep == 0:
                self.assertTrue(info["regime_decision_allowed"])
                self.assertTrue(info["regime_changed"])
            elif timestep < 10:
                self.assertFalse(info["regime_decision_allowed"])
                self.assertEqual(info["agent_regime"], PricingRegime.BBP)
            else:
                self.assertTrue(info["regime_decision_allowed"])
                self.assertTrue(info["regime_changed"])
                self.assertEqual(info["agent_regime"], PricingRegime.UNIFORM)
            decoded = PricingObservationCodec.decode(observation)
            expected_next_permission = 1.0 if timestep == 9 else -1.0
            if timestep == 10:
                expected_next_permission = -1.0
            self.assertEqual(
                decoded["regime_decision_allowed"],
                expected_next_permission,
            )

    def test_inactive_controls_do_not_change_posted_prices(self) -> None:
        environment = self.make_environment()
        environment.reset(options={"episode_index": 0})
        environment.step(
            gym_action(PricingRegime.UNIFORM, (0.0, 1.0, 1.0))
        )
        agent = environment.market.firms[0]
        self.assertEqual(agent.price_new, 2.25)
        self.assertEqual(agent.price_old, 3.0)
        environment.step(
            gym_action(PricingRegime.BBP, (1.0, -1.0, -1.0))
        )
        self.assertEqual(agent.pricing_regime, PricingRegime.UNIFORM)
        self.assertEqual(agent.uniform_price, 5.0)
        self.assertEqual(agent.price_new, 2.25)
        self.assertEqual(agent.price_old, 3.0)

    def test_effective_replay_action_uses_applied_regime(self) -> None:
        environment = self.make_environment()
        environment.reset(options={"episode_index": 0})
        environment.step(gym_action(PricingRegime.BBP))
        _, _, _, _, info = environment.step(
            gym_action(PricingRegime.UNIFORM, (-0.2, 0.3, 0.8))
        )
        replay_action = PricingActionCodec.from_replay_vector(
            info["effective_replay_action"]
        )
        self.assertEqual(replay_action.regime, PricingRegime.BBP)
        self.assertEqual(info["regime_decision_mask"], 0.0)

    def test_reward_and_required_information_are_economically_consistent(self) -> None:
        environment = self.make_environment(num_consumers=50)
        environment.reset(options={"episode_index": 0})
        _, reward, terminated, truncated, info = environment.step(
            gym_action(PricingRegime.UNIFORM)
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            set(REQUIRED_PRICING_STEP_INFO_FIELDS) - set(info),
            set(),
        )
        self.assertAlmostEqual(
            reward,
            info["raw_agent_profit"] / 250.0,
        )
        self.assertEqual(reward, info["normalized_reward"])
        self.assertAlmostEqual(
            info["profit_advantage"],
            info["raw_agent_profit"] - info["raw_opponent_profit"],
        )
        self.assertEqual(info["opponent_price_controls"].shape, (3,))
        self.assertTrue(np.all(np.isfinite(info["opponent_price_controls"])))

    def test_both_opponent_families_complete_full_episodes(self) -> None:
        environment = self.make_environment(episode_length=3)
        families = set()
        for episode_index in (0, 1):
            observation, reset_info = environment.reset(
                options={"episode_index": episode_index}
            )
            families.add(reset_info["opponent_family"])
            self.assertTrue(environment.observation_space.contains(observation))
            for timestep in range(3):
                observation, _, terminated, truncated, info = environment.step(
                    gym_action(PricingRegime(timestep % 2))
                )
                self.assertFalse(terminated)
                self.assertEqual(truncated, timestep == 2)
                self.assertTrue(
                    environment.observation_space.contains(observation)
                )
            self.assertIn("episode_summary", info)
        self.assertEqual(families, {"uniform", "bbp"})

    def test_ad_hoc_seed_and_protocol_index_are_mutually_exclusive(self) -> None:
        environment = self.make_environment()
        with self.assertRaises(ValueError):
            environment.reset(seed=12, options={"episode_index": 0})


if __name__ == "__main__":
    unittest.main()
