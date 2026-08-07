"""Controller isolation, recurrent architecture, and checkpoint tests for v2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from env.pricing_contracts import AgentArchitecture
from universal_pricing_v2.agents import HierarchicalPricingAgentFactory
from universal_pricing_v2.environment import (
    HierarchicalPricingEnvironmentFactoryV2,
)
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalTrainingPhase,
    PricingSkill,
    V2ExperimentMatrix,
    load_universal_pricing_v2_protocol,
)
from universal_pricing_v2.replay import (
    PricingSkillTransition,
    StrategyTransition,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v2.yaml"
)


def module_values(controller) -> list[torch.Tensor]:
    modules = (
        controller.actor,
        controller.critic_1,
        controller.critic_2,
        controller.target_critic_1,
        controller.target_critic_2,
    )
    return [
        parameter.detach().clone()
        for module in modules
        for parameter in module.parameters()
    ]


def identical(
    before: list[torch.Tensor], after: list[torch.Tensor]
) -> bool:
    return all(torch.equal(left, right) for left, right in zip(before, after))


class UniversalPricingV2AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_v2_protocol(PROTOCOL_PATH)
        cls.coordinate = V2ExperimentMatrix(cls.protocol).coordinates()[0]
        environment = HierarchicalPricingEnvironmentFactoryV2(
            cls.protocol, episode_length=2, num_consumers=10
        ).create_environment(cls.coordinate)
        cls.observation, _ = environment.reset(seed=71)
        environment.close()

    def small_sac_agent(self):
        profile = replace(
            self.protocol.agent_profiles[AgentArchitecture.SAC],
            batch_size=2,
            pricing_hidden_dimensions=(32, 32),
            strategy_hidden_dimensions=(16, 16),
            pricing_replay_capacity=20,
            strategy_replay_capacity=20,
        )
        return HierarchicalPricingAgentFactory.create(
            profile, self.protocol.run_seed_bundle(0), device="cpu"
        )

    @staticmethod
    def uniform_transition(value: float) -> PricingSkillTransition:
        observation = np.zeros(18, dtype=np.float32)
        return PricingSkillTransition(
            pricing_skill=PricingSkill.UNIFORM,
            observation=observation,
            price_action=np.asarray([value], dtype=np.float32),
            effective_action=np.asarray(
                [1.0, 0.0, value, 0.0, 0.0], dtype=np.float32
            ),
            reward=0.1 + value,
            next_observation=observation,
            done=False,
            active_controller_mask=1.0,
            opponent_price_controls=np.zeros(3, dtype=np.float32),
            stage_key="uniform-stage",
        )

    @staticmethod
    def strategy_transition(action: int) -> StrategyTransition:
        return StrategyTransition(
            observation=np.zeros(19, dtype=np.float32),
            regime_action=action,
            macro_reward=float(action + 1),
            next_observation=np.full(19, 0.1, dtype=np.float32),
            done=False,
            duration=10,
            stage_key="strategy",
        )

    def test_forced_uniform_update_changes_only_uniform_controller(self) -> None:
        agent = self.small_sac_agent()
        for value in (-0.25, 0.35):
            agent.record_pricing_transition(self.uniform_transition(value))
        uniform_before = module_values(agent.uniform_controller)
        bbp_before = module_values(agent.bbp_controller)
        strategy_before = module_values(agent.strategy_controller)
        metrics = agent.update_for_phase(
            HierarchicalTrainingPhase.UNIFORM_PRICING,
            stage_key="uniform-stage",
        )
        self.assertTrue(any(name.startswith("uniform_") for name in metrics))
        self.assertFalse(
            identical(uniform_before, module_values(agent.uniform_controller))
        )
        self.assertTrue(
            identical(bbp_before, module_values(agent.bbp_controller))
        )
        self.assertTrue(
            identical(strategy_before, module_values(agent.strategy_controller))
        )

    def test_frozen_strategy_update_keeps_pricing_byte_identical(self) -> None:
        agent = self.small_sac_agent()
        for regime in (0, 1):
            agent.record_strategy_transition(
                self.strategy_transition(regime)
            )
        uniform_before = module_values(agent.uniform_controller)
        bbp_before = module_values(agent.bbp_controller)
        strategy_before = module_values(agent.strategy_controller)
        metrics = agent.update_for_phase(
            HierarchicalTrainingPhase.STRATEGY_FROZEN,
            stage_key="strategy",
        )
        self.assertTrue(any(name.startswith("strategy_") for name in metrics))
        self.assertTrue(
            identical(uniform_before, module_values(agent.uniform_controller))
        )
        self.assertTrue(
            identical(bbp_before, module_values(agent.bbp_controller))
        )
        self.assertFalse(
            identical(strategy_before, module_values(agent.strategy_controller))
        )

    def test_joint_phase_uses_exactly_one_tenth_price_actor_rate(self) -> None:
        agent = self.small_sac_agent()
        agent.set_training_phase(
            HierarchicalTrainingPhase.JOINT_CONSOLIDATION
        )
        expected = agent.profile.actor_learning_rate * 0.1
        self.assertAlmostEqual(
            agent.profile.joint_price_learning_rate, expected
        )
        self.assertAlmostEqual(
            agent.uniform_controller.actor_optimizer.param_groups[0]["lr"],
            expected,
        )
        self.assertAlmostEqual(
            agent.bbp_controller.actor_optimizer.param_groups[0]["lr"],
            expected,
        )
        self.assertEqual(
            agent.strategy_controller.actor_optimizer.param_groups[0]["lr"],
            agent.profile.actor_learning_rate,
        )

    def test_recurrent_dimensions_and_macro_discount_are_frozen(self) -> None:
        for architecture, embedding_dimension in (
            (AgentArchitecture.RSAC, 0),
            (AgentArchitecture.OE_RSAC, 32),
        ):
            with self.subTest(architecture=architecture.value):
                profile = self.protocol.agent_profiles[architecture]
                agent = HierarchicalPricingAgentFactory.create(
                    profile,
                    self.protocol.run_seed_bundle(0),
                    device="cpu",
                )
                self.assertEqual(
                    agent.uniform_controller.recurrent_input_dimension,
                    21 + embedding_dimension,
                )
                self.assertEqual(
                    agent.bbp_controller.recurrent_input_dimension,
                    22 + embedding_dimension,
                )
                self.assertEqual(
                    agent.strategy_controller.actor.gru.input_size,
                    22 + embedding_dimension,
                )
                self.assertAlmostEqual(
                    agent.strategy_controller.gamma_price,
                    0.99**10,
                )
                for duration in (1, 4, 10):
                    applied = (
                        agent.strategy_controller.gamma_price
                        ** (duration / 10.0)
                    )
                    self.assertAlmostEqual(applied, 0.99**duration)
                if architecture is AgentArchitecture.OE_RSAC:
                    self.assertEqual(
                        agent.opponent_encoder.gru.input_size, 27
                    )
                    self.assertEqual(
                        agent.opponent_encoder.embedding_head.out_features,
                        32,
                    )

    def test_all_architectures_restore_deterministic_online_actions(self) -> None:
        for architecture in AgentArchitecture:
            with self.subTest(architecture=architecture.value):
                profile = self.protocol.agent_profiles[architecture]
                agent = HierarchicalPricingAgentFactory.create(
                    profile,
                    self.protocol.run_seed_bundle(0),
                    device="cpu",
                )
                agent.select_action(
                    self.observation,
                    regime_mode=AgentRegimeMode.LEARNED,
                    deterministic=True,
                )
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "a.pt"
                    agent.save(checkpoint)
                    expected = agent.select_action(
                        self.observation,
                        regime_mode=AgentRegimeMode.LEARNED,
                        deterministic=True,
                    )
                    restored = HierarchicalPricingAgentFactory.create(
                        profile,
                        self.protocol.run_seed_bundle(0),
                        device="cpu",
                    )
                    restored.load(checkpoint)
                    actual = restored.select_action(
                        self.observation,
                        regime_mode=AgentRegimeMode.LEARNED,
                        deterministic=True,
                    )
                self.assertEqual(actual.regime, expected.regime)
                np.testing.assert_allclose(
                    [
                        actual.uniform_control,
                        actual.bbp_new_control,
                        actual.bbp_premium_control,
                    ],
                    [
                        expected.uniform_control,
                        expected.bbp_new_control,
                        expected.bbp_premium_control,
                    ],
                    atol=0.0,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
