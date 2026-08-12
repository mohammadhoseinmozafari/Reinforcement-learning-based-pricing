"""Contract, optimization, and checkpoint tests for ``SACPricingAgent``."""

from pathlib import Path
import random
import tempfile
from types import MethodType
import unittest

import numpy as np
import torch

from env.pricing_contracts import (
    PricingActionCodec,
    PricingObservationFeature,
    PricingRegime,
)
from env.pricing_factory import UniversalPricingEnvironmentFactory
from models.sac_pricing import (
    HybridPricingActionTensorCodec,
    HybridPricingPolicyOutput,
    SACPricingAgent,
    SACPricingAgentConfig,
)
from models.universal_pricing_replay import (
    UniversalPricingReplayBuffer,
    UniversalPricingTransition,
)
from train.universal_pricing_protocol import (
    AgentArchitecture,
    ExperimentMatrix,
    load_universal_pricing_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


def observation(
    *,
    regime: PricingRegime = PricingRegime.UNIFORM,
    decision_allowed: bool = True,
) -> np.ndarray:
    value = np.zeros(18, dtype=np.float32)
    value[PricingObservationFeature.OWN_REGIME.index] = (
        -1.0 if regime is PricingRegime.UNIFORM else 1.0
    )
    value[PricingObservationFeature.REGIME_DECISION_ALLOWED.index] = (
        1.0 if decision_allowed else -1.0
    )
    return value


def replay_batch(
    *,
    batch_size: int = 8,
    regime: PricingRegime = PricingRegime.UNIFORM,
    decision_allowed: bool = True,
    done: bool = False,
) -> dict[str, np.ndarray]:
    current = observation(
        regime=regime,
        decision_allowed=decision_allowed,
    )
    one_hot = (
        [1.0, 0.0]
        if regime is PricingRegime.UNIFORM
        else [0.0, 1.0]
    )
    return {
        "observations": np.tile(current, (batch_size, 1)),
        "effective_actions": np.asarray(
            [one_hot + [0.2, -0.3, 0.4]] * batch_size,
            dtype=np.float32,
        ),
        "rewards": np.linspace(
            0.0, 0.5, batch_size, dtype=np.float32
        ).reshape(-1, 1),
        "next_observations": np.tile(current, (batch_size, 1)),
        "dones": np.full((batch_size, 1), float(done), dtype=np.float32),
        "regime_decision_masks": np.full(
            (batch_size, 1),
            float(decision_allowed),
            dtype=np.float32,
        ),
        "opponent_price_controls": np.zeros(
            (batch_size, 3),
            dtype=np.float32,
        ),
    }


class SACPricingAgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.seed_bundle = cls.protocol.run_seed_bundle(0)

    def make_agent(
        self,
        *,
        network_seed_offset: int = 0,
    ) -> SACPricingAgent:
        config = SACPricingAgentConfig(
            actor_hidden_dimensions=(32, 32),
            critic_hidden_dimensions=(32, 32),
            replay_capacity=100,
            batch_size=8,
        )
        bundle = self.seed_bundle
        return SACPricingAgent(
            config,
            network_initialization_seed=(
                bundle.network_initialization_seed + network_seed_offset
            ),
            exploration_seed=bundle.exploration_seed,
            torch_cpu_seed=bundle.torch_cpu_seed,
            torch_cuda_seed=bundle.torch_cuda_seed,
        )


class HybridActionTests(SACPricingAgentTestCase):
    def test_canonical_action_masking(self) -> None:
        actions = torch.tensor(
            [
                [1.0, 0.0, 0.25, -0.8, 0.7],
                [0.0, 1.0, -0.9, -0.4, 0.6],
            ]
        )
        canonical = (
            HybridPricingActionTensorCodec.canonicalize_replay_actions(
                actions
            )
        )
        torch.testing.assert_close(
            canonical,
            torch.tensor(
                [
                    [1.0, 0.0, 0.25, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.4, 0.6],
                ]
            ),
        )

    def test_actor_shapes_bounds_and_deterministic_actions(self) -> None:
        agent = self.make_agent()
        values = torch.zeros((7, 18))
        output = agent.actor(values)
        self.assertEqual(output.regime_logits.shape, (7, 2))
        self.assertEqual(output.uniform_mean.shape, (7, 1))
        self.assertEqual(output.bbp_mean.shape, (7, 2))
        probabilities = torch.softmax(output.regime_logits, dim=-1)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(7),
        )

        first = agent.select_action(
            observation(decision_allowed=True),
            deterministic=True,
        )
        repeated = agent.select_action(
            observation(decision_allowed=True),
            deterministic=True,
        )
        self.assertEqual(first, repeated)
        for control in (
            first.uniform_control,
            first.bbp_new_control,
            first.bbp_premium_control,
        ):
            self.assertGreaterEqual(control, -1.0)
            self.assertLessEqual(control, 1.0)

    def test_locked_state_uses_observed_regime_and_zeros_inactive_head(self) -> None:
        agent = self.make_agent()
        uniform = agent.select_action(
            observation(
                regime=PricingRegime.UNIFORM,
                decision_allowed=False,
            ),
            deterministic=True,
        )
        self.assertEqual(uniform.regime, PricingRegime.UNIFORM)
        self.assertEqual(uniform.bbp_new_control, 0.0)
        self.assertEqual(uniform.bbp_premium_control, 0.0)

        bbp = agent.select_action(
            observation(
                regime=PricingRegime.BBP,
                decision_allowed=False,
            ),
            deterministic=True,
        )
        self.assertEqual(bbp.regime, PricingRegime.BBP)
        self.assertEqual(bbp.uniform_control, 0.0)

    def test_update_returns_finite_named_metrics(self) -> None:
        metrics = self.make_agent().update(replay_batch())
        expected = {
            "critic_loss",
            "actor_loss",
            "regime_temperature_loss",
            "uniform_temperature_loss",
            "bbp_temperature_loss",
            "regime_temperature",
            "uniform_price_temperature",
            "bbp_price_temperature",
            "target_q_mean",
            "q1_mean",
            "q2_mean",
            "decision_fraction",
            "critic_gradient_norm",
            "actor_gradient_norm",
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_critic_target_exactly_marginalizes_eligible_regimes(self) -> None:
        agent = self.make_agent()

        class RegimeValueCritic(torch.nn.Module):
            def forward(self, observations, actions):
                del observations
                return actions[:, :1] * 1.0 + actions[:, 1:2] * 3.0

        agent.target_critic_1 = RegimeValueCritic()
        agent.target_critic_2 = RegimeValueCritic()

        def fixed_policy_branches(self, observations):
            batch_size = observations.shape[0]
            logits = torch.log(
                torch.tensor(
                    [[0.25, 0.75]],
                    dtype=observations.dtype,
                    device=observations.device,
                ).repeat(batch_size, 1)
            )
            zeros_1 = torch.zeros(
                (batch_size, 1),
                dtype=observations.dtype,
                device=observations.device,
            )
            zeros_2 = torch.zeros(
                (batch_size, 2),
                dtype=observations.dtype,
                device=observations.device,
            )
            return (
                HybridPricingPolicyOutput(
                    regime_logits=logits,
                    uniform_mean=zeros_1,
                    uniform_log_std=zeros_1,
                    bbp_mean=zeros_2,
                    bbp_log_std=zeros_2,
                ),
                HybridPricingActionTensorCodec.uniform_actions(zeros_1),
                torch.full_like(zeros_1, -0.5),
                HybridPricingActionTensorCodec.bbp_actions(zeros_2),
                torch.full_like(zeros_1, -1.0),
            )

        agent._sample_policy_branches = MethodType(
            fixed_policy_branches,
            agent,
        )
        next_observations = torch.as_tensor(
            np.stack(
                [
                    observation(decision_allowed=True),
                    observation(
                        regime=PricingRegime.BBP,
                        decision_allowed=False,
                    ),
                ]
            )
        )
        rewards = torch.tensor([[0.1], [0.2]])
        dones = torch.zeros((2, 1))
        actual = agent.compute_critic_target(
            rewards,
            next_observations,
            dones,
        )
        temperature = 0.2
        uniform_soft_q = 1.0 - temperature * -0.5
        bbp_soft_q = 3.0 - temperature * -1.0
        eligible_value = (
            0.25
            * (uniform_soft_q - temperature * np.log(0.25))
            + 0.75 * (bbp_soft_q - temperature * np.log(0.75))
        )
        expected = torch.tensor(
            [
                [0.1 + agent.config.gamma * eligible_value],
                [0.2 + agent.config.gamma * bbp_soft_q],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(actual.requires_grad)


class GradientMaskTests(SACPricingAgentTestCase):
    @staticmethod
    def gradient_norm(parameters) -> float:
        return float(
            sum(
                parameter.grad.abs().sum()
                for parameter in parameters
                if parameter.grad is not None
            )
        )

    def actor_head_gradient_norms(
        self,
        regime: PricingRegime,
        decision_allowed: bool,
    ) -> tuple[float, float, float]:
        agent = self.make_agent()
        batch = replay_batch(
            regime=regime,
            decision_allowed=decision_allowed,
        )
        observations = torch.as_tensor(batch["observations"])
        masks = torch.as_tensor(batch["regime_decision_masks"])
        agent.actor.zero_grad(set_to_none=True)
        agent.compute_actor_loss(observations, masks).backward()
        regime_norm = self.gradient_norm(
            agent.actor.regime_logits.parameters()
        )
        uniform_norm = self.gradient_norm(
            list(agent.actor.uniform_mean.parameters())
            + list(agent.actor.uniform_log_std.parameters())
        )
        bbp_norm = self.gradient_norm(
            list(agent.actor.bbp_mean.parameters())
            + list(agent.actor.bbp_log_std.parameters())
        )
        return regime_norm, uniform_norm, bbp_norm

    def test_decision_batch_updates_all_actor_heads(self) -> None:
        norms = self.actor_head_gradient_norms(
            PricingRegime.UNIFORM,
            True,
        )
        self.assertTrue(all(norm > 0.0 for norm in norms))

    def test_locked_uniform_masks_regime_and_bbp_heads(self) -> None:
        regime, uniform, bbp = self.actor_head_gradient_norms(
            PricingRegime.UNIFORM,
            False,
        )
        self.assertEqual(regime, 0.0)
        self.assertGreater(uniform, 0.0)
        self.assertEqual(bbp, 0.0)

    def test_locked_bbp_masks_regime_and_uniform_heads(self) -> None:
        regime, uniform, bbp = self.actor_head_gradient_norms(
            PricingRegime.BBP,
            False,
        )
        self.assertEqual(regime, 0.0)
        self.assertEqual(uniform, 0.0)
        self.assertGreater(bbp, 0.0)

    def test_locked_batch_skips_regime_temperature_optimizer(self) -> None:
        agent = self.make_agent()
        before = agent.log_regime_temperature.detach().clone()
        metrics = agent.update(
            replay_batch(
                regime=PricingRegime.UNIFORM,
                decision_allowed=False,
            )
        )
        torch.testing.assert_close(before, agent.log_regime_temperature)
        self.assertEqual(metrics["regime_temperature_loss"], 0.0)
        self.assertEqual(metrics["bbp_temperature_loss"], 0.0)


class ReplayAndSeedTests(SACPricingAgentTestCase):
    @staticmethod
    def make_transition(index: int) -> UniversalPricingTransition:
        regime = (
            PricingRegime.UNIFORM
            if index % 2 == 0
            else PricingRegime.BBP
        )
        one_hot = (
            [1.0, 0.0]
            if regime is PricingRegime.UNIFORM
            else [0.0, 1.0]
        )
        return UniversalPricingTransition(
            observation=observation(regime=regime),
            effective_action=np.asarray(
                one_hot + [0.1, -0.2, 0.3],
                dtype=np.float32,
            ),
            reward=float(index) / 10.0,
            next_observation=observation(
                regime=regime,
                decision_allowed=False,
            ),
            done=False,
            regime_decision_mask=1.0,
            opponent_price_controls=np.zeros(3, dtype=np.float32),
        )

    def test_replay_sampling_is_private_and_reproducible(self) -> None:
        first = UniversalPricingReplayBuffer(20, 12345)
        repeated = UniversalPricingReplayBuffer(20, 12345)
        for index in range(15):
            first.push(self.make_transition(index))
            repeated.push(self.make_transition(index))
        np.random.seed(10)
        np.random.random(100)
        random.seed(20)
        for _ in range(100):
            random.random()
        torch.manual_seed(30)
        torch.rand(100)
        first_batch = first.sample(8)
        repeated_batch = repeated.sample(8)
        for field_name in first_batch.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(first_batch, field_name),
                getattr(repeated_batch, field_name),
            )

    def test_invalid_transition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UniversalPricingTransition(
                observation=observation(),
                effective_action=np.asarray(
                    [1.0, 0.0, np.nan, 0.0, 0.0]
                ),
                reward=0.0,
                next_observation=observation(),
                done=False,
                regime_decision_mask=1.0,
                opponent_price_controls=np.zeros(3),
            )

    def test_initialization_and_actions_ignore_global_rng_state(self) -> None:
        expected = self.make_agent()
        expected_action = expected.select_action(observation())
        expected_parameters = {
            name: value.detach().clone()
            for name, value in expected.actor.state_dict().items()
        }

        np.random.seed(90)
        np.random.random(100)
        random.seed(91)
        for _ in range(100):
            random.random()
        torch.manual_seed(92)
        torch.rand(100)

        repeated = self.make_agent()
        self.assertEqual(
            expected_action,
            repeated.select_action(observation()),
        )
        for name, value in repeated.actor.state_dict().items():
            torch.testing.assert_close(value, expected_parameters[name])


class CheckpointAndIntegrationTests(SACPricingAgentTestCase):
    def test_checkpoint_restores_parameters_optimizers_and_rng_sequences(
        self,
    ) -> None:
        agent = self.make_agent()
        agent.update(replay_batch())
        agent.select_action(observation())
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "agent.pt"
            agent.save(checkpoint)
            expected_action = agent.select_action(observation())
            expected_update = agent.update(replay_batch())

            restored = self.make_agent(network_seed_offset=100)
            restored.load(checkpoint)
            self.assertEqual(
                expected_action,
                restored.select_action(observation()),
            )
            restored_update = restored.update(replay_batch())
            for metric_name in expected_update:
                self.assertAlmostEqual(
                    expected_update[metric_name],
                    restored_update[metric_name],
                    places=6,
                )

    def test_checkpoint_rejects_incompatible_configuration(self) -> None:
        agent = self.make_agent()
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "agent.pt"
            agent.save(checkpoint)
            incompatible = SACPricingAgent(
                SACPricingAgentConfig(
                    actor_hidden_dimensions=(16, 16),
                    critic_hidden_dimensions=(16, 16),
                    replay_capacity=100,
                    batch_size=8,
                ),
                network_initialization_seed=1,
                exploration_seed=2,
                torch_cpu_seed=3,
                torch_cuda_seed=4,
            )
            with self.assertRaises(ValueError):
                incompatible.load(checkpoint)

    def test_short_real_environment_rollout_populates_replay(self) -> None:
        coordinate = next(
            coordinate
            for coordinate in ExperimentMatrix(self.protocol).coordinates()
            if coordinate.agent_architecture is AgentArchitecture.SAC
        )
        environment = UniversalPricingEnvironmentFactory(
            self.protocol,
            num_consumers=20,
            episode_length=4,
        ).create_environment(coordinate)
        agent = self.make_agent()
        replay = UniversalPricingReplayBuffer(
            10,
            self.seed_bundle.replay_sampling_seed,
        )
        current_observation, _ = environment.reset(
            options={"episode_index": 0}
        )
        for _ in range(4):
            selected_action = agent.select_action(current_observation)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(PricingActionCodec.to_gym(selected_action))
            replay.push(
                UniversalPricingTransition.from_environment_step(
                    observation=current_observation,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            current_observation = next_observation
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.sample(4).effective_actions.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
