"""Small deterministic learning gates for the Day 3 hybrid SAC."""

from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from env.pricing_contracts import PricingObservationFeature
from models.sac_pricing import (
    HybridPricingActionTensorCodec,
    SACPricingAgent,
    SACPricingAgentConfig,
)
from train.universal_pricing_protocol import load_universal_pricing_protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


class AnalyticPricingCritic(nn.Module):
    """Differentiable synthetic Q function for a declared pricing scenario."""

    def __init__(
        self,
        *,
        uniform_value: float,
        bbp_value: float,
        uniform_target: float,
        bbp_target: tuple[float, float],
    ) -> None:
        super().__init__()
        self.uniform_value = uniform_value
        self.bbp_value = bbp_value
        self.uniform_target = uniform_target
        self.bbp_target = bbp_target

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        del observations
        uniform_q = self.uniform_value - (
            actions[:, 2:3] - self.uniform_target
        ).pow(2)
        bbp_q = (
            self.bbp_value
            - (actions[:, 3:4] - self.bbp_target[0]).pow(2)
            - (actions[:, 4:5] - self.bbp_target[1]).pow(2)
        )
        return (
            actions[:, :1] * uniform_q
            + actions[:, 1:2] * bbp_q
        )


class SACPricingLearningGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.seed_bundle = protocol.run_seed_bundle(0)

    def make_agent(
        self,
        *,
        actor_learning_rate: float = 0.01,
        critic_learning_rate: float = 0.005,
    ) -> SACPricingAgent:
        config = SACPricingAgentConfig(
            actor_hidden_dimensions=(16, 16),
            critic_hidden_dimensions=(16, 16),
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            entropy_learning_rate=0.001,
            initial_regime_temperature=0.01,
            initial_uniform_price_temperature=0.01,
            initial_bbp_price_temperature=0.01,
            replay_capacity=100,
            batch_size=16,
        )
        return SACPricingAgent.from_run_seed_bundle(
            config,
            self.seed_bundle,
        )

    @staticmethod
    def decision_observations(batch_size: int = 64) -> torch.Tensor:
        observations = torch.zeros((batch_size, 18))
        observations[:, PricingObservationFeature.OWN_REGIME.index] = -1.0
        observations[
            :,
            PricingObservationFeature.REGIME_DECISION_ALLOWED.index,
        ] = 1.0
        return observations

    @staticmethod
    def train_actor(
        agent: SACPricingAgent,
        critic: nn.Module,
        updates: int = 250,
    ) -> None:
        agent.critic_1 = critic
        agent.critic_2 = critic
        observations = SACPricingLearningGateTests.decision_observations()
        masks = torch.ones((observations.shape[0], 1))
        for _ in range(updates):
            agent.actor_optimizer.zero_grad(set_to_none=True)
            loss = agent.compute_actor_loss(observations, masks)
            loss.backward()
            agent.actor_optimizer.step()

    def test_equal_value_task_learns_both_price_optima(self) -> None:
        """Equivalent regimes impose no categorical-regime pass criterion."""

        agent = self.make_agent()
        critic = AnalyticPricingCritic(
            uniform_value=1.0,
            bbp_value=1.0,
            uniform_target=0.25,
            bbp_target=(0.25, 0.25),
        )
        self.train_actor(agent, critic)
        with torch.no_grad():
            output = agent.actor(self.decision_observations(1))
            uniform_control = torch.tanh(output.uniform_mean)
            bbp_controls = torch.tanh(output.bbp_mean)
            uniform_q = 1.0 - (uniform_control - 0.25).pow(2)
            bbp_q = (
                1.0
                - (bbp_controls[:, :1] - 0.25).pow(2)
                - (bbp_controls[:, 1:] - 0.25).pow(2)
            )
        self.assertGreater(float(uniform_q), 0.9)
        self.assertGreater(float(bbp_q), 0.9)

    def test_bbp_preferred_task_learns_regime_and_conditional_prices(
        self,
    ) -> None:
        agent = self.make_agent()
        critic = AnalyticPricingCritic(
            uniform_value=1.0,
            bbp_value=3.0,
            uniform_target=0.2,
            bbp_target=(-0.3, 0.4),
        )
        self.train_actor(agent, critic)
        with torch.no_grad():
            output = agent.actor(self.decision_observations(1))
            probabilities = torch.softmax(output.regime_logits, dim=-1)
            bbp_controls = torch.tanh(output.bbp_mean)
        self.assertGreater(float(probabilities[0, 1]), 0.8)
        self.assertAlmostEqual(
            float(bbp_controls[0, 0]),
            -0.3,
            delta=0.12,
        )
        self.assertAlmostEqual(
            float(bbp_controls[0, 1]),
            0.4,
            delta=0.12,
        )
        self.assertGreater(float(bbp_controls[0, 1]), 0.0)

    def test_tiny_replay_batch_overfits_immediate_reward(self) -> None:
        agent = self.make_agent(
            actor_learning_rate=0.001,
            critic_learning_rate=0.005,
        )
        observations = np.zeros((16, 18), dtype=np.float32)
        observations[:, PricingObservationFeature.OWN_REGIME.index] = -1.0
        observations[
            :,
            PricingObservationFeature.REGIME_DECISION_ALLOWED.index,
        ] = 1.0
        actions = np.tile(
            np.asarray([1.0, 0.0, 0.3, 0.8, -0.8], dtype=np.float32),
            (16, 1),
        )
        batch = {
            "observations": observations,
            "effective_actions": actions,
            "rewards": np.full((16, 1), 0.7, dtype=np.float32),
            "next_observations": observations.copy(),
            "dones": np.ones((16, 1), dtype=np.float32),
            "regime_decision_masks": np.ones(
                (16, 1),
                dtype=np.float32,
            ),
            "opponent_price_controls": np.zeros(
                (16, 3),
                dtype=np.float32,
            ),
        }
        observation_tensor = torch.as_tensor(observations)
        action_tensor = (
            HybridPricingActionTensorCodec.canonicalize_replay_actions(
                torch.as_tensor(actions)
            )
        )

        def squared_error() -> float:
            with torch.no_grad():
                return float(
                    (
                        agent.critic_1(
                            observation_tensor,
                            action_tensor,
                        )
                        - 0.7
                    )
                    .pow(2)
                    .mean()
                    + (
                        agent.critic_2(
                            observation_tensor,
                            action_tensor,
                        )
                        - 0.7
                    )
                    .pow(2)
                    .mean()
                )

        initial_error = squared_error()
        for _ in range(80):
            agent.update(batch)
        final_error = squared_error()
        self.assertLess(final_error, 0.001)
        self.assertLess(final_error, initial_error / 100.0)


if __name__ == "__main__":
    unittest.main()
