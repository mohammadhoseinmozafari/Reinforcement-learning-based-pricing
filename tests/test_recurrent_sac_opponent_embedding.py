"""Tests for recurrent SAC and integration with episode sequence replay.

The tests intentionally use the repository's real ``EpisodeReplayBuffer`` and
``CurriculumSequenceReplayBuffer``. Small network dimensions keep the suite
fast while exercising padding, masks, recurrent state, optimization, and
checkpoint restoration.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.buffer import EpisodeReplayBuffer, CurriculumSequenceReplayBuffer
from models.recurrent_sac_opponent_embedding import (
    OpponentEncoder,
    RecurrentActor,
    RecurrentCritic,
    RecurrentSACOpponentEmbeddingAgent,
)


OBS_DIM = 4
ACTION_DIM = 2
OPPONENT_ACTION_DIM = 2
SEQUENCE_LENGTH = 4


class FakeCurriculum:
    """Minimal two-stage curriculum implementing the replay-buffer contract."""

    def __init__(self):
        self.stages = [
            SimpleNamespace(opponent_type="easy"),
            SimpleNamespace(opponent_type="hard"),
        ]

    def get_sequence(self):
        return self.stages


def make_episode(length: int, offset: float = 0.0) -> dict:
    """Create a complete recurrent episode with consistent feature shapes."""
    rng = np.random.default_rng(7 + int(offset))
    return {
        "obs": rng.normal(size=(length, OBS_DIM)).astype(np.float32) + offset,
        "actions": np.tanh(rng.normal(size=(length, ACTION_DIM))).astype(np.float32),
        "rewards": rng.normal(size=(length, 1)).astype(np.float32),
        "next_obs": rng.normal(size=(length, OBS_DIM)).astype(np.float32) + offset,
        "dones": np.zeros((length, 1), dtype=np.float32),
        "opponent_actions": np.tanh(
            rng.normal(size=(length, OPPONENT_ACTION_DIM))
        ).astype(np.float32),
    }


def make_agent() -> RecurrentSACOpponentEmbeddingAgent:
    """Return a small CPU agent suitable for deterministic unit tests."""
    torch.manual_seed(3)
    return RecurrentSACOpponentEmbeddingAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        opponent_action_dim=OPPONENT_ACTION_DIM,
        opponent_embedding_dim=5,
        encoder_hidden_dim=8,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        device="cpu",
    )


class RecurrentModuleShapeTests(unittest.TestCase):
    """Verify the public sequence shapes of encoder, actor, and twin critic."""

    def test_network_shapes_and_action_bounds(self):
        batch_size, timesteps, embedding_dim = 3, 5, 6
        obs = torch.randn(batch_size, timesteps, OBS_DIM)
        actions = torch.randn(batch_size, timesteps, ACTION_DIM)
        rewards = torch.randn(batch_size, timesteps, 1)
        opponent_actions = torch.randn(
            batch_size, timesteps, OPPONENT_ACTION_DIM
        )
        interaction = torch.cat([obs, actions, rewards, opponent_actions], dim=-1)

        encoder = OpponentEncoder(
            OBS_DIM, ACTION_DIM, OPPONENT_ACTION_DIM,
            hidden_dim=9, opponent_embedding_dim=embedding_dim,
        )
        z_seq, z_last, prediction, _ = encoder(interaction)
        self.assertEqual(z_seq.shape, (batch_size, timesteps, embedding_dim))
        self.assertEqual(z_last.shape, (batch_size, embedding_dim))
        self.assertEqual(
            prediction.shape, (batch_size, timesteps, OPPONENT_ACTION_DIM)
        )

        actor = RecurrentActor(
            OBS_DIM, ACTION_DIM, embedding_dim,
            hidden_dim=9,
        )
        sampled, log_prob, mean_action, _ = actor(obs, z_seq)
        self.assertEqual(sampled.shape, (batch_size, timesteps, ACTION_DIM))
        self.assertEqual(log_prob.shape, (batch_size, timesteps, 1))
        self.assertEqual(mean_action.shape, sampled.shape)
        self.assertTrue(torch.all(sampled.abs() <= 1.0))

        critic = RecurrentCritic(OBS_DIM, ACTION_DIM, embedding_dim, hidden_dim=9)
        previous_actions = torch.zeros_like(sampled)
        previous_actions[:, 1:] = sampled[:, :-1]
        q1, q2 = critic(obs, previous_actions, sampled, z_seq)
        self.assertEqual(q1.shape, (batch_size, timesteps, 1))
        self.assertEqual(q2.shape, q1.shape)

    def test_critic_hidden_state_does_not_depend_on_current_action(self):
        """Current candidate actions affect Q heads, not recurrent belief state."""
        batch_size, timesteps, embedding_dim = 2, 3, 5
        obs = torch.randn(batch_size, timesteps, OBS_DIM)
        previous_actions = torch.randn(batch_size, timesteps, ACTION_DIM)
        embeddings = torch.randn(batch_size, timesteps, embedding_dim)
        first_actions = torch.zeros(batch_size, timesteps, ACTION_DIM)
        second_actions = torch.ones(batch_size, timesteps, ACTION_DIM)
        critic = RecurrentCritic(
            OBS_DIM, ACTION_DIM, embedding_dim, hidden_dim=8
        )

        first_q1, first_q2, first_hidden = critic(
            obs,
            previous_actions,
            first_actions,
            embeddings,
            return_hidden=True,
        )
        second_q1, second_q2, second_hidden = critic(
            obs,
            previous_actions,
            second_actions,
            embeddings,
            return_hidden=True,
        )

        torch.testing.assert_close(first_hidden[0], second_hidden[0])
        torch.testing.assert_close(first_hidden[1], second_hidden[1])
        self.assertFalse(torch.equal(first_q1, second_q1))
        self.assertFalse(torch.equal(first_q2, second_q2))

    def test_auxiliary_loss_ignores_padded_targets(self):
        predictions = torch.zeros(1, 4, OPPONENT_ACTION_DIM)
        opponent_actions = torch.ones(1, 4, OPPONENT_ACTION_DIM)
        mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
        baseline = OpponentEncoder.prediction_loss(
            predictions, opponent_actions, mask
        )
        opponent_actions[:, 2:] = 10_000.0
        changed_padding = OpponentEncoder.prediction_loss(
            predictions, opponent_actions, mask
        )
        self.assertEqual(float(baseline), float(changed_padding))

    def test_auxiliary_prediction_targets_same_timestep(self):
        opponent_actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones(1, 2, 1)
        loss = OpponentEncoder.prediction_loss(
            opponent_actions.clone(), opponent_actions, mask
        )
        self.assertEqual(float(loss), 0.0)


class RecurrentAgentReplayIntegrationTests(unittest.TestCase):
    """Exercise updates using both sequence replay buffers in this repository."""

    def test_update_from_episode_replay_buffer(self):
        replay = EpisodeReplayBuffer(
            capacity_episodes=4, sequence_length=SEQUENCE_LENGTH
        )
        source = make_episode(length=2)
        source.update(opponent_type="easy", stage_id=0)
        replay.push(source)
        sequence = replay.sample()
        batch = {
            key: np.expand_dims(value, axis=0)
            for key, value in sequence.items()
            if key in {
                "obs", "actions", "rewards", "next_obs", "dones",
                "opponent_actions", "mask",
            }
        }

        metrics = make_agent().update_from_batch(batch)
        for name in (
            "critic_loss", "actor_loss", "opponent_prediction_loss",
            "alpha_loss", "alpha", "total_loss",
        ):
            self.assertTrue(np.isfinite(metrics[name]), name)

    def test_update_from_curriculum_sequence_replay_buffer(self):
        replay = CurriculumSequenceReplayBuffer(
            capacity=4,
            batch_size=3,
            curriculum=FakeCurriculum(),
            sequence_length=SEQUENCE_LENGTH,
        )
        replay.push(make_episode(length=3))
        replay.set_stage("hard")
        replay.push(make_episode(length=4, offset=1.0))
        batch = replay.sample()

        agent = make_agent()
        metrics = agent.update_from_batch(batch)
        self.assertEqual(agent.total_updates, 1)
        self.assertTrue(np.isfinite(metrics["total_loss"]))
        self.assertEqual(batch["mask"].shape, (3, SEQUENCE_LENGTH, 1))

    def test_zero_argument_update_waits_for_episode_batch(self):
        replay = CurriculumSequenceReplayBuffer(
            capacity=4,
            batch_size=2,
            curriculum=FakeCurriculum(),
            sequence_length=SEQUENCE_LENGTH,
        )
        agent = RecurrentSACOpponentEmbeddingAgent(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            opponent_action_dim=OPPONENT_ACTION_DIM,
            opponent_embedding_dim=5,
            encoder_hidden_dim=8,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            replay_buffer=replay,
            device="cpu",
        )
        agent.update_from_batch = Mock(return_value={"critic_loss": 1.0})

        replay.push(make_episode(length=2))
        self.assertIsNone(agent.update())
        agent.update_from_batch.assert_not_called()

        replay.push(make_episode(length=2, offset=1.0))
        result = agent.update()
        self.assertEqual(result, {"critic_loss": 1.0})
        agent.update_from_batch.assert_called_once()

    def test_update_uses_pre_decision_and_post_transition_contexts(self):
        """Current embeddings exclude transition t; next embeddings include it."""
        agent = make_agent()
        episode = make_episode(length=3)
        batch = {
            key: np.expand_dims(value, axis=0)
            for key, value in episode.items()
        }
        batch["mask"] = np.ones((1, 3, 1), dtype=np.float32)
        observed_inputs = []

        def capture_encoder_input(module, args):
            observed_inputs.append(args[0].detach().cpu())

        hook = agent.opponent_encoder.register_forward_pre_hook(
            capture_encoder_input
        )
        try:
            agent.update_from_batch(batch)
        finally:
            hook.remove()

        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        opponent_actions = torch.as_tensor(batch["opponent_actions"])
        expected_current = torch.cat([
            obs,
            agent._shift_sequence(actions),
            agent._shift_sequence(rewards),
            agent._shift_sequence(opponent_actions),
        ], dim=-1)
        expected_next = torch.cat(
            [next_obs, actions, rewards, opponent_actions], dim=-1
        )
        torch.testing.assert_close(observed_inputs[0], expected_current)
        torch.testing.assert_close(observed_inputs[1], expected_next)

    def test_online_hidden_state_and_checkpoint_round_trip(self):
        agent = make_agent()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action = agent.select_action(
            obs,
            prev_action=np.zeros(ACTION_DIM),
            prev_reward=0.0,
            opponent_action=np.zeros(OPPONENT_ACTION_DIM),
            deterministic=True,
        )
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertIsNotNone(agent.encoder_hidden)
        self.assertIsNotNone(agent.actor_hidden)

        agent.reset_hidden()
        expected = agent.select_action(obs, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "recurrent_agent.pt"
            agent.save(checkpoint)
            restored = make_agent()
            restored.load(checkpoint)
            restored.reset_hidden()
            actual = restored.select_action(obs, deterministic=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
