"""Behavior tests for episodic recurrent curriculum collection and evaluation."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.buffer import EpisodeBuilder
from train.metrics import TrainingMetrics
from train.recurrent_curriculum_trainer import RecurrentCurriculumTrainer


class RecordingReplay:
    """Small episode-only replay double used to inspect insertion timing."""

    batch_size = 2

    def __init__(self):
        self.episodes = []
        self.current_stage = "easy"

    def create_episode_builder(self):
        return EpisodeBuilder()

    def push(self, episode):
        self.episodes.append(episode)

    def set_stage(self, stage):
        self.current_stage = stage

    def get_info(self):
        return {self.current_stage: len(self.episodes)}

    def __len__(self):
        return len(self.episodes)


class FixedActionSpace:
    def sample(self):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class ShortEpisodeEnv:
    """Gym-like environment with deterministic two-transition episodes."""

    def __init__(self, episode_length=2):
        self.action_space = FixedActionSpace()
        self.episode_length = episode_length
        self.reset_seeds = []
        self.steps = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= self.episode_length
        info = {
            "opponent_price_uniform": 0.5,
            "opponent_price_new": 0.5,
            "opponent_price_old": 1.0,
            "profit": 1.0,
            "opponent_profit": 2.0,
            "uniform_price": 1.0,
            "bbp_price_new": 1.0,
            "bbp_price_old": 1.5,
            "market_share": 0.4,
            "regime": 0,
            "opponent_regime": 1,
        }
        observation = np.full(2, self.steps, dtype=np.float32)
        return observation, 0.25, terminated, False, info


class RecordingAgent:
    action_dim = 3
    opponent_action_dim = 3
    alpha = 0.2

    def __init__(self, replay):
        self.replay_buffer = replay
        self.min_episodes_before_update = replay.batch_size
        self._minimum_episodes_was_defaulted = False
        self.reset_calls = 0
        self.contexts = []
        self.replay_sizes_during_updates = []

    def reset_hidden(self):
        self.reset_calls += 1

    def select_action(
        self,
        obs,
        prev_action=None,
        prev_reward=None,
        opponent_action=None,
        deterministic=False,
    ):
        self.contexts.append((
            np.asarray(prev_action).copy(),
            float(prev_reward),
            np.asarray(opponent_action).copy(),
            deterministic,
        ))
        return np.array([0.2, 0.3, 0.4], dtype=np.float32)

    def update(self):
        self.replay_sizes_during_updates.append(len(self.replay_buffer))
        return {"critic_loss": 1.0, "actor_loss": 2.0}

    def get_policy_stats(self):
        return {
            name: np.array([0.1, 0.2, 0.3], dtype=np.float32)
            for name in ("mean", "std", "raw_log_std", "log_std", "action")
        }


def make_trainer(env, replay, agent):
    config = SimpleNamespace(episode_length=4, updates_per_step=2)
    return RecurrentCurriculumTrainer(
        config=config,
        curriculum_config=SimpleNamespace(),
        env_factory=SimpleNamespace(),
        base_env=env,
        env=env,
        replay_buffer=replay,
        agent=agent,
    )


class RecurrentCurriculumTrainerTests(unittest.TestCase):
    """Check recurrent context, exact warmup limits, and replay isolation."""

    def test_warmup_pushes_complete_and_final_partial_episodes(self):
        replay = RecordingReplay()
        agent = RecordingAgent(replay)
        env = ShortEpisodeEnv(episode_length=2)
        trainer = make_trainer(env, replay, agent)

        trainer.warmup(env, replay, steps=5, seed=11)

        self.assertEqual([len(ep["obs"]) for ep in replay.episodes], [2, 2, 1])
        self.assertEqual(env.reset_seeds, [11, 12, 13])
        self.assertEqual(sum(len(ep["obs"]) for ep in replay.episodes), 5)
        self.assertEqual(float(replay.episodes[-1]["dones"][-1, 0]), 0.0)
        np.testing.assert_array_equal(
            replay.episodes[0]["opponent_actions"][0], [-1.0, -1.0, -1.0]
        )

    def test_run_episode_updates_only_from_previously_completed_episodes(self):
        replay = RecordingReplay()
        replay.push({"obs": np.zeros((1, 2), dtype=np.float32)})
        agent = RecordingAgent(replay)
        env = ShortEpisodeEnv(episode_length=2)
        trainer = make_trainer(env, replay, agent)

        reward, critic_losses, actor_losses = trainer.run_episode(
            env, agent, TrainingMetrics()
        )

        self.assertEqual(reward, 0.5)
        self.assertEqual(critic_losses, [1.0] * 4)
        self.assertEqual(actor_losses, [2.0] * 4)
        self.assertEqual(agent.replay_sizes_during_updates, [1, 1, 1, 1])
        self.assertEqual(len(replay), 2)
        np.testing.assert_array_equal(agent.contexts[0][0], np.zeros(3))
        self.assertEqual(agent.contexts[1][1], 0.25)
        np.testing.assert_array_equal(agent.contexts[1][2], [-1.0, -1.0, -1.0])

    def test_evaluation_is_deterministic_and_does_not_mutate_replay(self):
        replay = RecordingReplay()
        agent = RecordingAgent(replay)
        env = ShortEpisodeEnv(episode_length=2)
        trainer = make_trainer(env, replay, agent)

        reward, stats = trainer.evaluate_recurrent_agent(
            env, agent, num_episodes=2, max_steps=4
        )

        self.assertEqual(reward, 0.5)
        self.assertEqual(len(replay), 0)
        self.assertTrue(all(context[3] for context in agent.contexts))
        self.assertEqual(set(stats), {"uniform", "new", "old"})
        self.assertGreaterEqual(agent.reset_calls, 4)


if __name__ == "__main__":
    unittest.main()
