"""Integration tests for curriculum-driven early stopping in both trainers."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.curriculum import (
    Curriculum,
    CurriculumConfig,
    OpponentDifficulty,
    OpponentStage,
)
from train.recurrent_curriculum_trainer import RecurrentCurriculumTrainer
from train.trainer import CurriculumTrainer


class NullLogger:
    """Logger double accepting every existing trainer logging call."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeEnv:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeReplay:
    def __init__(self):
        self.current_stage = "only"

    def set_stage(self, stage):
        self.current_stage = stage


class FakeAgent:
    alpha = 0.2
    opponent_action_dim = 3

    def __init__(self, replay):
        self.replay_buffer = replay

    def reset_hidden(self):
        pass


def make_training_config(num_episodes=10, save_freq=100):
    return SimpleNamespace(
        seed=7,
        verbose=False,
        warmup_steps=1,
        num_episodes=num_episodes,
        episode_length=1,
        updates_per_step=1,
        eval_freq=100,
        eval_episodes=1,
        eval_seed_count=None,
        save_freq=save_freq,
    )


def make_curriculum(minimum=1, maximum=1):
    stage = OpponentStage(
        name="Only",
        opponent_type="only",
        difficulty=OpponentDifficulty.EASY,
        description="single final stage",
        min_episodes=minimum,
        max_episodes=maximum,
    )
    return CurriculumConfig(
        curriculum=Curriculum(),
        stages=[stage],
        window_size=4,
        verbose=False,
    )


def episode_result(metrics, with_updates=True):
    metrics.reset_episode()
    metrics.record_step({})
    losses = [1.0] if with_updates else []
    return 1.0, losses, losses


class CurriculumTrainingCompletionTests(unittest.TestCase):
    """Verify actual episode counts and absence of post-final env switches."""

    def test_feedforward_trainer_stops_at_final_stage_completion(self):
        env = FakeEnv()
        replay = FakeReplay()
        agent = FakeAgent(replay)
        factory = Mock()
        trainer = CurriculumTrainer(
            make_training_config(),
            make_curriculum(minimum=2, maximum=2),
            factory,
            env,
            env,
            replay,
            agent,
        )
        trainer.warmup = Mock()
        trainer.run_episode = Mock(
            side_effect=lambda _env, _agent, metrics: episode_result(metrics)
        )

        with (
            patch("train.trainer.CurriculumTrainingLogger", NullLogger),
            patch("train.trainer.save_checkpoint") as save_checkpoint,
        ):
            trainer.train()

        self.assertEqual(trainer.run_episode.call_count, 2)
        factory.create_environment.assert_not_called()
        save_checkpoint.assert_called_once()
        self.assertEqual(save_checkpoint.call_args.args[3], 2)
        self.assertTrue(save_checkpoint.call_args.kwargs["final"])

    def test_recurrent_replay_fill_episodes_do_not_count_toward_completion(self):
        env = FakeEnv()
        replay = FakeReplay()
        agent = FakeAgent(replay)
        factory = Mock()
        trainer = RecurrentCurriculumTrainer(
            make_training_config(),
            make_curriculum(minimum=1, maximum=1),
            factory,
            env,
            env,
            replay,
            agent,
        )
        trainer.warmup = Mock()
        update_flags = iter((False, False, True))
        trainer.run_episode = Mock(
            side_effect=lambda _env, _agent, metrics: episode_result(
                metrics, next(update_flags)
            )
        )

        with (
            patch(
                "train.recurrent_curriculum_trainer.CurriculumTrainingLogger",
                NullLogger,
            ),
            patch("train.utils.save_checkpoint") as save_checkpoint,
        ):
            trainer.train()

        self.assertEqual(trainer.run_episode.call_count, 3)
        factory.create_environment.assert_not_called()
        save_checkpoint.assert_called_once()
        self.assertEqual(save_checkpoint.call_args.args[3], 3)
        self.assertTrue(save_checkpoint.call_args.kwargs["final"])

    def test_periodic_checkpointing_is_independent_of_evaluation(self):
        env = FakeEnv()
        replay = FakeReplay()
        agent = FakeAgent(replay)
        trainer = CurriculumTrainer(
            make_training_config(num_episodes=2, save_freq=1),
            make_curriculum(minimum=100, maximum=100),
            Mock(),
            env,
            env,
            replay,
            agent,
        )
        trainer.warmup = Mock()
        trainer.run_episode = Mock(
            side_effect=lambda _env, _agent, metrics: episode_result(metrics)
        )

        with (
            patch("train.trainer.CurriculumTrainingLogger", NullLogger),
            patch("train.trainer.save_checkpoint") as save_checkpoint,
            patch("train.trainer.evaluate_agent") as evaluate_agent,
        ):
            trainer.train()

        evaluate_agent.assert_not_called()
        self.assertEqual(save_checkpoint.call_count, 3)
        self.assertEqual(
            [call.args[3] for call in save_checkpoint.call_args_list],
            [1, 2, 2],
        )
        self.assertTrue(save_checkpoint.call_args_list[-1].kwargs["final"])


if __name__ == "__main__":
    unittest.main()
