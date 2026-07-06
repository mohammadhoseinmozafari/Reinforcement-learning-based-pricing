"""Tests for feed-forward/recurrent agent selection in the experiment runner."""

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.buffer import CurriculumReplayBuffer, CurriculumSequenceReplayBuffer
from models.recurrent_sac_opponent_embedding import (
    RecurrentSACOpponentEmbeddingAgent,
)
from models.sac import SAC
from train.experiment import ExperimentOverrides, build_agent, load_experiment


class AgentSelectionTests(unittest.TestCase):
    """Ensure configuration selects matching agent and replay implementations."""

    @classmethod
    def setUpClass(cls):
        cls.experiment_path = PROJECT_ROOT / "config/sac/uniform_vs_bbp.yaml"
        cls.env = type(
            "ShapeOnlyEnv",
            (),
            {
                "observation_space": type("Space", (), {"shape": (13,)})(),
                "action_space": type("Space", (), {"shape": (3,)})(),
            },
        )()

    def test_default_profile_builds_normal_sac(self):
        experiment = load_experiment(self.experiment_path)
        replay, agent = build_agent(experiment, self.env)
        self.assertEqual(experiment.training_config.agent_type, "sac")
        self.assertIsInstance(replay, CurriculumReplayBuffer)
        self.assertIsInstance(agent, SAC)

    def test_training_profile_override_builds_recurrent_sac(self):
        experiment = load_experiment(
            self.experiment_path,
            ExperimentOverrides(
                training_config=(
                    PROJECT_ROOT / "config/training/recurrent_sac.yaml"
                ),
                device="cpu",
            ),
        )
        replay, agent = build_agent(experiment, self.env)
        self.assertIsInstance(replay, CurriculumSequenceReplayBuffer)
        self.assertIsInstance(agent, RecurrentSACOpponentEmbeddingAgent)
        self.assertIs(agent.replay_buffer, replay)

    def test_recurrent_training_profile_is_loadable(self):
        raw = yaml.safe_load(self.experiment_path.read_text(encoding="utf-8"))
        raw["training_config"] = str(
            PROJECT_ROOT / "config/training/recurrent_sac.yaml"
        )
        raw["curriculum_config"] = str(
            PROJECT_ROOT / "config/curricula/bbp.yaml"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recurrent_experiment.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            experiment = load_experiment(path)
        self.assertEqual(experiment.training_config.agent_type, "recurrent_sac")
        self.assertEqual(experiment.training_config.sequence_length, 16)
        self.assertEqual(experiment.training_config.batch_size, 32)

    def test_every_architecture_has_six_isolated_experiments(self):
        matchups = (
            "uniform_vs_uniform", "uniform_vs_bbp", "uniform_vs_mixed",
            "bbp_vs_uniform", "bbp_vs_bbp", "bbp_vs_mixed",
        )
        expectations = {
            "sac": ("sac.yaml", 256),
            "recurrent_sac": ("recurrent_sac.yaml", 32),
        }
        for architecture, (profile, batch_size) in expectations.items():
            for matchup in matchups:
                with self.subTest(architecture=architecture, matchup=matchup):
                    experiment = load_experiment(
                        PROJECT_ROOT / "config" / architecture / f"{matchup}.yaml"
                    )
                    self.assertEqual(experiment.training_source.name, profile)
                    self.assertEqual(
                        experiment.training_config.agent_type, architecture
                    )
                    self.assertEqual(
                        experiment.training_config.batch_size, batch_size
                    )
                    self.assertEqual(
                        experiment.training_config.save_dir,
                        f"experiments/{architecture}/{matchup}/runs/1",
                    )


if __name__ == "__main__":
    unittest.main()
