"""Contract tests for curriculum-aware episode sequence replay."""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.buffer import CurriculumSequenceReplayBuffer, EpisodeBuilder


class FakeCurriculum:
    def __init__(self):
        self.stages = [
            SimpleNamespace(opponent_type="easy"),
            SimpleNamespace(opponent_type="medium"),
            SimpleNamespace(opponent_type="hard"),
        ]

    def get_sequence(self):
        return self.stages


def episode(length=2):
    return {
        "obs": np.ones((length, 3), dtype=np.float32),
        "actions": np.ones((length, 2), dtype=np.float32),
        "rewards": np.ones((length, 1), dtype=np.float32),
        "next_obs": np.ones((length, 3), dtype=np.float32),
        "dones": np.zeros((length, 1), dtype=np.float32),
        "opponent_actions": np.ones((length, 2), dtype=np.float32),
    }


class CurriculumSequenceReplayBufferTests(unittest.TestCase):
    def setUp(self):
        self.buffer = CurriculumSequenceReplayBuffer(
            capacity=10,
            batch_size=4,
            curriculum=FakeCurriculum(),
            sequence_length=3,
        )

    def test_initial_interface_matches_curriculum_buffer(self):
        self.assertEqual(self.buffer.current_stage, "easy")
        self.assertEqual(self.buffer.stage_size("easy"), 0)
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.get_info(), {"easy": 0, "medium": 0, "hard": 0})

    def test_episode_builder_shapes_scalar_fields_and_copies_inputs(self):
        builder = self.buffer.create_episode_builder()
        obs = np.array([1.0, 2.0], dtype=np.float32)
        builder.append(obs, [0.1, 0.2], 3.0, [2.0, 3.0], False, [-1.0, 1.0])
        obs[:] = 99.0
        episode_data = builder.build()

        self.assertIsInstance(builder, EpisodeBuilder)
        self.assertEqual(len(builder), 1)
        self.assertEqual(episode_data["rewards"].shape, (1, 1))
        self.assertEqual(episode_data["dones"].shape, (1, 1))
        np.testing.assert_array_equal(episode_data["obs"][0], [1.0, 2.0])

    def test_empty_episode_builder_has_all_replay_fields(self):
        builder = self.buffer.create_episode_builder()
        self.assertEqual(len(builder), 0)
        self.assertEqual(set(builder.build()), set(EpisodeBuilder.FIELDS))

    def test_push_routes_episode_through_active_stage(self):
        source = episode()
        self.buffer.set_stage("medium")
        self.buffer.push(source)

        self.assertEqual(self.buffer.current_stage, "medium")
        self.assertEqual(self.buffer.current_stage_id, 1)
        self.assertEqual(self.buffer.stage_size("medium"), 1)
        self.assertNotIn("opponent_type", source)
        stored = self.buffer.buffers["medium"].episodes[0]
        self.assertEqual(stored["opponent_type"], "medium")
        self.assertEqual(stored["stage_id"], 1)

    def test_episode_metadata_survives_builder_and_sequence_sampling(self):
        builder = self.buffer.create_episode_builder(
            episode_seed=123,
            opponent_type="easy",
            stage_id=0,
        )
        builder.append([0, 0, 0], [0, 0], 0, [0, 0, 0], False, [0, 0])
        self.buffer.push_episode(builder.build())

        stored = self.buffer.buffers["easy"].episodes[0]
        self.assertEqual(stored["episode_seed"], 123)
        batch = self.buffer.sample(1)
        self.assertEqual(batch["episode_seeds"], [123])

    def test_sample_returns_stacked_padded_sequences(self):
        self.buffer.push(episode(length=2))
        batch = self.buffer.sample()

        self.assertEqual(batch["obs"].shape, (4, 3, 3))
        self.assertEqual(batch["actions"].shape, (4, 3, 2))
        self.assertEqual(batch["mask"].shape, (4, 3, 1))
        self.assertEqual(batch["opponent_types"], ["easy"] * 4)
        np.testing.assert_array_equal(batch["stage_ids"], np.zeros(4, dtype=np.int64))
        np.testing.assert_array_equal(batch["mask"][:, :, 0], [[1, 1, 0]] * 4)

    def test_sampling_excludes_future_curriculum_stages(self):
        self.buffer.set_stage("hard")
        self.buffer.push(episode())
        self.buffer.set_stage("easy")

        with self.assertRaisesRegex(ValueError, "No episodes available"):
            self.buffer.sample(1)

    def test_unknown_stage_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "Unknown curriculum stage"):
            self.buffer.set_stage("missing")


if __name__ == "__main__":
    unittest.main()
