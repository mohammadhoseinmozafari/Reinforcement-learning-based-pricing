"""Tests for per-stage convergence, limits, and final completion."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.curriculum import (
    Curriculum,
    CurriculumConfig,
    OpponentCurriculumScheduler,
    OpponentDifficulty,
    OpponentStage,
)


def make_stage(name, minimum=2, maximum=10):
    return OpponentStage(
        name=name,
        opponent_type=name.lower(),
        difficulty=OpponentDifficulty.EASY,
        description=f"{name} test stage",
        min_episodes=minimum,
        max_episodes=maximum,
    )


def make_scheduler(stages):
    config = CurriculumConfig(
        curriculum=Curriculum(),
        stages=stages,
        window_size=4,
        change_threshold=0.01,
        monitor_actor=True,
        monitor_critic=True,
        monitor_alpha=False,
        verbose=False,
    )
    return OpponentCurriculumScheduler(config)


class OpponentCurriculumSchedulerTests(unittest.TestCase):
    """Exercise the same exit rule on ordinary and final stages."""

    @staticmethod
    def step(scheduler, count, critic=1.0, actor=1.0):
        for _ in range(count):
            scheduler.step(critic, actor, 0.2)

    def test_stage_minimum_blocks_convergence_and_maximum(self):
        scheduler = make_scheduler([make_stage("Only", minimum=5, maximum=3)])
        self.step(scheduler, 4)

        should_advance, reason = scheduler.should_advance()

        self.assertFalse(should_advance)
        self.assertIn("Min episodes not met", reason)

    def test_stability_completes_stage_after_minimum(self):
        scheduler = make_scheduler([make_stage("Only", minimum=4, maximum=10)])
        self.step(scheduler, 4)

        should_advance, reason = scheduler.should_advance()

        self.assertTrue(should_advance)
        self.assertIn("All converged", reason)

    def test_maximum_forces_completion_when_metrics_are_unstable(self):
        scheduler = make_scheduler([make_stage("Only", minimum=2, maximum=4)])
        for loss in (1.0, 2.0, 3.0, 4.0):
            scheduler.step(loss, loss, 0.2)

        should_advance, reason = scheduler.should_advance()

        self.assertTrue(should_advance)
        self.assertIn("Max episodes reached", reason)

    def test_ordinary_advance_records_reason_and_resets_windows(self):
        scheduler = make_scheduler([
            make_stage("First", minimum=2, maximum=2),
            make_stage("Second", minimum=2, maximum=4),
        ])
        self.step(scheduler, 2)

        new_stage = scheduler.advance()

        self.assertEqual(new_stage.name, "Second")
        self.assertEqual(scheduler.current_stage_idx, 1)
        self.assertEqual(scheduler.episodes_in_stage, 0)
        self.assertEqual(len(scheduler.critic_losses), 0)
        self.assertEqual(len(scheduler.actor_losses), 0)
        self.assertEqual(len(scheduler.alphas), 0)
        self.assertIn("Max episodes reached", scheduler.advancement_records[0]["reason"])

    def test_final_stage_completion_is_bounded_and_idempotent(self):
        scheduler = make_scheduler([
            make_stage("First", minimum=1, maximum=1),
            make_stage("Final", minimum=4, maximum=10),
        ])
        self.step(scheduler, 1)
        scheduler.advance()
        self.step(scheduler, 4)

        self.assertIsNone(scheduler.advance())
        self.assertTrue(scheduler.is_complete)
        self.assertEqual(scheduler.current_stage_idx, 1)
        self.assertEqual(scheduler.episodes_in_stage, 4)
        self.assertEqual(len(scheduler.advancement_records), 2)
        completion_reason = scheduler.completion_reason

        self.assertIsNone(scheduler.advance())
        self.assertEqual(len(scheduler.advancement_records), 2)
        self.assertEqual(scheduler.completion_reason, completion_reason)
        self.assertFalse(scheduler.should_advance()[0])

    def test_single_stage_completes_at_maximum_and_reports_completion(self):
        scheduler = make_scheduler([make_stage("Only", minimum=2, maximum=2)])
        self.step(scheduler, 2, critic=2.0, actor=3.0)

        self.assertIsNone(scheduler.advance())

        self.assertTrue(scheduler.is_complete)
        self.assertEqual(scheduler.current_stage_idx, 0)
        self.assertEqual(scheduler.episodes_in_stage, 2)
        info = scheduler.get_info()
        self.assertEqual(info["is_complete"], True)
        self.assertEqual(info["total_stages_completed"], 1)
        summary = scheduler.get_summary()
        self.assertEqual(summary["total_stages_completed"], 1)
        self.assertEqual(summary["total_stages"], 1)
        self.assertIn("Max episodes reached", summary["completion_reason"])
        self.assertEqual(scheduler.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
