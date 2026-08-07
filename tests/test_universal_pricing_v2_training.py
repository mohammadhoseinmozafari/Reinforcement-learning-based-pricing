"""Curriculum progression, recurrent replay, and exact-resume tests for v2."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.pricing_contracts import AgentArchitecture
from train.universal_pricing_protocol import RunStatus
from universal_pricing_v2.curriculum import (
    HierarchicalCurriculumCoordinator,
    MasteryResult,
)
from universal_pricing_v2.protocol import (
    HierarchicalTrainingBudgetConfig,
    HierarchicalTrainingPhase,
    PricingSkill,
    PricingSkillCurriculumSpec,
    V2ArtifactLayout,
    V2ExperimentMatrix,
    load_universal_pricing_v2_protocol,
)
from universal_pricing_v2.replay import (
    PricingSkillEpisode,
    PricingSkillTransition,
    RecurrentPricingEpisodeReplay,
)
from universal_pricing_v2.trainer import HierarchicalPricingTrainer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v2.yaml"
)


def tiny_protocol(artifact_root: Path):
    base = load_universal_pricing_v2_protocol(PROTOCOL_PATH)
    uniform_stages = tuple(
        replace(stage, minimum_steps=1, maximum_steps=1)
        for stage in base.uniform_curriculum.stages
    )
    bbp_stages = tuple(
        replace(stage, minimum_steps=1, maximum_steps=1)
        for stage in base.bbp_curriculum.stages
    )
    budget = HierarchicalTrainingBudgetConfig(
        uniform_pricing_steps=9,
        bbp_pricing_steps=9,
        strategy_total_steps=20,
        strategy_frozen_minimum_steps=5,
        strategy_frozen_maximum_steps=10,
        warmup_steps=1_000,
        updates_per_step=1,
        checkpoint_interval_steps=3,
        evaluation_interval_steps=1,
    )
    return replace(
        base,
        artifact_root=artifact_root,
        mastery_gate=replace(
            base.mastery_gate, validation_interval_steps=1
        ),
        uniform_curriculum=PricingSkillCurriculumSpec(
            pricing_skill=PricingSkill.UNIFORM,
            phase_budget_steps=9,
            stages=uniform_stages,
        ),
        bbp_curriculum=PricingSkillCurriculumSpec(
            pricing_skill=PricingSkill.BBP,
            phase_budget_steps=9,
            stages=bbp_stages,
        ),
        training_budget=budget,
    )


def transition(
    value: float,
    *,
    done: bool,
    stage_key: str,
) -> PricingSkillTransition:
    observation = np.full(18, value, dtype=np.float32)
    next_observation = np.full(
        18, min(value + 0.01, 1.0), dtype=np.float32
    )
    return PricingSkillTransition(
        pricing_skill=PricingSkill.UNIFORM,
        observation=observation,
        price_action=np.asarray([value], dtype=np.float32),
        effective_action=np.asarray(
            [1.0, 0.0, value, 0.0, 0.0], dtype=np.float32
        ),
        reward=value,
        next_observation=next_observation,
        done=done,
        active_controller_mask=1.0,
        opponent_price_controls=np.zeros(3, dtype=np.float32),
        stage_key=stage_key,
    )


class UniversalPricingV2TrainingTests(unittest.TestCase):
    def test_early_mastery_allocates_saved_steps_to_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = tiny_protocol(Path(directory))
            uniform_stages = tuple(
                replace(stage, maximum_steps=2)
                for stage in base.uniform_curriculum.stages
            )
            protocol = replace(
                base,
                uniform_curriculum=PricingSkillCurriculumSpec(
                    pricing_skill=PricingSkill.UNIFORM,
                    phase_budget_steps=18,
                    stages=uniform_stages,
                ),
                training_budget=replace(
                    base.training_budget,
                    uniform_pricing_steps=18,
                ),
            )
            coordinator = HierarchicalCurriculumCoordinator(protocol)
            passing = MasteryResult(
                agent_net_profit=1.0,
                random_net_profit=0.0,
                oracle_net_profit=1.0,
                skill_score=1.0,
                passed=True,
                validation_episode_count=5,
            )
            for _ in range(9):
                coordinator.record_environment_steps(1)
                coordinator.record_mastery_result(passing)
                coordinator.record_mastery_result(passing)
            self.assertTrue(coordinator.state.consolidation)
            self.assertEqual(
                coordinator.state.phase,
                HierarchicalTrainingPhase.UNIFORM_PRICING,
            )
            self.assertEqual(coordinator.state.phase_environment_steps, 9)
            coordinator.record_environment_steps(9)
            self.assertEqual(
                coordinator.state.phase,
                HierarchicalTrainingPhase.BBP_PRICING,
            )

    def test_caps_advance_all_stages_and_preserve_total_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = tiny_protocol(Path(directory))
            coordinator = HierarchicalCurriculumCoordinator(protocol)
            for _ in range(9):
                coordinator.record_environment_steps(1)
                self.assertTrue(coordinator.advance_if_capped())
            self.assertEqual(
                coordinator.state.phase,
                HierarchicalTrainingPhase.BBP_PRICING,
            )
            for _ in range(9):
                coordinator.record_environment_steps(1)
                self.assertTrue(coordinator.advance_if_capped())
            self.assertEqual(
                coordinator.state.phase,
                HierarchicalTrainingPhase.STRATEGY_FROZEN,
            )
            coordinator.record_environment_steps(10)
            self.assertTrue(coordinator.advance_if_capped())
            self.assertEqual(
                coordinator.state.phase,
                HierarchicalTrainingPhase.JOINT_CONSOLIDATION,
            )
            coordinator.record_environment_steps(10)
            self.assertTrue(coordinator.is_completed)
            self.assertEqual(coordinator.state.total_environment_steps, 38)

    def test_recurrent_replay_never_crosses_episode_boundaries(self) -> None:
        replay = RecurrentPricingEpisodeReplay(
            pricing_skill=PricingSkill.UNIFORM,
            capacity_episodes=4,
            learning_sequence_length=2,
            burn_in_length=2,
            replay_sampling_seed=19,
        )
        for episode_index in range(2):
            first = transition(
                0.1 * episode_index,
                done=False,
                stage_key=f"stage-{episode_index}",
            )
            second = PricingSkillTransition(
                pricing_skill=PricingSkill.UNIFORM,
                observation=first.next_observation,
                price_action=np.asarray([0.2], dtype=np.float32),
                effective_action=np.asarray(
                    [1.0, 0.0, 0.2, 0.0, 0.0], dtype=np.float32
                ),
                reward=0.2,
                next_observation=np.full(
                    18, 0.3 + 0.1 * episode_index, dtype=np.float32
                ),
                done=True,
                active_controller_mask=1.0,
                opponent_price_controls=np.zeros(3, dtype=np.float32),
                stage_key=f"stage-{episode_index}",
            )
            replay.push_episode(
                PricingSkillEpisode(
                    pricing_skill=PricingSkill.UNIFORM,
                    transitions=(first, second),
                    stage_key=f"stage-{episode_index}",
                )
            )
        batch = replay.sample(4, current_stage_key="stage-1")
        self.assertEqual(batch.observations.shape, (4, 4, 18))
        self.assertTrue(
            np.all(batch.loss_masks <= batch.valid_masks)
        )
        self.assertTrue(
            np.all(np.sum(batch.valid_masks, axis=1) <= 2)
        )

    def test_interrupted_training_resumes_to_exact_curriculum_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = tiny_protocol(Path(directory))
            coordinate = next(
                item
                for item in V2ExperimentMatrix(protocol).coordinates()
                if item.agent_architecture is AgentArchitecture.SAC
            )
            interrupted = HierarchicalPricingTrainer(
                protocol,
                coordinate,
                device="cpu",
                verbose=False,
                enable_mastery_evaluation=False,
                maximum_environment_steps=5,
                episode_length=1,
            ).train()
            self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
            resumed = HierarchicalPricingTrainer(
                protocol,
                coordinate,
                device="cpu",
                verbose=False,
                enable_mastery_evaluation=False,
                resume=True,
                episode_length=1,
            ).train()
            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(
                resumed.curriculum_state["total_environment_steps"], 38
            )
            self.assertEqual(
                resumed.curriculum_state["phase"], "completed"
            )
            run_directory = V2ArtifactLayout(
                protocol.artifact_root
            ).run_directory(coordinate)
            metrics_path = run_directory / "metrics.jsonl"
            records = [
                json.loads(line)
                for line in metrics_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            episodes = [
                item
                for item in records
                if item["record_type"] == "training_episode"
            ]
            self.assertEqual(
                [item["episode_index"] for item in episodes],
                list(range(38)),
            )
            required_metrics = {
                "uniform_gross_agent_profit_total",
                "bbp_bbp_operating_cost_total",
                "strategy_macro_transition_count",
                "environment_steps_per_second",
                "episode_environment_seconds",
                "episode_inference_seconds",
                "episode_update_seconds",
                "wall_clock_seconds",
                "peak_gpu_memory_bytes",
                "uniform_controller_parameters",
                "bbp_controller_parameters",
                "strategy_controller_parameters",
            }
            self.assertFalse(required_metrics - set(episodes[-1]))
            self.assertFalse(
                any(
                    path.name.startswith(".s-")
                    for path in (run_directory / "ckpt").iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
