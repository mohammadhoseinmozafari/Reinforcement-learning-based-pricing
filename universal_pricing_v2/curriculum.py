"""Phase-aware v2 curriculum state and auditable mastery gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalTrainingPhase,
    MasteryGateConfig,
    OpponentCurriculumStageSpec,
    PricingSkillCurriculumSpec,
    UniversalPricingV2ProtocolConfig,
)


@dataclass(frozen=True)
class MasteryResult:
    """One paired validation result used by the coordinator."""

    agent_net_profit: float
    random_net_profit: float
    oracle_net_profit: float
    skill_score: float
    passed: bool
    validation_episode_count: int
    uniform_scenario_accuracy: float | None = None
    bbp_scenario_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if getattr(self, name) is not None
        }


class OracleNormalizedMasteryEvaluator:
    """Compute the protocol's random-to-oracle normalized skill score."""

    def __init__(self, gate: MasteryGateConfig) -> None:
        self.gate = gate

    def pricing_result(
        self,
        *,
        agent_net_profit: float,
        random_net_profit: float,
        oracle_net_profit: float,
        validation_episode_count: int,
    ) -> MasteryResult:
        values = np.asarray(
            [agent_net_profit, random_net_profit, oracle_net_profit],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Mastery profits must be finite")
        denominator = float(oracle_net_profit - random_net_profit)
        if denominator <= 0.0:
            raise ValueError(
                "Constant-price oracle must outperform random pricing"
            )
        score = float((agent_net_profit - random_net_profit) / denominator)
        return MasteryResult(
            agent_net_profit=float(agent_net_profit),
            random_net_profit=float(random_net_profit),
            oracle_net_profit=float(oracle_net_profit),
            skill_score=score,
            passed=score >= self.gate.score_threshold,
            validation_episode_count=int(validation_episode_count),
        )

    def strategy_result(
        self,
        *,
        learned_net_profit: float,
        random_regime_net_profit: float,
        best_forced_net_profit: float,
        uniform_scenario_accuracy: float,
        bbp_scenario_accuracy: float,
        validation_episode_count: int,
    ) -> MasteryResult:
        base = self.pricing_result(
            agent_net_profit=learned_net_profit,
            random_net_profit=random_regime_net_profit,
            oracle_net_profit=best_forced_net_profit,
            validation_episode_count=validation_episode_count,
        )
        uniform_accuracy = float(uniform_scenario_accuracy)
        bbp_accuracy = float(bbp_scenario_accuracy)
        if not (
            0.0 <= uniform_accuracy <= 1.0
            and 0.0 <= bbp_accuracy <= 1.0
        ):
            raise ValueError("Strategy scenario accuracies must be in [0, 1]")
        return MasteryResult(
            agent_net_profit=base.agent_net_profit,
            random_net_profit=base.random_net_profit,
            oracle_net_profit=base.oracle_net_profit,
            skill_score=base.skill_score,
            passed=(
                base.passed
                and uniform_accuracy >= self.gate.strategy_uniform_accuracy
                and bbp_accuracy >= self.gate.strategy_bbp_accuracy
            ),
            validation_episode_count=base.validation_episode_count,
            uniform_scenario_accuracy=uniform_accuracy,
            bbp_scenario_accuracy=bbp_accuracy,
        )


@dataclass
class HierarchicalCurriculumState:
    """Serializable mutable state for exact phase/stage continuation."""

    phase: HierarchicalTrainingPhase = (
        HierarchicalTrainingPhase.UNIFORM_PRICING
    )
    total_environment_steps: int = 0
    phase_environment_steps: int = 0
    strategy_environment_steps: int = 0
    stage_index: int = 0
    stage_environment_steps: int = 0
    stage_local_episode_index: int = 0
    phase_local_episode_index: int = 0
    consecutive_mastery_passes: int = 0
    last_validation_stage_step: int = 0
    consolidation: bool = False
    mastery_failures: list[dict[str, Any]] = field(default_factory=list)
    stage_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "total_environment_steps": self.total_environment_steps,
            "phase_environment_steps": self.phase_environment_steps,
            "strategy_environment_steps": self.strategy_environment_steps,
            "stage_index": self.stage_index,
            "stage_environment_steps": self.stage_environment_steps,
            "stage_local_episode_index": self.stage_local_episode_index,
            "phase_local_episode_index": self.phase_local_episode_index,
            "consecutive_mastery_passes": self.consecutive_mastery_passes,
            "last_validation_stage_step": self.last_validation_stage_step,
            "consolidation": self.consolidation,
            "mastery_failures": list(self.mastery_failures),
            "stage_outcomes": list(self.stage_outcomes),
        }

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Any]
    ) -> "HierarchicalCurriculumState":
        raw = dict(values)
        raw["phase"] = HierarchicalTrainingPhase(raw["phase"])
        raw["mastery_failures"] = list(raw.get("mastery_failures", ()))
        raw["stage_outcomes"] = list(raw.get("stage_outcomes", ()))
        return cls(**raw)


class HierarchicalCurriculumCoordinator:
    """Advance bounded mastery stages while preserving exact phase budgets."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        state: HierarchicalCurriculumState | None = None,
    ) -> None:
        self.protocol = protocol
        self.gate = protocol.mastery_gate
        self.state = state or HierarchicalCurriculumState()

    @property
    def is_completed(self) -> bool:
        return self.state.phase is HierarchicalTrainingPhase.COMPLETED

    @property
    def regime_mode(self) -> AgentRegimeMode:
        if self.state.phase is HierarchicalTrainingPhase.UNIFORM_PRICING:
            return AgentRegimeMode.FORCED_UNIFORM
        if self.state.phase is HierarchicalTrainingPhase.BBP_PRICING:
            return AgentRegimeMode.FORCED_BBP
        return AgentRegimeMode.LEARNED

    @property
    def pricing_controllers_frozen(self) -> bool:
        return self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN

    @property
    def joint_price_learning_rate(self) -> bool:
        return self.state.phase is HierarchicalTrainingPhase.JOINT_CONSOLIDATION

    def _curriculum(self) -> PricingSkillCurriculumSpec | None:
        if self.state.phase is HierarchicalTrainingPhase.UNIFORM_PRICING:
            return self.protocol.uniform_curriculum
        if self.state.phase is HierarchicalTrainingPhase.BBP_PRICING:
            return self.protocol.bbp_curriculum
        return None

    @property
    def current_stage(self) -> OpponentCurriculumStageSpec | None:
        curriculum = self._curriculum()
        if (
            curriculum is None
            or self.state.consolidation
            or self.state.stage_index >= len(curriculum.stages)
        ):
            return None
        return curriculum.stages[self.state.stage_index]

    @property
    def opponent_policy_name(self) -> str | None:
        stage = self.current_stage
        return stage.opponent_policy_name if stage is not None else None

    @property
    def stage_key(self) -> str:
        stage = self.current_stage
        return (
            f"{self.state.phase.value}:{self.state.stage_index}:"
            f"{stage.opponent_policy_name}"
            if stage is not None
            else f"{self.state.phase.value}:balanced_consolidation"
        )

    def register_episode(self) -> None:
        self.state.stage_local_episode_index += 1
        self.state.phase_local_episode_index += 1

    def record_environment_steps(self, step_count: int) -> None:
        if (
            not isinstance(step_count, int)
            or isinstance(step_count, bool)
            or step_count <= 0
        ):
            raise ValueError("step_count must be a positive integer")
        self.state.total_environment_steps += step_count
        self.state.phase_environment_steps += step_count
        if self.state.phase in {
            HierarchicalTrainingPhase.STRATEGY_FROZEN,
            HierarchicalTrainingPhase.JOINT_CONSOLIDATION,
        }:
            self.state.strategy_environment_steps += step_count
        if self.current_stage is not None:
            self.state.stage_environment_steps += step_count
        at_pricing_cap = (
            self.current_stage is not None
            and self.state.stage_environment_steps
            >= self.current_stage.maximum_steps
        )
        at_strategy_cap = (
            self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN
            and self.state.strategy_environment_steps
            >= self.protocol.training_budget.strategy_frozen_maximum_steps
        )
        if not at_pricing_cap and not at_strategy_cap:
            self._enforce_phase_budget()

    def advance_if_capped(self) -> bool:
        """Advance an unmastered stage after its final cap validation."""

        stage = self.current_stage
        if (
            stage is not None
            and self.state.stage_environment_steps >= stage.maximum_steps
        ):
            self._finish_current_stage(
                mastered=False, score=None, reason="maximum_exposure"
            )
            self._enforce_phase_budget()
            return True
        if (
            self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN
            and self.state.strategy_environment_steps
            >= self.protocol.training_budget.strategy_frozen_maximum_steps
        ):
            self._start_joint_consolidation(
                mastered=False, result=None
            )
            return True
        self._enforce_phase_budget()
        return False

    def should_validate(self) -> bool:
        if self.is_completed:
            return False
        if self.state.phase in {
            HierarchicalTrainingPhase.UNIFORM_PRICING,
            HierarchicalTrainingPhase.BBP_PRICING,
        }:
            stage = self.current_stage
            if stage is None or self.state.stage_environment_steps < (
                stage.minimum_steps
            ):
                return False
            current = self.state.stage_environment_steps
        elif self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN:
            if self.state.strategy_environment_steps < (
                self.protocol.training_budget.strategy_frozen_minimum_steps
            ):
                return False
            current = self.state.strategy_environment_steps
        else:
            return False
        reached_interval = (
            current - self.state.last_validation_stage_step
            >= self.gate.validation_interval_steps
        )
        at_cap = (
            self.current_stage is not None
            and current >= self.current_stage.maximum_steps
        ) or (
            self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN
            and current
            >= self.protocol.training_budget.strategy_frozen_maximum_steps
        )
        return reached_interval or at_cap

    def record_mastery_result(self, result: MasteryResult) -> None:
        if not isinstance(result, MasteryResult):
            raise TypeError("result must be MasteryResult")
        current_steps = (
            self.state.stage_environment_steps
            if self.current_stage is not None
            else self.state.strategy_environment_steps
        )
        self.state.last_validation_stage_step = current_steps
        self.state.consecutive_mastery_passes = (
            self.state.consecutive_mastery_passes + 1
            if result.passed
            else 0
        )
        if self.state.consecutive_mastery_passes < (
            self.gate.consecutive_passes
        ):
            return
        if self.current_stage is not None:
            self._finish_current_stage(
                mastered=True,
                score=result.skill_score,
                reason="mastery",
            )
            self._enforce_phase_budget()
        elif self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN:
            self._start_joint_consolidation(
                mastered=True, result=result
            )

    def _finish_current_stage(
        self,
        *,
        mastered: bool,
        score: float | None,
        reason: str,
    ) -> None:
        stage = self.current_stage
        if stage is None:
            return
        outcome = {
            "phase": self.state.phase.value,
            "stage_index": self.state.stage_index,
            "opponent_policy_name": stage.opponent_policy_name,
            "environment_steps": self.state.stage_environment_steps,
            "mastered": bool(mastered),
            "reason": reason,
        }
        if score is not None:
            outcome["skill_score"] = float(score)
        self.state.stage_outcomes.append(outcome)
        if not mastered:
            self.state.mastery_failures.append(dict(outcome))
        self.state.stage_index += 1
        self.state.stage_environment_steps = 0
        self.state.stage_local_episode_index = 0
        self.state.consecutive_mastery_passes = 0
        self.state.last_validation_stage_step = 0
        curriculum = self._curriculum()
        if curriculum is not None and self.state.stage_index >= len(
            curriculum.stages
        ):
            self.state.consolidation = True

    def _start_joint_consolidation(
        self,
        *,
        mastered: bool,
        result: MasteryResult | None,
    ) -> None:
        outcome: dict[str, Any] = {
            "phase": HierarchicalTrainingPhase.STRATEGY_FROZEN.value,
            "environment_steps": self.state.strategy_environment_steps,
            "mastered": mastered,
            "reason": "mastery" if mastered else "maximum_exposure",
        }
        if result is not None:
            outcome.update(result.to_dict())
        self.state.stage_outcomes.append(outcome)
        if not mastered:
            self.state.mastery_failures.append(dict(outcome))
        self.state.phase = HierarchicalTrainingPhase.JOINT_CONSOLIDATION
        self.state.phase_environment_steps = 0
        self.state.phase_local_episode_index = 0
        self.state.stage_local_episode_index = 0
        self.state.consecutive_mastery_passes = 0
        self.state.last_validation_stage_step = 0
        self.state.consolidation = True

    def _advance_pricing_phase(self) -> None:
        if self.state.phase is HierarchicalTrainingPhase.UNIFORM_PRICING:
            next_phase = HierarchicalTrainingPhase.BBP_PRICING
        elif self.state.phase is HierarchicalTrainingPhase.BBP_PRICING:
            next_phase = HierarchicalTrainingPhase.STRATEGY_FROZEN
        else:
            return
        self.state.phase = next_phase
        self.state.phase_environment_steps = 0
        self.state.stage_index = 0
        self.state.stage_environment_steps = 0
        self.state.stage_local_episode_index = 0
        self.state.phase_local_episode_index = 0
        self.state.consecutive_mastery_passes = 0
        self.state.last_validation_stage_step = 0
        self.state.consolidation = False

    def _enforce_phase_budget(self) -> None:
        budget = self.protocol.training_budget
        if (
            self.state.phase is HierarchicalTrainingPhase.UNIFORM_PRICING
            and self.state.phase_environment_steps
            >= budget.uniform_pricing_steps
        ):
            self._advance_pricing_phase()
        elif (
            self.state.phase is HierarchicalTrainingPhase.BBP_PRICING
            and self.state.phase_environment_steps >= budget.bbp_pricing_steps
        ):
            self._advance_pricing_phase()
        elif (
            self.state.phase is HierarchicalTrainingPhase.STRATEGY_FROZEN
            and self.state.strategy_environment_steps
            >= budget.strategy_frozen_maximum_steps
        ):
            self._start_joint_consolidation(mastered=False, result=None)
        elif (
            self.state.phase
            in {
                HierarchicalTrainingPhase.STRATEGY_FROZEN,
                HierarchicalTrainingPhase.JOINT_CONSOLIDATION,
            }
            and self.state.strategy_environment_steps
            >= budget.strategy_total_steps
        ):
            self.state.phase = HierarchicalTrainingPhase.COMPLETED
            self.state.phase_environment_steps = 0

    def diagnostics(self) -> dict[str, Any]:
        stage = self.current_stage
        return {
            "curriculum_phase": self.state.phase.value,
            "curriculum_stage_index": self.state.stage_index,
            "curriculum_stage_name": (
                stage.name if stage is not None else "balanced_consolidation"
            ),
            "curriculum_opponent_policy": (
                stage.opponent_policy_name if stage is not None else "balanced"
            ),
            "curriculum_stage_steps": self.state.stage_environment_steps,
            "curriculum_phase_steps": self.state.phase_environment_steps,
            "curriculum_total_steps": self.state.total_environment_steps,
            "curriculum_mastery_failures": len(
                self.state.mastery_failures
            ),
            "curriculum_consecutive_mastery_passes": (
                self.state.consecutive_mastery_passes
            ),
            "curriculum_consolidation": float(self.state.consolidation),
        }
