"""Predeclared 300,000-step anchor-pilot protocol derivation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from env.pricing_contracts import AgentArchitecture
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    DistributionCombination,
    ProtocolConfigError,
)
from universal_pricing_v2.protocol import (
    HierarchicalTrainingBudgetConfig,
    OpponentCurriculumStageSpec,
    PricingSkillCurriculumSpec,
    UniversalPricingV2ProtocolConfig,
    V2ArtifactLayout,
    V2ExperimentCoordinate,
    V2ManifestRepository,
    load_universal_pricing_v2_protocol,
)
from train.universal_pricing_protocol import RunStatus


@dataclass(frozen=True)
class PilotPopulationAnchor:
    name: str
    distribution_combination: DistributionCombination


@dataclass(frozen=True)
class UniversalPricingV2PilotConfig:
    base_protocol_path: Path
    artifact_root: Path
    budget_scale: float
    training_seed_index: int
    anchors: tuple[PilotPopulationAnchor, ...]

    def __post_init__(self) -> None:
        if self.budget_scale != 0.5:
            raise ProtocolConfigError("V2 pilot budget_scale is frozen at 0.5")
        if self.training_seed_index != 0:
            raise ProtocolConfigError("V2 pilots use training seed index 0")
        if len(self.anchors) != 3:
            raise ProtocolConfigError("V2 pilot requires three anchors")
        if len({item.name for item in self.anchors}) != 3:
            raise ProtocolConfigError("Pilot anchor names must be unique")

    @staticmethod
    def _scaled_curriculum(
        curriculum: PricingSkillCurriculumSpec,
        scaled_budget: int,
    ) -> PricingSkillCurriculumSpec:
        stages = tuple(
            OpponentCurriculumStageSpec(
                name=stage.name,
                opponent_policy_name=stage.opponent_policy_name,
                minimum_steps=max(
                    1, int(round(stage.minimum_steps * 0.5))
                ),
                maximum_steps=max(
                    1, int(round(stage.maximum_steps * 0.5))
                ),
            )
            for stage in curriculum.stages
        )
        difference = scaled_budget - sum(
            stage.maximum_steps for stage in stages
        )
        if difference:
            final = stages[-1]
            stages = (
                *stages[:-1],
                replace(
                    final,
                    maximum_steps=final.maximum_steps + difference,
                ),
            )
        return PricingSkillCurriculumSpec(
            pricing_skill=curriculum.pricing_skill,
            phase_budget_steps=scaled_budget,
            stages=stages,
        )

    def resolved_protocol(self) -> UniversalPricingV2ProtocolConfig:
        base = load_universal_pricing_v2_protocol(
            self.base_protocol_path
        )
        budget = HierarchicalTrainingBudgetConfig(
            uniform_pricing_steps=90_000,
            bbp_pricing_steps=110_000,
            strategy_total_steps=100_000,
            strategy_frozen_minimum_steps=25_000,
            strategy_frozen_maximum_steps=50_000,
            warmup_steps=500,
            updates_per_step=base.training_budget.updates_per_step,
            checkpoint_interval_steps=5_000,
            evaluation_interval_steps=5_000,
        )
        return replace(
            base,
            artifact_root=self.artifact_root,
            mastery_gate=replace(
                base.mastery_gate,
                validation_interval_steps=2_500,
            ),
            uniform_curriculum=self._scaled_curriculum(
                base.uniform_curriculum, budget.uniform_pricing_steps
            ),
            bbp_curriculum=self._scaled_curriculum(
                base.bbp_curriculum, budget.bbp_pricing_steps
            ),
            training_budget=budget,
        )

    def coordinates(self) -> tuple[V2ExperimentCoordinate, ...]:
        return tuple(
            V2ExperimentCoordinate(
                agent_architecture=architecture,
                distribution_combination=anchor.distribution_combination,
                training_seed_index=self.training_seed_index,
            )
            for architecture in AgentArchitecture
            for anchor in self.anchors
        )


@dataclass(frozen=True)
class V2PilotReadinessReport:
    """Auditable prerequisite for producing launch commands."""

    ready: bool
    completed_run_count: int
    required_run_count: int
    failures: tuple[str, ...]


class V2PilotReadinessGate:
    """Require all nine pilots to complete without mastery failures."""

    def evaluate(
        self, pilot: UniversalPricingV2PilotConfig
    ) -> V2PilotReadinessReport:
        protocol = pilot.resolved_protocol()
        layout = V2ArtifactLayout(protocol.artifact_root)
        repository = V2ManifestRepository()
        completed = 0
        failures: list[str] = []
        for coordinate in pilot.coordinates():
            path = layout.manifest_path(coordinate)
            if not path.is_file():
                failures.append(f"missing:{path}")
                continue
            try:
                manifest = repository.read(path)
            except (OSError, TypeError, ValueError) as exc:
                failures.append(f"invalid:{path}:{exc}")
                continue
            if manifest.status is not RunStatus.COMPLETED:
                failures.append(
                    f"status:{manifest.run_id.value}:{manifest.status.value}"
                )
                continue
            completed += 1
            failed_stages = [
                item
                for item in manifest.stage_outcomes
                if not bool(item.get("mastered", True))
            ]
            if failed_stages:
                failures.append(
                    f"mastery:{manifest.run_id.value}:"
                    f"{len(failed_stages)}"
                )
        return V2PilotReadinessReport(
            ready=not failures and completed == 9,
            completed_run_count=completed,
            required_run_count=9,
            failures=tuple(failures),
        )


def load_v2_pilot_config(
    path: str | Path,
) -> UniversalPricingV2PilotConfig:
    pilot_path = Path(path).resolve()
    if not pilot_path.is_file():
        raise ProtocolConfigError(
            f"Pilot configuration does not exist: {pilot_path}"
        )
    with pilot_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ProtocolConfigError("Pilot configuration must be a mapping")
    expected = {
        "base_protocol",
        "artifact_root",
        "budget_scale",
        "training_seed_index",
        "anchors",
    }
    if set(raw) != expected:
        raise ProtocolConfigError(
            "Pilot configuration fields must be exactly "
            + ", ".join(sorted(expected))
        )
    anchors_raw = raw["anchors"]
    if not isinstance(anchors_raw, list):
        raise ProtocolConfigError("Pilot anchors must be a list")
    anchors: list[PilotPopulationAnchor] = []
    for value in anchors_raw:
        if not isinstance(value, Mapping):
            raise ProtocolConfigError("Pilot anchor must be a mapping")
        anchors.append(
            PilotPopulationAnchor(
                name=str(value["name"]),
                distribution_combination=DistributionCombination(
                    location=ConsumerDistributionFamily(value["location"]),
                    strategicness=ConsumerDistributionFamily(
                        value["strategicness"]
                    ),
                    exclusivity=ConsumerDistributionFamily(
                        value["exclusivity"]
                    ),
                ),
            )
        )
    base_path = Path(str(raw["base_protocol"]))
    if not base_path.is_absolute():
        base_path = (pilot_path.parent / base_path).resolve()
    return UniversalPricingV2PilotConfig(
        base_protocol_path=base_path,
        artifact_root=Path(str(raw["artifact_root"])),
        budget_scale=float(raw["budget_scale"]),
        training_seed_index=int(raw["training_seed_index"]),
        anchors=tuple(anchors),
    )
