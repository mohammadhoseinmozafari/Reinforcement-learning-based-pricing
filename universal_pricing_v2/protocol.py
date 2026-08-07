"""Validated, deterministic contracts for ``universal_pricing_v2``."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from env.pricing_contracts import AgentArchitecture
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    ConsumerDistributionSpec,
    ConsumerPopulationSpec,
    DistributionCombination,
    OpponentPoolConfig,
    ProtocolConfigError,
    RunSeedBundle,
    RunStatus,
    SeedBankManifest,
    SeedDeriver,
    load_opponent_pool_config,
    load_seed_bank_manifest,
    stable_configuration_hash,
)


PROTOCOL_VERSION = "universal_pricing_v2"
ACTION_CONTRACT_VERSION = "pricing_action_v1"
OBSERVATION_CONTRACT_VERSION = "hierarchical_pricing_observation_v2"
PRIMARY_CURRICULUM_ID = "hierarchical_full_curriculum"
REWARD_SPECIFICATION = "normalized_net_profit"
MARKET_TIMING = "simultaneous"
DEFAULT_PROTOCOL_ROOT_SEED = 20260805


def _immutable(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolConfigError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ProtocolConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolConfigError(f"Configuration must be a mapping: {path}")
    return value


def _require_exact(
    values: Mapping[str, Any],
    required: set[str],
    location: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(values)
    unknown = set(values) - required - optional
    if missing:
        raise ProtocolConfigError(
            f"Missing key(s) in {location}: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ProtocolConfigError(
            f"Unknown key(s) in {location}: {', '.join(sorted(unknown))}"
        )


def _positive_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProtocolConfigError(f"{field_name} must be a positive integer")
    return int(value)


def _finite_float(value: Any, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ProtocolConfigError(f"{field_name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ProtocolConfigError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ProtocolConfigError(f"{field_name} must be at least {minimum}")
    return result


class HierarchicalTrainingPhase(str, Enum):
    """Stable phases in the general uniform → BBP → strategy curriculum."""

    UNIFORM_PRICING = "uniform_pricing"
    BBP_PRICING = "bbp_pricing"
    STRATEGY_FROZEN = "strategy_frozen"
    JOINT_CONSOLIDATION = "joint_consolidation"
    COMPLETED = "completed"


class AgentRegimeMode(str, Enum):
    """How v2 resolves the agent regime for one episode."""

    FORCED_UNIFORM = "forced_uniform"
    FORCED_BBP = "forced_bbp"
    LEARNED = "learned"


class PricingSkill(str, Enum):
    """Independent period-level pricing skills."""

    UNIFORM = "uniform"
    BBP = "bbp"


class V2SeedNamespace(IntEnum):
    """Call-order-independent v2 seed stream identifiers."""

    CONSUMER = 101
    OPPONENT = 102
    BALANCED_SCHEDULE = 103
    UNIFORM_REPLAY = 104
    BBP_REPLAY = 105
    STRATEGY_REPLAY = 106
    EXPLORATION = 107
    TRAINING = 108
    VALIDATION = 109


_PHASE_STREAM_ID = {
    HierarchicalTrainingPhase.UNIFORM_PRICING: 1,
    HierarchicalTrainingPhase.BBP_PRICING: 2,
    HierarchicalTrainingPhase.STRATEGY_FROZEN: 3,
    HierarchicalTrainingPhase.JOINT_CONSOLIDATION: 4,
    HierarchicalTrainingPhase.COMPLETED: 5,
}


@dataclass(frozen=True)
class StageEpisodeSeedBundle:
    """Seeds for one phase/stage-local training episode."""

    consumer_seed: int
    opponent_seed: int
    schedule_seed: int


class HierarchicalSeedDeriver:
    """Pure phase/stage derivation that never reads global RNG state."""

    @staticmethod
    def derive(
        run_seed: int,
        namespace: V2SeedNamespace,
        *,
        phase: HierarchicalTrainingPhase | None = None,
        stage_index: int = 0,
        local_index: int = 0,
    ) -> int:
        for name, value in (
            ("run_seed", run_seed),
            ("stage_index", stage_index),
            ("local_index", local_index),
        ):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or int(value) < 0
            ):
                raise ProtocolConfigError(f"{name} must be nonnegative")
        phase_id = 0 if phase is None else _PHASE_STREAM_ID[
            HierarchicalTrainingPhase(phase)
        ]
        sequence = np.random.SeedSequence(
            [
                int(run_seed),
                int(V2SeedNamespace(namespace)),
                phase_id,
                int(stage_index),
                int(local_index),
            ]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    @classmethod
    def episode_bundle(
        cls,
        run_seed: int,
        phase: HierarchicalTrainingPhase,
        stage_index: int,
        local_episode_index: int,
    ) -> StageEpisodeSeedBundle:
        return StageEpisodeSeedBundle(
            consumer_seed=cls.derive(
                run_seed,
                V2SeedNamespace.CONSUMER,
                phase=phase,
                stage_index=stage_index,
                local_index=local_episode_index,
            ),
            opponent_seed=cls.derive(
                run_seed,
                V2SeedNamespace.OPPONENT,
                phase=phase,
                stage_index=stage_index,
                local_index=local_episode_index,
            ),
            schedule_seed=cls.derive(
                run_seed,
                V2SeedNamespace.BALANCED_SCHEDULE,
                phase=phase,
                stage_index=stage_index,
            ),
        )


@dataclass(frozen=True)
class MasteryGateConfig:
    """Economic gate applied after the minimum stage exposure."""

    validation_interval_steps: int = 5_000
    score_threshold: float = 0.90
    consecutive_passes: int = 2
    validation_seed_count: int = 5
    strategy_uniform_accuracy: float = 0.80
    strategy_bbp_accuracy: float = 0.80

    def __post_init__(self) -> None:
        for name in (
            "validation_interval_steps",
            "consecutive_passes",
            "validation_seed_count",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        for name in (
            "score_threshold",
            "strategy_uniform_accuracy",
            "strategy_bbp_accuracy",
        ):
            value = _finite_float(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ProtocolConfigError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class OpponentCurriculumStageSpec:
    """One ordered opponent stage and its bounded exposure."""

    name: str
    opponent_policy_name: str
    minimum_steps: int
    maximum_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProtocolConfigError("Curriculum stage name must be non-empty")
        if (
            not isinstance(self.opponent_policy_name, str)
            or not self.opponent_policy_name.strip()
        ):
            raise ProtocolConfigError(
                "Curriculum opponent policy name must be non-empty"
            )
        minimum = _positive_integer(self.minimum_steps, "minimum_steps")
        maximum = _positive_integer(self.maximum_steps, "maximum_steps")
        if minimum > maximum:
            raise ProtocolConfigError(
                "Curriculum minimum_steps cannot exceed maximum_steps"
            )
        object.__setattr__(self, "minimum_steps", minimum)
        object.__setattr__(self, "maximum_steps", maximum)

    @property
    def opponent_family(self) -> str:
        if self.opponent_policy_name.startswith("uniform_"):
            return "uniform"
        if self.opponent_policy_name.startswith("bbp_"):
            return "bbp"
        raise ProtocolConfigError(
            f"Cannot infer family for {self.opponent_policy_name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "opponent_policy_name": self.opponent_policy_name,
            "minimum_steps": self.minimum_steps,
            "maximum_steps": self.maximum_steps,
        }


EXPECTED_OPPONENT_ORDER = (
    "uniform_fixed",
    "uniform_random",
    "uniform_undercutter",
    "uniform_tit_for_tat",
    "uniform_myopic",
    "bbp_fixed_discriminator",
    "bbp_acquisition_predator",
    "bbp_loyalty_harvester",
    "bbp_myopic_segment_optimizer",
)


@dataclass(frozen=True)
class PricingSkillCurriculumSpec:
    """The complete nine-stage curriculum for one forced pricing skill."""

    pricing_skill: PricingSkill
    phase_budget_steps: int
    stages: tuple[OpponentCurriculumStageSpec, ...]

    def __post_init__(self) -> None:
        skill = PricingSkill(self.pricing_skill)
        stages = tuple(self.stages)
        if tuple(stage.opponent_policy_name for stage in stages) != (
            EXPECTED_OPPONENT_ORDER
        ):
            raise ProtocolConfigError(
                f"{skill.value} curriculum must preserve the frozen "
                "nine-opponent order"
            )
        budget = _positive_integer(
            self.phase_budget_steps, "phase_budget_steps"
        )
        maximum_total = sum(stage.maximum_steps for stage in stages)
        if maximum_total != budget:
            raise ProtocolConfigError(
                f"{skill.value} stage maximums must sum to phase budget "
                f"{budget}, got {maximum_total}"
            )
        object.__setattr__(self, "pricing_skill", skill)
        object.__setattr__(self, "phase_budget_steps", budget)
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pricing_skill": self.pricing_skill.value,
            "phase_budget_steps": self.phase_budget_steps,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class HierarchicalTrainingBudgetConfig:
    """Fixed v2 phase budgets and shared update/checkpoint settings."""

    uniform_pricing_steps: int
    bbp_pricing_steps: int
    strategy_total_steps: int
    strategy_frozen_minimum_steps: int
    strategy_frozen_maximum_steps: int
    warmup_steps: int
    updates_per_step: int
    checkpoint_interval_steps: int
    evaluation_interval_steps: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "warmup_steps":
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ProtocolConfigError(
                        "warmup_steps must be a nonnegative integer"
                    )
                continue
            object.__setattr__(self, name, _positive_integer(value, name))
        if self.strategy_frozen_minimum_steps > (
            self.strategy_frozen_maximum_steps
        ):
            raise ProtocolConfigError(
                "strategy frozen minimum cannot exceed its maximum"
            )
        if self.strategy_frozen_maximum_steps >= self.strategy_total_steps:
            raise ProtocolConfigError(
                "strategy frozen maximum must leave joint-consolidation steps"
            )

    @property
    def environment_steps(self) -> int:
        return (
            self.uniform_pricing_steps
            + self.bbp_pricing_steps
            + self.strategy_total_steps
        )

    def to_dict(self) -> dict[str, int]:
        result = {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        result["environment_steps"] = self.environment_steps
        return result


@dataclass(frozen=True)
class HierarchicalAgentProfileConfig:
    """Validated architecture and optimizer settings for all controllers."""

    architecture: AgentArchitecture
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    joint_price_learning_rate: float = 3e-5
    gamma_price: float = 0.99
    gamma_strategy: float = 0.99**10
    tau: float = 0.005
    initial_temperature: float = 0.2
    gradient_clip_norm: float = 10.0
    batch_size: int = 256
    pricing_replay_capacity: int = 300_000
    strategy_replay_capacity: int = 100_000
    pricing_hidden_dimensions: tuple[int, ...] = (256, 256)
    strategy_hidden_dimensions: tuple[int, ...] = (128, 128)
    recurrent_hidden_dimension: int | None = None
    price_sequence_length: int | None = None
    price_burn_in_length: int | None = None
    strategy_sequence_length: int | None = None
    strategy_burn_in_length: int | None = None
    episode_replay_capacity: int | None = None
    opponent_embedding_dimension: int | None = None
    encoder_hidden_dimension: int | None = None
    encoder_learning_rate: float | None = None
    auxiliary_loss_weight: float | None = None

    def __post_init__(self) -> None:
        architecture = AgentArchitecture(self.architecture)
        object.__setattr__(self, "architecture", architecture)
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_learning_rate",
            "joint_price_learning_rate",
            "gamma_price",
            "gamma_strategy",
            "tau",
            "initial_temperature",
            "gradient_clip_norm",
        ):
            value = _finite_float(getattr(self, name), name, minimum=0.0)
            if name in {"gamma_price", "gamma_strategy", "tau"} and value > 1.0:
                raise ProtocolConfigError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "batch_size",
            "pricing_replay_capacity",
            "strategy_replay_capacity",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        for name in ("pricing_hidden_dimensions", "strategy_hidden_dimensions"):
            values = tuple(getattr(self, name))
            if not values or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item <= 0
                for item in values
            ):
                raise ProtocolConfigError(f"{name} must contain positive integers")
            object.__setattr__(self, name, values)

        recurrent_names = (
            "recurrent_hidden_dimension",
            "price_sequence_length",
            "price_burn_in_length",
            "strategy_sequence_length",
            "strategy_burn_in_length",
            "episode_replay_capacity",
        )
        encoder_names = (
            "opponent_embedding_dimension",
            "encoder_hidden_dimension",
            "encoder_learning_rate",
            "auxiliary_loss_weight",
        )
        if architecture is AgentArchitecture.SAC:
            present = [
                name
                for name in recurrent_names + encoder_names
                if getattr(self, name) is not None
            ]
            if present:
                raise ProtocolConfigError(
                    "sac rejects recurrent/encoder fields: "
                    + ", ".join(present)
                )
        else:
            for name in recurrent_names:
                value = getattr(self, name)
                object.__setattr__(self, name, _positive_integer(value, name))
            if architecture is AgentArchitecture.RSAC:
                present = [
                    name for name in encoder_names if getattr(self, name) is not None
                ]
                if present:
                    raise ProtocolConfigError(
                        "rsac rejects encoder fields: " + ", ".join(present)
                    )
            else:
                for name in (
                    "opponent_embedding_dimension",
                    "encoder_hidden_dimension",
                ):
                    object.__setattr__(
                        self, name, _positive_integer(getattr(self, name), name)
                    )
                for name in ("encoder_learning_rate", "auxiliary_loss_weight"):
                    object.__setattr__(
                        self,
                        name,
                        _finite_float(getattr(self, name), name, minimum=0.0),
                    )

    @classmethod
    def from_mapping(
        cls, architecture_name: str, values: Mapping[str, Any]
    ) -> "HierarchicalAgentProfileConfig":
        if not isinstance(values, Mapping):
            raise ProtocolConfigError(
                f"agent_profiles.{architecture_name} must be a mapping"
            )
        raw = dict(values)
        architecture = AgentArchitecture(raw.pop("architecture", architecture_name))
        for name in ("pricing_hidden_dimensions", "strategy_hidden_dimensions"):
            if name in raw:
                raw[name] = tuple(raw[name])
        try:
            return cls(architecture=architecture, **raw)
        except TypeError as exc:
            raise ProtocolConfigError(
                f"Invalid {architecture_name} profile: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, Enum):
                result[name] = value.value
            elif isinstance(value, tuple):
                result[name] = list(value)
            else:
                result[name] = value
        return result


@dataclass(frozen=True)
class V2ExperimentCoordinate:
    """One architecture, distribution cell, and independent training seed."""

    agent_architecture: AgentArchitecture
    distribution_combination: DistributionCombination
    training_seed_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_architecture",
            AgentArchitecture(self.agent_architecture),
        )
        if not isinstance(self.distribution_combination, DistributionCombination):
            raise ProtocolConfigError(
                "distribution_combination must be DistributionCombination"
            )
        if (
            not isinstance(self.training_seed_index, int)
            or isinstance(self.training_seed_index, bool)
            or not 0 <= self.training_seed_index < 10
        ):
            raise ProtocolConfigError("training_seed_index must be from 0 to 9")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_architecture": self.agent_architecture.value,
            "distribution_combination": self.distribution_combination.to_dict(),
            "training_seed_index": self.training_seed_index,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "V2ExperimentCoordinate":
        combination = values["distribution_combination"]
        return cls(
            agent_architecture=AgentArchitecture(values["agent_architecture"]),
            distribution_combination=DistributionCombination(
                location=ConsumerDistributionFamily(combination["location"]),
                strategicness=ConsumerDistributionFamily(
                    combination["strategicness"]
                ),
                exclusivity=ConsumerDistributionFamily(
                    combination["exclusivity"]
                ),
            ),
            training_seed_index=int(values["training_seed_index"]),
        )


@dataclass(frozen=True)
class UniversalPricingV2ProtocolConfig:
    """Complete validated v2 protocol."""

    protocol_version: str
    action_contract_version: str
    observation_contract_version: str
    curriculum_id: str
    reward_specification: str
    market_timing: str
    artifact_root: Path
    regime_commitment_length: int
    bbp_operating_cost_rate: float
    agent_profiles: Mapping[AgentArchitecture, HierarchicalAgentProfileConfig]
    consumer_distributions: Mapping[
        str,
        Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec],
    ]
    opponent_pool: OpponentPoolConfig
    seed_manifest: SeedBankManifest
    mastery_gate: MasteryGateConfig
    uniform_curriculum: PricingSkillCurriculumSpec
    bbp_curriculum: PricingSkillCurriculumSpec
    training_budget: HierarchicalTrainingBudgetConfig

    def __post_init__(self) -> None:
        expected = (
            (self.protocol_version, PROTOCOL_VERSION, "protocol_version"),
            (
                self.action_contract_version,
                ACTION_CONTRACT_VERSION,
                "action_contract_version",
            ),
            (
                self.observation_contract_version,
                OBSERVATION_CONTRACT_VERSION,
                "observation_contract_version",
            ),
            (self.curriculum_id, PRIMARY_CURRICULUM_ID, "curriculum_id"),
            (
                self.reward_specification,
                REWARD_SPECIFICATION,
                "reward_specification",
            ),
            (self.market_timing, MARKET_TIMING, "market_timing"),
        )
        for actual, wanted, name in expected:
            if actual != wanted:
                raise ProtocolConfigError(f"{name} must be {wanted!r}")
        object.__setattr__(
            self,
            "regime_commitment_length",
            _positive_integer(
                self.regime_commitment_length, "regime_commitment_length"
            ),
        )
        rate = _finite_float(
            self.bbp_operating_cost_rate,
            "bbp_operating_cost_rate",
            minimum=0.0,
        )
        if not np.isclose(rate, 0.01):
            raise ProtocolConfigError(
                "universal_pricing_v2 freezes bbp_operating_cost_rate at 0.01"
            )
        object.__setattr__(self, "bbp_operating_cost_rate", rate)
        profiles = {
            AgentArchitecture(key): value
            for key, value in self.agent_profiles.items()
        }
        if set(profiles) != set(AgentArchitecture):
            raise ProtocolConfigError(
                "agent_profiles must define exactly sac, rsac, and oe_rsac"
            )
        for key, profile in profiles.items():
            if profile.architecture is not key:
                raise ProtocolConfigError("Agent profile architecture mismatch")
        object.__setattr__(self, "agent_profiles", _immutable(profiles))
        if (
            self.uniform_curriculum.pricing_skill is not PricingSkill.UNIFORM
            or self.bbp_curriculum.pricing_skill is not PricingSkill.BBP
        ):
            raise ProtocolConfigError("Pricing curricula use wrong skill IDs")
        if (
            self.uniform_curriculum.phase_budget_steps
            != self.training_budget.uniform_pricing_steps
            or self.bbp_curriculum.phase_budget_steps
            != self.training_budget.bbp_pricing_steps
        ):
            raise ProtocolConfigError(
                "Curriculum phase budgets disagree with training budget"
            )
        expected_attributes = {"location", "strategicness", "exclusivity"}
        if set(self.consumer_distributions) != expected_attributes:
            raise ProtocolConfigError(
                "consumer_distributions must define all three attributes"
            )
        for attribute, specs in self.consumer_distributions.items():
            if set(specs) != set(ConsumerDistributionFamily):
                raise ProtocolConfigError(
                    f"{attribute} must define all three distribution families"
                )
        if len(self.seed_manifest.training_roots) != 10:
            raise ProtocolConfigError("v2 requires ten training roots")
        if self.mastery_gate.validation_seed_count > len(
            self.seed_manifest.validation_environment_seeds
        ):
            raise ProtocolConfigError(
                "mastery validation seed count exceeds seed bank"
            )
        artifact_root = Path(self.artifact_root)
        if str(artifact_root).strip() in {"", "."}:
            raise ProtocolConfigError("artifact_root must name a directory")
        object.__setattr__(self, "artifact_root", artifact_root)

    def population_spec(
        self, combination: DistributionCombination
    ) -> ConsumerPopulationSpec:
        return ConsumerPopulationSpec(
            location=self.consumer_distributions["location"][
                combination.location
            ],
            strategicness=self.consumer_distributions["strategicness"][
                combination.strategicness
            ],
            exclusivity=self.consumer_distributions["exclusivity"][
                combination.exclusivity
            ],
        )

    def run_seed_bundle(self, seed_index: int) -> RunSeedBundle:
        if not 0 <= seed_index < len(self.seed_manifest.training_roots):
            raise ProtocolConfigError("training seed index is out of range")
        return SeedDeriver.derive_run_bundle(
            self.seed_manifest.training_roots[seed_index]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "action_contract_version": self.action_contract_version,
            "observation_contract_version": self.observation_contract_version,
            "curriculum_id": self.curriculum_id,
            "reward_specification": self.reward_specification,
            "market_timing": self.market_timing,
            "artifact_root": str(self.artifact_root),
            "regime_commitment_length": self.regime_commitment_length,
            "bbp_operating_cost_rate": self.bbp_operating_cost_rate,
            "agent_profiles": {
                key.value: value.to_dict()
                for key, value in sorted(
                    self.agent_profiles.items(), key=lambda item: item[0].value
                )
            },
            "consumer_distributions": {
                attribute: {
                    family.value: dict(spec.parameters)
                    for family, spec in sorted(
                        specs.items(), key=lambda item: item[0].value
                    )
                }
                for attribute, specs in sorted(
                    self.consumer_distributions.items()
                )
            },
            "opponent_pool": self.opponent_pool.to_dict(),
            "seed_manifest": self.seed_manifest.to_dict(),
            "mastery_gate": self.mastery_gate.to_dict(),
            "uniform_curriculum": self.uniform_curriculum.to_dict(),
            "bbp_curriculum": self.bbp_curriculum.to_dict(),
            "training_budget": self.training_budget.to_dict(),
        }


class V2ExperimentMatrix:
    """Deterministically enumerate the 810 primary v2 runs."""

    def __init__(self, protocol: UniversalPricingV2ProtocolConfig) -> None:
        self.protocol = protocol

    def coordinates(self) -> tuple[V2ExperimentCoordinate, ...]:
        families = tuple(ConsumerDistributionFamily)
        return tuple(
            V2ExperimentCoordinate(
                agent_architecture=architecture,
                distribution_combination=DistributionCombination(
                    location=location,
                    strategicness=strategicness,
                    exclusivity=exclusivity,
                ),
                training_seed_index=seed_index,
            )
            for architecture in AgentArchitecture
            for location, strategicness, exclusivity in product(
                families, repeat=3
            )
            for seed_index in range(10)
        )


@dataclass(frozen=True)
class V2ExperimentRunId:
    """Stable full run identity; it is not used as a directory name."""

    value: str

    _PATTERN = re.compile(
        r"universal_pricing_v2__(?P<agent>sac|rsac|oe_rsac)"
        r"__location-(?P<location>uniform|truncated_normal|truncated_skew_normal)"
        r"__strategicness-(?P<strategicness>uniform|truncated_normal|truncated_skew_normal)"
        r"__exclusivity-(?P<exclusivity>uniform|truncated_normal|truncated_skew_normal)"
        r"__seed-(?P<seed>\d{2})"
    )

    @classmethod
    def from_coordinate(
        cls, coordinate: V2ExperimentCoordinate
    ) -> "V2ExperimentRunId":
        combination = coordinate.distribution_combination
        return cls(
            f"{PROTOCOL_VERSION}__{coordinate.agent_architecture.value}"
            f"__location-{combination.location.value}"
            f"__strategicness-{combination.strategicness.value}"
            f"__exclusivity-{combination.exclusivity.value}"
            f"__seed-{coordinate.training_seed_index:02d}"
        )

    @classmethod
    def parse(cls, value: str) -> V2ExperimentCoordinate:
        match = cls._PATTERN.fullmatch(value)
        if match is None:
            raise ProtocolConfigError(f"Malformed v2 run ID: {value!r}")
        return V2ExperimentCoordinate(
            agent_architecture=AgentArchitecture(match.group("agent")),
            distribution_combination=DistributionCombination(
                location=ConsumerDistributionFamily(match.group("location")),
                strategicness=ConsumerDistributionFamily(
                    match.group("strategicness")
                ),
                exclusivity=ConsumerDistributionFamily(
                    match.group("exclusivity")
                ),
            ),
            training_seed_index=int(match.group("seed")),
        )

    def __str__(self) -> str:
        return self.value


_FAMILY_CODE = {
    ConsumerDistributionFamily.UNIFORM: "u",
    ConsumerDistributionFamily.TRUNCATED_NORMAL: "tn",
    ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL: "tsn",
}


@dataclass(frozen=True)
class V2ArtifactLayout:
    """Short, Windows-safe v2 paths with identity retained in the manifest."""

    artifact_root: Path

    def distribution_code(self, combination: DistributionCombination) -> str:
        return (
            f"l-{_FAMILY_CODE[combination.location]}"
            f"__s-{_FAMILY_CODE[combination.strategicness]}"
            f"__e-{_FAMILY_CODE[combination.exclusivity]}"
        )

    def run_directory(self, coordinate: V2ExperimentCoordinate) -> Path:
        return (
            Path(self.artifact_root)
            / coordinate.agent_architecture.value
            / self.distribution_code(coordinate.distribution_combination)
            / f"s{coordinate.training_seed_index:02d}"
        )

    def manifest_path(self, coordinate: V2ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "manifest.json"

    def metrics_path(self, coordinate: V2ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "metrics.jsonl"

    def checkpoint_directory(self, coordinate: V2ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "ckpt"

    def latest_snapshot_path(self, coordinate: V2ExperimentCoordinate) -> Path:
        return self.checkpoint_directory(coordinate) / "latest.pt"

    def final_checkpoint_path(self, coordinate: V2ExperimentCoordinate) -> Path:
        return self.checkpoint_directory(coordinate) / "final.pt"


@dataclass(frozen=True)
class UniversalPricingV2RunManifest:
    """Immutable v2 run identity plus evolving status and stage outcomes."""

    protocol_version: str
    run_id: V2ExperimentRunId
    coordinate: V2ExperimentCoordinate
    run_seed_bundle: RunSeedBundle
    resolved_protocol: Mapping[str, Any]
    configuration_hash: str
    git_commit: str
    hardware_metadata: Mapping[str, Any]
    status: RunStatus
    curriculum_state: Mapping[str, Any] = field(default_factory=dict)
    stage_outcomes: tuple[Mapping[str, Any], ...] = ()
    artifact_references: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolConfigError(
                f"Manifest protocol must be {PROTOCOL_VERSION}"
            )
        if V2ExperimentRunId.parse(self.run_id.value) != self.coordinate:
            raise ProtocolConfigError("Manifest identity mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_hash):
            raise ProtocolConfigError("configuration_hash must be SHA-256")
        resolved = dict(self.resolved_protocol)
        if stable_configuration_hash(resolved) != self.configuration_hash:
            raise ProtocolConfigError("Manifest configuration hash mismatch")
        object.__setattr__(self, "status", RunStatus(self.status))
        object.__setattr__(self, "resolved_protocol", _immutable(resolved))
        object.__setattr__(
            self, "hardware_metadata", _immutable(self.hardware_metadata)
        )
        object.__setattr__(
            self, "curriculum_state", _immutable(self.curriculum_state)
        )
        object.__setattr__(
            self,
            "stage_outcomes",
            tuple(_immutable(item) for item in self.stage_outcomes),
        )
        object.__setattr__(
            self,
            "artifact_references",
            _immutable(self.artifact_references),
        )

    def identity(self) -> tuple[Any, ...]:
        return (
            self.protocol_version,
            self.run_id.value,
            self.coordinate,
            self.run_seed_bundle,
            self.configuration_hash,
            self.git_commit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "run_id": self.run_id.value,
            "coordinate": self.coordinate.to_dict(),
            "run_seed_bundle": self.run_seed_bundle.to_dict(),
            "resolved_protocol": dict(self.resolved_protocol),
            "configuration_hash": self.configuration_hash,
            "git_commit": self.git_commit,
            "hardware_metadata": dict(self.hardware_metadata),
            "status": self.status.value,
            "curriculum_state": dict(self.curriculum_state),
            "stage_outcomes": [dict(item) for item in self.stage_outcomes],
            "artifact_references": dict(self.artifact_references),
        }

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Any]
    ) -> "UniversalPricingV2RunManifest":
        try:
            seeds = values["run_seed_bundle"]
            return cls(
                protocol_version=str(values["protocol_version"]),
                run_id=V2ExperimentRunId(str(values["run_id"])),
                coordinate=V2ExperimentCoordinate.from_dict(
                    values["coordinate"]
                ),
                run_seed_bundle=RunSeedBundle(
                    **{
                        name: int(seeds[name])
                        for name in RunSeedBundle.__dataclass_fields__
                    }
                ),
                resolved_protocol=values["resolved_protocol"],
                configuration_hash=str(values["configuration_hash"]),
                git_commit=str(values["git_commit"]),
                hardware_metadata=values["hardware_metadata"],
                status=RunStatus(values["status"]),
                curriculum_state=values.get("curriculum_state", {}),
                stage_outcomes=tuple(values.get("stage_outcomes", ())),
                artifact_references=values.get("artifact_references", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProtocolConfigError):
                raise
            raise ProtocolConfigError(f"Malformed v2 manifest: {exc}") from exc


class V2ManifestRepository:
    """Atomically persist v2 manifests without long temporary names."""

    def read(self, path: str | Path) -> UniversalPricingV2RunManifest:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise ProtocolConfigError(
                f"Manifest does not exist: {manifest_path}"
            )
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                values = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolConfigError(
                f"Cannot read manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(values, dict):
            raise ProtocolConfigError("Manifest must be a JSON object")
        return UniversalPricingV2RunManifest.from_dict(values)

    def write(
        self,
        path: str | Path,
        manifest: UniversalPricingV2RunManifest,
    ) -> None:
        manifest_path = Path(path)
        if manifest_path.exists():
            existing = self.read(manifest_path)
            if existing.identity() != manifest.identity():
                raise ProtocolConfigError(
                    "Existing v2 manifest identity cannot change"
                )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".m-", suffix=".tmp", dir=manifest_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    manifest.to_dict(),
                    stream,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


def _resolve_reference(owner: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolConfigError(f"{name} must be a path string")
    path = Path(value)
    return (owner.parent / path).resolve() if not path.is_absolute() else path


def _parse_distributions(
    values: Any,
) -> Mapping[
    str, Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec]
]:
    if not isinstance(values, Mapping):
        raise ProtocolConfigError("consumer_distributions must be a mapping")
    result: dict[
        str, Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec]
    ] = {}
    for attribute, family_values in values.items():
        if not isinstance(family_values, Mapping):
            raise ProtocolConfigError(
                f"consumer_distributions.{attribute} must be a mapping"
            )
        result[str(attribute)] = _immutable(
            {
                ConsumerDistributionFamily(name): ConsumerDistributionSpec(
                    family=ConsumerDistributionFamily(name),
                    parameters=parameters,
                )
                for name, parameters in family_values.items()
            }
        )
    return _immutable(result)


def _parse_stage_list(values: Any) -> tuple[OpponentCurriculumStageSpec, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ProtocolConfigError("curriculum stages must be a list")
    result: list[OpponentCurriculumStageSpec] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ProtocolConfigError(f"curriculum stage {index} must be a mapping")
        try:
            result.append(OpponentCurriculumStageSpec(**dict(value)))
        except TypeError as exc:
            raise ProtocolConfigError(
                f"Invalid curriculum stage {index}: {exc}"
            ) from exc
    return tuple(result)


def load_universal_pricing_v2_protocol(
    path: str | Path,
) -> UniversalPricingV2ProtocolConfig:
    """Load the complete isolated v2 YAML protocol."""

    protocol_path = Path(path).resolve()
    raw = _load_yaml(protocol_path)
    required = {
        "protocol_version",
        "action_contract_version",
        "observation_contract_version",
        "curriculum_id",
        "reward_specification",
        "market_timing",
        "artifact_root",
        "regime_commitment_length",
        "bbp_operating_cost_rate",
        "agent_profiles",
        "consumer_distributions",
        "opponent_pool_config",
        "seed_manifest",
        "mastery_gate",
        "curricula",
        "training_budget",
    }
    _require_exact(raw, required, str(protocol_path))

    profiles_raw = raw["agent_profiles"]
    if not isinstance(profiles_raw, Mapping):
        raise ProtocolConfigError("agent_profiles must be a mapping")
    profiles = {
        AgentArchitecture(name): HierarchicalAgentProfileConfig.from_mapping(
            name, value
        )
        for name, value in profiles_raw.items()
    }

    budget_raw = raw["training_budget"]
    if not isinstance(budget_raw, Mapping):
        raise ProtocolConfigError("training_budget must be a mapping")
    try:
        budget = HierarchicalTrainingBudgetConfig(**dict(budget_raw))
    except TypeError as exc:
        raise ProtocolConfigError(f"Invalid training_budget: {exc}") from exc

    curricula_raw = raw["curricula"]
    if not isinstance(curricula_raw, Mapping):
        raise ProtocolConfigError("curricula must be a mapping")
    _require_exact(
        curricula_raw,
        {"uniform_pricing", "bbp_pricing"},
        "curricula",
    )
    curriculum_specs: dict[PricingSkill, PricingSkillCurriculumSpec] = {}
    for skill, field_name, phase_budget in (
        (PricingSkill.UNIFORM, "uniform_pricing", budget.uniform_pricing_steps),
        (PricingSkill.BBP, "bbp_pricing", budget.bbp_pricing_steps),
    ):
        stage_values = curricula_raw[field_name]
        curriculum_specs[skill] = PricingSkillCurriculumSpec(
            pricing_skill=skill,
            phase_budget_steps=phase_budget,
            stages=_parse_stage_list(stage_values),
        )

    gate_raw = raw["mastery_gate"]
    if not isinstance(gate_raw, Mapping):
        raise ProtocolConfigError("mastery_gate must be a mapping")
    try:
        gate = MasteryGateConfig(**dict(gate_raw))
    except TypeError as exc:
        raise ProtocolConfigError(f"Invalid mastery_gate: {exc}") from exc

    artifact_root = Path(str(raw["artifact_root"]))
    if not artifact_root.is_absolute():
        # Artifact paths are intentionally resolved from the invocation/repo
        # working directory, matching v1 behavior and portable manifests.
        artifact_root = Path(raw["artifact_root"])

    return UniversalPricingV2ProtocolConfig(
        protocol_version=str(raw["protocol_version"]),
        action_contract_version=str(raw["action_contract_version"]),
        observation_contract_version=str(raw["observation_contract_version"]),
        curriculum_id=str(raw["curriculum_id"]),
        reward_specification=str(raw["reward_specification"]),
        market_timing=str(raw["market_timing"]),
        artifact_root=artifact_root,
        regime_commitment_length=raw["regime_commitment_length"],
        bbp_operating_cost_rate=raw["bbp_operating_cost_rate"],
        agent_profiles=profiles,
        consumer_distributions=_parse_distributions(
            raw["consumer_distributions"]
        ),
        opponent_pool=load_opponent_pool_config(
            _resolve_reference(
                protocol_path,
                raw["opponent_pool_config"],
                "opponent_pool_config",
            )
        ),
        seed_manifest=load_seed_bank_manifest(
            _resolve_reference(
                protocol_path, raw["seed_manifest"], "seed_manifest"
            )
        ),
        mastery_gate=gate,
        uniform_curriculum=curriculum_specs[PricingSkill.UNIFORM],
        bbp_curriculum=curriculum_specs[PricingSkill.BBP],
        training_budget=budget,
    )


def select_v2_experiment_coordinate(
    protocol: UniversalPricingV2ProtocolConfig,
    *,
    agent_architecture: str | AgentArchitecture,
    location_distribution: str | ConsumerDistributionFamily,
    strategicness_distribution: str | ConsumerDistributionFamily,
    exclusivity_distribution: str | ConsumerDistributionFamily,
    training_seed_index: int,
) -> V2ExperimentCoordinate:
    coordinate = V2ExperimentCoordinate(
        agent_architecture=AgentArchitecture(agent_architecture),
        distribution_combination=DistributionCombination(
            location=ConsumerDistributionFamily(location_distribution),
            strategicness=ConsumerDistributionFamily(
                strategicness_distribution
            ),
            exclusivity=ConsumerDistributionFamily(
                exclusivity_distribution
            ),
        ),
        training_seed_index=training_seed_index,
    )
    if coordinate.agent_architecture not in protocol.agent_profiles:
        raise ProtocolConfigError("Architecture is absent from v2 protocol")
    return coordinate
