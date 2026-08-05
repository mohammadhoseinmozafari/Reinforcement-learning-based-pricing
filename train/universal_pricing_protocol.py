"""Versioned research protocol infrastructure for universal pricing experiments.

This module deliberately constructs no environments and no learning agents.
It validates and enumerates the immutable inputs that those implementations
will consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from env.opponent_policies import OPPONENT_PRESETS
from env.pricing_contracts import (
    AGENT_ARCHITECTURE_SPECS,
    AgentArchitecture,
)
from models.sac_pricing import SACPricingAgentConfig


PROTOCOL_VERSION = "universal_pricing_v1"
ACTION_CONTRACT_VERSION = "pricing_action_v1"
OBSERVATION_CONTRACT_VERSION = "pricing_observation_v1"
PRIMARY_CURRICULUM_ID = "mixed_balanced"
PROTOCOL_ROOT_SEED = 20260805
TRAINING_SEED_COUNT = 10
VALIDATION_SEED_COUNT = 25
FINAL_EVALUATION_SEED_COUNT = 100


class ProtocolConfigError(ValueError):
    """Raised when a universal pricing protocol is malformed."""


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
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


def _reject_unknown(
    values: Mapping[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ProtocolConfigError(
            f"Unknown key(s) in {location}: {', '.join(sorted(unknown))}"
        )


def _require(
    values: Mapping[str, Any],
    required: set[str],
    location: str,
) -> None:
    missing = required - set(values)
    if missing:
        raise ProtocolConfigError(
            f"Missing key(s) in {location}: {', '.join(sorted(missing))}"
        )


def _resolve_reference(owner: Path, reference: Any, field_name: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ProtocolConfigError(f"{field_name} must be a path string")
    path = Path(reference)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


class ConsumerDistributionFamily(str, Enum):
    """Supported independent distribution families for consumer attributes."""

    UNIFORM = "uniform"
    TRUNCATED_NORMAL = "truncated_normal"
    TRUNCATED_SKEW_NORMAL = "truncated_skew_normal"


@dataclass(frozen=True)
class ConsumerDistributionSpec:
    """Validated family and sampling parameters for one consumer attribute."""

    family: ConsumerDistributionFamily
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        try:
            family = ConsumerDistributionFamily(self.family)
        except (TypeError, ValueError) as exc:
            raise ProtocolConfigError(
                f"Unknown consumer distribution family: {self.family!r}"
            ) from exc
        object.__setattr__(self, "family", family)

        if not isinstance(self.parameters, Mapping):
            raise ProtocolConfigError("Distribution parameters must be a mapping")
        parameters: dict[str, float] = {}
        for parameter_name, raw_value in self.parameters.items():
            if not isinstance(parameter_name, str) or not parameter_name:
                raise ProtocolConfigError(
                    "Distribution parameter names must be non-empty strings"
                )
            if isinstance(raw_value, bool):
                raise ProtocolConfigError(
                    f"Distribution parameter {parameter_name} must be numeric"
                )
            value = float(raw_value)
            if not np.isfinite(value):
                raise ProtocolConfigError(
                    f"Distribution parameter {parameter_name} must be finite"
                )
            parameters[parameter_name] = value

        required_parameters = {
            ConsumerDistributionFamily.UNIFORM: {"low", "high"},
            ConsumerDistributionFamily.TRUNCATED_NORMAL: {
                "mean", "standard_deviation", "low", "high"
            },
            ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL: {
                "location", "scale", "shape", "low", "high"
            },
        }[family]
        if set(parameters) != required_parameters:
            raise ProtocolConfigError(
                f"{family.value} parameters must be exactly "
                f"{sorted(required_parameters)}"
            )
        if parameters["low"] >= parameters["high"]:
            raise ProtocolConfigError(
                "Distribution lower bound must be below its upper bound"
            )
        if (
            "standard_deviation" in parameters
            and parameters["standard_deviation"] <= 0
        ):
            raise ProtocolConfigError("standard_deviation must be positive")
        if "scale" in parameters and parameters["scale"] <= 0:
            raise ProtocolConfigError("scale must be positive")
        if (
            family is ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL
            and parameters["shape"] <= 0
        ):
            raise ProtocolConfigError(
                "truncated_skew_normal shape must be positive"
            )
        object.__setattr__(self, "parameters", _immutable_mapping(parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ConsumerPopulationSpec:
    """Distribution specifications for the three independent attributes."""

    location: ConsumerDistributionSpec
    strategicness: ConsumerDistributionSpec
    exclusivity: ConsumerDistributionSpec


@dataclass(frozen=True)
class DistributionCombination:
    """Immutable coordinate in the three-family distribution grid."""

    location: ConsumerDistributionFamily
    strategicness: ConsumerDistributionFamily
    exclusivity: ConsumerDistributionFamily

    def __post_init__(self) -> None:
        for field_name in ("location", "strategicness", "exclusivity"):
            try:
                value = ConsumerDistributionFamily(getattr(self, field_name))
            except (TypeError, ValueError) as exc:
                raise ProtocolConfigError(
                    f"Invalid {field_name} distribution family"
                ) from exc
            object.__setattr__(self, field_name, value)

    @property
    def identifier(self) -> str:
        return (
            f"location-{self.location.value}"
            f"__strategicness-{self.strategicness.value}"
            f"__exclusivity-{self.exclusivity.value}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "location": self.location.value,
            "strategicness": self.strategicness.value,
            "exclusivity": self.exclusivity.value,
        }


@dataclass(frozen=True)
class TrainingBudgetConfig:
    """Fixed environment-step budget and research checkpoints."""

    environment_steps: int
    warmup_steps: int
    updates_per_step: int
    evaluation_interval_steps: int
    checkpoint_interval_steps: int

    def __post_init__(self) -> None:
        integer_fields = (
            "environment_steps",
            "warmup_steps",
            "updates_per_step",
            "evaluation_interval_steps",
            "checkpoint_interval_steps",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ProtocolConfigError(f"{field_name} must be an integer")
        if self.environment_steps <= 0:
            raise ProtocolConfigError("environment_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.environment_steps:
            raise ProtocolConfigError(
                "warmup_steps must be nonnegative and below environment_steps"
            )
        for field_name in integer_fields[2:]:
            if getattr(self, field_name) <= 0:
                raise ProtocolConfigError(f"{field_name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in (
                "environment_steps",
                "warmup_steps",
                "updates_per_step",
                "evaluation_interval_steps",
                "checkpoint_interval_steps",
            )
        }


@dataclass(frozen=True)
class AgentProfileConfig:
    """Architecture-specific configuration validated before construction."""

    architecture: AgentArchitecture
    sequence_length: int | None = None
    episode_replay_capacity: int | None = None
    opponent_embedding_dim: int | None = None
    encoder_hidden_dim: int | None = None
    auxiliary_loss_weight: float | None = None
    sac_pricing_config: SACPricingAgentConfig | None = None

    def __post_init__(self) -> None:
        try:
            architecture = AgentArchitecture(self.architecture)
        except (TypeError, ValueError) as exc:
            raise ProtocolConfigError(
                f"Unknown agent architecture: {self.architecture!r}"
            ) from exc
        object.__setattr__(self, "architecture", architecture)

        sequence_fields = ("sequence_length", "episode_replay_capacity")
        encoder_fields = (
            "opponent_embedding_dim",
            "encoder_hidden_dim",
            "auxiliary_loss_weight",
        )
        if architecture is AgentArchitecture.SAC:
            prohibited = sequence_fields + encoder_fields
            self._reject_present(prohibited)
            if self.sac_pricing_config is None:
                object.__setattr__(
                    self,
                    "sac_pricing_config",
                    SACPricingAgentConfig(),
                )
            elif not isinstance(
                self.sac_pricing_config,
                SACPricingAgentConfig,
            ):
                raise ProtocolConfigError(
                    "sac_pricing_config must be SACPricingAgentConfig"
                )
        elif architecture is AgentArchitecture.RSAC:
            if self.sac_pricing_config is not None:
                raise ProtocolConfigError(
                    "rsac rejects SAC pricing hyperparameters"
                )
            self._require_positive_integers(sequence_fields)
            self._reject_present(encoder_fields)
        else:
            if self.sac_pricing_config is not None:
                raise ProtocolConfigError(
                    "oe_rsac rejects SAC pricing hyperparameters"
                )
            self._require_positive_integers(
                sequence_fields
                + ("opponent_embedding_dim", "encoder_hidden_dim")
            )
            if self.auxiliary_loss_weight is None:
                raise ProtocolConfigError(
                    "oe_rsac requires auxiliary_loss_weight"
                )
            if (
                isinstance(self.auxiliary_loss_weight, bool)
                or not np.isfinite(float(self.auxiliary_loss_weight))
                or float(self.auxiliary_loss_weight) < 0
            ):
                raise ProtocolConfigError(
                    "auxiliary_loss_weight must be finite and nonnegative"
                )
            object.__setattr__(
                self,
                "auxiliary_loss_weight",
                float(self.auxiliary_loss_weight),
            )

    def _reject_present(self, field_names: Iterable[str]) -> None:
        present = [name for name in field_names if getattr(self, name) is not None]
        if present:
            raise ProtocolConfigError(
                f"{self.architecture.value} rejects fields: "
                f"{', '.join(present)}"
            )

    def _require_positive_integers(self, field_names: Iterable[str]) -> None:
        for field_name in field_names:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProtocolConfigError(
                    f"{self.architecture.value} requires positive {field_name}"
                )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"architecture": self.architecture.value}
        if self.sac_pricing_config is not None:
            result.update(self.sac_pricing_config.to_dict())
        for field_name in (
            "sequence_length",
            "episode_replay_capacity",
            "opponent_embedding_dim",
            "encoder_hidden_dim",
            "auxiliary_loss_weight",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


class OpponentFamily(str, Enum):
    """Opponent pricing families balanced by the episode schedule."""

    UNIFORM = "uniform"
    BBP = "bbp"


@dataclass(frozen=True)
class OpponentPoolConfig:
    """Registered policy names and equal family sampling weights."""

    uniform_policies: tuple[str, ...]
    bbp_policies: tuple[str, ...]
    uniform_weight: float = 0.5
    bbp_weight: float = 0.5

    def __post_init__(self) -> None:
        uniform = tuple(self.uniform_policies)
        bbp = tuple(self.bbp_policies)
        if not uniform or not bbp:
            raise ProtocolConfigError(
                "Both uniform and BBP opponent families must be non-empty"
            )
        all_policies = uniform + bbp
        if len(set(all_policies)) != len(all_policies):
            raise ProtocolConfigError("Opponent policy names must be unique")
        unknown = set(all_policies) - set(OPPONENT_PRESETS)
        if unknown:
            raise ProtocolConfigError(
                f"Unknown opponent policy name(s): {', '.join(sorted(unknown))}"
            )
        for policy_name in uniform:
            if not policy_name.startswith("uniform_"):
                raise ProtocolConfigError(
                    f"Uniform policy has incompatible name: {policy_name}"
                )
        for policy_name in bbp:
            if not policy_name.startswith("bbp_"):
                raise ProtocolConfigError(
                    f"BBP policy has incompatible name: {policy_name}"
                )
        weights = (float(self.uniform_weight), float(self.bbp_weight))
        if not all(np.isfinite(weight) and weight > 0 for weight in weights):
            raise ProtocolConfigError("Opponent family weights must be positive")
        if not np.isclose(weights[0], 0.5) or not np.isclose(weights[1], 0.5):
            raise ProtocolConfigError(
                "universal_pricing_v1 requires 0.5 weight for each family"
            )
        object.__setattr__(self, "uniform_policies", uniform)
        object.__setattr__(self, "bbp_policies", bbp)
        object.__setattr__(self, "uniform_weight", weights[0])
        object.__setattr__(self, "bbp_weight", weights[1])

    def policies_for(self, family: OpponentFamily) -> tuple[str, ...]:
        family = OpponentFamily(family)
        return (
            self.uniform_policies
            if family is OpponentFamily.UNIFORM
            else self.bbp_policies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_weights": {
                OpponentFamily.UNIFORM.value: self.uniform_weight,
                OpponentFamily.BBP.value: self.bbp_weight,
            },
            "families": {
                OpponentFamily.UNIFORM.value: list(self.uniform_policies),
                OpponentFamily.BBP.value: list(self.bbp_policies),
            },
        }


@dataclass(frozen=True)
class OpponentEpisodeAssignment:
    """Resolved opponent policy and seed for one episode."""

    episode_index: int
    opponent_family: OpponentFamily
    policy_name: str
    opponent_seed: int


class BalancedOpponentSchedule:
    """Deterministic 50/50 family blocks with shuffled family round-robin."""

    _FAMILY_STREAM = {
        OpponentFamily.UNIFORM: 1,
        OpponentFamily.BBP: 2,
    }

    def __init__(
        self,
        opponent_pool: OpponentPoolConfig,
        schedule_seed: int,
    ) -> None:
        self.opponent_pool = opponent_pool
        self.schedule_seed = int(schedule_seed)
        if self.schedule_seed < 0:
            raise ProtocolConfigError("schedule_seed must be nonnegative")

    def assignment(self, episode_index: int) -> OpponentEpisodeAssignment:
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            raise ProtocolConfigError("episode_index must be an integer")
        if episode_index < 0:
            raise ProtocolConfigError("episode_index must be nonnegative")

        block_index, position = divmod(episode_index, 2)
        family_order = [
            OpponentFamily.UNIFORM,
            OpponentFamily.BBP,
        ]
        block_generator = np.random.default_rng(
            np.random.SeedSequence([self.schedule_seed, 0, block_index])
        )
        block_generator.shuffle(family_order)
        family = family_order[position]
        policies = self.opponent_pool.policies_for(family)

        cycle_index, cycle_position = divmod(block_index, len(policies))
        policy_order = list(policies)
        policy_generator = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.schedule_seed,
                    self._FAMILY_STREAM[family],
                    cycle_index,
                ]
            )
        )
        policy_generator.shuffle(policy_order)
        policy_name = policy_order[cycle_position]
        opponent_seed = SeedDeriver.derive_episode_seed(
            self.schedule_seed,
            episode_index,
        )
        return OpponentEpisodeAssignment(
            episode_index=episode_index,
            opponent_family=family,
            policy_name=policy_name,
            opponent_seed=opponent_seed,
        )

    def assignments(
        self,
        episode_count: int,
    ) -> tuple[OpponentEpisodeAssignment, ...]:
        if not isinstance(episode_count, int) or isinstance(episode_count, bool):
            raise ProtocolConfigError("episode_count must be an integer")
        if episode_count < 0:
            raise ProtocolConfigError("episode_count must be nonnegative")
        return tuple(self.assignment(index) for index in range(episode_count))


class SeedPurpose(IntEnum):
    """Stable stream identifiers; values are part of the public protocol."""

    NETWORK_INITIALIZATION = 1
    CONSUMER_POPULATION = 2
    OPPONENT_SCHEDULE = 3
    EXPLORATION = 4
    REPLAY_SAMPLING = 5
    TORCH_CPU = 6
    TORCH_CUDA = 7


@dataclass(frozen=True)
class SeedBankManifest:
    """Committed confirmatory, validation, and locked evaluation seed banks."""

    protocol_root_seed: int
    training_roots: tuple[int, ...]
    validation_environment_seeds: tuple[int, ...]
    final_evaluation_environment_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_root_seed", int(self.protocol_root_seed))
        for field_name, expected_count in (
            ("training_roots", TRAINING_SEED_COUNT),
            ("validation_environment_seeds", VALIDATION_SEED_COUNT),
            (
                "final_evaluation_environment_seeds",
                FINAL_EVALUATION_SEED_COUNT,
            ),
        ):
            seeds = tuple(int(seed) for seed in getattr(self, field_name))
            if len(seeds) != expected_count:
                raise ProtocolConfigError(
                    f"{field_name} must contain {expected_count} seeds"
                )
            if any(seed < 0 for seed in seeds):
                raise ProtocolConfigError(f"{field_name} seeds must be nonnegative")
            object.__setattr__(self, field_name, seeds)
        all_seeds = (
            self.training_roots
            + self.validation_environment_seeds
            + self.final_evaluation_environment_seeds
        )
        if len(set(all_seeds)) != len(all_seeds):
            raise ProtocolConfigError("Seed banks must be mutually disjoint")

    @classmethod
    def from_root_seed(
        cls,
        root_seed: int = PROTOCOL_ROOT_SEED,
    ) -> "SeedBankManifest":
        children = np.random.SeedSequence(int(root_seed)).spawn(
            TRAINING_SEED_COUNT
            + VALIDATION_SEED_COUNT
            + FINAL_EVALUATION_SEED_COUNT
        )
        seeds = tuple(
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in children
        )
        training_end = TRAINING_SEED_COUNT
        validation_end = training_end + VALIDATION_SEED_COUNT
        return cls(
            protocol_root_seed=int(root_seed),
            training_roots=seeds[:training_end],
            validation_environment_seeds=seeds[
                training_end:validation_end
            ],
            final_evaluation_environment_seeds=seeds[validation_end:],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_root_seed": self.protocol_root_seed,
            "training_roots": list(self.training_roots),
            "validation_environment_seeds": list(
                self.validation_environment_seeds
            ),
            "final_evaluation_environment_seeds": list(
                self.final_evaluation_environment_seeds
            ),
        }


@dataclass(frozen=True)
class RunSeedBundle:
    """All independent deterministic streams for one training replicate."""

    run_seed: int
    network_initialization_seed: int
    consumer_population_seed: int
    opponent_schedule_seed: int
    exploration_seed: int
    replay_sampling_seed: int
    torch_cpu_seed: int
    torch_cuda_seed: int

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class EpisodeSeedBundle:
    """Consumer and opponent streams derived for one episode."""

    consumer_seed: int
    opponent_seed: int


class SeedDeriver:
    """Pure call-order-independent NumPy SeedSequence derivation."""

    @staticmethod
    def derive_stream_seed(run_seed: int, purpose: SeedPurpose) -> int:
        sequence = np.random.SeedSequence(
            [int(run_seed), int(SeedPurpose(purpose))]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def derive_episode_seed(stream_seed: int, episode_index: int) -> int:
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            raise ProtocolConfigError("episode_index must be an integer")
        if episode_index < 0:
            raise ProtocolConfigError("episode_index must be nonnegative")
        sequence = np.random.SeedSequence(
            [int(stream_seed), int(episode_index)]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    @classmethod
    def derive_run_bundle(cls, run_seed: int) -> RunSeedBundle:
        return RunSeedBundle(
            run_seed=int(run_seed),
            network_initialization_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.NETWORK_INITIALIZATION
            ),
            consumer_population_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.CONSUMER_POPULATION
            ),
            opponent_schedule_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.OPPONENT_SCHEDULE
            ),
            exploration_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.EXPLORATION
            ),
            replay_sampling_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.REPLAY_SAMPLING
            ),
            torch_cpu_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.TORCH_CPU
            ),
            torch_cuda_seed=cls.derive_stream_seed(
                run_seed, SeedPurpose.TORCH_CUDA
            ),
        )

    @classmethod
    def derive_episode_bundle(
        cls,
        run_seed_bundle: RunSeedBundle,
        episode_index: int,
    ) -> EpisodeSeedBundle:
        return EpisodeSeedBundle(
            consumer_seed=cls.derive_episode_seed(
                run_seed_bundle.consumer_population_seed,
                episode_index,
            ),
            opponent_seed=cls.derive_episode_seed(
                run_seed_bundle.opponent_schedule_seed,
                episode_index,
            ),
        )


@dataclass(frozen=True)
class UniversalPricingProtocolConfig:
    """Fully resolved and validated universal pricing protocol."""

    protocol_version: str
    action_contract_version: str
    observation_contract_version: str
    curriculum_id: str
    regime_commitment_length: int
    reward_specification: str
    artifact_root: Path
    agent_profiles: Mapping[AgentArchitecture, AgentProfileConfig]
    consumer_distributions: Mapping[
        str,
        Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec],
    ]
    opponent_pool: OpponentPoolConfig
    seed_manifest: SeedBankManifest
    training_budget: TrainingBudgetConfig

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolConfigError(
                f"Expected protocol_version {PROTOCOL_VERSION!r}"
            )
        if self.action_contract_version != ACTION_CONTRACT_VERSION:
            raise ProtocolConfigError(
                f"Expected action contract {ACTION_CONTRACT_VERSION!r}"
            )
        if self.observation_contract_version != OBSERVATION_CONTRACT_VERSION:
            raise ProtocolConfigError(
                f"Expected observation contract {OBSERVATION_CONTRACT_VERSION!r}"
            )
        if self.curriculum_id != PRIMARY_CURRICULUM_ID:
            raise ProtocolConfigError(
                f"Expected curriculum_id {PRIMARY_CURRICULUM_ID!r}"
            )
        if (
            not isinstance(self.regime_commitment_length, int)
            or isinstance(self.regime_commitment_length, bool)
            or self.regime_commitment_length <= 0
        ):
            raise ProtocolConfigError(
                "regime_commitment_length must be a positive integer"
            )
        if self.reward_specification != "normalized_raw_profit":
            raise ProtocolConfigError(
                "reward_specification must be 'normalized_raw_profit'"
            )

        profiles = {
            AgentArchitecture(key): value
            for key, value in self.agent_profiles.items()
        }
        expected_architectures = set(AgentArchitecture)
        if set(profiles) != expected_architectures:
            raise ProtocolConfigError(
                "Agent profiles must define exactly sac, rsac, and oe_rsac"
            )
        for architecture, profile in profiles.items():
            if profile.architecture is not architecture:
                raise ProtocolConfigError(
                    f"Agent profile key does not match {profile.architecture.value}"
                )
            if architecture not in AGENT_ARCHITECTURE_SPECS:
                raise ProtocolConfigError(
                    f"Architecture specification is missing: {architecture.value}"
                )

        expected_attributes = {"location", "strategicness", "exclusivity"}
        if set(self.consumer_distributions) != expected_attributes:
            raise ProtocolConfigError(
                "Consumer distributions must define location, strategicness, "
                "and exclusivity"
            )
        distribution_catalog: dict[
            str,
            Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec],
        ] = {}
        for attribute_name, raw_specs in self.consumer_distributions.items():
            specs = {
                ConsumerDistributionFamily(family): spec
                for family, spec in raw_specs.items()
            }
            if set(specs) != set(ConsumerDistributionFamily):
                raise ProtocolConfigError(
                    f"{attribute_name} must define all three distribution families"
                )
            for family, spec in specs.items():
                if spec.family is not family:
                    raise ProtocolConfigError(
                        f"{attribute_name} distribution key/family mismatch"
                    )
            distribution_catalog[attribute_name] = MappingProxyType(specs)

        if self.seed_manifest.protocol_root_seed != PROTOCOL_ROOT_SEED:
            raise ProtocolConfigError(
                f"Protocol root seed must be {PROTOCOL_ROOT_SEED}"
            )
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        object.__setattr__(self, "agent_profiles", MappingProxyType(profiles))
        object.__setattr__(
            self,
            "consumer_distributions",
            MappingProxyType(distribution_catalog),
        )

    def population_spec(
        self,
        combination: DistributionCombination,
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
        if not isinstance(seed_index, int) or isinstance(seed_index, bool):
            raise ProtocolConfigError("seed_index must be an integer")
        if seed_index < 0 or seed_index >= len(
            self.seed_manifest.training_roots
        ):
            raise ProtocolConfigError("seed_index must be between 0 and 9")
        return SeedDeriver.derive_run_bundle(
            self.seed_manifest.training_roots[seed_index]
        )

    def to_dict(self) -> dict[str, Any]:
        distributions: dict[str, Any] = {}
        for attribute_name, specs in self.consumer_distributions.items():
            distributions[attribute_name] = {
                family.value: dict(spec.parameters)
                for family, spec in specs.items()
            }
        return {
            "protocol_version": self.protocol_version,
            "action_contract_version": self.action_contract_version,
            "observation_contract_version": self.observation_contract_version,
            "curriculum_id": self.curriculum_id,
            "regime_commitment_length": self.regime_commitment_length,
            "reward_specification": self.reward_specification,
            "artifact_root": str(self.artifact_root),
            "agent_profiles": {
                architecture.value: profile.to_dict()
                for architecture, profile in self.agent_profiles.items()
            },
            "consumer_distributions": distributions,
            "opponent_pool": self.opponent_pool.to_dict(),
            "seed_manifest": self.seed_manifest.to_dict(),
            "training_budget": self.training_budget.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentCoordinate:
    """One primary model/distribution/curriculum/training-seed coordinate."""

    agent_architecture: AgentArchitecture
    distribution_combination: DistributionCombination
    curriculum_id: str
    training_seed_index: int

    def __post_init__(self) -> None:
        try:
            architecture = AgentArchitecture(self.agent_architecture)
        except (TypeError, ValueError) as exc:
            raise ProtocolConfigError("Invalid coordinate architecture") from exc
        object.__setattr__(self, "agent_architecture", architecture)
        if self.curriculum_id != PRIMARY_CURRICULUM_ID:
            raise ProtocolConfigError(
                f"Coordinate curriculum must be {PRIMARY_CURRICULUM_ID}"
            )
        if (
            not isinstance(self.training_seed_index, int)
            or isinstance(self.training_seed_index, bool)
            or not 0 <= self.training_seed_index < TRAINING_SEED_COUNT
        ):
            raise ProtocolConfigError(
                "training_seed_index must be between 0 and 9"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_architecture": self.agent_architecture.value,
            "distribution_combination": (
                self.distribution_combination.to_dict()
            ),
            "curriculum_id": self.curriculum_id,
            "training_seed_index": self.training_seed_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentCoordinate":
        distributions = value["distribution_combination"]
        return cls(
            agent_architecture=AgentArchitecture(value["agent_architecture"]),
            distribution_combination=DistributionCombination(
                location=ConsumerDistributionFamily(distributions["location"]),
                strategicness=ConsumerDistributionFamily(
                    distributions["strategicness"]
                ),
                exclusivity=ConsumerDistributionFamily(
                    distributions["exclusivity"]
                ),
            ),
            curriculum_id=str(value["curriculum_id"]),
            training_seed_index=int(value["training_seed_index"]),
        )


class ExperimentMatrix:
    """Deterministically enumerate the 810 primary experiment coordinates."""

    def __init__(self, protocol: UniversalPricingProtocolConfig) -> None:
        self.protocol = protocol

    def distribution_combinations(
        self,
    ) -> tuple[DistributionCombination, ...]:
        families = tuple(ConsumerDistributionFamily)
        return tuple(
            DistributionCombination(location, strategicness, exclusivity)
            for location in families
            for strategicness in families
            for exclusivity in families
        )

    def coordinates(self) -> tuple[ExperimentCoordinate, ...]:
        coordinates = tuple(
            ExperimentCoordinate(
                agent_architecture=architecture,
                distribution_combination=combination,
                curriculum_id=self.protocol.curriculum_id,
                training_seed_index=seed_index,
            )
            for architecture in AgentArchitecture
            for combination in self.distribution_combinations()
            for seed_index in range(TRAINING_SEED_COUNT)
        )
        if len(coordinates) != 810 or len(set(coordinates)) != 810:
            raise ProtocolConfigError(
                "Primary experiment matrix must contain 810 unique coordinates"
            )
        return coordinates


class RunStatus(str, Enum):
    """Lifecycle status recorded in an experiment run manifest."""

    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ExperimentRunId:
    """Stable serial form of an :class:`ExperimentCoordinate`."""

    value: str

    _PATTERN = re.compile(
        r"^universal_pricing_v1__mixed_balanced__"
        r"(?P<agent>sac|rsac|oe_rsac)"
        r"__location-(?P<location>uniform|truncated_normal|truncated_skew_normal)"
        r"__strategicness-(?P<strategicness>uniform|truncated_normal|truncated_skew_normal)"
        r"__exclusivity-(?P<exclusivity>uniform|truncated_normal|truncated_skew_normal)"
        r"__seed-(?P<seed>\d{2})$"
    )

    def __post_init__(self) -> None:
        self.parse(self.value)

    @classmethod
    def from_coordinate(
        cls,
        coordinate: ExperimentCoordinate,
    ) -> "ExperimentRunId":
        combination = coordinate.distribution_combination
        return cls(
            f"{PROTOCOL_VERSION}"
            f"__{coordinate.curriculum_id}"
            f"__{coordinate.agent_architecture.value}"
            f"__location-{combination.location.value}"
            f"__strategicness-{combination.strategicness.value}"
            f"__exclusivity-{combination.exclusivity.value}"
            f"__seed-{coordinate.training_seed_index:02d}"
        )

    @classmethod
    def parse(cls, value: str) -> ExperimentCoordinate:
        match = cls._PATTERN.fullmatch(value)
        if match is None:
            raise ProtocolConfigError(f"Malformed experiment run ID: {value!r}")
        seed_index = int(match.group("seed"))
        return ExperimentCoordinate(
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
            curriculum_id=PRIMARY_CURRICULUM_ID,
            training_seed_index=seed_index,
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ArtifactLayout:
    """Pure artifact path builder; methods never create filesystem entries."""

    artifact_root: Path

    def run_directory(self, coordinate: ExperimentCoordinate) -> Path:
        return (
            Path(self.artifact_root)
            / coordinate.curriculum_id
            / coordinate.agent_architecture.value
            / coordinate.distribution_combination.identifier
            / f"seed-{coordinate.training_seed_index}"
        )

    def manifest_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "manifest.json"

    def checkpoint_directory(self, coordinate: ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "checkpoints"

    def metrics_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.run_directory(coordinate) / "metrics.jsonl"


def stable_configuration_hash(values: Mapping[str, Any]) -> str:
    """Hash structured configuration using canonical UTF-8 JSON."""

    canonical = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ExperimentRunManifest:
    """Immutable run identity plus versioned resolved research facts."""

    protocol_version: str
    run_id: ExperimentRunId
    coordinate: ExperimentCoordinate
    run_seed_bundle: RunSeedBundle
    resolved_protocol: Mapping[str, Any]
    configuration_hash: str
    git_commit: str
    hardware_metadata: Mapping[str, Any]
    status: RunStatus
    artifact_references: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolConfigError(
                f"Manifest protocol must be {PROTOCOL_VERSION}"
            )
        if self.run_id.parse(self.run_id.value) != self.coordinate:
            raise ProtocolConfigError(
                "Manifest run ID does not match its experiment coordinate"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_hash):
            raise ProtocolConfigError(
                "Manifest configuration_hash must be lowercase SHA-256"
            )
        try:
            status = RunStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ProtocolConfigError("Invalid run manifest status") from exc
        if not isinstance(self.git_commit, str) or not self.git_commit.strip():
            raise ProtocolConfigError("Manifest git_commit must be non-empty")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "resolved_protocol",
            _immutable_mapping(self.resolved_protocol),
        )
        if stable_configuration_hash(
            dict(self.resolved_protocol)
        ) != self.configuration_hash:
            raise ProtocolConfigError(
                "Manifest configuration_hash does not match resolved_protocol"
            )
        object.__setattr__(
            self,
            "hardware_metadata",
            _immutable_mapping(self.hardware_metadata),
        )
        object.__setattr__(
            self,
            "artifact_references",
            _immutable_mapping(self.artifact_references),
        )

    def identity(self) -> tuple[str, str, ExperimentCoordinate]:
        return (self.protocol_version, self.run_id.value, self.coordinate)

    def immutable_facts(self) -> tuple[Any, ...]:
        return (
            self.identity(),
            self.run_seed_bundle,
            self.configuration_hash,
            self.git_commit,
            json.dumps(
                dict(self.hardware_metadata),
                sort_keys=True,
                separators=(",", ":"),
            ),
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
            "artifact_references": dict(self.artifact_references),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentRunManifest":
        try:
            seed_values = value["run_seed_bundle"]
            return cls(
                protocol_version=str(value["protocol_version"]),
                run_id=ExperimentRunId(str(value["run_id"])),
                coordinate=ExperimentCoordinate.from_dict(value["coordinate"]),
                run_seed_bundle=RunSeedBundle(
                    **{
                        field_name: int(seed_values[field_name])
                        for field_name in RunSeedBundle.__dataclass_fields__
                    }
                ),
                resolved_protocol=value["resolved_protocol"],
                configuration_hash=str(value["configuration_hash"]),
                git_commit=str(value["git_commit"]),
                hardware_metadata=value["hardware_metadata"],
                status=RunStatus(value["status"]),
                artifact_references=value.get("artifact_references", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProtocolConfigError):
                raise
            raise ProtocolConfigError(
                f"Malformed experiment run manifest: {exc}"
            ) from exc


class ManifestRepository:
    """Read and atomically replace compatible run manifests."""

    def read(self, manifest_path: str | Path) -> ExperimentRunManifest:
        path = Path(manifest_path)
        if not path.is_file():
            raise ProtocolConfigError(f"Manifest does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolConfigError(f"Cannot read manifest {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProtocolConfigError("Manifest root must be a JSON object")
        return ExperimentRunManifest.from_dict(value)

    def write(
        self,
        manifest_path: str | Path,
        manifest: ExperimentRunManifest,
    ) -> None:
        path = Path(manifest_path)
        if path.exists():
            existing = self.read(path)
            if existing.identity() != manifest.identity():
                raise ProtocolConfigError(
                    "Existing manifest identity cannot be changed"
                )
            if existing.immutable_facts() != manifest.immutable_facts():
                raise ProtocolConfigError(
                    "Existing manifest immutable facts cannot be changed"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    manifest.to_dict(),
                    stream,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def _parse_agent_profiles(
    raw_profiles: Any,
    location: str,
) -> Mapping[AgentArchitecture, AgentProfileConfig]:
    if not isinstance(raw_profiles, dict):
        raise ProtocolConfigError(f"{location} must be a mapping")
    profiles: dict[AgentArchitecture, AgentProfileConfig] = {}
    recurrent_fields = {
        "architecture",
        "sequence_length",
        "episode_replay_capacity",
        "opponent_embedding_dim",
        "encoder_hidden_dim",
        "auxiliary_loss_weight",
    }
    sac_fields = set(SACPricingAgentConfig.__dataclass_fields__)
    allowed = recurrent_fields | sac_fields
    for raw_name, raw_profile in raw_profiles.items():
        try:
            architecture = AgentArchitecture(raw_name)
        except ValueError as exc:
            raise ProtocolConfigError(
                f"Unknown agent profile: {raw_name}"
            ) from exc
        if not isinstance(raw_profile, dict):
            raise ProtocolConfigError(
                f"{location}:{raw_name} must be a mapping"
            )
        _reject_unknown(raw_profile, allowed, f"{location}:{raw_name}")
        _require(raw_profile, {"architecture"}, f"{location}:{raw_name}")
        profile_values = dict(raw_profile)
        sac_values = {
            field_name: profile_values.pop(field_name)
            for field_name in tuple(profile_values)
            if field_name in sac_fields
        }
        if sac_values and architecture is not AgentArchitecture.SAC:
            raise ProtocolConfigError(
                f"{architecture.value} rejects SAC pricing fields: "
                + ", ".join(sorted(sac_values))
            )
        try:
            sac_config = (
                SACPricingAgentConfig(**sac_values)
                if architecture is AgentArchitecture.SAC
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolConfigError(
                f"Invalid SAC profile in {location}:{raw_name}: {exc}"
            ) from exc
        profiles[architecture] = AgentProfileConfig(
            **profile_values,
            sac_pricing_config=sac_config,
        )
    return profiles


def _parse_consumer_distributions(
    raw_distributions: Any,
    location: str,
) -> Mapping[
    str,
    Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec],
]:
    if not isinstance(raw_distributions, dict):
        raise ProtocolConfigError(f"{location} must be a mapping")
    catalog: dict[
        str,
        Mapping[ConsumerDistributionFamily, ConsumerDistributionSpec],
    ] = {}
    for attribute_name, raw_specs in raw_distributions.items():
        if not isinstance(raw_specs, dict):
            raise ProtocolConfigError(
                f"{location}:{attribute_name} must be a mapping"
            )
        specs: dict[
            ConsumerDistributionFamily,
            ConsumerDistributionSpec,
        ] = {}
        for raw_family, raw_parameters in raw_specs.items():
            try:
                family = ConsumerDistributionFamily(raw_family)
            except ValueError as exc:
                raise ProtocolConfigError(
                    f"Unknown distribution family: {raw_family}"
                ) from exc
            specs[family] = ConsumerDistributionSpec(
                family=family,
                parameters=raw_parameters,
            )
        catalog[attribute_name] = specs
    return catalog


def load_opponent_pool_config(path: str | Path) -> OpponentPoolConfig:
    """Load the balanced nine-policy universal opponent pool."""

    pool_path = Path(path).resolve()
    raw = _load_yaml_mapping(pool_path)
    _reject_unknown(raw, {"family_weights", "families"}, str(pool_path))
    _require(raw, {"family_weights", "families"}, str(pool_path))
    weights = raw["family_weights"]
    families = raw["families"]
    if not isinstance(weights, dict) or not isinstance(families, dict):
        raise ProtocolConfigError(
            "Opponent family_weights and families must be mappings"
        )
    expected = {family.value for family in OpponentFamily}
    if set(weights) != expected or set(families) != expected:
        raise ProtocolConfigError(
            "Opponent pool must define exactly uniform and bbp families"
        )
    for family_name, policies in families.items():
        if not isinstance(policies, list) or not all(
            isinstance(policy, str) and policy for policy in policies
        ):
            raise ProtocolConfigError(
                f"Opponent family {family_name} must be a list of names"
            )
    return OpponentPoolConfig(
        uniform_policies=tuple(families["uniform"]),
        bbp_policies=tuple(families["bbp"]),
        uniform_weight=weights["uniform"],
        bbp_weight=weights["bbp"],
    )


def load_seed_bank_manifest(path: str | Path) -> SeedBankManifest:
    """Load and verify the committed seed banks against the root seed."""

    seed_path = Path(path).resolve()
    raw = _load_yaml_mapping(seed_path)
    allowed = {
        "protocol_root_seed",
        "training_roots",
        "validation_environment_seeds",
        "final_evaluation_environment_seeds",
    }
    _reject_unknown(raw, allowed, str(seed_path))
    _require(raw, allowed, str(seed_path))
    manifest = SeedBankManifest(
        protocol_root_seed=raw["protocol_root_seed"],
        training_roots=tuple(raw["training_roots"]),
        validation_environment_seeds=tuple(
            raw["validation_environment_seeds"]
        ),
        final_evaluation_environment_seeds=tuple(
            raw["final_evaluation_environment_seeds"]
        ),
    )
    expected = SeedBankManifest.from_root_seed(manifest.protocol_root_seed)
    if manifest != expected:
        raise ProtocolConfigError(
            f"Committed seed bank does not match root seed in {seed_path}"
        )
    return manifest


def load_universal_pricing_protocol(
    path: str | Path,
) -> UniversalPricingProtocolConfig:
    """Resolve and validate a universal_pricing_v1 YAML protocol."""

    protocol_path = Path(path).resolve()
    raw = _load_yaml_mapping(protocol_path)
    allowed = {
        "protocol_version",
        "action_contract_version",
        "observation_contract_version",
        "curriculum_id",
        "regime_commitment_length",
        "reward_specification",
        "artifact_root",
        "agent_profiles",
        "consumer_distributions",
        "opponent_pool_config",
        "seed_manifest",
        "training_budget",
    }
    _reject_unknown(raw, allowed, str(protocol_path))
    _require(raw, allowed, str(protocol_path))

    budget = raw["training_budget"]
    if not isinstance(budget, dict):
        raise ProtocolConfigError("training_budget must be a mapping")
    budget_fields = set(TrainingBudgetConfig.__dataclass_fields__)
    _reject_unknown(budget, budget_fields, "training_budget")
    _require(budget, budget_fields, "training_budget")

    artifact_root = Path(raw["artifact_root"])
    if str(artifact_root).strip() in {"", "."}:
        raise ProtocolConfigError("artifact_root must name an output directory")

    return UniversalPricingProtocolConfig(
        protocol_version=raw["protocol_version"],
        action_contract_version=raw["action_contract_version"],
        observation_contract_version=raw["observation_contract_version"],
        curriculum_id=raw["curriculum_id"],
        regime_commitment_length=raw["regime_commitment_length"],
        reward_specification=raw["reward_specification"],
        artifact_root=artifact_root,
        agent_profiles=_parse_agent_profiles(
            raw["agent_profiles"],
            f"{protocol_path}:agent_profiles",
        ),
        consumer_distributions=_parse_consumer_distributions(
            raw["consumer_distributions"],
            f"{protocol_path}:consumer_distributions",
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
                protocol_path,
                raw["seed_manifest"],
                "seed_manifest",
            )
        ),
        training_budget=TrainingBudgetConfig(**budget),
    )


def select_experiment_coordinate(
    protocol: UniversalPricingProtocolConfig,
    *,
    agent_architecture: str | AgentArchitecture,
    location_distribution: str | ConsumerDistributionFamily,
    strategicness_distribution: str | ConsumerDistributionFamily,
    exclusivity_distribution: str | ConsumerDistributionFamily,
    training_seed_index: int,
) -> ExperimentCoordinate:
    """Validate canonical CLI selectors without constructing runtime objects."""

    coordinate = ExperimentCoordinate(
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
        curriculum_id=protocol.curriculum_id,
        training_seed_index=training_seed_index,
    )
    if coordinate not in ExperimentMatrix(protocol).coordinates():
        raise ProtocolConfigError("Coordinate is outside the primary matrix")
    return coordinate
