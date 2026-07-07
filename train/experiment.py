"""Load composed YAML experiments and build their initial runtime objects."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from env.opponent_policies import OPPONENT_PRESETS
from env.type import EnvironmentType
from train.config import TrainingConfig
from train.curriculum import (
    Curriculum,
    CurriculumConfig,
    OpponentDifficulty,
    OpponentStage,
)


class ExperimentConfigError(ValueError):
    """Raised when a composed experiment configuration is invalid."""


class ConfiguredCurriculum(Curriculum):
    """A curriculum whose ordered stages were loaded from configuration."""

    def __init__(self, stages: list[OpponentStage]) -> None:
        super().__init__()
        self.opponent_sequence = stages

    def get_sequence(self) -> list[OpponentStage]:
        return list(self.opponent_sequence)

    @property
    def opponent_types(self) -> list[str]:
        return [stage.opponent_type for stage in self.opponent_sequence]


@dataclass(frozen=True)
class ExperimentOverrides:
    """Optional command-line values applied after all YAML layers."""

    episodes: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    save_dir: Optional[str] = None
    training_config: Optional[Path] = None


@dataclass(frozen=True)
class ResolvedExperiment:
    """Validated runtime configuration composed from three YAML layers."""

    name: str
    source: Path
    training_source: Path
    training_config: TrainingConfig
    curriculum_config: CurriculumConfig


EXPERIMENT_KEYS = {
    "name", "agent_strategy", "training_config", "curriculum_config", "save_dir"
}
TRAINING_SECTIONS = {
    "environment": {"num_consumers", "episode_length"},
    "agent": {
        "agent_type", "hidden_dim", "lr_actor", "lr_critic", "lr_alpha", "target_entropy",
        "gamma", "tau", "auto_alpha", "alpha", "log_std_min", "log_std_max",
        "buffer_size", "batch_size", "lr_scheduler", "lr_scheduler_kwargs", "device",
        "sequence_length", "episode_buffer_capacity", "opponent_action_dim",
        "opponent_embedding_dim", "encoder_hidden_dim", "actor_hidden_dim",
        "critic_hidden_dim", "lr_encoder", "opponent_aux_loss_weight",
        "grad_clip_norm", "min_episodes_before_update", "current_stage_weight",
    },
    "training": {
        "num_episodes", "warmup_steps", "updates_per_step",
        "stage_warmup_random_prob",
    },
    "evaluation": {"eval_freq", "eval_episodes", "eval_seed", "eval_seed_count"},
    "logging": {"log_freq", "save_freq", "verbose"},
    "reproducibility": {"seed"},
}
CURRICULUM_KEYS = {"scheduler", "stages"}
SCHEDULER_KEYS = {
    "window_size", "change_threshold", "monitor_critic", "monitor_actor",
    "monitor_alpha", "min_episodes_per_stage", "max_episodes_per_stage",
}
STAGE_KEYS = {
    "name", "opponent_type", "difficulty", "description",
    "min_episodes", "max_episodes",
    "min_episodes_per_stage", "max_episodes_per_stage",
}


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentConfigError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"Configuration must be a mapping: {path}")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ExperimentConfigError(
            f"Unknown key(s) in {location}: {', '.join(sorted(unknown))}"
        )


def _require(data: Mapping[str, Any], keys: set[str], location: str) -> None:
    missing = keys - set(data)
    if missing:
        raise ExperimentConfigError(
            f"Missing key(s) in {location}: {', '.join(sorted(missing))}"
        )


def _resolve_reference(owner: Path, reference: Any, field_name: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ExperimentConfigError(f"{field_name} in {owner} must be a path string")
    path = Path(reference)
    return (owner.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _load_training_values(path: Path) -> dict[str, Any]:
    raw = _load_mapping(path)
    _reject_unknown(raw, set(TRAINING_SECTIONS), str(path))
    values: dict[str, Any] = {}
    for section, allowed in TRAINING_SECTIONS.items():
        section_data = raw.get(section, {})
        if not isinstance(section_data, dict):
            raise ExperimentConfigError(f"{path}:{section} must be a mapping")
        _reject_unknown(section_data, allowed, f"{path}:{section}")
        values.update(section_data)
    return values


def load_curriculum_config(
    path: str | Path,
    environment_type: EnvironmentType,
    num_consumers: int,
    episode_length: int,
    verbose: bool = True,
) -> CurriculumConfig:
    """Load and validate one reusable curriculum YAML file."""
    curriculum_path = Path(path).resolve()
    raw = _load_mapping(curriculum_path)
    _reject_unknown(raw, CURRICULUM_KEYS, str(curriculum_path))
    _require(raw, {"stages"}, str(curriculum_path))

    scheduler = raw.get("scheduler", {})
    if not isinstance(scheduler, dict):
        raise ExperimentConfigError(f"{curriculum_path}:scheduler must be a mapping")
    _reject_unknown(scheduler, SCHEDULER_KEYS, f"{curriculum_path}:scheduler")

    raw_stages = raw["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ExperimentConfigError(f"{curriculum_path}:stages must be a non-empty list")

    stages: list[OpponentStage] = []
    names: set[str] = set()
    opponents: set[str] = set()
    for index, raw_stage in enumerate(raw_stages):
        location = f"{curriculum_path}:stages[{index}]"
        if not isinstance(raw_stage, dict):
            raise ExperimentConfigError(f"{location} must be a mapping")
        _reject_unknown(raw_stage, STAGE_KEYS, location)
        _require(raw_stage, {"name", "opponent_type", "difficulty", "description"}, location)

        name = raw_stage["name"]
        opponent = raw_stage["opponent_type"]
        if name in names:
            raise ExperimentConfigError(f"Duplicate stage name in {curriculum_path}: {name}")
        if opponent in opponents:
            raise ExperimentConfigError(
                f"Duplicate opponent type in {curriculum_path}: {opponent}"
            )
        if opponent not in OPPONENT_PRESETS:
            raise ExperimentConfigError(f"Unknown opponent preset in {location}: {opponent}")
        try:
            difficulty = OpponentDifficulty[str(raw_stage["difficulty"]).upper()]
        except KeyError as exc:
            allowed = ", ".join(item.name.lower() for item in OpponentDifficulty)
            raise ExperimentConfigError(
                f"Invalid difficulty in {location}; expected one of: {allowed}"
            ) from exc

        names.add(name)
        opponents.add(opponent)
        min_episodes = int(raw_stage.get(
            "min_episodes",
            raw_stage.get("min_episodes_per_stage", 100),
        ))
        max_episodes = raw_stage.get(
            "max_episodes",
            raw_stage.get("max_episodes_per_stage"),
        )
        if min_episodes <= 0:
            raise ExperimentConfigError(f"{location}: minimum episodes must be positive")
        if max_episodes is not None:
            max_episodes = int(max_episodes)
            if max_episodes < min_episodes:
                raise ExperimentConfigError(
                    f"{location}: maximum episodes cannot be below minimum"
                )

        stages.append(OpponentStage(
            name=name,
            opponent_type=opponent,
            difficulty=difficulty,
            description=str(raw_stage["description"]),
            min_episodes=min_episodes,
            max_episodes=max_episodes,
        ))

    curriculum = ConfiguredCurriculum(stages)
    config = CurriculumConfig(
        curriculum=curriculum,
        stages=stages,
        environment_type=environment_type,
        num_consumers=num_consumers,
        episode_length=episode_length,
        verbose=verbose,
        **scheduler,
    )
    _validate_curriculum(config, curriculum_path)
    return config


def _validate_training(config: TrainingConfig, path: Path) -> None:
    if config.agent_type not in {"sac", "recurrent_sac"}:
        raise ExperimentConfigError(
            f"{path}: agent_type must be 'sac' or 'recurrent_sac'"
        )
    positive = (
        "num_consumers", "episode_length", "hidden_dim", "buffer_size", "batch_size",
        "num_episodes", "warmup_steps", "updates_per_step", "eval_freq",
        "eval_episodes", "log_freq", "save_freq",
    )
    for name in positive:
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ExperimentConfigError(f"{path}: {name} must be positive")
    for name in ("lr_actor", "lr_critic", "lr_alpha", "alpha"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ExperimentConfigError(f"{path}: {name} must be positive")
    if config.buffer_size < config.batch_size:
        raise ExperimentConfigError(f"{path}: buffer_size cannot be below batch_size")
    if not isinstance(config.gamma, (int, float)) or not 0 < config.gamma <= 1:
        raise ExperimentConfigError(f"{path}: gamma must be in (0, 1]")
    if not isinstance(config.tau, (int, float)) or not 0 < config.tau <= 1:
        raise ExperimentConfigError(f"{path}: tau must be in (0, 1]")
    if (not isinstance(config.log_std_min, (int, float)) or
            not isinstance(config.log_std_max, (int, float)) or
            config.log_std_min >= config.log_std_max):
        raise ExperimentConfigError(f"{path}: log_std_min must be less than log_std_max")
    if config.lr_scheduler not in {None, "cosine", "step", "exponential"}:
        raise ExperimentConfigError(
            f"{path}: lr_scheduler must be null, cosine, step, or exponential"
        )
    if (
        not isinstance(config.stage_warmup_random_prob, (int, float))
        or isinstance(config.stage_warmup_random_prob, bool)
        or not 0.0 <= config.stage_warmup_random_prob <= 1.0
    ):
        raise ExperimentConfigError(
            f"{path}: stage_warmup_random_prob must be in [0, 1]"
        )
    if (
        config.eval_seed_count is not None
        and (
            not isinstance(config.eval_seed_count, int)
            or isinstance(config.eval_seed_count, bool)
            or config.eval_seed_count <= 0
        )
    ):
        raise ExperimentConfigError(
            f"{path}: eval_seed_count must be positive or null"
        )
    if (
        config.eval_seed is not None
        and (
            not isinstance(config.eval_seed, int)
            or isinstance(config.eval_seed, bool)
            or config.eval_seed < 0
        )
    ):
        raise ExperimentConfigError(f"{path}: eval_seed must be a non-negative integer")
    if config.agent_type == "recurrent_sac":
        recurrent_positive = (
            "sequence_length", "episode_buffer_capacity", "opponent_action_dim",
            "opponent_embedding_dim", "encoder_hidden_dim", "actor_hidden_dim",
            "critic_hidden_dim", "lr_encoder",
        )
        for name in recurrent_positive:
            value = getattr(config, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ExperimentConfigError(f"{path}: {name} must be positive")
        if config.opponent_aux_loss_weight < 0:
            raise ExperimentConfigError(
                f"{path}: opponent_aux_loss_weight cannot be negative"
            )
        if not 0.0 < config.current_stage_weight <= 1.0:
            raise ExperimentConfigError(
                f"{path}: current_stage_weight must be in (0, 1]"
            )
        if (
            config.min_episodes_before_update is not None
            and config.min_episodes_before_update <= 0
        ):
            raise ExperimentConfigError(
                f"{path}: min_episodes_before_update must be positive or null"
            )


def _validate_curriculum(config: CurriculumConfig, path: Path) -> None:
    if config.window_size <= 1:
        raise ExperimentConfigError(f"{path}: window_size must be greater than 1")
    if config.change_threshold < 0:
        raise ExperimentConfigError(f"{path}: change_threshold cannot be negative")

def load_experiment(
    path: str | Path,
    overrides: ExperimentOverrides | None = None,
) -> ResolvedExperiment:
    """Compose experiment, training, curriculum, and CLI configuration layers."""
    source = Path(path).resolve()
    raw = _load_mapping(source)
    _reject_unknown(raw, EXPERIMENT_KEYS, str(source))
    _require(raw, EXPERIMENT_KEYS, str(source))

    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ExperimentConfigError(f"name in {source} must be a non-empty string")
    if not isinstance(raw["save_dir"], str) or not raw["save_dir"].strip():
        raise ExperimentConfigError(f"save_dir in {source} must be a non-empty string")

    try:
        environment_type = EnvironmentType(raw["agent_strategy"])
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EnvironmentType)
        raise ExperimentConfigError(
            f"Invalid agent_strategy in {source}; expected one of: {allowed}"
        ) from exc

    overrides = overrides or ExperimentOverrides()
    training_path = (
        overrides.training_config.resolve()
        if overrides.training_config is not None
        else _resolve_reference(source, raw["training_config"], "training_config")
    )
    curriculum_path = _resolve_reference(source, raw["curriculum_config"], "curriculum_config")
    values = _load_training_values(training_path)
    values.update(environment_type=environment_type, save_dir=str(raw["save_dir"]))

    if overrides.episodes is not None:
        values["num_episodes"] = overrides.episodes
    if overrides.seed is not None:
        values["seed"] = overrides.seed
    if overrides.device is not None:
        values["device"] = overrides.device
    if overrides.save_dir is not None:
        values["save_dir"] = overrides.save_dir

    allowed_config_fields = {field.name for field in fields(TrainingConfig)}
    unexpected = set(values) - allowed_config_fields
    if unexpected:
        raise ExperimentConfigError(
            f"Training values do not map to TrainingConfig: {', '.join(sorted(unexpected))}"
        )
    training_config = TrainingConfig(**values)
    _validate_training(training_config, training_path)
    curriculum_config = load_curriculum_config(
        curriculum_path,
        environment_type,
        training_config.num_consumers,
        training_config.episode_length,
        training_config.verbose,
    )
    return ResolvedExperiment(
        name=str(raw["name"]),
        source=source,
        training_source=training_path,
        training_config=training_config,
        curriculum_config=curriculum_config,
    )


def build_environment(experiment: ResolvedExperiment):
    """Build the reusable factory and initial base/wrapped environments."""
    from env.factory import EnvironmentFactory
    from models.reward_normalizer import FixedRewardNormalizer

    config = experiment.training_config
    factory = EnvironmentFactory(config.environment_type, FixedRewardNormalizer)
    first_opponent = experiment.curriculum_config.stages[0].opponent_type
    base_env, env = factory.create_environment(first_opponent, config)
    return factory, base_env, env


def build_agent(experiment: ResolvedExperiment, env):
    """Build the configured replay buffer and SAC agent implementation."""
    from models.buffer import CurriculumReplayBuffer, CurriculumSequenceReplayBuffer

    config = experiment.training_config
    if config.agent_type == "recurrent_sac":
        from models.recurrent_sac_opponent_embedding import (
            RecurrentSACOpponentEmbeddingAgent,
        )

        replay_buffer = CurriculumSequenceReplayBuffer(
            capacity=config.episode_buffer_capacity,
            batch_size=config.batch_size,
            curriculum=experiment.curriculum_config.curriculum,
            sequence_length=config.sequence_length,
            current_stage_weight=config.current_stage_weight,
        )
        agent = RecurrentSACOpponentEmbeddingAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            opponent_action_dim=config.opponent_action_dim,
            opponent_embedding_dim=config.opponent_embedding_dim,
            encoder_hidden_dim=config.encoder_hidden_dim,
            actor_hidden_dim=config.actor_hidden_dim,
            critic_hidden_dim=config.critic_hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            lr_encoder=config.lr_encoder,
            lr_alpha=config.lr_alpha,
            gamma=config.gamma,
            tau=config.tau,
            alpha=config.alpha,
            auto_alpha=config.auto_alpha,
            target_entropy=config.target_entropy,
            opponent_aux_loss_weight=config.opponent_aux_loss_weight,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            grad_clip_norm=config.grad_clip_norm,
            device=config.device,
            replay_buffer=replay_buffer,
            min_episodes_before_update=config.min_episodes_before_update,
        )
        return replay_buffer, agent

    from models.sac import SAC

    replay_buffer = CurriculumReplayBuffer(
        capacity=config.buffer_size,
        batch_size=config.batch_size,
        curriculum=experiment.curriculum_config.curriculum,
    )
    agent = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        replay_buffer=replay_buffer,
        hidden_dim=config.hidden_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        lr_alpha=config.lr_alpha,
        gamma=config.gamma,
        tau=config.tau,
        alpha=config.alpha,
        auto_alpha=config.auto_alpha,
        target_entropy=config.target_entropy,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        batch_size=config.batch_size,
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        device=config.device,
    )
    return replay_buffer, agent
