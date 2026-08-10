"""Controller-specific deterministic replay for hierarchical pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from env.pricing_contracts import (
    AgentArchitecture,
    PricingActionCodec,
    PricingObservationCodec,
    PricingRegime,
)
from universal_pricing_v2.observations import StrategyObservationCodec
from universal_pricing_v2.protocol import PricingSkill


def _vector(
    value: Any,
    *,
    shape: tuple[int, ...],
    field_name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must be finite")
    if lower is not None and np.any(result < lower):
        raise ValueError(f"{field_name} is below {lower}")
    if upper is not None and np.any(result > upper):
        raise ValueError(f"{field_name} is above {upper}")
    return result.copy()


def _stage_fraction_diagnostics(
    stage_keys: Sequence[str],
) -> dict[str, float]:
    if not stage_keys:
        return {}
    unique, counts = np.unique(
        np.asarray(stage_keys, dtype=object), return_counts=True
    )
    total = float(len(stage_keys))
    return {
        f"stage_fraction__{str(stage_key)}": float(count / total)
        for stage_key, count in zip(unique, counts)
    }


@dataclass(frozen=True)
class PricingSkillTransition:
    """One micro transition for an independently optimized price skill."""

    pricing_skill: PricingSkill
    observation: np.ndarray
    price_action: np.ndarray
    effective_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    active_controller_mask: float
    opponent_price_controls: np.ndarray
    stage_key: str

    def __post_init__(self) -> None:
        skill = PricingSkill(self.pricing_skill)
        action_dimension = 1 if skill is PricingSkill.UNIFORM else 2
        observation = PricingObservationCodec.validate_vector(self.observation)
        next_observation = PricingObservationCodec.validate_vector(
            self.next_observation
        )
        price_action = _vector(
            self.price_action,
            shape=(action_dimension,),
            field_name="price_action",
            lower=-1.0,
            upper=1.0,
        )
        effective_action = _vector(
            self.effective_action,
            shape=(PricingActionCodec.REPLAY_VECTOR_LENGTH,),
            field_name="effective_action",
            lower=-1.0,
            upper=1.0,
        )
        if not (
            np.array_equal(effective_action[:2], [1.0, 0.0])
            or np.array_equal(effective_action[:2], [0.0, 1.0])
        ):
            raise ValueError("effective_action regime must be one-hot")
        opponent_controls = _vector(
            self.opponent_price_controls,
            shape=(PricingActionCodec.PRICE_CONTROL_COUNT,),
            field_name="opponent_price_controls",
            lower=-1.0,
            upper=1.0,
        )
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        mask = float(self.active_controller_mask)
        if mask not in (0.0, 1.0):
            raise ValueError("active_controller_mask must be 0 or 1")
        if not isinstance(self.done, (bool, np.bool_)):
            raise ValueError("done must be boolean")
        if not isinstance(self.stage_key, str) or not self.stage_key:
            raise ValueError("stage_key must be non-empty")
        expected_mask = (
            effective_action[0]
            if skill is PricingSkill.UNIFORM
            else effective_action[1]
        )
        if not np.isclose(mask, expected_mask):
            raise ValueError(
                "active_controller_mask disagrees with effective regime"
            )
        object.__setattr__(self, "pricing_skill", skill)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "price_action", price_action)
        object.__setattr__(self, "effective_action", effective_action)
        object.__setattr__(
            self, "opponent_price_controls", opponent_controls
        )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "done", bool(self.done))
        object.__setattr__(self, "active_controller_mask", mask)

    @classmethod
    def from_environment_step(
        cls,
        *,
        pricing_skill: PricingSkill,
        observation: Mapping[str, np.ndarray],
        reward: float,
        next_observation: Mapping[str, np.ndarray],
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        stage_key: str,
    ) -> "PricingSkillTransition":
        required = {"effective_replay_action", "opponent_price_controls"}
        missing = required - set(info)
        if missing:
            raise ValueError(
                "V2 environment info is missing: "
                + ", ".join(sorted(missing))
            )
        skill = PricingSkill(pricing_skill)
        effective = np.asarray(
            info["effective_replay_action"], dtype=np.float32
        )
        if skill is PricingSkill.UNIFORM:
            action = effective[2:3]
            mask = effective[0]
        else:
            action = effective[3:5]
            mask = effective[1]
        return cls(
            pricing_skill=skill,
            observation=observation["pricing"],
            price_action=action,
            effective_action=effective,
            reward=reward,
            next_observation=next_observation["pricing"],
            done=bool(terminated or truncated),
            active_controller_mask=float(mask),
            opponent_price_controls=info["opponent_price_controls"],
            stage_key=stage_key,
        )


@dataclass(frozen=True)
class StrategyTransition:
    """One semi-Markov transition spanning a commitment window."""

    observation: np.ndarray
    regime_action: int
    macro_reward: float
    next_observation: np.ndarray
    done: bool
    duration: int
    stage_key: str
    opponent_embedding: np.ndarray | None = None
    next_opponent_embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        observation = StrategyObservationCodec.validate_vector(self.observation)
        next_observation = StrategyObservationCodec.validate_vector(
            self.next_observation
        )
        if (
            not isinstance(self.regime_action, (int, np.integer))
            or isinstance(self.regime_action, (bool, np.bool_))
            or int(self.regime_action) not in (0, 1)
        ):
            raise ValueError("regime_action must be 0 or 1")
        reward = float(self.macro_reward)
        if not np.isfinite(reward):
            raise ValueError("macro_reward must be finite")
        if (
            not isinstance(self.duration, int)
            or isinstance(self.duration, bool)
            or not 1 <= self.duration <= 10
        ):
            raise ValueError("duration must be from 1 through 10")
        if not isinstance(self.done, (bool, np.bool_)):
            raise ValueError("done must be boolean")
        if not self.stage_key:
            raise ValueError("stage_key must be non-empty")
        embeddings: list[np.ndarray | None] = []
        for name in ("opponent_embedding", "next_opponent_embedding"):
            value = getattr(self, name)
            if value is None:
                embeddings.append(None)
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector")
            embeddings.append(array.copy())
        if (embeddings[0] is None) != (embeddings[1] is None):
            raise ValueError("Both strategy embeddings must be present or absent")
        if (
            embeddings[0] is not None
            and embeddings[0].shape != embeddings[1].shape
        ):
            raise ValueError("Strategy embedding dimensions must match")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "regime_action", int(self.regime_action))
        object.__setattr__(self, "macro_reward", reward)
        object.__setattr__(self, "done", bool(self.done))
        object.__setattr__(self, "opponent_embedding", embeddings[0])
        object.__setattr__(self, "next_opponent_embedding", embeddings[1])


@dataclass(frozen=True)
class PricingReplayBatch:
    observations: np.ndarray
    price_actions: np.ndarray
    effective_actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    active_controller_masks: np.ndarray
    opponent_price_controls: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class StrategyReplayBatch:
    observations: np.ndarray
    regime_actions: np.ndarray
    macro_rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    durations: np.ndarray
    opponent_embeddings: np.ndarray
    next_opponent_embeddings: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class FeedForwardPricingReplayBuffer:
    """Private-RNG micro replay with current/previous-stage sampling."""

    def __init__(
        self,
        *,
        pricing_skill: PricingSkill,
        capacity: int,
        replay_sampling_seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.pricing_skill = PricingSkill(pricing_skill)
        self.capacity = int(capacity)
        self.action_dimension = (
            1 if self.pricing_skill is PricingSkill.UNIFORM else 2
        )
        self._transitions: list[PricingSkillTransition] = []
        self._rng = np.random.default_rng(int(replay_sampling_seed))

    def __len__(self) -> int:
        return len(self._transitions)

    def push(self, transition: PricingSkillTransition) -> None:
        if (
            not isinstance(transition, PricingSkillTransition)
            or transition.pricing_skill is not self.pricing_skill
        ):
            raise TypeError("Transition has incompatible pricing skill")
        self._transitions.append(transition)
        if len(self._transitions) > self.capacity:
            del self._transitions[0]

    def _indices(
        self, batch_size: int, current_stage_key: str | None
    ) -> np.ndarray:
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {len(self)}"
            )
        all_indices = np.arange(len(self), dtype=np.int64)
        if current_stage_key is None:
            return self._rng.choice(
                all_indices, size=batch_size, replace=False
            )
        current = np.asarray(
            [
                index
                for index, transition in enumerate(self._transitions)
                if transition.stage_key == current_stage_key
            ],
            dtype=np.int64,
        )
        previous = np.setdiff1d(all_indices, current, assume_unique=True)
        current_count = min(len(current), (batch_size + 1) // 2)
        previous_count = min(len(previous), batch_size - current_count)
        remaining = batch_size - current_count - previous_count
        current_count += min(remaining, len(current) - current_count)
        remaining = batch_size - current_count - previous_count
        previous_count += min(remaining, len(previous) - previous_count)
        selected = np.concatenate(
            [
                self._rng.choice(
                    current, size=current_count, replace=False
                )
                if current_count
                else np.empty(0, dtype=np.int64),
                self._rng.choice(
                    previous, size=previous_count, replace=False
                )
                if previous_count
                else np.empty(0, dtype=np.int64),
            ]
        )
        self._rng.shuffle(selected)
        return selected

    def sample(
        self,
        batch_size: int,
        *,
        current_stage_key: str | None = None,
    ) -> PricingReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        transitions = [
            self._transitions[int(index)]
            for index in self._indices(batch_size, current_stage_key)
        ]
        return PricingReplayBatch(
            observations=np.stack([item.observation for item in transitions]),
            price_actions=np.stack([item.price_action for item in transitions]),
            effective_actions=np.stack(
                [item.effective_action for item in transitions]
            ),
            rewards=np.asarray(
                [[item.reward] for item in transitions], dtype=np.float32
            ),
            next_observations=np.stack(
                [item.next_observation for item in transitions]
            ),
            dones=np.asarray(
                [[float(item.done)] for item in transitions],
                dtype=np.float32,
            ),
            active_controller_masks=np.asarray(
                [[item.active_controller_mask] for item in transitions],
                dtype=np.float32,
            ),
            opponent_price_controls=np.stack(
                [item.opponent_price_controls for item in transitions]
            ),
        )

    def diagnostics(self) -> dict[str, float]:
        return {
            "transition_count": float(len(self)),
            "active_fraction": float(
                np.mean(
                    [
                        item.active_controller_mask
                        for item in self._transitions
                    ]
                )
            )
            if self._transitions
            else 0.0,
            "mean_reward": float(
                np.mean([item.reward for item in self._transitions])
            )
            if self._transitions
            else 0.0,
            "stage_count": float(
                len({item.stage_key for item in self._transitions})
            ),
            **_stage_fraction_diagnostics(
                [item.stage_key for item in self._transitions]
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "pricing_skill": self.pricing_skill.value,
            "capacity": self.capacity,
            "transitions": list(self._transitions),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if (
            values["pricing_skill"] != self.pricing_skill.value
            or int(values["capacity"]) != self.capacity
        ):
            raise ValueError("Incompatible pricing replay state")
        transitions = list(values["transitions"])
        if any(
            not isinstance(item, PricingSkillTransition)
            or item.pricing_skill is not self.pricing_skill
            for item in transitions
        ):
            raise ValueError("Pricing replay contains invalid transitions")
        self._transitions = transitions[-self.capacity :]
        self._rng.bit_generator.state = dict(values["rng_state"])


class StrategyReplayBuffer:
    """Feed-forward macro replay with optional OE boundary embeddings."""

    def __init__(
        self,
        *,
        capacity: int,
        replay_sampling_seed: int,
        opponent_embedding_dimension: int = 0,
    ) -> None:
        if capacity <= 0 or opponent_embedding_dimension < 0:
            raise ValueError("Invalid strategy replay capacity/dimension")
        self.capacity = int(capacity)
        self.opponent_embedding_dimension = int(
            opponent_embedding_dimension
        )
        self._transitions: list[StrategyTransition] = []
        self._rng = np.random.default_rng(int(replay_sampling_seed))

    def __len__(self) -> int:
        return len(self._transitions)

    def push(self, transition: StrategyTransition) -> None:
        if not isinstance(transition, StrategyTransition):
            raise TypeError("transition must be StrategyTransition")
        dimension = (
            0
            if transition.opponent_embedding is None
            else transition.opponent_embedding.size
        )
        if dimension != self.opponent_embedding_dimension:
            raise ValueError("Strategy transition embedding dimension mismatch")
        self._transitions.append(transition)
        if len(self._transitions) > self.capacity:
            del self._transitions[0]

    def sample(self, batch_size: int) -> StrategyReplayBatch:
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Invalid strategy replay batch size")
        indices = self._rng.choice(
            len(self), size=batch_size, replace=False
        )
        transitions = [self._transitions[int(index)] for index in indices]
        embedding_dimension = self.opponent_embedding_dimension
        return StrategyReplayBatch(
            observations=np.stack([item.observation for item in transitions]),
            regime_actions=np.asarray(
                [[item.regime_action] for item in transitions],
                dtype=np.int64,
            ),
            macro_rewards=np.asarray(
                [[item.macro_reward] for item in transitions],
                dtype=np.float32,
            ),
            next_observations=np.stack(
                [item.next_observation for item in transitions]
            ),
            dones=np.asarray(
                [[float(item.done)] for item in transitions],
                dtype=np.float32,
            ),
            durations=np.asarray(
                [[item.duration] for item in transitions],
                dtype=np.float32,
            ),
            opponent_embeddings=np.stack(
                [
                    item.opponent_embedding
                    if item.opponent_embedding is not None
                    else np.empty((embedding_dimension,), dtype=np.float32)
                    for item in transitions
                ]
            ),
            next_opponent_embeddings=np.stack(
                [
                    item.next_opponent_embedding
                    if item.next_opponent_embedding is not None
                    else np.empty((embedding_dimension,), dtype=np.float32)
                    for item in transitions
                ]
            ),
        )

    def diagnostics(self) -> dict[str, float]:
        return {
            "transition_count": float(len(self)),
            "mean_macro_reward": float(
                np.mean(
                    [transition.macro_reward for transition in self._transitions]
                )
            )
            if self._transitions
            else 0.0,
            "bbp_action_fraction": float(
                np.mean(
                    [transition.regime_action for transition in self._transitions]
                )
            )
            if self._transitions
            else 0.0,
            **_stage_fraction_diagnostics(
                [item.stage_key for item in self._transitions]
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "opponent_embedding_dimension": self.opponent_embedding_dimension,
            "transitions": list(self._transitions),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if (
            int(values["capacity"]) != self.capacity
            or int(values["opponent_embedding_dimension"])
            != self.opponent_embedding_dimension
        ):
            raise ValueError("Incompatible strategy replay state")
        transitions = list(values["transitions"])
        if any(not isinstance(item, StrategyTransition) for item in transitions):
            raise ValueError("Strategy replay contains invalid transitions")
        self._transitions = transitions[-self.capacity :]
        self._rng.bit_generator.state = dict(values["rng_state"])


@dataclass(frozen=True)
class PricingSkillEpisode:
    """One complete micro episode as seen by one price controller."""

    pricing_skill: PricingSkill
    transitions: tuple[PricingSkillTransition, ...]
    stage_key: str

    def __post_init__(self) -> None:
        skill = PricingSkill(self.pricing_skill)
        transitions = tuple(self.transitions)
        if not transitions or not transitions[-1].done:
            raise ValueError("Pricing episode must be non-empty and complete")
        if any(item.pricing_skill is not skill for item in transitions):
            raise ValueError("Pricing episode contains another skill")
        for previous, current in zip(transitions, transitions[1:]):
            if not np.array_equal(
                previous.next_observation, current.observation
            ):
                raise ValueError("Pricing episode must be contiguous")
            if previous.done:
                raise ValueError("Only final pricing transition may be done")
        object.__setattr__(self, "pricing_skill", skill)
        object.__setattr__(self, "transitions", transitions)


@dataclass(frozen=True)
class PricingSequenceBatch:
    observations: np.ndarray
    previous_price_actions: np.ndarray
    previous_rewards: np.ndarray
    previous_active_masks: np.ndarray
    previous_effective_actions: np.ndarray
    previous_opponent_price_controls: np.ndarray
    price_actions: np.ndarray
    effective_actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    active_controller_masks: np.ndarray
    opponent_price_controls: np.ndarray
    valid_masks: np.ndarray
    loss_masks: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class RecurrentPricingEpisodeReplay:
    """Sample burn-in plus learning windows without crossing episodes."""

    def __init__(
        self,
        *,
        pricing_skill: PricingSkill,
        capacity_episodes: int,
        learning_sequence_length: int,
        burn_in_length: int,
        replay_sampling_seed: int,
    ) -> None:
        for name, value in (
            ("capacity_episodes", capacity_episodes),
            ("learning_sequence_length", learning_sequence_length),
            ("burn_in_length", burn_in_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.pricing_skill = PricingSkill(pricing_skill)
        self.action_dimension = (
            1 if self.pricing_skill is PricingSkill.UNIFORM else 2
        )
        self.capacity_episodes = int(capacity_episodes)
        self.learning_sequence_length = int(learning_sequence_length)
        self.burn_in_length = int(burn_in_length)
        self.maximum_sequence_length = (
            self.learning_sequence_length + self.burn_in_length
        )
        self._episodes: list[PricingSkillEpisode] = []
        self._rng = np.random.default_rng(int(replay_sampling_seed))

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return sum(len(episode.transitions) for episode in self._episodes)

    def push_episode(self, episode: PricingSkillEpisode) -> None:
        if (
            not isinstance(episode, PricingSkillEpisode)
            or episode.pricing_skill is not self.pricing_skill
        ):
            raise TypeError("Incompatible pricing episode")
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity_episodes:
            del self._episodes[0]

    def sample(
        self,
        batch_size: int,
        *,
        current_stage_key: str | None = None,
        target_current_fraction : float = 0.7,
        min_current_pool : Optional[int] = None
    ) -> PricingSequenceBatch:
        if batch_size <= 0 or not self._episodes:
            raise ValueError("Cannot sample recurrent pricing replay")
        current = [
            index
            for index, episode in enumerate(self._episodes)
            if episode.stage_key == current_stage_key
        ]
        previous = [
            index
            for index, episode in enumerate(self._episodes)
            if episode.stage_key != current_stage_key
        ]
        if current_stage_key is None or not current:
            selected = self._rng.integers(
                0, len(self._episodes), size=batch_size
            )
        else:
            if min_current_pool is None:
                min_current_pool = max(1, round(batch_size * target_current_fraction))
            pool_ratio = min(1.0, len(current) / min_current_pool)
            effective_fraction = target_current_fraction * pool_ratio

            current_count = round(batch_size * effective_fraction)
            previous_count = batch_size - current_count
            selected = np.concatenate(
                [
                    self._rng.choice(
                        current,
                        size=current_count,
                        replace=len(current) < current_count,
                    ),
                    self._rng.choice(
                        previous if previous else current,
                        size=previous_count,
                        replace=True,
                    ),
                ]
            )
            self._rng.shuffle(selected)
        samples = [
            self._sample_episode(self._episodes[int(index)])
            for index in selected
        ]
        return PricingSequenceBatch(
            **{
                name: np.stack([sample[name] for sample in samples])
                for name in PricingSequenceBatch.__dataclass_fields__
            }
        )

    def _sample_episode(
        self, episode: PricingSkillEpisode
    ) -> dict[str, np.ndarray]:
        transitions = episode.transitions
        learning_start = int(self._rng.integers(0, len(transitions)))
        learning_end = min(
            learning_start + self.learning_sequence_length, len(transitions)
        )
        segment_start = max(0, learning_start - self.burn_in_length)
        selected = transitions[segment_start:learning_end]
        maximum = self.maximum_sequence_length
        valid_length = len(selected)
        learning_offset = learning_start - segment_start
        fields = {
            "observations": np.zeros((maximum, 18), np.float32),
            "previous_price_actions": np.zeros(
                (maximum, self.action_dimension), np.float32
            ),
            "previous_rewards": np.zeros((maximum, 1), np.float32),
            "previous_active_masks": np.zeros((maximum, 1), np.float32),
            "previous_effective_actions": np.zeros((maximum, 5), np.float32),
            "previous_opponent_price_controls": np.zeros(
                (maximum, 3), np.float32
            ),
            "price_actions": np.zeros(
                (maximum, self.action_dimension), np.float32
            ),
            "effective_actions": np.zeros((maximum, 5), np.float32),
            "rewards": np.zeros((maximum, 1), np.float32),
            "next_observations": np.zeros((maximum, 18), np.float32),
            "dones": np.zeros((maximum, 1), np.float32),
            "active_controller_masks": np.zeros((maximum, 1), np.float32),
            "opponent_price_controls": np.zeros((maximum, 3), np.float32),
            "valid_masks": np.zeros((maximum, 1), np.float32),
            "loss_masks": np.zeros((maximum, 1), np.float32),
        }
        for local, transition in enumerate(selected):
            global_index = segment_start + local
            previous_transition = (
                transitions[global_index - 1] if global_index > 0 else None
            )
            fields["observations"][local] = transition.observation
            fields["price_actions"][local] = transition.price_action
            fields["effective_actions"][local] = transition.effective_action
            fields["rewards"][local, 0] = transition.reward
            fields["next_observations"][local] = transition.next_observation
            fields["dones"][local, 0] = float(transition.done)
            fields["active_controller_masks"][
                local, 0
            ] = transition.active_controller_mask
            fields["opponent_price_controls"][
                local
            ] = transition.opponent_price_controls
            fields["valid_masks"][local, 0] = 1.0
            if previous_transition is not None:
                fields["previous_price_actions"][
                    local
                ] = previous_transition.price_action
                fields["previous_rewards"][
                    local, 0
                ] = previous_transition.reward
                fields["previous_active_masks"][
                    local, 0
                ] = previous_transition.active_controller_mask
                fields["previous_effective_actions"][
                    local
                ] = previous_transition.effective_action
                fields["previous_opponent_price_controls"][
                    local
                ] = previous_transition.opponent_price_controls
        fields["loss_masks"][learning_offset:valid_length, 0] = 1.0
        return fields

    def diagnostics(self) -> dict[str, float]:
        transitions = [
            item
            for episode in self._episodes
            for item in episode.transitions
        ]
        return {
            "episode_count": float(len(self)),
            "transition_count": float(len(transitions)),
            "active_fraction": float(
                np.mean([item.active_controller_mask for item in transitions])
            )
            if transitions
            else 0.0,
            "stage_count": float(
                len({episode.stage_key for episode in self._episodes})
            ),
            **_stage_fraction_diagnostics(
                [episode.stage_key for episode in self._episodes]
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "pricing_skill": self.pricing_skill.value,
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
            "episodes": list(self._episodes),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        expected = {
            "pricing_skill": self.pricing_skill.value,
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
        }
        if any(values.get(name) != value for name, value in expected.items()):
            raise ValueError("Incompatible recurrent pricing replay state")
        episodes = list(values["episodes"])
        if any(
            not isinstance(item, PricingSkillEpisode)
            or item.pricing_skill is not self.pricing_skill
            for item in episodes
        ):
            raise ValueError("Invalid recurrent pricing episodes")
        self._episodes = episodes[-self.capacity_episodes :]
        self._rng.bit_generator.state = dict(values["rng_state"])


@dataclass(frozen=True)
class StrategyEpisode:
    transitions: tuple[StrategyTransition, ...]

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions or not transitions[-1].done:
            raise ValueError("Strategy episode must be complete")
        if any(item.done for item in transitions[:-1]):
            raise ValueError("Only final strategy transition may be done")
        object.__setattr__(self, "transitions", transitions)


@dataclass(frozen=True)
class StrategySequenceBatch:
    observations: np.ndarray
    previous_regime_one_hot: np.ndarray
    previous_macro_rewards: np.ndarray
    regime_actions: np.ndarray
    macro_rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    durations: np.ndarray
    opponent_embeddings: np.ndarray
    next_opponent_embeddings: np.ndarray
    valid_masks: np.ndarray
    loss_masks: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class RecurrentStrategyEpisodeReplay:
    """Macro sequence replay: two burn-in plus eight learning decisions."""

    def __init__(
        self,
        *,
        capacity_episodes: int,
        learning_sequence_length: int,
        burn_in_length: int,
        replay_sampling_seed: int,
        opponent_embedding_dimension: int = 0,
    ) -> None:
        for value in (
            capacity_episodes,
            learning_sequence_length,
            burn_in_length,
        ):
            if value <= 0:
                raise ValueError("Strategy sequence values must be positive")
        if opponent_embedding_dimension < 0:
            raise ValueError("opponent_embedding_dimension cannot be negative")
        self.capacity_episodes = int(capacity_episodes)
        self.learning_sequence_length = int(learning_sequence_length)
        self.burn_in_length = int(burn_in_length)
        self.maximum_sequence_length = (
            self.learning_sequence_length + self.burn_in_length
        )
        self.opponent_embedding_dimension = int(
            opponent_embedding_dimension
        )
        self._episodes: list[StrategyEpisode] = []
        self._rng = np.random.default_rng(int(replay_sampling_seed))

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return sum(len(item.transitions) for item in self._episodes)

    def push_episode(self, episode: StrategyEpisode) -> None:
        if not isinstance(episode, StrategyEpisode):
            raise TypeError("episode must be StrategyEpisode")
        for transition in episode.transitions:
            dimension = (
                0
                if transition.opponent_embedding is None
                else transition.opponent_embedding.size
            )
            if dimension != self.opponent_embedding_dimension:
                raise ValueError("Strategy episode embedding dimension mismatch")
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity_episodes:
            del self._episodes[0]

    def sample(self, batch_size: int) -> StrategySequenceBatch:
        if batch_size <= 0 or not self._episodes:
            raise ValueError("Cannot sample strategy sequence replay")
        indices = self._rng.integers(
            0, len(self._episodes), size=batch_size
        )
        samples = [
            self._sample_episode(self._episodes[int(index)])
            for index in indices
        ]
        return StrategySequenceBatch(
            **{
                name: np.stack([sample[name] for sample in samples])
                for name in StrategySequenceBatch.__dataclass_fields__
            }
        )

    def _sample_episode(
        self, episode: StrategyEpisode
    ) -> dict[str, np.ndarray]:
        transitions = episode.transitions
        learning_start = int(self._rng.integers(0, len(transitions)))
        learning_end = min(
            learning_start + self.learning_sequence_length,
            len(transitions),
        )
        start = max(0, learning_start - self.burn_in_length)
        selected = transitions[start:learning_end]
        maximum = self.maximum_sequence_length
        embedding = self.opponent_embedding_dimension
        fields = {
            "observations": np.zeros((maximum, 19), np.float32),
            "previous_regime_one_hot": np.zeros((maximum, 2), np.float32),
            "previous_macro_rewards": np.zeros((maximum, 1), np.float32),
            "regime_actions": np.zeros((maximum, 1), np.int64),
            "macro_rewards": np.zeros((maximum, 1), np.float32),
            "next_observations": np.zeros((maximum, 19), np.float32),
            "dones": np.zeros((maximum, 1), np.float32),
            "durations": np.ones((maximum, 1), np.float32),
            "opponent_embeddings": np.zeros(
                (maximum, embedding), np.float32
            ),
            "next_opponent_embeddings": np.zeros(
                (maximum, embedding), np.float32
            ),
            "valid_masks": np.zeros((maximum, 1), np.float32),
            "loss_masks": np.zeros((maximum, 1), np.float32),
        }
        for local, transition in enumerate(selected):
            global_index = start + local
            previous = transitions[global_index - 1] if global_index else None
            fields["observations"][local] = transition.observation
            fields["regime_actions"][local, 0] = transition.regime_action
            fields["macro_rewards"][local, 0] = transition.macro_reward
            fields["next_observations"][local] = transition.next_observation
            fields["dones"][local, 0] = float(transition.done)
            fields["durations"][local, 0] = transition.duration
            if embedding:
                fields["opponent_embeddings"][
                    local
                ] = transition.opponent_embedding
                fields["next_opponent_embeddings"][
                    local
                ] = transition.next_opponent_embedding
            if previous is not None:
                fields["previous_regime_one_hot"][
                    local, previous.regime_action
                ] = 1.0
                fields["previous_macro_rewards"][
                    local, 0
                ] = previous.macro_reward
            fields["valid_masks"][local, 0] = 1.0
        learning_offset = learning_start - start
        fields["loss_masks"][learning_offset : len(selected), 0] = 1.0
        return fields

    def diagnostics(self) -> dict[str, float]:
        transitions = [
            item
            for episode in self._episodes
            for item in episode.transitions
        ]
        return {
            "episode_count": float(len(self)),
            "transition_count": float(len(transitions)),
            "mean_macro_reward": float(
                np.mean([item.macro_reward for item in transitions])
            )
            if transitions
            else 0.0,
            "bbp_action_fraction": float(
                np.mean([item.regime_action for item in transitions])
            )
            if transitions
            else 0.0,
            **_stage_fraction_diagnostics(
                [item.stage_key for item in transitions]
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
            "opponent_embedding_dimension": self.opponent_embedding_dimension,
            "episodes": list(self._episodes),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        expected = {
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
            "opponent_embedding_dimension": self.opponent_embedding_dimension,
        }
        if any(values.get(name) != value for name, value in expected.items()):
            raise ValueError("Incompatible recurrent strategy replay state")
        episodes = list(values["episodes"])
        if any(not isinstance(item, StrategyEpisode) for item in episodes):
            raise ValueError("Invalid recurrent strategy episodes")
        self._episodes = episodes[-self.capacity_episodes :]
        self._rng.bit_generator.state = dict(values["rng_state"])


class CurriculumReplayRepository:
    """Own all three replay stores behind one checkpointable interface."""

    def __init__(
        self,
        *,
        architecture: AgentArchitecture,
        pricing_capacity: int,
        strategy_capacity: int,
        episode_capacity: int | None,
        price_sequence_length: int | None,
        price_burn_in_length: int | None,
        strategy_sequence_length: int | None,
        strategy_burn_in_length: int | None,
        uniform_seed: int,
        bbp_seed: int,
        strategy_seed: int,
        opponent_embedding_dimension: int = 0,
    ) -> None:
        self.architecture = AgentArchitecture(architecture)
        recurrent = self.architecture is not AgentArchitecture.SAC
        if recurrent:
            required = (
                episode_capacity,
                price_sequence_length,
                price_burn_in_length,
                strategy_sequence_length,
                strategy_burn_in_length,
            )
            if any(value is None for value in required):
                raise ValueError("Recurrent replay settings are required")
            self.uniform_pricing = RecurrentPricingEpisodeReplay(
                pricing_skill=PricingSkill.UNIFORM,
                capacity_episodes=int(episode_capacity),
                learning_sequence_length=int(price_sequence_length),
                burn_in_length=int(price_burn_in_length),
                replay_sampling_seed=uniform_seed,
            )
            self.bbp_pricing = RecurrentPricingEpisodeReplay(
                pricing_skill=PricingSkill.BBP,
                capacity_episodes=int(episode_capacity),
                learning_sequence_length=int(price_sequence_length),
                burn_in_length=int(price_burn_in_length),
                replay_sampling_seed=bbp_seed,
            )
            self.strategy = RecurrentStrategyEpisodeReplay(
                capacity_episodes=int(episode_capacity),
                learning_sequence_length=int(strategy_sequence_length),
                burn_in_length=int(strategy_burn_in_length),
                replay_sampling_seed=strategy_seed,
                opponent_embedding_dimension=opponent_embedding_dimension,
            )
        else:
            self.uniform_pricing = FeedForwardPricingReplayBuffer(
                pricing_skill=PricingSkill.UNIFORM,
                capacity=pricing_capacity,
                replay_sampling_seed=uniform_seed,
            )
            self.bbp_pricing = FeedForwardPricingReplayBuffer(
                pricing_skill=PricingSkill.BBP,
                capacity=pricing_capacity,
                replay_sampling_seed=bbp_seed,
            )
            self.strategy = StrategyReplayBuffer(
                capacity=strategy_capacity,
                replay_sampling_seed=strategy_seed,
                opponent_embedding_dimension=opponent_embedding_dimension,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "uniform_pricing": self.uniform_pricing.state_dict(),
            "bbp_pricing": self.bbp_pricing.state_dict(),
            "strategy": self.strategy.state_dict(),
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if values["architecture"] != self.architecture.value:
            raise ValueError("Replay architecture mismatch")
        self.uniform_pricing.load_state_dict(values["uniform_pricing"])
        self.bbp_pricing.load_state_dict(values["bbp_pricing"])
        self.strategy.load_state_dict(values["strategy"])

    def diagnostics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for prefix, replay in (
            ("uniform_replay", self.uniform_pricing),
            ("bbp_replay", self.bbp_pricing),
            ("strategy_replay", self.strategy),
        ):
            result.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in replay.diagnostics().items()
                }
            )
        return result
