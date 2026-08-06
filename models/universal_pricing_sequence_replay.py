"""Deterministic episodic replay for universal recurrent pricing agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from models.universal_pricing_replay import UniversalPricingTransition


@dataclass(frozen=True)
class UniversalPricingEpisode:
    """One complete, contiguous universal-pricing episode."""

    transitions: tuple[UniversalPricingTransition, ...]
    episode_index: int
    consumer_seed: int
    opponent_seed: int
    opponent_family: str
    opponent_policy_name: str

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError("An episode must contain at least one transition")
        for index, transition in enumerate(transitions):
            if not isinstance(transition, UniversalPricingTransition):
                raise TypeError("Episode transitions must be universal transitions")
            if index and not np.array_equal(
                transitions[index - 1].next_observation,
                transition.observation,
            ):
                raise ValueError("Episode transitions must be contiguous")
            if index < len(transitions) - 1 and transition.done:
                raise ValueError("Only the final episode transition may be done")
        if not transitions[-1].done:
            raise ValueError("The final episode transition must be done")
        if not isinstance(self.episode_index, int) or self.episode_index < 0:
            raise ValueError("episode_index must be nonnegative")
        if self.opponent_family not in {"uniform", "bbp"}:
            raise ValueError("opponent_family must be uniform or bbp")
        if not self.opponent_policy_name:
            raise ValueError("opponent_policy_name must be non-empty")
        object.__setattr__(self, "transitions", transitions)


class UniversalPricingEpisodeBuilder:
    """Accumulate transitions and seal them into one validated episode."""

    def __init__(
        self,
        *,
        episode_index: int,
        consumer_seed: int,
        opponent_seed: int,
        opponent_family: str,
        opponent_policy_name: str,
    ) -> None:
        self._metadata = {
            "episode_index": episode_index,
            "consumer_seed": consumer_seed,
            "opponent_seed": opponent_seed,
            "opponent_family": opponent_family,
            "opponent_policy_name": opponent_policy_name,
        }
        self._transitions: list[UniversalPricingTransition] = []

    def append(self, transition: UniversalPricingTransition) -> None:
        if self._transitions and self._transitions[-1].done:
            raise ValueError("Cannot append after the episode is complete")
        self._transitions.append(transition)

    def __len__(self) -> int:
        return len(self._transitions)

    def build(self) -> UniversalPricingEpisode:
        return UniversalPricingEpisode(
            transitions=tuple(self._transitions),
            **self._metadata,
        )


@dataclass(frozen=True)
class UniversalPricingSequenceBatch:
    """Padded burn-in and learning sequences with explicit temporal context."""

    observations: np.ndarray
    previous_effective_actions: np.ndarray
    previous_rewards: np.ndarray
    previous_opponent_price_controls: np.ndarray
    effective_actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    regime_decision_masks: np.ndarray
    opponent_price_controls: np.ndarray
    valid_masks: np.ndarray
    loss_masks: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


class UniversalPricingSequenceReplayBuffer:
    """Episode replay with private RNG and reproducible burn-in windows."""

    def __init__(
        self,
        *,
        capacity_episodes: int,
        learning_sequence_length: int,
        burn_in_length: int,
        batch_size: int,
        replay_sampling_seed: int,
    ) -> None:
        for name, value in (
            ("capacity_episodes", capacity_episodes),
            ("learning_sequence_length", learning_sequence_length),
            ("burn_in_length", burn_in_length),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.capacity_episodes = capacity_episodes
        self.learning_sequence_length = learning_sequence_length
        self.burn_in_length = burn_in_length
        self.batch_size = batch_size
        self.maximum_sequence_length = learning_sequence_length + burn_in_length
        self._episodes: list[UniversalPricingEpisode] = []
        self._rng = np.random.default_rng(int(replay_sampling_seed))

    def __len__(self) -> int:
        return len(self._episodes)

    def push_episode(self, episode: UniversalPricingEpisode) -> None:
        if not isinstance(episode, UniversalPricingEpisode):
            raise TypeError("episode must be UniversalPricingEpisode")
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity_episodes:
            del self._episodes[0]

    def sample(self, batch_size: int | None = None) -> UniversalPricingSequenceBatch:
        if not self._episodes:
            raise ValueError("Cannot sample from empty sequence replay")
        size = self.batch_size if batch_size is None else batch_size
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("batch_size must be a positive integer")
        episode_indices = self._rng.integers(0, len(self._episodes), size=size)
        samples = [
            self._sample_episode(self._episodes[int(index)])
            for index in episode_indices
        ]
        return UniversalPricingSequenceBatch(
            **{
                field_name: np.stack(
                    [sample[field_name] for sample in samples],
                    axis=0,
                )
                for field_name in UniversalPricingSequenceBatch.__dataclass_fields__
            }
        )

    def _sample_episode(self, episode: UniversalPricingEpisode) -> dict[str, np.ndarray]:
        transitions = episode.transitions
        learning_start = int(self._rng.integers(0, len(transitions)))
        learning_end = min(
            learning_start + self.learning_sequence_length,
            len(transitions),
        )
        segment_start = max(0, learning_start - self.burn_in_length)
        selected = transitions[segment_start:learning_end]
        valid_length = len(selected)
        learning_offset = learning_start - segment_start
        maximum = self.maximum_sequence_length

        fields = {
            "observations": np.zeros((maximum, 18), dtype=np.float32),
            "previous_effective_actions": np.zeros((maximum, 5), dtype=np.float32),
            "previous_rewards": np.zeros((maximum, 1), dtype=np.float32),
            "previous_opponent_price_controls": np.zeros(
                (maximum, 3), dtype=np.float32
            ),
            "effective_actions": np.zeros((maximum, 5), dtype=np.float32),
            "rewards": np.zeros((maximum, 1), dtype=np.float32),
            "next_observations": np.zeros((maximum, 18), dtype=np.float32),
            "dones": np.zeros((maximum, 1), dtype=np.float32),
            "regime_decision_masks": np.zeros((maximum, 1), dtype=np.float32),
            "opponent_price_controls": np.zeros((maximum, 3), dtype=np.float32),
            "valid_masks": np.zeros((maximum, 1), dtype=np.float32),
            "loss_masks": np.zeros((maximum, 1), dtype=np.float32),
        }
        for local_index, transition in enumerate(selected):
            global_index = segment_start + local_index
            previous = transitions[global_index - 1] if global_index > 0 else None
            fields["observations"][local_index] = transition.observation
            fields["effective_actions"][local_index] = transition.effective_action
            fields["rewards"][local_index, 0] = transition.reward
            fields["next_observations"][local_index] = transition.next_observation
            fields["dones"][local_index, 0] = float(transition.done)
            fields["regime_decision_masks"][
                local_index, 0
            ] = transition.regime_decision_mask
            fields["opponent_price_controls"][
                local_index
            ] = transition.opponent_price_controls
            if previous is not None:
                fields["previous_effective_actions"][
                    local_index
                ] = previous.effective_action
                fields["previous_rewards"][local_index, 0] = previous.reward
                fields["previous_opponent_price_controls"][
                    local_index
                ] = previous.opponent_price_controls
            fields["valid_masks"][local_index, 0] = 1.0
        fields["loss_masks"][
            learning_offset:valid_length, 0
        ] = 1.0
        return fields

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
            "batch_size": self.batch_size,
            "episodes": list(self._episodes),
            "rng_state": self._rng.bit_generator.state,
        }

    def diagnostics(self) -> dict[str, float]:
        """Return episode and regime composition without sampling replay."""

        transitions = [
            transition
            for episode in self._episodes
            for transition in episode.transitions
        ]
        episode_count = len(self._episodes)
        if not transitions:
            return {
                "replay_episode_count": float(episode_count),
                "replay_transition_count": 0.0,
                "replay_bbp_fraction": 0.0,
                "replay_decision_fraction": 0.0,
                "replay_mean_reward": 0.0,
                "replay_bbp_opponent_episode_fraction": 0.0,
            }
        return {
            "replay_episode_count": float(episode_count),
            "replay_transition_count": float(len(transitions)),
            "replay_bbp_fraction": float(
                np.mean(
                    [
                        transition.effective_action[1]
                        for transition in transitions
                    ]
                )
            ),
            "replay_decision_fraction": float(
                np.mean(
                    [
                        transition.regime_decision_mask
                        for transition in transitions
                    ]
                )
            ),
            "replay_mean_reward": float(
                np.mean([transition.reward for transition in transitions])
            ),
            "replay_bbp_opponent_episode_fraction": float(
                np.mean(
                    [
                        episode.opponent_family == "bbp"
                        for episode in self._episodes
                    ]
                )
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "capacity_episodes": self.capacity_episodes,
            "learning_sequence_length": self.learning_sequence_length,
            "burn_in_length": self.burn_in_length,
            "batch_size": self.batch_size,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"Incompatible sequence replay {name}")
        episodes = list(state["episodes"])
        if any(not isinstance(item, UniversalPricingEpisode) for item in episodes):
            raise ValueError("Sequence replay state contains invalid episodes")
        self._episodes = episodes[-self.capacity_episodes :]
        self._rng.bit_generator.state = dict(state["rng_state"])
