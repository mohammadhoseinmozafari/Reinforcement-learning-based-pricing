"""Replay contracts for the feed-forward universal-pricing agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from env.pricing_contracts import PricingActionCodec, PricingObservationCodec


def _finite_float32_vector(
    value: Any,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain only finite values")
    return vector.copy()


@dataclass(frozen=True)
class UniversalPricingTransition:
    """One complete transition under the frozen universal replay contract."""

    observation: np.ndarray
    effective_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    regime_decision_mask: float
    opponent_price_controls: np.ndarray

    def __post_init__(self) -> None:
        observation = PricingObservationCodec.validate_vector(self.observation)
        next_observation = PricingObservationCodec.validate_vector(
            self.next_observation
        )
        effective_action = _finite_float32_vector(
            self.effective_action,
            shape=(PricingActionCodec.REPLAY_VECTOR_LENGTH,),
            field_name="effective_action",
        )
        if not (
            np.array_equal(effective_action[:2], [1.0, 0.0])
            or np.array_equal(effective_action[:2], [0.0, 1.0])
        ):
            raise ValueError(
                "effective_action must contain a valid regime one-hot pair"
            )
        if np.any(effective_action[2:] < -1.0) or np.any(
            effective_action[2:] > 1.0
        ):
            raise ValueError(
                "effective_action price controls must be in [-1, 1]"
            )
        opponent_controls = _finite_float32_vector(
            self.opponent_price_controls,
            shape=(PricingActionCodec.PRICE_CONTROL_COUNT,),
            field_name="opponent_price_controls",
        )
        if np.any(opponent_controls < -1.0) or np.any(opponent_controls > 1.0):
            raise ValueError(
                "opponent_price_controls must be in [-1, 1]"
            )
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        decision_mask = float(self.regime_decision_mask)
        if decision_mask not in (0.0, 1.0):
            raise ValueError("regime_decision_mask must be 0 or 1")
        if not isinstance(self.done, (bool, np.bool_)):
            raise ValueError("done must be boolean")

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "effective_action", effective_action)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "done", bool(self.done))
        object.__setattr__(self, "regime_decision_mask", decision_mask)
        object.__setattr__(
            self,
            "opponent_price_controls",
            opponent_controls,
        )

    @classmethod
    def from_environment_step(
        cls,
        *,
        observation: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> "UniversalPricingTransition":
        """Build a transition directly from ``UniversalPricingEnv.step``."""

        required = {
            "effective_replay_action",
            "regime_decision_mask",
            "opponent_price_controls",
        }
        missing = required - set(info)
        if missing:
            raise ValueError(
                "Environment info is missing replay fields: "
                + ", ".join(sorted(missing))
            )
        return cls(
            observation=observation,
            effective_action=info["effective_replay_action"],
            reward=reward,
            next_observation=next_observation,
            done=bool(terminated or truncated),
            regime_decision_mask=info["regime_decision_mask"],
            opponent_price_controls=info["opponent_price_controls"],
        )


@dataclass(frozen=True)
class UniversalPricingReplayBatch:
    """Validated, consistently shaped batch sampled from universal replay."""

    observations: np.ndarray
    effective_actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    regime_decision_masks: np.ndarray
    opponent_price_controls: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "effective_actions": self.effective_actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "regime_decision_masks": self.regime_decision_masks,
            "opponent_price_controls": self.opponent_price_controls,
        }


class UniversalPricingReplayBuffer:
    """Fixed-capacity replay using a private, committed NumPy RNG stream."""

    def __init__(self, capacity: int, replay_sampling_seed: int) -> None:
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._rng = np.random.default_rng(int(replay_sampling_seed))
        self._observations = np.empty(
            (capacity, PricingObservationCodec.FEATURE_COUNT),
            dtype=np.float32,
        )
        self._actions = np.empty(
            (capacity, PricingActionCodec.REPLAY_VECTOR_LENGTH),
            dtype=np.float32,
        )
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty_like(self._observations)
        self._dones = np.empty((capacity, 1), dtype=np.float32)
        self._decision_masks = np.empty((capacity, 1), dtype=np.float32)
        self._opponent_controls = np.empty(
            (capacity, PricingActionCodec.PRICE_CONTROL_COUNT),
            dtype=np.float32,
        )
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        return self._size

    def push(self, transition: UniversalPricingTransition) -> None:
        if not isinstance(transition, UniversalPricingTransition):
            raise TypeError("transition must be a UniversalPricingTransition")
        index = self._next_index
        self._observations[index] = transition.observation
        self._actions[index] = transition.effective_action
        self._rewards[index, 0] = transition.reward
        self._next_observations[index] = transition.next_observation
        self._dones[index, 0] = float(transition.done)
        self._decision_masks[index, 0] = transition.regime_decision_mask
        self._opponent_controls[index] = transition.opponent_price_controls
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> UniversalPricingReplayBatch:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {self._size}"
            )
        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=False,
        )
        return UniversalPricingReplayBatch(
            observations=self._observations[indices].copy(),
            effective_actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_observations=self._next_observations[indices].copy(),
            dones=self._dones[indices].copy(),
            regime_decision_masks=self._decision_masks[indices].copy(),
            opponent_price_controls=self._opponent_controls[indices].copy(),
        )

    def rng_state(self) -> dict[str, Any]:
        """Return a copy of the sampling state for reproducible checkpoints."""

        return dict(self._rng.bit_generator.state)

    def set_rng_state(self, state: Mapping[str, Any]) -> None:
        self._rng.bit_generator.state = dict(state)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "next_index": self._next_index,
            "observations": self._observations[: self._size].copy(),
            "actions": self._actions[: self._size].copy(),
            "rewards": self._rewards[: self._size].copy(),
            "next_observations": self._next_observations[: self._size].copy(),
            "dones": self._dones[: self._size].copy(),
            "decision_masks": self._decision_masks[: self._size].copy(),
            "opponent_controls": self._opponent_controls[: self._size].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("capacity") != self.capacity:
            raise ValueError("Incompatible replay capacity")
        size = int(state["size"])
        if size < 0 or size > self.capacity:
            raise ValueError("Invalid replay size")
        mappings = (
            ("observations", self._observations),
            ("actions", self._actions),
            ("rewards", self._rewards),
            ("next_observations", self._next_observations),
            ("dones", self._dones),
            ("decision_masks", self._decision_masks),
            ("opponent_controls", self._opponent_controls),
        )
        for name, target in mappings:
            values = np.asarray(state[name], dtype=np.float32)
            if values.shape != target[:size].shape:
                raise ValueError(f"Incompatible replay state field: {name}")
            target[:size] = values
        self._size = size
        self._next_index = int(state["next_index"])
        self._rng.bit_generator.state = dict(state["rng_state"])
