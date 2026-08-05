"""Public action, observation, and agent contracts for universal pricing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


class PricingRegime(IntEnum):
    """Pricing regimes exposed by the universal pricing environment."""

    UNIFORM = 0
    BBP = 1


@dataclass(frozen=True)
class PricingAction:
    """Structured regime selection and normalized price controls."""

    regime: PricingRegime
    uniform_control: float
    bbp_new_control: float
    bbp_premium_control: float

    def __post_init__(self) -> None:
        raw_regime = self.regime
        if isinstance(raw_regime, (bool, np.bool_)) or not isinstance(
            raw_regime,
            (int, np.integer, PricingRegime),
        ):
            raise ValueError("Pricing regime must be integer 0 or 1")
        try:
            regime = PricingRegime(int(raw_regime))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown pricing regime: {raw_regime!r}") from exc
        object.__setattr__(self, "regime", regime)

        for field_name in (
            "uniform_control",
            "bbp_new_control",
            "bbp_premium_control",
        ):
            raw_value = getattr(self, field_name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError(f"{field_name} must be numeric")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [-1, 1]")
            object.__setattr__(self, field_name, value)


class PricingActionCodec:
    """Convert pricing actions between structured, Gym, and replay forms."""

    GYM_REGIME_KEY = "regime"
    GYM_PRICE_CONTROLS_KEY = "price_controls"
    PRICE_CONTROL_COUNT = 3
    REPLAY_VECTOR_LENGTH = 5
    REPLAY_REGIME_DECISION_MASK_FIELD = "regime_decision_mask"
    REPLAY_OPPONENT_PRICE_CONTROLS_FIELD = "opponent_price_controls"

    @classmethod
    def to_gym(cls, action: PricingAction) -> dict[str, Any]:
        action = cls._validated_action(action)
        return {
            cls.GYM_REGIME_KEY: int(action.regime),
            cls.GYM_PRICE_CONTROLS_KEY: np.asarray(
                [
                    action.uniform_control,
                    action.bbp_new_control,
                    action.bbp_premium_control,
                ],
                dtype=np.float32,
            ),
        }

    @classmethod
    def from_gym(cls, gym_action: Mapping[str, Any]) -> PricingAction:
        if not isinstance(gym_action, Mapping):
            raise TypeError("Gym pricing action must be a mapping")
        if set(gym_action) != {
            cls.GYM_REGIME_KEY,
            cls.GYM_PRICE_CONTROLS_KEY,
        }:
            raise ValueError(
                "Gym pricing action must contain exactly 'regime' and "
                "'price_controls'"
            )

        controls = np.asarray(
            gym_action[cls.GYM_PRICE_CONTROLS_KEY],
            dtype=np.float64,
        )
        if controls.shape != (cls.PRICE_CONTROL_COUNT,):
            raise ValueError("price_controls must have shape (3,)")
        if not np.all(np.isfinite(controls)):
            raise ValueError("price_controls must contain only finite values")

        return PricingAction(
            regime=cls._parse_regime(gym_action[cls.GYM_REGIME_KEY]),
            uniform_control=float(controls[0]),
            bbp_new_control=float(controls[1]),
            bbp_premium_control=float(controls[2]),
        )

    @classmethod
    def to_replay_vector(cls, action: PricingAction) -> np.ndarray:
        action = cls._validated_action(action)
        regime_one_hot = (
            (1.0, 0.0)
            if action.regime is PricingRegime.UNIFORM
            else (0.0, 1.0)
        )
        return np.asarray(
            [
                *regime_one_hot,
                action.uniform_control,
                action.bbp_new_control,
                action.bbp_premium_control,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_replay_vector(cls, replay_vector: Any) -> PricingAction:
        values = np.asarray(replay_vector, dtype=np.float64)
        if values.shape != (cls.REPLAY_VECTOR_LENGTH,):
            raise ValueError("Replay action vector must have shape (5,)")
        if not np.all(np.isfinite(values)):
            raise ValueError("Replay action vector must contain only finite values")
        if not (
            np.array_equal(values[:2], np.asarray([1.0, 0.0]))
            or np.array_equal(values[:2], np.asarray([0.0, 1.0]))
        ):
            raise ValueError("Replay regime fields must be a valid one-hot pair")

        regime = (
            PricingRegime.UNIFORM
            if values[0] == 1.0
            else PricingRegime.BBP
        )
        return PricingAction(
            regime=regime,
            uniform_control=float(values[2]),
            bbp_new_control=float(values[3]),
            bbp_premium_control=float(values[4]),
        )

    @staticmethod
    def _validated_action(action: PricingAction) -> PricingAction:
        if not isinstance(action, PricingAction):
            raise TypeError("Expected a PricingAction")
        return PricingAction(
            regime=action.regime,
            uniform_control=action.uniform_control,
            bbp_new_control=action.bbp_new_control,
            bbp_premium_control=action.bbp_premium_control,
        )

    @staticmethod
    def _parse_regime(value: Any) -> PricingRegime:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Pricing regime must be integer 0 or 1")
        if not isinstance(value, (int, np.integer)):
            raise ValueError("Pricing regime must be integer 0 or 1")
        try:
            return PricingRegime(int(value))
        except ValueError as exc:
            raise ValueError("Pricing regime must be integer 0 or 1") from exc


class PricingObservationFeature(str, Enum):
    """Stable indices for the universal 18-feature observation."""

    OWN_MARKET_SHARE = "own_market_share"
    OPPONENT_MARKET_SHARE = "opponent_market_share"
    OWN_UNIFORM_PRICE = "own_uniform_price"
    OWN_BBP_NEW_PRICE = "own_bbp_new_price"
    OWN_BBP_OLD_PRICE = "own_bbp_old_price"
    OPPONENT_UNIFORM_PRICE = "opponent_uniform_price"
    OPPONENT_BBP_NEW_PRICE = "opponent_bbp_new_price"
    OPPONENT_BBP_OLD_PRICE = "opponent_bbp_old_price"
    OWN_DEMAND_RATIO = "own_demand_ratio"
    OWN_NEW_CUSTOMER_RATIO = "own_new_customer_ratio"
    OWN_RETENTION_RATE = "own_retention_rate"
    OWN_REGIME = "own_regime"
    OPPONENT_REGIME = "opponent_regime"
    OWN_PROFIT_TREND = "own_profit_trend"
    OWN_POPULARITY_CHANGE = "own_popularity_change"
    EPISODE_PROGRESS = "episode_progress"
    REGIME_COMMITMENT_PROGRESS = "regime_commitment_progress"
    REGIME_DECISION_ALLOWED = "regime_decision_allowed"

    @property
    def field_name(self) -> str:
        return self.value

    @property
    def index(self) -> int:
        return tuple(type(self)).index(self)


class PricingObservationCodec:
    """Encode named normalized market features in the frozen feature order."""

    FEATURE_COUNT = 18
    FEATURE_ORDER = tuple(PricingObservationFeature)
    FEATURE_NAMES = tuple(feature.field_name for feature in FEATURE_ORDER)

    @classmethod
    def encode(
        cls,
        market_features: Mapping[str | PricingObservationFeature, float],
    ) -> np.ndarray:
        if not isinstance(market_features, Mapping):
            raise TypeError("Market features must be a mapping")

        normalized_features: dict[str, float] = {}
        for key, value in market_features.items():
            field_name = key.field_name if isinstance(
                key, PricingObservationFeature
            ) else str(key)
            if field_name in normalized_features:
                raise ValueError(f"Duplicate observation feature: {field_name}")
            normalized_features[field_name] = value

        missing = set(cls.FEATURE_NAMES) - set(normalized_features)
        extra = set(normalized_features) - set(cls.FEATURE_NAMES)
        if missing or extra:
            raise ValueError(
                f"Observation features mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        return cls.validate_vector(
            [normalized_features[name] for name in cls.FEATURE_NAMES]
        )

    @classmethod
    def decode(cls, observation_vector: Any) -> dict[str, float]:
        vector = cls.validate_vector(observation_vector)
        return {
            name: float(vector[index])
            for index, name in enumerate(cls.FEATURE_NAMES)
        }

    @classmethod
    def validate_vector(cls, observation_vector: Any) -> np.ndarray:
        vector = np.asarray(observation_vector, dtype=np.float64)
        if vector.shape != (cls.FEATURE_COUNT,):
            raise ValueError("Pricing observation must have shape (18,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Pricing observation must contain only finite values")
        if np.any(vector < -1.0) or np.any(vector > 1.0):
            raise ValueError("Pricing observation features must be in [-1, 1]")
        return vector.astype(np.float32, copy=True)


INITIAL_PRICING_REGIME = PricingRegime.UNIFORM
INITIAL_REGIME_DECISION_ALLOWED = True
REGIME_SWITCHING_COST = 0.0
REQUIRED_PRICING_STEP_INFO_FIELDS = (
    "raw_agent_profit",
    "raw_opponent_profit",
    "profit_advantage",
    "normalized_reward",
    "agent_regime",
    "opponent_regime",
    "regime_changed",
    "regime_decision_allowed",
    "opponent_policy_name",
)


@runtime_checkable
class PricingAgent(Protocol):
    """Runtime interface implemented by every universal pricing agent."""

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> PricingAction:
        ...

    def update(self, replay_batch: Mapping[str, Any]) -> Mapping[str, float]:
        ...

    def reset_recurrent_state(self) -> None:
        ...

    def save(self, checkpoint_path: str | Path) -> None:
        ...

    def load(self, checkpoint_path: str | Path) -> None:
        ...

    def policy_diagnostics(self) -> Mapping[str, float]:
        ...


class AgentArchitecture(str, Enum):
    """Stable architecture identifiers used by configurations and run IDs."""

    SAC = "sac"
    RSAC = "rsac"
    OE_RSAC = "oe_rsac"


@dataclass(frozen=True)
class AgentArchitectureSpec:
    """Declarative capabilities of a future agent implementation."""

    architecture: AgentArchitecture
    implementation_class_name: str
    is_recurrent: bool
    uses_sequence_replay: bool
    uses_opponent_encoder: bool


AGENT_ARCHITECTURE_SPECS: Mapping[
    AgentArchitecture, AgentArchitectureSpec
] = {
    AgentArchitecture.SAC: AgentArchitectureSpec(
        architecture=AgentArchitecture.SAC,
        implementation_class_name="SACPricingAgent",
        is_recurrent=False,
        uses_sequence_replay=False,
        uses_opponent_encoder=False,
    ),
    AgentArchitecture.RSAC: AgentArchitectureSpec(
        architecture=AgentArchitecture.RSAC,
        implementation_class_name="RecurrentSACPricingAgent",
        is_recurrent=True,
        uses_sequence_replay=True,
        uses_opponent_encoder=False,
    ),
    AgentArchitecture.OE_RSAC: AgentArchitectureSpec(
        architecture=AgentArchitecture.OE_RSAC,
        implementation_class_name=(
            "OpponentEmbeddingRecurrentSACPricingAgent"
        ),
        is_recurrent=True,
        uses_sequence_replay=True,
        uses_opponent_encoder=True,
    ),
}
