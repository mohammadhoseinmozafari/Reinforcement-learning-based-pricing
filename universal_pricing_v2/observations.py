"""Macro strategy observation and ten-period evidence window."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from env.models import HotellingMarket
from env.pricing_contracts import PricingRegime
from train.universal_pricing_protocol import ProtocolConfigError


class StrategyObservationFeature(str, Enum):
    """Frozen 19-feature macro-controller order."""

    CURRENT_AGENT_REGIME = "current_agent_regime"
    OPPONENT_REGIME = "opponent_regime"
    CURRENT_OWN_MARKET_SHARE = "current_own_market_share"
    CURRENT_OPPONENT_MARKET_SHARE = "current_opponent_market_share"
    CURRENT_OWN_ACTIVE_PRICE_LEVEL = "current_own_active_price_level"
    CURRENT_OPPONENT_ACTIVE_PRICE_LEVEL = "current_opponent_active_price_level"
    WINDOW_MEAN_OWN_MARKET_SHARE = "window_mean_own_market_share"
    WINDOW_MARKET_SHARE_CHANGE = "window_market_share_change"
    WINDOW_MEAN_DEMAND_RATIO = "window_mean_demand_ratio"
    WINDOW_MEAN_NEW_CUSTOMER_RATIO = "window_mean_new_customer_ratio"
    WINDOW_MEAN_RETENTION = "window_mean_retention"
    WINDOW_MEAN_NORMALIZED_NET_PROFIT = "window_mean_normalized_net_profit"
    WINDOW_MEAN_NORMALIZED_NET_PROFIT_ADVANTAGE = (
        "window_mean_normalized_net_profit_advantage"
    )
    WINDOW_NET_PROFIT_TREND = "window_net_profit_trend"
    WINDOW_POPULARITY_CHANGE = "window_popularity_change"
    WINDOW_MEAN_OWN_ACTIVE_PRICE = "window_mean_own_active_price"
    WINDOW_MEAN_OPPONENT_ACTIVE_PRICE = "window_mean_opponent_active_price"
    EPISODE_PROGRESS = "episode_progress"
    WINDOW_HISTORY_AVAILABLE = "window_history_available"

    @property
    def index(self) -> int:
        return tuple(type(self)).index(self)


class StrategyObservationCodec:
    """Encode, decode, and validate finite normalized strategy features."""

    FEATURE_ORDER = tuple(StrategyObservationFeature)
    FEATURE_NAMES = tuple(item.value for item in FEATURE_ORDER)
    FEATURE_COUNT = len(FEATURE_ORDER)

    @classmethod
    def encode(
        cls,
        features: Mapping[str | StrategyObservationFeature, float],
    ) -> np.ndarray:
        if not isinstance(features, Mapping):
            raise TypeError("Strategy features must be a mapping")
        normalized: dict[str, float] = {}
        for key, value in features.items():
            name = key.value if isinstance(key, StrategyObservationFeature) else str(key)
            if name in normalized:
                raise ValueError(f"Duplicate strategy feature: {name}")
            normalized[name] = float(value)
        missing = set(cls.FEATURE_NAMES) - set(normalized)
        extra = set(normalized) - set(cls.FEATURE_NAMES)
        if missing or extra:
            raise ValueError(
                f"Strategy feature mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return cls.validate_vector([normalized[name] for name in cls.FEATURE_NAMES])

    @classmethod
    def validate_vector(cls, values: Any) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (cls.FEATURE_COUNT,):
            raise ValueError(
                f"Strategy observation must have shape ({cls.FEATURE_COUNT},)"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Strategy observation must be finite")
        if np.any(vector < -1.0) or np.any(vector > 1.0):
            raise ValueError("Strategy observation must be in [-1, 1]")
        return vector.astype(np.float32, copy=True)

    @classmethod
    def decode(cls, values: Any) -> dict[str, float]:
        vector = cls.validate_vector(values)
        return {
            name: float(vector[index])
            for index, name in enumerate(cls.FEATURE_NAMES)
        }


@dataclass(frozen=True)
class StrategyPeriodRecord:
    """Observable period evidence retained for the macro boundary."""

    own_market_share: float
    demand_ratio: float
    new_customer_ratio: float
    retention_rate: float
    normalized_net_profit: float
    normalized_net_profit_advantage: float
    own_active_price: float
    opponent_active_price: float


class StrategyObservationWindow:
    """Accumulate the current commitment window without global state."""

    def __init__(self, commitment_length: int = 10) -> None:
        if (
            not isinstance(commitment_length, int)
            or isinstance(commitment_length, bool)
            or commitment_length <= 0
        ):
            raise ProtocolConfigError(
                "commitment_length must be a positive integer"
            )
        self.commitment_length = commitment_length
        self._records: list[StrategyPeriodRecord] = []

    def reset(self) -> None:
        self._records = []

    def record(self, record: StrategyPeriodRecord) -> None:
        if not isinstance(record, StrategyPeriodRecord):
            raise TypeError("record must be StrategyPeriodRecord")
        values = [float(getattr(record, name)) for name in record.__dataclass_fields__]
        if not np.all(np.isfinite(values)):
            raise ValueError("Strategy period record must be finite")
        self._records.append(record)
        if len(self._records) > self.commitment_length:
            del self._records[0]

    @property
    def has_history(self) -> bool:
        return bool(self._records)

    @property
    def records(self) -> tuple[StrategyPeriodRecord, ...]:
        return tuple(self._records)

    def state_dict(self) -> dict[str, Any]:
        return {
            "commitment_length": self.commitment_length,
            "records": [
                {
                    name: float(getattr(record, name))
                    for name in record.__dataclass_fields__
                }
                for record in self._records
            ],
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if int(values["commitment_length"]) != self.commitment_length:
            raise ValueError("Strategy window commitment length mismatch")
        self._records = [
            StrategyPeriodRecord(**record) for record in values["records"]
        ]


class StrategyObservationBuilder:
    """Build the macro state from current market facts and a recent window."""

    def __init__(
        self,
        *,
        num_consumers: int,
        episode_length: int,
        maximum_price: float = 5.0,
        minimum_price: float = 0.5,
    ) -> None:
        if num_consumers <= 0 or episode_length <= 0:
            raise ProtocolConfigError(
                "num_consumers and episode_length must be positive"
            )
        if minimum_price >= maximum_price:
            raise ProtocolConfigError("Invalid strategy price bounds")
        self.num_consumers = int(num_consumers)
        self.episode_length = int(episode_length)
        self.maximum_price = float(maximum_price)
        self.minimum_price = float(minimum_price)

    @staticmethod
    def _ratio(value: float) -> float:
        return float(2.0 * np.clip(value, 0.0, 1.0) - 1.0)

    @staticmethod
    def _regime(value: int | PricingRegime) -> float:
        return -1.0 if PricingRegime(value) is PricingRegime.UNIFORM else 1.0

    def normalize_price(self, value: float) -> float:
        fraction = (float(value) - self.minimum_price) / (
            self.maximum_price - self.minimum_price
        )
        return self._ratio(fraction)

    @staticmethod
    def active_price(firm: Any) -> float:
        if PricingRegime(firm.pricing_regime) is PricingRegime.UNIFORM:
            return float(firm.uniform_price)
        return float((firm.price_new + firm.price_old) / 2.0)

    @staticmethod
    def _mean(records: tuple[StrategyPeriodRecord, ...], name: str) -> float:
        return (
            float(np.mean([getattr(record, name) for record in records]))
            if records
            else 0.0
        )

    @staticmethod
    def _change(
        records: tuple[StrategyPeriodRecord, ...],
        name: str,
    ) -> float:
        if len(records) < 2:
            return 0.0
        return float(
            np.clip(
                getattr(records[-1], name) - getattr(records[0], name),
                -1.0,
                1.0,
            )
        )

    def build(
        self,
        market: HotellingMarket,
        timestep: int,
        window: StrategyObservationWindow,
    ) -> np.ndarray:
        agent, opponent = market.firms
        own_share = 0.5 if timestep == 0 else float(agent.market_share)
        opponent_share = 0.5 if timestep == 0 else float(opponent.market_share)
        records = window.records
        own_active = self.active_price(agent)
        opponent_active = self.active_price(opponent)
        features = {
            StrategyObservationFeature.CURRENT_AGENT_REGIME: self._regime(
                agent.pricing_regime
            ),
            StrategyObservationFeature.OPPONENT_REGIME: self._regime(
                opponent.pricing_regime
            ),
            StrategyObservationFeature.CURRENT_OWN_MARKET_SHARE: self._ratio(
                own_share
            ),
            StrategyObservationFeature.CURRENT_OPPONENT_MARKET_SHARE: self._ratio(
                opponent_share
            ),
            StrategyObservationFeature.CURRENT_OWN_ACTIVE_PRICE_LEVEL: (
                self.normalize_price(own_active)
            ),
            StrategyObservationFeature.CURRENT_OPPONENT_ACTIVE_PRICE_LEVEL: (
                self.normalize_price(opponent_active)
            ),
            StrategyObservationFeature.WINDOW_MEAN_OWN_MARKET_SHARE: self._ratio(
                self._mean(records, "own_market_share") if records else 0.5
            ),
            StrategyObservationFeature.WINDOW_MARKET_SHARE_CHANGE: self._change(
                records, "own_market_share"
            ),
            StrategyObservationFeature.WINDOW_MEAN_DEMAND_RATIO: self._ratio(
                self._mean(records, "demand_ratio") if records else 0.0
            ),
            StrategyObservationFeature.WINDOW_MEAN_NEW_CUSTOMER_RATIO: self._ratio(
                self._mean(records, "new_customer_ratio") if records else 0.0
            ),
            StrategyObservationFeature.WINDOW_MEAN_RETENTION: self._ratio(
                self._mean(records, "retention_rate") if records else 0.0
            ),
            StrategyObservationFeature.WINDOW_MEAN_NORMALIZED_NET_PROFIT: float(
                np.clip(
                    self._mean(records, "normalized_net_profit"),
                    -1.0,
                    1.0,
                )
            ),
            StrategyObservationFeature.WINDOW_MEAN_NORMALIZED_NET_PROFIT_ADVANTAGE: float(
                np.clip(
                    self._mean(records, "normalized_net_profit_advantage"),
                    -1.0,
                    1.0,
                )
            ),
            StrategyObservationFeature.WINDOW_NET_PROFIT_TREND: self._change(
                records, "normalized_net_profit"
            ),
            StrategyObservationFeature.WINDOW_POPULARITY_CHANGE: self._change(
                records, "own_market_share"
            ),
            StrategyObservationFeature.WINDOW_MEAN_OWN_ACTIVE_PRICE: (
                self.normalize_price(
                    self._mean(records, "own_active_price")
                    if records
                    else own_active
                )
            ),
            StrategyObservationFeature.WINDOW_MEAN_OPPONENT_ACTIVE_PRICE: (
                self.normalize_price(
                    self._mean(records, "opponent_active_price")
                    if records
                    else opponent_active
                )
            ),
            StrategyObservationFeature.EPISODE_PROGRESS: self._ratio(
                min(max(timestep / self.episode_length, 0.0), 1.0)
            ),
            StrategyObservationFeature.WINDOW_HISTORY_AVAILABLE: (
                1.0 if window.has_history else -1.0
            ),
        }
        return StrategyObservationCodec.encode(features)
