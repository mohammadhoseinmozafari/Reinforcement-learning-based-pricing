"""V2-only BBP operating-cost and net-profit accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from env.pricing_contracts import PricingRegime
from train.universal_pricing_protocol import ProtocolConfigError


@dataclass(frozen=True)
class BBPOperatingCostConfig:
    """Symmetric fixed cost charged for each completed BBP market period."""

    capacity_rate: float = 0.01
    maximum_price: float = 5.0
    marginal_cost: float = 0.0

    def __post_init__(self) -> None:
        values = (
            float(self.capacity_rate),
            float(self.maximum_price),
            float(self.marginal_cost),
        )
        if not all(np.isfinite(value) for value in values):
            raise ProtocolConfigError("BBP cost parameters must be finite")
        if values[0] < 0.0 or values[0] > 1.0:
            raise ProtocolConfigError("capacity_rate must be in [0, 1]")
        if values[1] <= values[2]:
            raise ProtocolConfigError(
                "maximum_price must exceed marginal_cost"
            )
        object.__setattr__(self, "capacity_rate", values[0])
        object.__setattr__(self, "maximum_price", values[1])
        object.__setattr__(self, "marginal_cost", values[2])

    def capacity(self, num_consumers: int) -> float:
        if (
            not isinstance(num_consumers, int)
            or isinstance(num_consumers, bool)
            or num_consumers <= 0
        ):
            raise ProtocolConfigError(
                "num_consumers must be a positive integer"
            )
        return float(
            num_consumers * (self.maximum_price - self.marginal_cost)
        )

    def cost_for(
        self,
        regime: PricingRegime | int,
        num_consumers: int,
    ) -> float:
        return (
            self.capacity_rate * self.capacity(num_consumers)
            if PricingRegime(regime) is PricingRegime.BBP
            else 0.0
        )


@dataclass(frozen=True)
class PeriodProfitAccounting:
    """Gross, cost, and net facts for one market period."""

    gross_agent_profit: float
    agent_bbp_operating_cost: float
    net_agent_profit: float
    gross_opponent_profit: float
    opponent_bbp_operating_cost: float
    net_opponent_profit: float
    net_profit_advantage: float
    normalized_net_reward: float

    def to_info(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


class BBPProfitAccounting:
    """Apply v2 accounting after the unchanged pure market has cleared."""

    def __init__(
        self,
        num_consumers: int,
        cost_config: BBPOperatingCostConfig | None = None,
    ) -> None:
        self.num_consumers = num_consumers
        self.cost_config = cost_config or BBPOperatingCostConfig()
        self.capacity = self.cost_config.capacity(num_consumers)

    def calculate(
        self,
        *,
        gross_agent_profit: float,
        agent_regime: PricingRegime | int,
        gross_opponent_profit: float,
        opponent_regime: PricingRegime | int,
    ) -> PeriodProfitAccounting:
        gross_agent = float(gross_agent_profit)
        gross_opponent = float(gross_opponent_profit)
        if not np.isfinite(gross_agent) or not np.isfinite(gross_opponent):
            raise ValueError("Gross profits must be finite")
        agent_cost = self.cost_config.cost_for(
            agent_regime, self.num_consumers
        )
        opponent_cost = self.cost_config.cost_for(
            opponent_regime, self.num_consumers
        )
        net_agent = gross_agent - agent_cost
        net_opponent = gross_opponent - opponent_cost
        return PeriodProfitAccounting(
            gross_agent_profit=gross_agent,
            agent_bbp_operating_cost=agent_cost,
            net_agent_profit=net_agent,
            gross_opponent_profit=gross_opponent,
            opponent_bbp_operating_cost=opponent_cost,
            net_opponent_profit=net_opponent,
            net_profit_advantage=net_agent - net_opponent,
            normalized_net_reward=net_agent / self.capacity,
        )
