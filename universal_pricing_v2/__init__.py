"""Hierarchical universal-pricing research protocol.

The package is intentionally additive.  It depends on stable, pure contracts
from ``universal_pricing_v1`` but does not alter the legacy or v1 runtimes.
"""

from universal_pricing_v2.protocol import (
    ACTION_CONTRACT_VERSION,
    MARKET_TIMING,
    OBSERVATION_CONTRACT_VERSION,
    PROTOCOL_VERSION,
    AgentRegimeMode,
    HierarchicalTrainingPhase,
    UniversalPricingV2ProtocolConfig,
    V2ExperimentCoordinate,
    V2ExperimentMatrix,
    load_universal_pricing_v2_protocol,
)

__all__ = [
    "ACTION_CONTRACT_VERSION",
    "MARKET_TIMING",
    "OBSERVATION_CONTRACT_VERSION",
    "PROTOCOL_VERSION",
    "AgentRegimeMode",
    "HierarchicalTrainingPhase",
    "UniversalPricingV2ProtocolConfig",
    "V2ExperimentCoordinate",
    "V2ExperimentMatrix",
    "load_universal_pricing_v2_protocol",
]
