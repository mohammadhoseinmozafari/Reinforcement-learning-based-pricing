"""Public environment API with lazy imports for optional runtime dependencies."""

from env.type import EnvironmentType

__all__ = [
    "EnvironmentType",
    "PricingEnv",
    "UniversalPricingEnv",
    "UniversalPricingEnvironmentFactory",
    "make_pricing_env",
]


def __getattr__(name):
    if name in {"PricingEnv", "make_pricing_env"}:
        from env.pricing_env import PricingEnv, make_pricing_env
        return {"PricingEnv": PricingEnv, "make_pricing_env": make_pricing_env}[name]
    if name == "UniversalPricingEnv":
        from env.universal_pricing_env import UniversalPricingEnv
        return UniversalPricingEnv
    if name == "UniversalPricingEnvironmentFactory":
        from env.universal_pricing_factory import (
            UniversalPricingEnvironmentFactory,
        )
        return UniversalPricingEnvironmentFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
