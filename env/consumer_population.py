"""Reproducible consumer-population generation for universal pricing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import skewnorm, truncnorm

from env.models.market import Consumer
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    ConsumerDistributionSpec,
    ConsumerPopulationSpec,
    ProtocolConfigError,
)


class ConsumerAttribute(str, Enum):
    """Consumer attributes with stable independent random-stream IDs."""

    LOCATION = "location"
    STRATEGICNESS = "strategicness"
    EXCLUSIVITY = "exclusivity"

    @property
    def stream_id(self) -> int:
        return {
            ConsumerAttribute.LOCATION: 1,
            ConsumerAttribute.STRATEGICNESS: 2,
            ConsumerAttribute.EXCLUSIVITY: 3,
        }[self]


@runtime_checkable
class ConsumerAttributeSampler(Protocol):
    """Sampling interface for one bounded consumer attribute."""

    family: ConsumerDistributionFamily

    def sample(
        self,
        distribution_spec: ConsumerDistributionSpec,
        sample_count: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        ...


def _validate_sampling_request(
    distribution_spec: ConsumerDistributionSpec,
    expected_family: ConsumerDistributionFamily,
    sample_count: int,
    random_generator: np.random.Generator,
) -> None:
    if distribution_spec.family is not expected_family:
        raise ProtocolConfigError(
            f"{expected_family.value} sampler received "
            f"{distribution_spec.family.value} specification"
        )
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ProtocolConfigError("sample_count must be a positive integer")
    if not isinstance(random_generator, np.random.Generator):
        raise TypeError("random_generator must be numpy.random.Generator")


def _bounded_inverse_probabilities(
    random_generator: np.random.Generator,
    sample_count: int,
    lower_cdf: float,
    upper_cdf: float,
) -> np.ndarray:
    if (
        not np.isfinite(lower_cdf)
        or not np.isfinite(upper_cdf)
        or not 0.0 <= lower_cdf < upper_cdf <= 1.0
    ):
        raise ProtocolConfigError("Distribution has invalid truncation mass")
    probabilities = lower_cdf + (
        upper_cdf - lower_cdf
    ) * random_generator.random(sample_count)
    return np.clip(
        probabilities,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )


class UniformConsumerAttributeSampler:
    """Sample a bounded uniform consumer attribute."""

    family = ConsumerDistributionFamily.UNIFORM

    def sample(
        self,
        distribution_spec: ConsumerDistributionSpec,
        sample_count: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        _validate_sampling_request(
            distribution_spec,
            self.family,
            sample_count,
            random_generator,
        )
        parameters = distribution_spec.parameters
        return random_generator.uniform(
            parameters["low"],
            parameters["high"],
            size=sample_count,
        ).astype(np.float64, copy=False)


class TruncatedNormalConsumerAttributeSampler:
    """Sample a true bounded normal distribution through its inverse CDF."""

    family = ConsumerDistributionFamily.TRUNCATED_NORMAL

    def sample(
        self,
        distribution_spec: ConsumerDistributionSpec,
        sample_count: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        _validate_sampling_request(
            distribution_spec,
            self.family,
            sample_count,
            random_generator,
        )
        parameters = distribution_spec.parameters
        mean = parameters["mean"]
        standard_deviation = parameters["standard_deviation"]
        standardized_low = (parameters["low"] - mean) / standard_deviation
        standardized_high = (parameters["high"] - mean) / standard_deviation
        probabilities = _bounded_inverse_probabilities(
            random_generator,
            sample_count,
            0.0,
            1.0,
        )
        samples = truncnorm.ppf(
            probabilities,
            standardized_low,
            standardized_high,
            loc=mean,
            scale=standard_deviation,
        )
        return np.clip(
            samples,
            parameters["low"],
            parameters["high"],
        ).astype(np.float64, copy=False)


class TruncatedSkewNormalConsumerAttributeSampler:
    """Sample a bounded skew-normal distribution through its inverse CDF."""

    family = ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL

    def sample(
        self,
        distribution_spec: ConsumerDistributionSpec,
        sample_count: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        _validate_sampling_request(
            distribution_spec,
            self.family,
            sample_count,
            random_generator,
        )
        parameters = distribution_spec.parameters
        distribution = skewnorm(
            parameters["shape"],
            loc=parameters["location"],
            scale=parameters["scale"],
        )
        probabilities = _bounded_inverse_probabilities(
            random_generator,
            sample_count,
            float(distribution.cdf(parameters["low"])),
            float(distribution.cdf(parameters["high"])),
        )
        samples = distribution.ppf(probabilities)
        return np.clip(
            samples,
            parameters["low"],
            parameters["high"],
        ).astype(np.float64, copy=False)


class TruncatedSkewNormalMomentCalibrator:
    """Reconstruct skew-normal location and scale from truncated moments."""

    @staticmethod
    def truncated_moments(
        *,
        location: float,
        scale: float,
        shape: float,
        low: float,
        high: float,
    ) -> tuple[float, float]:
        if scale <= 0 or low >= high:
            raise ProtocolConfigError("Invalid skew-normal calibration bounds")
        distribution = skewnorm(shape, loc=location, scale=scale)
        truncation_mass = float(
            distribution.cdf(high) - distribution.cdf(low)
        )
        if not np.isfinite(truncation_mass) or truncation_mass <= 0:
            raise ProtocolConfigError(
                "Skew-normal calibration has no finite truncation mass"
            )
        mean = float(
            distribution.expect(
                lambda value: value,
                lb=low,
                ub=high,
                conditional=True,
            )
        )
        second_moment = float(
            distribution.expect(
                lambda value: value * value,
                lb=low,
                ub=high,
                conditional=True,
            )
        )
        variance = max(0.0, second_moment - mean * mean)
        return mean, float(np.sqrt(variance))

    @classmethod
    def calibrate(
        cls,
        *,
        target_mean: float,
        target_standard_deviation: float,
        shape: float,
        low: float,
        high: float,
        initial_location: float,
        initial_scale: float,
    ) -> tuple[float, float]:
        if (
            not low < target_mean < high
            or target_standard_deviation <= 0
            or initial_scale <= 0
        ):
            raise ProtocolConfigError("Invalid skew-normal moment targets")

        def residuals(parameters: np.ndarray) -> np.ndarray:
            location = float(parameters[0])
            scale = float(np.exp(parameters[1]))
            mean, standard_deviation = cls.truncated_moments(
                location=location,
                scale=scale,
                shape=shape,
                low=low,
                high=high,
            )
            return np.asarray(
                [
                    mean - target_mean,
                    standard_deviation - target_standard_deviation,
                ],
                dtype=np.float64,
            )

        result = least_squares(
            residuals,
            x0=np.asarray(
                [initial_location, np.log(initial_scale)],
                dtype=np.float64,
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=200,
        )
        location = float(result.x[0])
        scale = float(np.exp(result.x[1]))
        if not result.success or np.max(np.abs(result.fun)) > 1e-8:
            raise ProtocolConfigError(
                "Could not calibrate truncated skew-normal moments"
            )
        return location, scale


class ConsumerAttributeSamplerRegistry:
    """Validated family-to-sampler registry."""

    def __init__(
        self,
        samplers: Mapping[
            ConsumerDistributionFamily,
            ConsumerAttributeSampler,
        ] | None = None,
    ) -> None:
        if samplers is None:
            samplers = {
                ConsumerDistributionFamily.UNIFORM: (
                    UniformConsumerAttributeSampler()
                ),
                ConsumerDistributionFamily.TRUNCATED_NORMAL: (
                    TruncatedNormalConsumerAttributeSampler()
                ),
                ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL: (
                    TruncatedSkewNormalConsumerAttributeSampler()
                ),
            }
        resolved = {
            ConsumerDistributionFamily(family): sampler
            for family, sampler in samplers.items()
        }
        if set(resolved) != set(ConsumerDistributionFamily):
            raise ProtocolConfigError(
                "Consumer sampler registry must define all distribution families"
            )
        for family, sampler in resolved.items():
            if not isinstance(sampler, ConsumerAttributeSampler):
                raise TypeError(
                    f"Sampler for {family.value} does not satisfy the protocol"
                )
            if sampler.family is not family:
                raise ProtocolConfigError(
                    f"Sampler registration mismatch for {family.value}"
                )
        self._samplers = MappingProxyType(resolved)

    def sampler_for(
        self,
        family: ConsumerDistributionFamily,
    ) -> ConsumerAttributeSampler:
        return self._samplers[ConsumerDistributionFamily(family)]


@dataclass(frozen=True, eq=False)
class ConsumerPopulationSnapshot:
    """Immutable sampled attributes for one episode population."""

    locations: np.ndarray
    strategicness: np.ndarray
    exclusivity: np.ndarray

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for field_name in ("locations", "strategicness", "exclusivity"):
            values = np.asarray(
                getattr(self, field_name),
                dtype=np.float64,
            ).copy()
            if values.ndim != 1 or values.size == 0:
                raise ProtocolConfigError(
                    f"{field_name} must be a non-empty one-dimensional array"
                )
            if not np.all(np.isfinite(values)):
                raise ProtocolConfigError(f"{field_name} must be finite")
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ProtocolConfigError(f"{field_name} must be in [0, 1]")
            values.setflags(write=False)
            arrays[field_name] = values
        sizes = {values.size for values in arrays.values()}
        if len(sizes) != 1:
            raise ProtocolConfigError(
                "Consumer population attribute lengths must match"
            )
        for field_name, values in arrays.items():
            object.__setattr__(self, field_name, values)

    @property
    def population_size(self) -> int:
        return int(self.locations.size)

    def to_consumers(self) -> list[Consumer]:
        return [
            Consumer(
                consumer_id=index,
                location=float(self.locations[index]),
                exclusivity_preference=float(self.exclusivity[index]),
                strategic_foresight=float(self.strategicness[index]),
            )
            for index in range(self.population_size)
        ]


class ConsumerPopulationGenerator:
    """Generate independently streamed consumer attributes for one episode."""

    def __init__(
        self,
        sampler_registry: ConsumerAttributeSamplerRegistry | None = None,
    ) -> None:
        self.sampler_registry = (
            sampler_registry or ConsumerAttributeSamplerRegistry()
        )

    def generate(
        self,
        population_spec: ConsumerPopulationSpec,
        population_size: int,
        consumer_seed: int,
    ) -> ConsumerPopulationSnapshot:
        if (
            not isinstance(population_size, int)
            or isinstance(population_size, bool)
            or population_size <= 0
        ):
            raise ProtocolConfigError(
                "population_size must be a positive integer"
            )
        if (
            not isinstance(consumer_seed, (int, np.integer))
            or isinstance(consumer_seed, (bool, np.bool_))
            or int(consumer_seed) < 0
        ):
            raise ProtocolConfigError(
                "consumer_seed must be a nonnegative integer"
            )

        samples: dict[ConsumerAttribute, np.ndarray] = {}
        for attribute in ConsumerAttribute:
            distribution_spec = getattr(population_spec, attribute.value)
            random_generator = np.random.default_rng(
                np.random.SeedSequence(
                    [int(consumer_seed), attribute.stream_id]
                )
            )
            samples[attribute] = self.sampler_registry.sampler_for(
                distribution_spec.family
            ).sample(
                distribution_spec,
                population_size,
                random_generator,
            )
        return ConsumerPopulationSnapshot(
            locations=samples[ConsumerAttribute.LOCATION],
            strategicness=samples[ConsumerAttribute.STRATEGICNESS],
            exclusivity=samples[ConsumerAttribute.EXCLUSIVITY],
        )
