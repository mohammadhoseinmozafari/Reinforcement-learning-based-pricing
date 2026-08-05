"""Tests for reproducible universal consumer populations."""

import math
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from env.consumer_population import (
    ConsumerAttributeSamplerRegistry,
    ConsumerPopulationGenerator,
    ConsumerPopulationSnapshot,
    TruncatedSkewNormalMomentCalibrator,
)
from env.models import HotellingMarket
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    DistributionCombination,
    ExperimentMatrix,
    ProtocolConfigError,
    load_universal_pricing_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


class ConsumerPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.generator = ConsumerPopulationGenerator()
        cls.uniform_combination = DistributionCombination(
            ConsumerDistributionFamily.UNIFORM,
            ConsumerDistributionFamily.UNIFORM,
            ConsumerDistributionFamily.UNIFORM,
        )

    def test_same_seed_produces_byte_identical_population(self) -> None:
        population_spec = self.protocol.population_spec(
            self.uniform_combination
        )
        first = self.generator.generate(population_spec, 250, 8123)
        repeated = self.generator.generate(population_spec, 250, 8123)
        self.assertEqual(first.locations.tobytes(), repeated.locations.tobytes())
        self.assertEqual(
            first.strategicness.tobytes(),
            repeated.strategicness.tobytes(),
        )
        self.assertEqual(
            first.exclusivity.tobytes(),
            repeated.exclusivity.tobytes(),
        )

    def test_different_seed_changes_population(self) -> None:
        population_spec = self.protocol.population_spec(
            self.uniform_combination
        )
        first = self.generator.generate(population_spec, 250, 100)
        changed = self.generator.generate(population_spec, 250, 101)
        self.assertFalse(np.array_equal(first.locations, changed.locations))
        self.assertFalse(
            np.array_equal(first.strategicness, changed.strategicness)
        )
        self.assertFalse(
            np.array_equal(first.exclusivity, changed.exclusivity)
        )

    def test_attribute_streams_are_isolated_between_combinations(self) -> None:
        first_spec = self.protocol.population_spec(
            DistributionCombination(
                ConsumerDistributionFamily.UNIFORM,
                ConsumerDistributionFamily.TRUNCATED_NORMAL,
                ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL,
            )
        )
        changed_location_spec = self.protocol.population_spec(
            DistributionCombination(
                ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL,
                ConsumerDistributionFamily.TRUNCATED_NORMAL,
                ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL,
            )
        )
        first = self.generator.generate(first_spec, 500, 42)
        changed = self.generator.generate(
            changed_location_spec,
            500,
            42,
        )
        self.assertFalse(np.array_equal(first.locations, changed.locations))
        np.testing.assert_array_equal(
            first.strategicness,
            changed.strategicness,
        )
        np.testing.assert_array_equal(
            first.exclusivity,
            changed.exclusivity,
        )

    def test_all_27_combinations_are_finite_and_bounded(self) -> None:
        for index, combination in enumerate(
            ExperimentMatrix(self.protocol).distribution_combinations()
        ):
            with self.subTest(combination=combination.identifier):
                snapshot = self.generator.generate(
                    self.protocol.population_spec(combination),
                    100,
                    9000 + index,
                )
                for values in (
                    snapshot.locations,
                    snapshot.strategicness,
                    snapshot.exclusivity,
                ):
                    self.assertTrue(np.all(np.isfinite(values)))
                    self.assertGreaterEqual(float(np.min(values)), 0.0)
                    self.assertLessEqual(float(np.max(values)), 1.0)

    def test_population_snapshot_is_immutable(self) -> None:
        snapshot = ConsumerPopulationSnapshot(
            locations=[0.2, 0.8],
            strategicness=[0.1, 0.9],
            exclusivity=[0.3, 0.7],
        )
        with self.assertRaises(ValueError):
            snapshot.locations[0] = 0.5
        consumers = snapshot.to_consumers()
        self.assertEqual([consumer.id for consumer in consumers], [0, 1])
        self.assertEqual(consumers[0].strategicness, 0.1)
        self.assertEqual(consumers[1].exclusivity_pref, 0.7)

    def test_global_rng_state_does_not_affect_population(self) -> None:
        population_spec = self.protocol.population_spec(
            self.uniform_combination
        )
        expected = self.generator.generate(population_spec, 100, 1234)
        np.random.seed(987)
        np.random.random(1000)
        random.seed(876)
        for _ in range(1000):
            random.random()
        torch.manual_seed(765)
        torch.rand(1000)
        repeated = self.generator.generate(population_spec, 100, 1234)
        np.testing.assert_array_equal(expected.locations, repeated.locations)
        np.testing.assert_array_equal(
            expected.strategicness,
            repeated.strategicness,
        )
        np.testing.assert_array_equal(
            expected.exclusivity,
            repeated.exclusivity,
        )

    def test_distribution_moments_and_positive_skew(self) -> None:
        sample_count = 100_000
        seed = 34567
        snapshots = {}
        for family in ConsumerDistributionFamily:
            combination = DistributionCombination(family, family, family)
            snapshots[family] = self.generator.generate(
                self.protocol.population_spec(combination),
                sample_count,
                seed,
            ).locations

        uniform_values = snapshots[ConsumerDistributionFamily.UNIFORM]
        self.assertAlmostEqual(float(np.mean(uniform_values)), 0.5, delta=0.003)
        self.assertAlmostEqual(
            float(np.var(uniform_values)),
            1.0 / 12.0,
            delta=0.003,
        )

        normal_values = snapshots[
            ConsumerDistributionFamily.TRUNCATED_NORMAL
        ]
        skew_values = snapshots[
            ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL
        ]
        target_standard_deviation = 0.19091949726891616
        self.assertAlmostEqual(
            float(np.mean(normal_values)),
            0.5,
            delta=0.003,
        )
        self.assertAlmostEqual(
            float(np.std(normal_values)),
            target_standard_deviation,
            delta=0.003,
        )
        self.assertAlmostEqual(
            float(np.mean(skew_values)),
            float(np.mean(normal_values)),
            delta=0.003,
        )
        self.assertAlmostEqual(
            float(np.std(skew_values)),
            float(np.std(normal_values)),
            delta=0.003,
        )
        centered = skew_values - np.mean(skew_values)
        sample_skewness = float(
            np.mean(centered ** 3) / np.std(skew_values) ** 3
        )
        self.assertGreater(sample_skewness, 0.35)

    def test_attribute_streams_are_empirically_independent(self) -> None:
        snapshot = self.generator.generate(
            self.protocol.population_spec(self.uniform_combination),
            100_000,
            24680,
        )
        correlation_matrix = np.corrcoef(
            np.vstack(
                [
                    snapshot.locations,
                    snapshot.strategicness,
                    snapshot.exclusivity,
                ]
            )
        )
        off_diagonal = correlation_matrix[
            np.triu_indices_from(correlation_matrix, k=1)
        ]
        self.assertTrue(np.all(np.abs(off_diagonal) < 0.02))

    def test_calibration_reconstructs_frozen_parameters(self) -> None:
        location, scale = TruncatedSkewNormalMomentCalibrator.calibrate(
            target_mean=0.5,
            target_standard_deviation=0.19091949726891616,
            shape=5.0,
            low=0.0,
            high=1.0,
            initial_location=0.2457160185,
            initial_scale=0.3497947666,
        )
        self.assertAlmostEqual(location, 0.2457160185, delta=1e-6)
        self.assertAlmostEqual(scale, 0.3497947666, delta=1e-6)
        mean, standard_deviation = (
            TruncatedSkewNormalMomentCalibrator.truncated_moments(
                location=location,
                scale=scale,
                shape=5.0,
                low=0.0,
                high=1.0,
            )
        )
        self.assertAlmostEqual(mean, 0.5, delta=1e-8)
        self.assertAlmostEqual(
            standard_deviation,
            0.19091949726891616,
            delta=1e-8,
        )

    def test_registry_rejects_incomplete_family_mapping(self) -> None:
        with self.assertRaises(ProtocolConfigError):
            ConsumerAttributeSamplerRegistry({})

    def test_market_population_installation_preserves_legacy_reset(self) -> None:
        legacy_reference = HotellingMarket(num_consumers=20, seed=44)
        expected_locations = np.asarray(
            [consumer.location for consumer in legacy_reference.consumers]
        )
        market = HotellingMarket(num_consumers=20, seed=44)
        snapshot = self.generator.generate(
            self.protocol.population_spec(self.uniform_combination),
            20,
            999,
        )
        market.install_consumer_population(snapshot)
        np.testing.assert_array_equal(
            [consumer.location for consumer in market.consumers],
            snapshot.locations,
        )
        market.reset(seed=44)
        np.testing.assert_array_equal(
            [consumer.location for consumer in market.consumers],
            expected_locations,
        )


if __name__ == "__main__":
    unittest.main()
