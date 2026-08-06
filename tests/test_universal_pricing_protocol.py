"""Tests for universal_pricing_v1 experiment infrastructure."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from env.pricing_contracts import AgentArchitecture
from env.type import EnvironmentType
from train.experiment import load_curriculum_config
from train.universal_pricing_protocol import (
    AgentProfileConfig,
    ArtifactLayout,
    BalancedOpponentSchedule,
    ConsumerDistributionFamily,
    DistributionCombination,
    ExperimentCoordinate,
    ExperimentMatrix,
    ExperimentRunId,
    ExperimentRunManifest,
    ManifestRepository,
    OpponentFamily,
    ProtocolConfigError,
    RunStatus,
    SeedBankManifest,
    SeedDeriver,
    SeedPurpose,
    load_opponent_pool_config,
    load_universal_pricing_protocol,
    stable_configuration_hash,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)


class ProtocolConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)

    def test_distribution_and_coordinate_counts_are_exact(self) -> None:
        matrix = ExperimentMatrix(self.protocol)
        combinations = matrix.distribution_combinations()
        coordinates = matrix.coordinates()
        self.assertEqual(len(combinations), 27)
        self.assertEqual(len(set(combinations)), 27)
        self.assertEqual(len(coordinates), 810)
        self.assertEqual(len(set(coordinates)), 810)

    def test_run_ids_and_artifact_paths_are_unique(self) -> None:
        coordinates = ExperimentMatrix(self.protocol).coordinates()
        layout = ArtifactLayout(self.protocol.artifact_root)
        run_ids = {
            str(ExperimentRunId.from_coordinate(coordinate))
            for coordinate in coordinates
        }
        paths = {
            str(layout.run_directory(coordinate))
            for coordinate in coordinates
        }
        self.assertEqual(len(run_ids), 810)
        self.assertEqual(len(paths), 810)

    def test_run_id_round_trip(self) -> None:
        coordinate = ExperimentCoordinate(
            agent_architecture=AgentArchitecture.OE_RSAC,
            distribution_combination=DistributionCombination(
                ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL,
                ConsumerDistributionFamily.TRUNCATED_NORMAL,
                ConsumerDistributionFamily.UNIFORM,
            ),
            curriculum_id="mixed_balanced",
            training_seed_index=9,
        )
        run_id = ExperimentRunId.from_coordinate(coordinate)
        self.assertEqual(ExperimentRunId.parse(str(run_id)), coordinate)

    def test_repeated_enumeration_is_byte_deterministic(self) -> None:
        matrix = ExperimentMatrix(self.protocol)
        first = "\n".join(
            str(ExperimentRunId.from_coordinate(coordinate))
            for coordinate in matrix.coordinates()
        ).encode()
        second = "\n".join(
            str(ExperimentRunId.from_coordinate(coordinate))
            for coordinate in matrix.coordinates()
        ).encode()
        self.assertEqual(first, second)

    def test_enumeration_does_not_create_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "not-created"
            layout = ArtifactLayout(root)
            for coordinate in ExperimentMatrix(self.protocol).coordinates():
                layout.manifest_path(coordinate)
            self.assertFalse(root.exists())

    def test_population_spec_selects_all_three_attributes(self) -> None:
        combination = DistributionCombination(
            ConsumerDistributionFamily.UNIFORM,
            ConsumerDistributionFamily.TRUNCATED_NORMAL,
            ConsumerDistributionFamily.TRUNCATED_SKEW_NORMAL,
        )
        population = self.protocol.population_spec(combination)
        self.assertEqual(population.location.family.value, "uniform")
        self.assertEqual(
            population.strategicness.family.value,
            "truncated_normal",
        )
        self.assertEqual(
            population.exclusivity.family.value,
            "truncated_skew_normal",
        )

    def test_agent_profile_fields_are_architecture_specific(self) -> None:
        with self.assertRaises(ProtocolConfigError):
            AgentProfileConfig(
                AgentArchitecture.SAC,
                sequence_length=16,
            )
        with self.assertRaises(ProtocolConfigError):
            AgentProfileConfig(AgentArchitecture.RSAC)
        with self.assertRaises(ProtocolConfigError):
            AgentProfileConfig(
                AgentArchitecture.RSAC,
                sequence_length=16,
                episode_replay_capacity=100,
                opponent_embedding_dim=32,
            )
        with self.assertRaises(ProtocolConfigError):
            AgentProfileConfig(
                AgentArchitecture.OE_RSAC,
                sequence_length=16,
                episode_replay_capacity=100,
            )

    def test_legacy_mixed_stages_inherit_scheduler_limits(self) -> None:
        curriculum = load_curriculum_config(
            REPOSITORY_ROOT / "config/curricula/mixed.yaml",
            EnvironmentType.UNIFORM_PRICING,
            num_consumers=100,
            episode_length=200,
            verbose=False,
        )
        self.assertEqual(
            [stage.min_episodes for stage in curriculum.stages],
            [100] * 9,
        )
        self.assertEqual(
            [stage.max_episodes for stage in curriculum.stages],
            [250] * 9,
        )


class SeedInfrastructureTests(unittest.TestCase):
    def test_seed_bank_sizes_disjointness_and_recreation(self) -> None:
        manifest = SeedBankManifest.from_root_seed(20260805)
        self.assertEqual(len(manifest.training_roots), 10)
        self.assertEqual(len(manifest.validation_environment_seeds), 25)
        self.assertEqual(
            len(manifest.final_evaluation_environment_seeds),
            100,
        )
        all_seeds = (
            manifest.training_roots
            + manifest.validation_environment_seeds
            + manifest.final_evaluation_environment_seeds
        )
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertEqual(
            manifest,
            SeedBankManifest.from_root_seed(20260805),
        )

    def test_purposes_and_episode_indices_create_distinct_streams(self) -> None:
        root = SeedBankManifest.from_root_seed().training_roots[0]
        streams = {
            SeedDeriver.derive_stream_seed(root, purpose)
            for purpose in SeedPurpose
        }
        self.assertEqual(len(streams), len(SeedPurpose))
        episode_seeds = {
            SeedDeriver.derive_episode_seed(next(iter(streams)), index)
            for index in range(20)
        }
        self.assertEqual(len(episode_seeds), 20)

    def test_global_rng_state_does_not_affect_derivation(self) -> None:
        run_seed = SeedBankManifest.from_root_seed().training_roots[3]
        expected = SeedDeriver.derive_run_bundle(run_seed)
        np.random.seed(99)
        np.random.random(100)
        random.seed(400)
        for _ in range(100):
            random.random()
        torch.manual_seed(123)
        torch.rand(100)
        self.assertEqual(expected, SeedDeriver.derive_run_bundle(run_seed))

    def test_agents_share_run_and_episode_seeds(self) -> None:
        protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        bundles = {
            architecture: protocol.run_seed_bundle(4)
            for architecture in AgentArchitecture
        }
        self.assertEqual(len(set(bundles.values())), 1)
        episodes = {
            architecture: SeedDeriver.derive_episode_bundle(bundle, 17)
            for architecture, bundle in bundles.items()
        }
        self.assertEqual(len(set(episodes.values())), 1)


class OpponentScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = load_universal_pricing_protocol(
            PROTOCOL_PATH
        ).opponent_pool

    def test_each_complete_pair_is_family_balanced(self) -> None:
        assignments = BalancedOpponentSchedule(
            self.pool, 812345
        ).assignments(200)
        for index in range(0, len(assignments), 2):
            self.assertEqual(
                {
                    assignments[index].opponent_family,
                    assignments[index + 1].opponent_family,
                },
                {OpponentFamily.UNIFORM, OpponentFamily.BBP},
            )

    def test_round_robin_covers_every_policy_in_each_family(self) -> None:
        assignments = BalancedOpponentSchedule(
            self.pool, 812345
        ).assignments(10)
        by_family = {
            family: {
                assignment.policy_name
                for assignment in assignments
                if assignment.opponent_family is family
            }
            for family in OpponentFamily
        }
        self.assertEqual(
            by_family[OpponentFamily.UNIFORM],
            set(self.pool.uniform_policies),
        )
        self.assertEqual(
            by_family[OpponentFamily.BBP],
            set(self.pool.bbp_policies),
        )

    def test_schedule_reproducibility_and_seed_sensitivity(self) -> None:
        first = BalancedOpponentSchedule(self.pool, 1234).assignments(50)
        repeated = BalancedOpponentSchedule(self.pool, 1234).assignments(50)
        changed = BalancedOpponentSchedule(self.pool, 5678).assignments(50)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_duplicate_or_unknown_policy_is_rejected(self) -> None:
        invalid_documents = (
            {
                "family_weights": {"uniform": 0.5, "bbp": 0.5},
                "families": {
                    "uniform": ["uniform_fixed", "uniform_fixed"],
                    "bbp": ["bbp_fixed_discriminator"],
                },
            },
            {
                "family_weights": {"uniform": 0.5, "bbp": 0.5},
                "families": {
                    "uniform": ["uniform_unknown"],
                    "bbp": ["bbp_fixed_discriminator"],
                },
            },
        )
        for document in invalid_documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "pool.yaml"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ProtocolConfigError):
                    load_opponent_pool_config(path)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_universal_pricing_protocol(PROTOCOL_PATH)
        cls.coordinate = ExperimentMatrix(cls.protocol).coordinates()[0]

    def make_manifest(self) -> ExperimentRunManifest:
        resolved = self.protocol.to_dict()
        return ExperimentRunManifest(
            protocol_version=self.protocol.protocol_version,
            run_id=ExperimentRunId.from_coordinate(self.coordinate),
            coordinate=self.coordinate,
            run_seed_bundle=self.protocol.run_seed_bundle(0),
            resolved_protocol=resolved,
            configuration_hash=stable_configuration_hash(resolved),
            git_commit="test-commit",
            hardware_metadata={"device": "cpu"},
            status=RunStatus.REGISTERED,
            artifact_references={},
        )

    def test_configuration_hash_is_stable(self) -> None:
        first = self.protocol.to_dict()
        second = json.loads(json.dumps(first))
        self.assertEqual(
            stable_configuration_hash(first),
            stable_configuration_hash(second),
        )

    def test_repository_round_trip_and_status_update(self) -> None:
        repository = ManifestRepository()
        manifest = self.make_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run" / "manifest.json"
            repository.write(path, manifest)
            self.assertEqual(repository.read(path).to_dict(), manifest.to_dict())
            repository.write(
                path,
                replace(manifest, status=RunStatus.RUNNING),
            )
            self.assertEqual(repository.read(path).status, RunStatus.RUNNING)

    def test_repository_rejects_identity_or_immutable_fact_changes(self) -> None:
        repository = ManifestRepository()
        manifest = self.make_manifest()
        alternate_coordinate = replace(
            self.coordinate,
            training_seed_index=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            repository.write(path, manifest)
            with self.assertRaises(ProtocolConfigError):
                repository.write(
                    path,
                    replace(
                        manifest,
                        coordinate=alternate_coordinate,
                        run_id=ExperimentRunId.from_coordinate(
                            alternate_coordinate
                        ),
                    ),
                )
            with self.assertRaises(ProtocolConfigError):
                repository.write(
                    path,
                    replace(manifest, git_commit="other-commit"),
                )


if __name__ == "__main__":
    unittest.main()
