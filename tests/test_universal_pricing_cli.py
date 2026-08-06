"""Command-line validation tests for universal pricing Day 1."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import pricing_evaluate
import pricing_train
from train.universal_pricing_protocol import (
    ExperimentMatrix,
    ExperimentRunId,
    ExperimentRunManifest,
    ManifestRepository,
    ProtocolConfigError,
    RunStatus,
    SeedDeriver,
    load_universal_pricing_protocol,
    stable_configuration_hash,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "config/protocols/universal_pricing_v1.yaml"
)
PILOT_PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "config/protocols/universal_pricing_v1_pilot.yaml"
)


class UniversalPricingCliTests(unittest.TestCase):
    def test_pilot_protocol_has_isolated_root_and_fifty_thousand_steps(
        self,
    ) -> None:
        production = load_universal_pricing_protocol(PROTOCOL_PATH)
        pilot = load_universal_pricing_protocol(PILOT_PROTOCOL_PATH)
        self.assertEqual(pilot.training_budget.environment_steps, 50_000)
        self.assertNotEqual(pilot.artifact_root, production.artifact_root)
        self.assertEqual(
            pilot.seed_manifest,
            production.seed_manifest,
        )
        self.assertEqual(
            set(pilot.agent_profiles),
            set(production.agent_profiles),
        )

    def test_legacy_and_protocol_modes_are_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                pricing_train.parse_args(
                    [
                        "--config",
                        "legacy.yaml",
                        "--protocol",
                        str(PROTOCOL_PATH),
                    ]
                )

    def test_protocol_selectors_are_required_for_one_run(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                pricing_train.parse_args(
                    ["--protocol", str(PROTOCOL_PATH), "--agent", "sac"]
                )

    def test_enumeration_prints_810_ids(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            pricing_train.main(
                ["--protocol", str(PROTOCOL_PATH), "--enumerate"]
            )
        run_ids = output.getvalue().splitlines()
        self.assertEqual(len(run_ids), 810)
        self.assertEqual(len(set(run_ids)), 810)
        self.assertTrue(
            all(identifier.startswith("universal_pricing_v1__") for identifier in run_ids)
        )

    def test_one_run_validation_does_not_construct_environment(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            pricing_train.main(
                [
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--agent",
                    "rsac",
                    "--location-distribution",
                    "uniform",
                    "--strategicness-distribution",
                    "truncated_normal",
                    "--exclusivity-distribution",
                    "truncated_skew_normal",
                    "--seed-index",
                    "0",
                    "--validate-only",
                ]
            )
        self.assertIn("Validated universal pricing run", output.getvalue())
        self.assertIn("__rsac__", output.getvalue())


class UniversalPricingEvaluationCliTests(unittest.TestCase):
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
            status=RunStatus.COMPLETED,
            artifact_references={},
        )

    def test_validation_and_final_suite_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            ManifestRepository().write(
                run_directory / "manifest.json",
                self.make_manifest(),
            )
            self.assertEqual(
                len(
                    pricing_evaluate.validate_evaluation_request(
                        run_directory,
                        "validation",
                    )
                ),
                25,
            )
            self.assertEqual(
                len(
                    pricing_evaluate.validate_evaluation_request(
                        run_directory,
                        "final",
                    )
                ),
                100,
            )

    def test_evaluation_parser_accepts_pilot_protocol(self) -> None:
        arguments = pricing_evaluate.parse_args(
            [
                "--run-directory",
                "pilot-run",
                "--protocol",
                str(PILOT_PROTOCOL_PATH),
                "--evaluation-suite",
                "validation",
            ]
        )
        self.assertEqual(arguments.protocol, PILOT_PROTOCOL_PATH)

    def test_malformed_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            (run_directory / "manifest.json").write_text(
                '{"protocol_version": "obsolete"}',
                encoding="utf-8",
            )
            with self.assertRaises(ProtocolConfigError):
                pricing_evaluate.validate_evaluation_request(
                    run_directory,
                    "final",
                )

    def test_nontraining_run_seed_is_rejected(self) -> None:
        manifest = self.make_manifest()
        final_seed = (
            self.protocol.seed_manifest.final_evaluation_environment_seeds[0]
        )
        invalid_manifest = ExperimentRunManifest(
            protocol_version=manifest.protocol_version,
            run_id=manifest.run_id,
            coordinate=manifest.coordinate,
            run_seed_bundle=SeedDeriver.derive_run_bundle(final_seed),
            resolved_protocol=dict(manifest.resolved_protocol),
            configuration_hash=manifest.configuration_hash,
            git_commit=manifest.git_commit,
            hardware_metadata=manifest.hardware_metadata,
            status=manifest.status,
            artifact_references=manifest.artifact_references,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            ManifestRepository().write(
                run_directory / "manifest.json",
                invalid_manifest,
            )
            with self.assertRaises(ProtocolConfigError):
                pricing_evaluate.validate_evaluation_request(
                    run_directory,
                    "final",
                )


if __name__ == "__main__":
    unittest.main()
