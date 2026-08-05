"""Validate a universal pricing run against a locked evaluation suite."""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import Sequence

from train.universal_pricing_protocol import (
    ManifestRepository,
    PROTOCOL_VERSION,
    ProtocolConfigError,
    load_seed_bank_manifest,
    load_universal_pricing_protocol,
)


class EvaluationSuite(str, Enum):
    """Seed bank selected for checkpoint evaluation."""

    VALIDATION = "validation"
    FINAL = "final"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Directory containing manifest.json for one universal run",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluation-suite",
        required=True,
        choices=[suite.value for suite in EvaluationSuite],
    )
    return parser.parse_args(argv)


def validate_evaluation_request(
    run_directory: str | Path,
    evaluation_suite: str | EvaluationSuite,
) -> tuple[int, ...]:
    """Validate manifest identity and return the selected committed seed bank."""

    run_directory = Path(run_directory)
    suite = EvaluationSuite(evaluation_suite)
    manifest = ManifestRepository().read(run_directory / "manifest.json")
    if manifest.protocol_version != PROTOCOL_VERSION:
        raise ProtocolConfigError(
            f"Evaluation requires protocol {PROTOCOL_VERSION}"
        )

    seed_manifest_path = (
        Path(__file__).resolve().parent
        / "config"
        / "seeds"
        / "universal_pricing_v1.yaml"
    )
    seed_manifest = load_seed_bank_manifest(seed_manifest_path)
    if manifest.run_seed_bundle.run_seed not in seed_manifest.training_roots:
        raise ProtocolConfigError(
            "Manifest run seed is not a committed training root"
        )
    if suite is EvaluationSuite.VALIDATION:
        seeds = seed_manifest.validation_environment_seeds
    else:
        seeds = seed_manifest.final_evaluation_environment_seeds
        if set(seeds) & set(seed_manifest.training_roots):
            raise ProtocolConfigError(
                "Final evaluation seeds overlap training roots"
            )
    return seeds


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        seeds = validate_evaluation_request(
            args.run_directory,
            args.evaluation_suite,
        )
    except (ProtocolConfigError, ValueError, TypeError) as exc:
        raise SystemExit(f"Evaluation configuration error: {exc}") from exc

    manifest = ManifestRepository().read(
        args.run_directory / "manifest.json"
    )
    if manifest.status.value != "completed":
        raise SystemExit("Evaluation requires a completed training run")
    protocol_path = (
        Path(__file__).resolve().parent
        / "config"
        / "protocols"
        / "universal_pricing_v1.yaml"
    )
    protocol = load_universal_pricing_protocol(protocol_path)
    if protocol.to_dict() != dict(manifest.resolved_protocol):
        raise SystemExit(
            "Evaluation protocol does not match the run manifest"
        )
    checkpoint = Path(
        manifest.artifact_references.get(
            "final_checkpoint",
            str(args.run_directory / "checkpoints" / "final.pt"),
        )
    )
    if not checkpoint.is_file():
        checkpoint = args.run_directory / "checkpoints" / "final.pt"
    from evaluation.universal_pricing_evaluator import (
        UniversalPricingEvaluator,
    )

    episodes, summary = UniversalPricingEvaluator(
        protocol,
        manifest.coordinate,
        device=args.device,
    ).evaluate_checkpoint(
        checkpoint,
        seeds,
        suite=args.evaluation_suite,
        output_directory=args.run_directory / "evaluation",
    )
    print(
        f"Evaluated {len(episodes)} balanced episodes from {len(seeds)} "
        f"locked {args.evaluation_suite} seeds."
    )
    print(
        "Mean normalized episode reward: "
        f"{summary['mean_normalized_reward_total']:.6f}"
    )


if __name__ == "__main__":
    main()
