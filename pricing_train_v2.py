"""Validate, enumerate, or train isolated universal-pricing v2 runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from env.pricing_contracts import AgentArchitecture
from train.universal_pricing_protocol import (
    ConsumerDistributionFamily,
    ProtocolConfigError,
)
from universal_pricing_v2.protocol import (
    V2ArtifactLayout,
    V2ExperimentMatrix,
    V2ExperimentRunId,
    load_universal_pricing_v2_protocol,
    select_v2_experiment_coordinate,
)


DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parent
    / "config"
    / "protocols"
    / "universal_pricing_v2.yaml"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL
    )
    parser.add_argument("--enumerate", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--agent",
        choices=[item.value for item in AgentArchitecture],
    )
    families = [item.value for item in ConsumerDistributionFamily]
    parser.add_argument(
        "--location-distribution", choices=families
    )
    parser.add_argument(
        "--strategicness-distribution", choices=families
    )
    parser.add_argument(
        "--exclusivity-distribution", choices=families
    )
    parser.add_argument("--seed-index", type=int)
    parser.add_argument(
        "--maximum-steps",
        type=int,
        help="Development-only early interruption bound",
    )
    parser.add_argument(
        "--disable-mastery",
        action="store_true",
        help="Development-only: advance stages only at their caps",
    )
    args = parser.parse_args(argv)
    selectors = (
        args.agent,
        args.location_distribution,
        args.strategicness_distribution,
        args.exclusivity_distribution,
        args.seed_index,
    )
    if args.enumerate:
        if (
            args.validate_only
            or args.resume
            or any(value is not None for value in selectors)
            or args.maximum_steps is not None
        ):
            parser.error(
                "--enumerate cannot be combined with one-run arguments"
            )
    elif any(value is None for value in selectors):
        parser.error(
            "training requires --agent, all three distribution selectors, "
            "and --seed-index"
        )
    if args.validate_only and args.resume:
        parser.error("--validate-only cannot be combined with --resume")
    if args.maximum_steps is not None and args.maximum_steps <= 0:
        parser.error("--maximum-steps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_universal_pricing_v2_protocol(args.protocol)
        matrix = V2ExperimentMatrix(protocol)
        if args.enumerate:
            for coordinate in matrix.coordinates():
                print(V2ExperimentRunId.from_coordinate(coordinate))
            return
        coordinate = select_v2_experiment_coordinate(
            protocol,
            agent_architecture=args.agent,
            location_distribution=args.location_distribution,
            strategicness_distribution=(
                args.strategicness_distribution
            ),
            exclusivity_distribution=args.exclusivity_distribution,
            training_seed_index=args.seed_index,
        )
        layout = V2ArtifactLayout(protocol.artifact_root)
    except (ProtocolConfigError, TypeError, ValueError) as exc:
        raise SystemExit(f"V2 protocol configuration error: {exc}") from exc

    print("\nValidated hierarchical universal-pricing v2 run")
    print(f"  Run ID:        {V2ExperimentRunId.from_coordinate(coordinate)}")
    print(f"  Architecture:  {coordinate.agent_architecture.value}")
    print(f"  Budget:        {protocol.training_budget.environment_steps} steps")
    print(f"  Run directory: {layout.run_directory(coordinate)}")
    if args.validate_only:
        print("\nValidation completed without creating artifacts.")
        return

    from universal_pricing_v2.trainer import HierarchicalPricingTrainer

    manifest = HierarchicalPricingTrainer(
        protocol,
        coordinate,
        device=args.device,
        resume=args.resume,
        verbose=not args.quiet,
        enable_mastery_evaluation=not args.disable_mastery,
        maximum_environment_steps=args.maximum_steps,
    ).train()
    print(f"\nV2 run finished with status: {manifest.status.value}")


if __name__ == "__main__":
    main()
