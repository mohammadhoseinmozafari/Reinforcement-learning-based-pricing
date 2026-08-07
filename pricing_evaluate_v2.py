"""Evaluate a completed v2 hierarchy on paired locked seed suites."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from train.universal_pricing_protocol import ProtocolConfigError, RunStatus
from universal_pricing_v2.agents import HierarchicalPricingAgentFactory
from universal_pricing_v2.evaluation import UniversalPricingV2Evaluator
from universal_pricing_v2.protocol import (
    V2ManifestRepository,
    load_universal_pricing_v2_protocol,
)


DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parent
    / "config"
    / "protocols"
    / "universal_pricing_v2.yaml"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL
    )
    parser.add_argument(
        "--evaluation-suite",
        required=True,
        choices=("validation", "final", "transfer"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        protocol = load_universal_pricing_v2_protocol(args.protocol)
        manifest = V2ManifestRepository().read(
            args.run_directory / "manifest.json"
        )
    except (ProtocolConfigError, TypeError, ValueError) as exc:
        raise SystemExit(f"V2 evaluation configuration error: {exc}") from exc
    if manifest.status is not RunStatus.COMPLETED:
        raise SystemExit("V2 evaluation requires a completed run")
    if protocol.to_dict() != dict(manifest.resolved_protocol):
        raise SystemExit("V2 evaluation protocol does not match manifest")
    checkpoint = Path(
        manifest.artifact_references.get(
            "final_checkpoint",
            args.run_directory / "ckpt" / "final.pt",
        )
    )
    if not checkpoint.is_file():
        raise SystemExit(f"Final checkpoint does not exist: {checkpoint}")
    evaluator = UniversalPricingV2Evaluator(
        protocol, manifest.coordinate, device=args.device
    )
    if args.evaluation_suite in {"final", "transfer"}:
        seeds = protocol.seed_manifest.final_evaluation_environment_seeds
    else:
        seeds = protocol.seed_manifest.validation_environment_seeds
    if args.evaluation_suite == "transfer":
        agent = HierarchicalPricingAgentFactory.create(
            protocol.agent_profiles[
                manifest.coordinate.agent_architecture
            ],
            manifest.run_seed_bundle,
            device=args.device,
        )
        agent.load(checkpoint)
        matrix = evaluator.transfer_matrix(agent, seeds)
        from universal_pricing_v2.evaluation import V2EvaluationRepository

        V2EvaluationRepository._json(
            args.run_directory / "evaluation" / "transfer_matrix.json",
            matrix,
        )
        print(f"Completed {len(matrix)} distribution-transfer cells.")
        return
    episodes, summary = evaluator.evaluate_checkpoint(
        checkpoint,
        seeds,
        suite=args.evaluation_suite,
        output_directory=args.run_directory / "evaluation",
        include_counterfactuals=True,
    )
    print(
        f"Evaluated {len(episodes)} paired counterfactual episodes "
        f"from {len(seeds)} locked {args.evaluation_suite} seeds."
    )
    learned = summary["counterfactuals"]["learned"]
    print(
        "Mean learned-strategy net profit: "
        f"{learned['mean_net_agent_profit_total']:.4f}"
    )


if __name__ == "__main__":
    main()
