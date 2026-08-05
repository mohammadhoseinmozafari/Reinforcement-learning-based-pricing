"""Command-line entrypoint for legacy and universal pricing experiments."""

import argparse
from pathlib import Path
from typing import Sequence

from train.experiment import (
    ExperimentConfigError,
    ExperimentOverrides,
    build_agent,
    build_environment,
    load_experiment,
)
from train.universal_pricing_protocol import (
    AgentArchitecture,
    ArtifactLayout,
    ConsumerDistributionFamily,
    ExperimentMatrix,
    ExperimentRunId,
    ProtocolConfigError,
    load_universal_pricing_protocol,
    select_experiment_coordinate,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the experiment path and supported high-value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config",
        type=Path,
        help="Legacy composed experiment YAML file",
    )
    mode.add_argument(
        "--protocol",
        type=Path,
        help="Versioned universal pricing protocol YAML file",
    )
    parser.add_argument("--episodes", type=int, help="Override training episode count")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", help="Override Torch device, such as cpu or cuda")
    parser.add_argument("--save-dir", help="Override experiment output directory")
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Override the complete training profile selected by the experiment",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Print the complete protocol matrix without creating artifacts",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate one protocol coordinate without creating artifacts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a protocol run from its latest exact snapshot",
    )
    parser.add_argument(
        "--agent",
        choices=[architecture.value for architecture in AgentArchitecture],
        help="Universal protocol agent architecture",
    )
    distribution_choices = [
        family.value for family in ConsumerDistributionFamily
    ]
    parser.add_argument(
        "--location-distribution",
        choices=distribution_choices,
        help="Consumer location distribution family",
    )
    parser.add_argument(
        "--strategicness-distribution",
        choices=distribution_choices,
        help="Consumer strategicness distribution family",
    )
    parser.add_argument(
        "--exclusivity-distribution",
        choices=distribution_choices,
        help="Consumer exclusivity distribution family",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        help="Confirmatory training-seed index from 0 through 9",
    )
    args = parser.parse_args(argv)

    protocol_selectors = (
        args.agent,
        args.location_distribution,
        args.strategicness_distribution,
        args.exclusivity_distribution,
        args.seed_index,
    )
    if args.config is not None:
        if (
            args.enumerate
            or args.validate_only
            or args.resume
            or any(value is not None for value in protocol_selectors)
        ):
            parser.error(
                "protocol selectors and --enumerate require --protocol"
            )
    else:
        legacy_overrides = (
            args.episodes,
            args.seed,
            args.save_dir,
            args.training_config,
        )
        if any(value is not None for value in legacy_overrides):
            parser.error(
                "legacy overrides cannot be combined with --protocol"
            )
        if args.validate_only and args.resume:
            parser.error("--validate-only cannot be combined with --resume")
        if args.enumerate:
            if (
                args.validate_only
                or args.resume
                or any(value is not None for value in protocol_selectors)
            ):
                parser.error(
                    "--enumerate cannot be combined with one-run selectors"
                )
        elif any(value is None for value in protocol_selectors):
            parser.error(
                "protocol training requires --agent, all three distribution "
                "selectors, and --seed-index"
            )
    return args


def _run_protocol_mode(args: argparse.Namespace) -> None:
    """Validate, enumerate, or train one universal protocol coordinate."""

    try:
        protocol = load_universal_pricing_protocol(args.protocol)
        matrix = ExperimentMatrix(protocol)
        artifact_layout = ArtifactLayout(protocol.artifact_root)
        if args.enumerate:
            for coordinate in matrix.coordinates():
                print(ExperimentRunId.from_coordinate(coordinate))
            return

        coordinate = select_experiment_coordinate(
            protocol,
            agent_architecture=args.agent,
            location_distribution=args.location_distribution,
            strategicness_distribution=args.strategicness_distribution,
            exclusivity_distribution=args.exclusivity_distribution,
            training_seed_index=args.seed_index,
        )
        run_id = ExperimentRunId.from_coordinate(coordinate)
        run_seed_bundle = protocol.run_seed_bundle(args.seed_index)
    except (ProtocolConfigError, ValueError, TypeError) as exc:
        raise SystemExit(f"Protocol configuration error: {exc}") from exc

    print("\nValidated universal pricing run")
    print(f"  Protocol:        {protocol.protocol_version}")
    print(f"  Run ID:          {run_id}")
    print(f"  Architecture:    {coordinate.agent_architecture.value}")
    print(f"  Training seed:   {run_seed_bundle.run_seed}")
    print(f"  Run directory:   {artifact_layout.run_directory(coordinate)}")
    if args.validate_only:
        print("\nValidation completed without creating run artifacts.")
        return
    from train.universal_pricing_trainer import UniversalPricingTrainer

    trainer = UniversalPricingTrainer(
        protocol,
        coordinate,
        device=args.device or "cpu",
        resume=args.resume,
    )
    manifest = trainer.train()
    print(f"\nRun finished with status: {manifest.status.value}")


def main(argv: Sequence[str] | None = None) -> None:
    """Compose configuration, build runtime objects, and start shared training."""
    args = parse_args(argv)
    if args.protocol is not None:
        _run_protocol_mode(args)
        return

    overrides = ExperimentOverrides(
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        training_config=args.training_config,
    )
    try:
        experiment = load_experiment(args.config, overrides)
    except ExperimentConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    config = experiment.training_config
    print("\nResolved training configuration")
    print(f"  Profile:         {experiment.training_source}")
    print(f"  Agent type:      {config.agent_type}")
    print(f"  Batch size:      {config.batch_size}")
    if config.agent_type == "recurrent_sac":
        print(f"  Sequence length: {config.sequence_length}")
    print(f"  Save directory:  {config.save_dir}\n")

    env_factory, base_env, env = build_environment(experiment)
    replay_buffer, agent = build_agent(experiment, env)
    if experiment.training_config.agent_type == "recurrent_sac":
        from train.recurrent_curriculum_trainer import RecurrentCurriculumTrainer
        trainer_class = RecurrentCurriculumTrainer
    else:
        from train.trainer import CurriculumTrainer
        trainer_class = CurriculumTrainer
    trainer = trainer_class(
        config=experiment.training_config,
        curriculum_config=experiment.curriculum_config,
        env_factory=env_factory,
        base_env=base_env,
        env=env,
        replay_buffer=replay_buffer,
        agent=agent,
    )
    trainer.train()


if __name__ == "__main__":
    main()
