"""Command-line entrypoint for YAML-configured pricing experiments."""

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the experiment path and supported high-value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument("--episodes", type=int, help="Override training episode count")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", help="Override Torch device, such as cpu or cuda")
    parser.add_argument("--save-dir", help="Override experiment output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Compose configuration, build runtime objects, and start shared training."""
    args = parse_args(argv)
    overrides = ExperimentOverrides(
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )
    try:
        experiment = load_experiment(args.config, overrides)
    except ExperimentConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    env_factory, base_env, env = build_environment(experiment)
    replay_buffer, agent = build_agent(experiment, env)
    from train.trainer import CurriculumTrainer

    trainer = CurriculumTrainer(
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
