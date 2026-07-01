"""Train a uniform-pricing SAC agent against uniform-price opponents."""

import numpy as np

from env import EnvironmentType
from train.config import TrainingConfig
from train.curriculum import CurriculumConfig
from train.pricing.curriculum import PricingCurriculum
from train.trainer import CurriculumTrainer


def main() -> None:
    """Run the uniform-opponent curriculum and print final summary metrics."""
    config = TrainingConfig(
        environment_type=EnvironmentType.UNIFORM_PRICING,
        num_episodes=1000,
        warmup_steps=500,
        eval_freq=10,
        save_freq=100,
        save_dir="experiments/uniform_pricing/uniform_opp/runs/1",
    )
    curriculum = PricingCurriculum()
    curriculum.opponent_sequence = [
        stage for stage in curriculum.get_sequence()
        if stage.opponent_type.endswith("_uniform")
    ]
    curriculum_config = CurriculumConfig(
        curriculum=curriculum,
        stages=curriculum.opponent_sequence,
        window_size=20,
        change_threshold=0.04,
        min_episodes_per_stage=100,
        max_episodes_per_stage=250,
    )

    _, metrics = CurriculumTrainer(config, curriculum_config).train()
    print("\nTraining complete")



if __name__ == "__main__":
    main()
