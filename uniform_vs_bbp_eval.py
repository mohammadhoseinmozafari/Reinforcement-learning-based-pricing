"""Evaluate a trained uniform-pricing agent against every BBP opponent."""

import json
import os

from env import EnvironmentType
from env.factory import EnvironmentFactory
from evaluation.config import EvaluationConfig
from evaluation.evaluator import Evaluator
from models.buffer import ReplayBuffer
from models.sac import SAC
from train.pricing.curriculum import PricingCurriculum

MODEL_PATH = "experiments/uniform_pricing/bbp_opp/runs/1/sac_uniform_final.pt"
SAVE_PATH = "experiments/uniform_pricing/bbp_opp/eval"


def main() -> None:
    """Load one checkpoint and write one JSON result per BBP opponent."""
    config = EvaluationConfig(
        env_type=EnvironmentType.UNIFORM_PRICING,
        model_path=MODEL_PATH,
    )
    factory = EnvironmentFactory(config.env_type)
    agent = SAC(
        state_dim=13,
        action_dim=3,
        hidden_dim=32,
        replay_buffer=ReplayBuffer(100),
    )
    agent.load(config.model_path)
    os.makedirs(SAVE_PATH, exist_ok=True)

    opponents = [
        name for name in PricingCurriculum().opponent_types
        if name.endswith("_bbp")
    ]
    for opponent in opponents:
        base_env, env = factory.create_environment(opponent, config)
        try:
            result = Evaluator(config).evaluate(agent, env)
        finally:
            base_env.close()

        result_path = os.path.join(SAVE_PATH, f"eval_result_{opponent}.json")
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2)


if __name__ == "__main__":
    main()
