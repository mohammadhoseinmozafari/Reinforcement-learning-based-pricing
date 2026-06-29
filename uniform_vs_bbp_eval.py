from env.factory import EnvironmentFactory, EnvironmentType
import os
import json

from evaluation.config import EvaluationConfig
from evaluation.evaluator import Evaluator
from models.SAC import SAC
from models.buffer import ReplayBuffer
from train.uniform_training.curriculum import BBPOpponentUniformPricingCurriculum

MODEL_PATH = "experiments/uniform_pricing/bbp_opp/runs/1/sac_uniform_final.pt"
SAVE_PATH = "experiments/uniform_pricing/bbp_opp/eval/"
opponent_types = BBPOpponentUniformPricingCurriculum().opponent_types

os.makedirs(SAVE_PATH, exist_ok=True)

config = EvaluationConfig(
    env_type= EnvironmentType.UNIFORM_PRICING,
    model_path= MODEL_PATH
)
env_factory = EnvironmentFactory(
    config.env_type
)


agent = SAC(
    state_dim=9, 
    action_dim=1,
    hidden_dim=32,
    replay_buffer=ReplayBuffer(100)
)

agent.load(config.model_path)

for opp_type in opponent_types :
    _, env = env_factory.create_environment(opp_type, config)
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(agent, env)
    

    result_save_path = os.path.join(SAVE_PATH, f"eval_result_{opp_type}.json")
    with open(result_save_path, 'w') as f:
        json.dump(eval_result, f, indent=2)

    

