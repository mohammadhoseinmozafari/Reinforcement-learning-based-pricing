



from dataclasses import dataclass

from config.constants import EPISODE_LENGTH, NUM_CONSUMERS, RANDOM_SEED
from env.factory import EnvironmentType


@dataclass
class EvaluationConfig:
    env_type : EnvironmentType
    model_path : str 

    num_episodes : int = 1
    episode_length = EPISODE_LENGTH
    num_consumers = NUM_CONSUMERS

    

    random_seed : int = RANDOM_SEED
    