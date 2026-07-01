



from dataclasses import dataclass

from config.constants import EPISODE_LENGTH, NUM_CONSUMERS, RANDOM_SEED
from env.type import EnvironmentType


@dataclass
class EvaluationConfig:
    env_type : EnvironmentType
    model_path : str 

    num_episodes : int = 1
    episode_length: int = EPISODE_LENGTH
    num_consumers: int = NUM_CONSUMERS

    random_seed : int = RANDOM_SEED
