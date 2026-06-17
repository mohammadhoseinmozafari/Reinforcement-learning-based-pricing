


from enum import Enum
from typing import Callable

from env import make_uniform_pricing_env



class EnvironmentType (str, Enum):
    UNIFORM_PRICING = 'uniform_pricing'

class EnvironmentFactory :
    
    def __init__(self, env_type : EnvironmentType , reward_normalizer : type) -> None:
        self._env_type = env_type
        self.env_creator : Callable = self.resolve_env_creator (self._env_type)
        self.reward_normalizer = reward_normalizer


    def create_environment (self,  opponent_type : str , config  ):
        base_env = self.env_creator(
            opponent = opponent_type,
            num_consumers = config.num_consumers,
            episode_length = config.episode_length,
            seed = config.seed
        )
        env = self.reward_normalizer(base_env)
        return base_env, env



    def resolve_env_creator (self, env_type : EnvironmentType):
        if env_type == "uniform_pricing":
            return make_uniform_pricing_env
        else :
            raise ValueError("Environment Type not valid")

