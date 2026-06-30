





from models.reward_normalizer import FixedRewardNormalizer
from .type import EnvironmentType


class EnvironmentFactory :
    
    def __init__(self,  reward_normalizer : type = FixedRewardNormalizer) -> None:
        
        self.reward_normalizer = reward_normalizer


    def create_environment (self, env_type: EnvironmentType,  opponent_type : str , config  ) :
        from env.pricing_env import make_pricing_env
        base_env = make_pricing_env(
                environment_type= env_type,
                opponent = opponent_type,
                num_consumers = config.num_consumers,
                episode_length = config.episode_length,
                seed = config.random_seed
            )
        

            
        env = self.reward_normalizer(base_env)
        return base_env, env




    

