





from models.reward_normalizer import FixedRewardNormalizer
from .type import EnvironmentType


class EnvironmentFactory:
    """Build a pricing environment and its reward-normalized training wrapper.

    ``env_type`` may be supplied once when the factory is created or per call to
    :meth:`create_environment`.  Keeping both forms makes the factory convenient
    for training (one environment type) and evaluation (many opponents).
    """

    def __init__(
        self,
        env_type: EnvironmentType | None = None,
        reward_normalizer: type = FixedRewardNormalizer,
    ) -> None:
        self.env_type = env_type
        self.reward_normalizer = reward_normalizer

    def create_environment(
        self,
        opponent_type: str,
        config,
        env_type: EnvironmentType | None = None,
    ):
        """Return ``(base_env, wrapped_env)`` for an opponent and config."""
        from env.pricing_env import make_pricing_env

        selected_env_type = env_type or self.env_type or getattr(
            config, "environment_type", getattr(config, "env_type", None)
        )
        if selected_env_type is None:
            raise ValueError("An environment type must be supplied to EnvironmentFactory")

        seed = getattr(config, "seed", getattr(config, "random_seed", None))
        base_env = make_pricing_env(
            environment_type=selected_env_type,
            opponent=opponent_type,
            num_consumers=config.num_consumers,
            episode_length=config.episode_length,
            seed=seed,
        )
        try:
            env = self.reward_normalizer(
                base_env, num_consumers=config.num_consumers
            )
        except TypeError:
            env = self.reward_normalizer(base_env)
        return base_env, env




    
