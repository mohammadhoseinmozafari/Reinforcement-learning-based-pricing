"""Construction of universal pricing environments from protocol coordinates."""

from __future__ import annotations

from config.constants import EPISODE_LENGTH, NUM_CONSUMERS
from env.universal_pricing_env import UniversalPricingEnv
from train.universal_pricing_protocol import (
    ExperimentCoordinate,
    ExperimentMatrix,
    ProtocolConfigError,
    RunSeedBundle,
    UniversalPricingProtocolConfig,
)


class UniversalPricingEnvironmentFactory:
    """Resolve one primary coordinate into an unwrapped universal environment."""

    def __init__(
        self,
        protocol: UniversalPricingProtocolConfig,
        *,
        num_consumers: int = NUM_CONSUMERS,
        episode_length: int = EPISODE_LENGTH,
    ) -> None:
        self.protocol = protocol
        self.num_consumers = num_consumers
        self.episode_length = episode_length

    def create_environment(
        self,
        coordinate: ExperimentCoordinate,
    ) -> UniversalPricingEnv:
        return self.create_environment_with_run_seed(
            coordinate,
            self.protocol.run_seed_bundle(
                coordinate.training_seed_index
            ),
        )

    def create_environment_with_run_seed(
        self,
        coordinate: ExperimentCoordinate,
        run_seed_bundle: RunSeedBundle,
    ) -> UniversalPricingEnv:
        """Construct an environment with an explicit training/evaluation bundle."""
        if coordinate not in ExperimentMatrix(self.protocol).coordinates():
            raise ProtocolConfigError(
                "Experiment coordinate is outside the protocol matrix"
            )
        return UniversalPricingEnv(
            consumer_population_spec=self.protocol.population_spec(
                coordinate.distribution_combination
            ),
            opponent_pool=self.protocol.opponent_pool,
            run_seed_bundle=run_seed_bundle,
            regime_commitment_length=(
                self.protocol.regime_commitment_length
            ),
            num_consumers=self.num_consumers,
            episode_length=self.episode_length,
        )
