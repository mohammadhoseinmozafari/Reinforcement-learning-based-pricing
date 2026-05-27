from dataclasses import dataclass
from config.constants import NUM_CONSUMERS, EPISODE_LENGTH
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Uniform Pricing training."""
    # Environment
    num_consumers: int = NUM_CONSUMERS
    episode_length: int = EPISODE_LENGTH
    opponent_type: str = "random_reactive_uniform"
    
    # SAC hyperparameters
    hidden_dim: int = 256
    lr_actor: float = 5e-5
    lr_critic: float = 3e-4
    lr_alpha: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    auto_alpha: bool = True
    buffer_size: int = 100000
    batch_size: int = 512
    
    # Training
    num_episodes: int = 1000
    warmup_steps: int = 1000
    updates_per_step: int = 1
    eval_freq: int = 10
    eval_episodes: int = 5
    
    # Logging
    log_freq: int = 1  # Log every N episodes
    save_freq: int = 100  # Save model every N episodes
    
    # Reproducibility
    seed: int = 42
    
    # Paths
    save_dir: str = "experiments/phase2/phase2_1_uniform_training_results"

