from dataclasses import dataclass
from config.constants import NUM_CONSUMERS, EPISODE_LENGTH
from typing import Optional, Dict
from env.type import EnvironmentType
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Uniform Pricing training."""
    agent_type: str = "sac"
    # Environment
    environment_type: EnvironmentType = EnvironmentType.UNIFORM_PRICING
    num_consumers: int = NUM_CONSUMERS
    episode_length: int = EPISODE_LENGTH
    opponent_type: str = "random_reactive_uniform"
    
    # SAC hyperparameters
    hidden_dim: int = 32
    lr_actor: float = 5e-4
    lr_critic: float = 5e-4
    lr_alpha: float = 1e-4
    target_entropy : float = -0.5
    gamma: float = 0.9
    tau: float = 0.005
    auto_alpha: bool = True
    log_std_min : float = -10.0
    log_std_max: float = 0.1

    buffer_size: int = 100000
    batch_size: int = 256
    alpha: float = 1.0

    lr_scheduler : Optional[str] = None
    lr_scheduler_kwargs: Optional[Dict] = None
    device: Optional[str] = None

    # Recurrent SAC (used only when agent_type == "recurrent_sac")
    sequence_length: int = 16
    episode_buffer_capacity: int = 1000
    opponent_action_dim: int = 3
    opponent_embedding_dim: int = 32
    encoder_hidden_dim: int = 128
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    lr_encoder: float = 3e-4
    opponent_aux_loss_weight: float = 1.0
    grad_clip_norm: Optional[float] = 10.0
    min_episodes_before_update: Optional[int] = None
    current_stage_weight: float = 0.75
    # Training
    num_episodes: int = 1000
    warmup_steps: int = 1000
    updates_per_step: int = 2
    eval_freq: int = 10
    eval_episodes: int = 5
    
    # Logging
    log_freq: int = 1  # Log every N episodes
    save_freq: int = 100  # Save model every N episodes
    
    # Reproducibility
    seed: int = 42
    verbose : bool = True
    # Paths
    save_dir: str = "experiments/phase2/phase2_1_uniform_training_results"
