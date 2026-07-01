import gymnasium as gym
from models.sac import SAC
from typing import Tuple, Dict , Any  
import os 
import json
import numpy as np
from dataclasses import asdict
from enum import Enum
from train.config import TrainingConfig
from train.metrics  import TrainingMetrics
def evaluate_agent(
    env: gym.Env,
    agent: SAC,
    num_episodes: int,
    max_steps: int
) -> Tuple[float , Dict[str, Any]]:
    """
    Evaluate agent with deterministic policy.
    
    Args:
        env: Environment
        agent: SAC agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Average reward
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    total_reward = 0.0
    mean_values = []
    std_values = []
    raw_log_std_values = []
    log_std_values = []
    actions_taken = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            
            stats = agent.get_policy_stats(state)
            mean_values.extend(np.asarray(stats["mean"], dtype=float).ravel())
            std_values.extend(np.asarray(stats["std"], dtype=float).ravel())
            raw_log_std_values.extend(
                np.asarray(stats["raw_log_std"], dtype=float).ravel()
            )
            log_std_values.extend(np.asarray(stats["log_std"], dtype=float).ravel())

            action = agent.select_action(state, deterministic=True)
            
            actions_taken.extend(np.asarray(action, dtype=float).ravel())
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            state = next_state
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    policy_stats ={
        "mean": float(np.mean(mean_values)),
        "raw_log_std": float(np.mean(raw_log_std_values)),
        "log_std": float(np.mean(log_std_values)),
        "std": float(np.mean(std_values)),
        "action": float(np.mean(actions_taken)),
        }
        
    return total_reward / num_episodes, policy_stats


# =============================================================================
# SAVING AND VISUALIZATION
# =============================================================================

def save_checkpoint(
    agent: SAC,
    metrics: TrainingMetrics,
    config: TrainingConfig,
    episode: int,
    final: bool = False
):
    """Save model and metrics checkpoint."""
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Save model
    suffix = "final" if final else f"ep{episode}"
    model_path = os.path.join(config.save_dir, f"sac_uniform_{suffix}.pt")
    agent.save(model_path)
    
    # Save metrics
    metrics_dict = {
        "episode_rewards": metrics.episode_rewards,
        
        "episode_profits": metrics.episode_profits,
        "episode_opponent_profits": metrics.episode_opp_profits,
        
        "episode_prices": metrics.episode_prices,
        "episode_opponent_prices_uniform": metrics.episode_opp_uniform_prices,
        "episode_opponent_prices_new": metrics.episode_opp_new_prices,
        "episode_opponent_prices_old": metrics.episode_opp_old_prices,
        
        "episode_market_shares": metrics.episode_market_shares,
        "eval_rewards": metrics.eval_rewards,
        "critic_losses": metrics.critic_losses,
        "actor_losses": metrics.actor_losses,
        "alphas": metrics.alphas,
    }
    
    metrics_path = os.path.join(config.save_dir, f"metrics_{suffix}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save config
    config_path = os.path.join(config.save_dir, "config.json")
    config_dict = asdict(config)
    config_dict = {
        key: value.value if isinstance(value, Enum) else value
        for key, value in config_dict.items()
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

