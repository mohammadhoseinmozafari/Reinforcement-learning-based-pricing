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
    stat_names = ("mean", "std", "raw_log_std", "log_std")
    policy_samples = {name: [] for name in stat_names}
    action_samples = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            
            stats = agent.get_policy_stats(state)
            for name in stat_names:
                values = np.asarray(stats[name], dtype=float).ravel()
                if values.size != 3:
                    raise ValueError(f"Expected 3 policy heads, got {values.size}")
                policy_samples[name].append(values)

            action = agent.select_action(state, deterministic=True)
            
            action_values = np.asarray(action, dtype=float).ravel()
            if action_values.size != 3:
                raise ValueError(f"Expected 3 action heads, got {action_values.size}")
            action_samples.append(action_values)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            state = next_state
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    averaged = {
        name: np.mean(np.asarray(samples), axis=0)
        for name, samples in policy_samples.items()
    }
    averaged_actions = np.mean(np.asarray(action_samples), axis=0)
    policy_stats = {}
    for index, head in enumerate(("uniform", "new", "old")):
        policy_stats[head] = {
            name: float(values[index]) for name, values in averaged.items()
        }
        policy_stats[head]["action"] = float(averaged_actions[index])
        
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
        
        "episode_uniform_prices": metrics.episode_uniform_prices,
        "episode_new_prices": metrics.episode_new_prices,
        "episode_old_prices": metrics.episode_old_prices,
        "episode_opponent_prices_uniform": metrics.episode_opp_uniform_prices,
        "episode_opponent_prices_new": metrics.episode_opp_new_prices,
        "episode_opponent_prices_old": metrics.episode_opp_old_prices,
        
        "episode_market_shares": metrics.episode_market_shares,
        "episode_regimes": metrics.episode_regimes,
        "episode_opponent_regimes": metrics.episode_opp_regimes,
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
