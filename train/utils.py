import gymnasium as gym
from models.sac import SAC
from typing import Tuple, Dict , Any  
import os 
import json
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
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        mean_values = []
        std_values = []
        raw_log_std_values= []
        log_std_values = []
        actions_taken = []

        for _ in range(max_steps):
            
            stats = agent.get_policy_stats(state)
            mean_values.append(float(stats["mean"][0]))
            std_values.append(float(stats["std"][0]))
            raw_log_std_values.append(float(stats['raw_log_std'][0]))
            log_std_values.append(float(stats["log_std"][0]))

            action = agent.select_action(state, deterministic=True)
            
            actions_taken.append(float(action[0]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            state = next_state
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    policy_stats ={
        "mean" : np.mean(mean_values),
        "raw_log_std" : np.mean(raw_log_std_values),
        "log_std" : np.mean(log_std_values),
        "std" : np.mean(std_values),
        "action" : np.mean(actions_taken)    
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
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)



