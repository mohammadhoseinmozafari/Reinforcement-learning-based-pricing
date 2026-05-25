"""
Phase 2.1 — Uniform Price Learning Training Script

Train SAC agent to learn optimal uniform pricing against stationary opponent.

Features:
- Single scalar action (uniform price)
- Comprehensive logging and metrics tracking
- Reproducible via seeds
- Configurable opponent policies
- Training visualization
"""





import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List,  Tuple
from dataclasses import dataclass, field
import json
import os

from env.uniform_pricing_env import  make_uniform_pricing_env
from models.SAC import SAC
from config.constants import    EPISODE_LENGTH, NUM_CONSUMERS
from models.reward_normalizer import EpisodeRewardNormalizer
import gymnasium as gym



# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Phase 2.1 training."""
    # Environment
    num_consumers: int = NUM_CONSUMERS
    episode_length: int = EPISODE_LENGTH
    opponent_type: str = "random_reactive_uniform"
    
    # SAC hyperparameters
    hidden_dim: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    auto_alpha: bool = True
    buffer_size: int = 100000
    batch_size: int = 256
    
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


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode_rewards: List[float] = field(default_factory=list)

    episode_profits: List[float] = field(default_factory=list)
    episode_opp_profits: List[float] = field(default_factory=list)

    episode_prices: List[float] = field(default_factory=list)
    episode_opp_prices: List[float] = field(default_factory=list)

    episode_market_shares: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    
    # Per-step tracking (for current episode)
    step_profits: List[float] = field(default_factory=list)
    step_prices: List[float] = field(default_factory=list)
    step_market_shares: List[float] = field(default_factory=list)
    
    step_opp_profits: List[float] = field(default_factory=list)
    step_opp_prices: List[float] = field(default_factory=list)
    
    def reset_episode(self):
        """Reset per-episode tracking."""
        self.step_profits = []
        self.step_prices = []
        self.step_market_shares = []

        self.step_opp_profits=[]
        self.step_opp_prices=[]
    
    def record_step(self, info: Dict):
        """Record metrics from a step."""
        self.step_profits.append(info.get("profit", 0.0))
        self.step_prices.append(info.get("price", 0.0))
        self.step_market_shares.append(info.get("market_share", 0.0))
    
        self.step_opp_profits.append(info.get("opponent_profit", 0.0))
        self.step_opp_prices.append(info.get("opponent_price", 0.0))


    def end_episode(self, total_reward: float):
        """Finalize episode metrics."""
        self.episode_rewards.append(total_reward)
        self.episode_profits.append(sum(self.step_profits))
        self.episode_prices.append(float(np.mean(self.step_prices)) if self.step_prices else 0.0)
        self.episode_market_shares.append(float(np.mean(self.step_market_shares)) if self.step_market_shares else 0.0)

        self.episode_opp_profits.append(sum(self.step_opp_profits))
        
        self.episode_opp_prices.append(float(np.mean(self.step_opp_prices)) if self.step_opp_prices else 0.0)
        

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_uniform_pricing(
    config: TrainingConfig,
    verbose: bool = True
) -> Tuple[SAC, TrainingMetrics]:
    """
    Train SAC agent for uniform pricing.
    
    Args:
        config: Training configuration
        verbose: Print progress
        
    Returns:
        Trained agent and metrics
    """
    # Set random seeds for reproducibility
    np.random.seed(config.seed)
    
    # Create environment
    base_env = make_uniform_pricing_env(
        opponent=config.opponent_type,
        num_consumers=config.num_consumers,
        episode_length=config.episode_length,
        seed=config.seed,
    )
    env = EpisodeRewardNormalizer(base_env)
    assert env.observation_space.shape
    assert env.action_space.shape
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if verbose:
        print("=" * 60)
        print("PHASE 2.1 — UNIFORM PRICE LEARNING")
        print("=" * 60)
        print(f"State dim: {state_dim}")
        print(f"Action dim: {action_dim}")
        print(f"Opponent: {config.opponent_type}")
        print(f"Episodes: {config.num_episodes}")
        print("=" * 60)
    
    # Create SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=1.0,  # Actions already in [0, 1]
        hidden_dim=config.hidden_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        lr_alpha=config.lr_alpha,
        gamma=config.gamma,
        tau=config.tau,
        auto_alpha=config.auto_alpha,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
    )
    
    if verbose:
        print(f"Device: {agent.device}")
    
    # Initialize metrics
    metrics = TrainingMetrics()
    
    # =========================================
    # WARMUP: Random exploration
    # =========================================
    if verbose:
        print(f"\nWarming up with {config.warmup_steps} random steps...")
    
    state, _ = env.reset(seed=config.seed)
    for _ in range(config.warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition (action already in [0, 1])
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    # =========================================
    # TRAINING LOOP
    # =========================================
    if verbose:
        print("Starting training...\n")
    num_episodes = config.num_episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        metrics.reset_episode()
        
        episode_reward: float = 0.0
        episode_critic_loss = []
        episode_actor_loss = []
   
        for step in range(config.episode_length):

            action = agent.select_action(state)
                        
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
           
               
            # Update agent
            for _ in range(config.updates_per_step):
                update_metrics = agent.update()
                if update_metrics is not None:
                    episode_critic_loss.append(update_metrics['critic_loss'])
                    episode_actor_loss.append(update_metrics['actor_loss'])
            
            # Track metrics
            metrics.record_step(info)
    
            episode_reward += float(reward)
            state = next_state
            
            if done:
                break
        
        
        # End episode
        metrics.end_episode(episode_reward)
        
        if episode_critic_loss:
            metrics.critic_losses.append(float(np.mean(episode_critic_loss)))
            metrics.actor_losses.append(float(np.mean(episode_actor_loss)))
            metrics.alphas.append(agent.alpha)
        
        # =========================================
        # EVALUATION
        # =========================================
        if (episode + 1) % config.eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, config.eval_episodes, config.episode_length)
            metrics.eval_rewards.append(eval_reward)
            
            if verbose:
                avg_reward = np.mean(metrics.episode_rewards[-config.eval_freq:])
                avg_price = np.mean(metrics.episode_prices[-config.eval_freq:])
                
                avg_opp_price = np.mean(metrics.episode_opp_prices[-config.eval_freq:])

                avg_share = np.mean(metrics.episode_market_shares[-config.eval_freq:])
                lrs = agent.get_current_lrs()
                print(f"Episode {episode + 1}/{config.num_episodes} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Eval: {eval_reward:.1f} | "
                      f"Price: { avg_price :.2f} | "
                      f"Share: {avg_share:.2f} | "
                      f"Opponent Price: {avg_opp_price:.2f} | "
                      f"α: {agent.alpha:.4f} | " 
                      f"LR: {lrs['actor_lr']:.2e}")
        
        # =========================================
        # SAVE CHECKPOINT
        # =========================================
        if (episode + 1) % config.save_freq == 0:
            save_checkpoint(agent, metrics, config, episode + 1)
    
    # Final save
    save_checkpoint(agent, metrics, config, config.num_episodes, final=True)
  
    env.close()
    return agent, metrics


def evaluate_agent(
    env: gym.Env,
    agent: SAC,
    num_episodes: int,
    max_steps: int
) -> float:
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
        
        for _ in range(max_steps):
            # Use deterministic action for evaluation
            action = agent.select_action(state, deterministic=True)
            action = np.clip(action, 0.0, 1.0)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            state = next_state
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


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
        "episode_opponent_prices": metrics.episode_opp_prices,
        
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



def plot_training_results(metrics: TrainingMetrics, config: TrainingConfig):
    """Plot training results with an aesthetic, publication‑ready style."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold', y=1.02)

    # ---- 1. Episode rewards + opponent profits + moving average ----
    ax = axes[0, 0]
    episodes = range(1, len(metrics.episode_rewards) + 1)
    ax.plot(episodes, metrics.episode_rewards, alpha=0.4, lw=1.5, label='Agent Reward')
    ax.plot(episodes, metrics.episode_opp_profits, alpha=0.4, lw=1.5, label='Opponent Profit')
    
    window = min(20, len(metrics.episode_rewards))
    if len(metrics.episode_rewards) >= window:
        moving_avg = np.convolve(metrics.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(metrics.episode_rewards)+1), moving_avg, 
                'r-', lw=2.5, label=f'{window}-ep MA')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total Reward / Profit', fontsize=11)
    ax.set_title('Rewards & Opponent Profit', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ---- 2. Evaluation rewards (scatter + line) ----
    ax = axes[0, 1]
    eval_eps = list(range(config.eval_freq, config.num_episodes + 1, config.eval_freq))
    if len(eval_eps) > len(metrics.eval_rewards):
        eval_eps = eval_eps[:len(metrics.eval_rewards)]
    ax.plot(eval_eps, metrics.eval_rewards, 's-', color='#2c7bb6', lw=2, markersize=6, 
            markerfacecolor='white', markeredgewidth=1.5, label='Evaluation reward')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Eval Reward', fontsize=11)
    ax.set_title('Evaluation Performance', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 3. Agent vs Opponent Prices ----
    ax = axes[0, 2]
    ax.plot(episodes, metrics.episode_prices, label='Agent Price', lw=2, color='#1b9e77')
    ax.plot(episodes, metrics.episode_opp_prices, label='Opponent Price', lw=2, color='#d95f02', linestyle='--')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('Pricing Strategy', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 4. Market share with 0.5 reference ----
    ax = axes[1, 0]
    ax.fill_between(episodes, 0.5, metrics.episode_market_shares, 
                    where=(np.array(metrics.episode_market_shares) >= 0.5), 
                    color='green', alpha=0.2, interpolate=True)
    ax.fill_between(episodes, 0.5, metrics.episode_market_shares, 
                    where=(np.array(metrics.episode_market_shares) < 0.5), 
                    color='red', alpha=0.2, interpolate=True)
    ax.plot(episodes, metrics.episode_market_shares, 'k-', lw=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Market Share', fontsize=11)
    ax.set_title('Market Share (above 0.5 = advantage)', fontsize=12)
    ax.grid(True, alpha=0.3)
    # Annotate final share
    final_share = metrics.episode_market_shares[-1] if metrics.episode_market_shares else 0.5
    ax.annotate(f'Final: {final_share:.2f}', xy=(episodes[-1], final_share), 
                xytext=(-30, 10), textcoords='offset points', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

    # ---- 5. Losses (smoothed if long) ----
    ax = axes[1, 1]
    if metrics.critic_losses:
        # optional smoothing for readability
        ax.plot(metrics.critic_losses, label='Critic Loss', alpha=0.6, lw=1.2)
        ax.plot( metrics.actor_losses, label='Actor Loss', alpha=0.6, lw=1.2)
        ax.set_yscale('log')  # losses often span orders of magnitude
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_title('Training Losses', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 6. Entropy coefficient α ----
    ax = axes[1, 2]
    if metrics.alphas:
        alpha_eps = range(1, len(metrics.alphas) + 1)
        ax.plot(alpha_eps, metrics.alphas, color='purple', lw=2.5)
        ax.fill_between(alpha_eps, 0, metrics.alphas, alpha=0.2, color='purple')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('α', fontsize=11)
    ax.set_title('Entropy Coefficient', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(config.save_dir, exist_ok=True)
    plt.savefig(os.path.join(config.save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
