from typing import List

import numpy as np

from env import make_uniform_pricing_env
from models.SAC import SAC
from train.config import TrainingConfig
from train.uniform_training.curriculum import UniformPricingCurriculum
from train.curriculum import CurriculumConfig, OpponentCurriculumScheduler
from models.reward_normalizer import EpisodeRewardNormalizer
from train.metrics import TrainingMetrics
from train.uniform_training.uniform_training import evaluate_agent, save_checkpoint
from models.reward_normalizer import FixedRewardNormalizer

def train_with_curriculum(
        config: TrainingConfig,
        curriculum_config: CurriculumConfig,
        verbose = True
):
    np.random.seed(config.seed)

    curriculum = OpponentCurriculumScheduler(curriculum_config)
    base_env = make_uniform_pricing_env(
        opponent=curriculum.current_opponent,
        num_consumers = config.num_consumers,
        episode_length = config.episode_length,
        seed = config.seed
        
    )
    env = FixedRewardNormalizer(base_env)
    assert env.observation_space.shape
    assert env.action_space.shape

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if verbose:
        print("=" * 55)
        print("CONVERGENCE-BASED CURRICULUM")
        print("=" * 55)
        print(f"Monitoring: ", end="")
        parts = []
        if curriculum_config.monitor_critic: parts.append("critic loss")
        if curriculum_config.monitor_actor: parts.append("actor loss")
        if curriculum_config.monitor_alpha: parts.append("alpha")
        print(" + ".join(parts))
        print(f"Threshold: {curriculum_config.change_threshold*100:.1f}% change")
        print(f"Window: {curriculum_config.window_size} episodes")
        print("-" * 55)
        for i, opp in enumerate(curriculum_config.stages):
            marker = " ← START" if i == 0 else ""
            print(f"  Stage {i+1}: {opp}{marker}")
        print("=" * 55)
    
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
        alpha= config.alpha,
        auto_alpha=config.auto_alpha,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_entropy=config.target_entropy,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max
    )
    if verbose:
        print("SAC Configuration")
        print(agent.get_info())

    metrics = TrainingMetrics()
    # =========================================
    # WARMUP: Random exploration
    # =========================================
    if verbose:
        print(f"\nWarming up with {config.warmup_steps} random steps...")
    

    state, _ = env.reset(seed = config.seed)
    for _ in range (config.warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    # =========================================
    # TRAINING LOOP
    # =========================================
    if verbose:
        print("Starting training...\n")

    num_episodes = config.num_episodes
    episode_length = config.episode_length
    updates_per_step = config.updates_per_step
    for episode in range(num_episodes):
        state, _ = env.reset()
        metrics.reset_episode()
        
        episode_reward: float = 0.0
        episode_critic_loss: List = []
        episode_actor_loss: List = []
        for step in range(episode_length):

            action = agent.select_action(state)
                        
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
           
           # Update agent
            for _ in range(updates_per_step):
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
        avg_critic = float(np.mean(episode_critic_loss) if episode_critic_loss else 0)
        avg_actor = float(np.mean(episode_actor_loss) if episode_actor_loss else 0)
        current_alpha = agent.alpha
        
        if episode_critic_loss:
            metrics.critic_losses.append(avg_critic)
            metrics.actor_losses.append(avg_actor)
            metrics.alphas.append(current_alpha)
        
        curriculum.step(
            critic_loss=avg_critic,
            actor_loss=avg_actor,
            alpha= current_alpha
        )
        # =========================================
        # EVALUATION
        # =========================================
        if (episode + 1) % config.eval_freq == 0:
            eval_reward = evaluate_agent(base_env, agent, config.eval_episodes, config.episode_length)
            metrics.eval_rewards.append(eval_reward)
            new_opponent = curriculum.advance()
            if new_opponent is not None:
                base_env = make_uniform_pricing_env(
                    opponent=new_opponent.opponent_type,
                    num_consumers = config.num_consumers,
                    episode_length = config.episode_length,
                    seed= config.seed+ episode,

                )
                env = EpisodeRewardNormalizer(base_env)
            if verbose:
                info = curriculum.get_info()
                conv = info['convergence_status']
                critic_stat = "✓" if conv.get('critic', False) == True else "✗"
                actor_stat = "✓" if conv.get('actor', False) == True else "✗"
                alpha_stat = "✓" if conv.get('alpha', False) == True else "✗"
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
                      f"LR: {lrs['actor_lr']:.2e} | Stage: {info['stage_name']} | "
                      f"Convergance: Critic{critic_stat}, Actor{actor_stat}, alpha{alpha_stat} ")
            if (episode + 1) % config.save_freq == 0:
                save_checkpoint(agent, metrics, config, episode + 1)
         # Final save
    save_checkpoint(agent, metrics, config, config.num_episodes, final=True)
  
    env.close()
    return agent, metrics


