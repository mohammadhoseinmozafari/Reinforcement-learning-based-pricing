from typing import Any, List
from train.uniform_training.logger import CurriculumTrainingLogger
import numpy as np

from env import make_uniform_pricing_env
from models.SAC import SAC
from models.buffer import BaseReplayBuffer, Curriculum, CurriculumReplayBuffer
from train.config import TrainingConfig
from train.curriculum import CurriculumConfig, OpponentCurriculumScheduler, OpponentStage
from train.metrics import TrainingMetrics
from train.uniform_training.uniform_training import evaluate_agent, save_checkpoint
from models.reward_normalizer import FixedRewardNormalizer

def create_environment (
        config : TrainingConfig,
        opponent
) :
    base_env = make_uniform_pricing_env(
        opponent= opponent,
        num_consumers = config.num_consumers,
        episode_length = config.episode_length,
        seed = config.seed
        
    )
    env = FixedRewardNormalizer(base_env)
    return base_env, env

def create_agent (
        config: TrainingConfig, env, replay_buffer : BaseReplayBuffer   
)  -> SAC:
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

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
        replay_buffer= replay_buffer,
        target_entropy=config.target_entropy,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max
    )

    return agent


def create_replay_buffer (
        config : TrainingConfig, 
        curriculum: Curriculum
) -> CurriculumReplayBuffer:
    
    return CurriculumReplayBuffer(
        capacity= config.buffer_size,
        batch_size=config.batch_size,
        curriculum= curriculum
    )


def warmup (env , 
            replay_buffer : BaseReplayBuffer , 
            steps : int , 
            seed: int  )  -> None: 
    
    state, _ = env.reset(seed = seed)
    
    for _ in range (steps):

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state if not done else env.reset()[0]



def train_with_curriculum(
        config: TrainingConfig,
        curriculum_config: CurriculumConfig,
        verbose = True
):
    
    np.random.seed(config.seed)

    curriculum = OpponentCurriculumScheduler(curriculum_config)
    logger = CurriculumTrainingLogger(curriculum_config, verbose= verbose)

    current_opponent = curriculum.current_opponent
    base_env , env = create_environment(
        config,
        current_opponent
    )


    logger.print_training_header()
    
    # create replay buffer
    replay_buffer = create_replay_buffer (config= config , curriculum=curriculum_config.curriculum)
    logger.log_replay_buffer(replay_buffer)
        

    # Create SAC agent
    agent = create_agent(
        config= config,
        env= env,
        replay_buffer= replay_buffer
    )
    logger.log_agent_config(agent)

    metrics = TrainingMetrics()
    # =========================================
    # WARMUP: Random exploration
    # =========================================
    logger.log_warmup_start(config.warmup_steps)
    warmup(env, agent.replay_buffer, config.warmup_steps, config.seed)


    # =========================================
    # TRAINING LOOP
    # =========================================
    if verbose:
        print("\033[32mStarting training...\033[0m\n")

    num_episodes = config.num_episodes
    episode_length = config.episode_length
    updates_per_step = config.updates_per_step
    for episode in range(num_episodes):
        current_stage = curriculum.current_stage
        
        if current_stage.opponent_type == 'mixed':
            assert current_stage.opponent_types
            opponent_type = np.random.choice(current_stage.opponent_types)
            base_env.close()
            
           
            base_env = make_uniform_pricing_env(
                opponent= opponent_type,
                num_consumers = config.num_consumers,
                episode_length = config.episode_length,
                seed = config.seed
                
            )
            env = FixedRewardNormalizer(base_env)

            agent.replay_buffer.set_stage(opponent_type)
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

            logger.log_episode_progress(episode, metrics, agent, eval_reward, curriculum, config)
    
            new_opponent = curriculum.advance()
            if new_opponent is not None:
                print("\n" + "=" * 60)
                print(f"Switching to opponent: {new_opponent.opponent_type}")
                print("=" * 60)
                
                base_env.close()
                if new_opponent.opponent_type =="mixed":
                    print(f"Entering the mixed stage with opponent types : {new_opponent.opponent_types}")
                else:
                    opponent_type = new_opponent.opponent_type
                base_env = make_uniform_pricing_env(
                    opponent=opponent_type,
                    num_consumers = config.num_consumers,
                    episode_length = config.episode_length,
                    seed= config.seed,

                )
                env = FixedRewardNormalizer(base_env)
            
                agent.replay_buffer.set_stage(
                    opponent_type
                )
                if verbose:
                    print(f"Replay buffer stage changed, current replay buffer stage: {agent.replay_buffer.current_stage}") 
                    print(agent.replay_buffer.get_info())
                
                
                if new_opponent.opponent_type != 'mixed':
                    
                    # agent.reset_optimizers()
                    # agent.reset_target_networks()
                    if verbose:
                        print(f"Warming up the agent with new opponent {new_opponent.opponent_type}")

                    warmup (env= env, replay_buffer= agent.replay_buffer, steps = config.warmup_steps, seed= config.seed)
                  
            
            if (episode + 1) % config.save_freq == 0:
                save_checkpoint(agent, metrics, config, episode + 1)
         # Final save
    save_checkpoint(agent, metrics, config, config.num_episodes, final=True)
  
    env.close()
    return agent, metrics


