from typing import Any, List

import numpy as np

from env import make_uniform_pricing_env
from models.SAC import SAC
from models.buffer import BaseReplayBuffer, Curriculum, CurriculumReplayBuffer
from train.config import TrainingConfig
from train.curriculum import CurriculumConfig, OpponentCurriculumScheduler, OpponentStage
from train.metrics import TrainingMetrics
from train.uniform_training.uniform_training import evaluate_agent, save_checkpoint
from models.reward_normalizer import FixedRewardNormalizer

class Color:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    END = '\033[0m'

class BoxStyle:
    """Box drawing characters for different styles"""
    SIMPLE = {
        'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
        'h': '─', 'v': '│', 'lc': '├', 'rc': '┤'
    }
    DOUBLE = {
        'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
        'h': '═', 'v': '║', 'lc': '╠', 'rc': '╣'
    }
    ROUNDED = {
        'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯',
        'h': '─', 'v': '│', 'lc': '├', 'rc': '┤'
    }

class CurriculumTrainingLogger :

    def __init__(self, curriculum_config: CurriculumConfig, verbose: bool = True) -> None:
        self.curriculum_config = curriculum_config
        self.verbose = verbose
        
    def c(self, color: str, text: str) -> str:
        return f"{color}{text}{Color.END}"
    
    def print_training_header(self) :
        
        if not self.verbose:
            return
        
        monitored = []

        if self.curriculum_config.monitor_critic:
            monitored.append("Critic Loss")
        if self.curriculum_config.monitor_actor:
            monitored.append("Actor Loss")
        if self.curriculum_config.monitor_alpha:
            monitored.append("Alpha")
        
        # Calculate box width based on content
        max_stage_width = max(
            (len(f"Stage {i+1}: {opp.name}") for i, opp in enumerate(self.curriculum_config.stages)),
            default=30
        )
        box_width = max(55, max_stage_width + 15)
        
        # Choose box style
        style = BoxStyle.ROUNDED
        
        # Header
        print(f"\n{self.c(Color.CYAN, style['tl'] + style['h'] * (box_width - 2) + style['tr'])}")
        
        # Title section
        title = "CONVERGENCE-BASED CURRICULUM"
        padding = (box_width - 2 - len(title)) // 2
        print(f"{self.c(Color.CYAN, style['v'])}{' ' * padding}{self.c(Color.BOLD + Color.YELLOW, title)}{' ' * (box_width - 2 - padding - len(title))}{self.c(Color.CYAN, style['v'])}")
        
        print(f"{self.c(Color.CYAN, style['lc'] + style['h'] * (box_width - 2) + style['rc'])}")
        
        # Monitoring section
        monitoring_label = "Monitoring:"
        monitoring_value = " + ".join(monitored) if monitored else "None"
        print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD, monitoring_label)} {self.c(Color.GREEN, monitoring_value)}{' ' * (box_width - len(monitoring_label) - len(monitoring_value) - 4)}{self.c(Color.CYAN, style['v'])}")
        
        # Threshold
        threshold_label = "Threshold:"
        threshold_value = f"{self.curriculum_config.change_threshold * 100:.1f}% change"
        print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD, threshold_label)} {self.c(Color.GREEN, threshold_value)}{' ' * (box_width - len(threshold_label) - len(threshold_value) - 4)}{self.c(Color.CYAN, style['v'])}")
        
        # Window size
        window_label = "Window:"
        window_value = f"{self.curriculum_config.window_size} episode{'s' if self.curriculum_config.window_size != 1 else ''}"
        print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD, window_label)} {self.c(Color.GREEN, window_value)}{' ' * (box_width - len(window_label) - len(window_value) - 4)}{self.c(Color.CYAN, style['v'])}")
        
        print(f"{self.c(Color.CYAN, style['lc'] + style['h'] * (box_width - 2) + style['rc'])}")
        
        # Stages section
        stages_label = "Curriculum Stages:"
        print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD + Color.BLUE, stages_label)}{' ' * (box_width - len(stages_label) - 3)}{self.c(Color.CYAN, style['v'])}")
        print(f"{self.c(Color.CYAN, style['v'])}{' ' * (box_width - 2)}{self.c(Color.CYAN, style['v'])}")
        
        for i, opp in enumerate(self.curriculum_config.stages):
            stage_num = f"Stage {i+1}"
            if i == 0:
                stage_text = f"  ▶ {self.c(Color.YELLOW, stage_num)}: {self.c(Color.BOLD, opp.name)} {self.c(Color.RED, '← CURRENT')}"
            else:
                stage_text = f"    {self.c(Color.CYAN, stage_num)}: {self.c(Color.BOLD, opp.name)}"
            
            # Truncate description if too long
            desc = opp.description
            max_desc_width = box_width - 8
            if len(desc) > max_desc_width:
                desc = desc[:max_desc_width - 3] + "..."
            if i ==0:
                print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD,stage_text)}{' ' * (box_width - len(stage_num) - len(opp.name) - 19)}{self.c(Color.CYAN, style['v'])}")

            else:
                print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD,stage_text)}{' ' * (box_width - len(stage_num) - len(opp.name) - 9)}{self.c(Color.CYAN, style['v'])}")
            if desc:
                print(f"{self.c(Color.CYAN, style['v'])}      {self.c(Color.BOLD, desc)}{' ' * (box_width - len(desc) - 8)}{self.c(Color.CYAN, style['v'])}")
            
            if i < len(self.curriculum_config.stages) - 1:
                print(f"{self.c(Color.CYAN, style['v'])}      {self.c(Color.BOLD, '↓')}{' ' * (box_width - 9)}{self.c(Color.CYAN, style['v'])}")
        
        # Footer
        print(f"{self.c(Color.CYAN, style['bl'] + style['h'] * (box_width - 2) + style['br'])}\n")
        
        # Summary stats
        total_stages = len(self.curriculum_config.stages)
        active_monitors = len(monitored)
        print(f"{self.c(Color.DIM, f'✨ {total_stages} stages configured • {active_monitors} metrics monitored • Window size: {self.curriculum_config.window_size}')}\n")
    
    def log_replay_buffer(self, replay_buffer: Any) -> None:
        """
        Display replay buffer initialization information in a visually appealing format.
        
        Args:
            replay_buffer: Replay buffer object with get_info() method that returns
                        a dictionary of buffer names and their lengths
        """
        if not self.verbose:
            return
        
        buffer_info = replay_buffer.get_info()
        
        # Calculate box width based on content
        max_label_width = max(
            (len(f"Replay buffer: {buffer}") for buffer in buffer_info.keys()),
            default=30
        )
        max_length_width = max(
            (len(f"Length: {length}") for length in buffer_info.values()),
            default=20
        )
        content_width = max_label_width + max_length_width + 3
        box_width = max(55, content_width + 4)
        
        # Choose box style
        style = BoxStyle.ROUNDED
        
        # Header
        print(f"\n{self.c(Color.CYAN, style['tl'] + style['h'] * (box_width - 2) + style['tr'])}")
        
        # Title section
        title = "REPLAY BUFFER INITIALIZATION"
        padding = (box_width - 2 - len(title)) // 2
        print(f"{self.c(Color.CYAN, style['v'])}{' ' * padding}{self.c(Color.BOLD + Color.YELLOW, title)}{' ' * (box_width - 2 - padding - len(title))}{self.c(Color.CYAN, style['v'])}")
        
        print(f"{self.c(Color.CYAN, style['lc'] + style['h'] * (box_width - 2) + style['rc'])}")
        
        # Buffer info header
        header_text = "Buffer Details:"
        print(f"{self.c(Color.CYAN, style['v'])} {self.c(Color.BOLD + Color.BLUE, header_text)}{' ' * (box_width - len(header_text) - 3)}{self.c(Color.CYAN, style['v'])}")
        print(f"{self.c(Color.CYAN, style['v'])}{' ' * (box_width - 2)}{self.c(Color.CYAN, style['v'])}")
        
        # Display each buffer
        for i, (buffer_name, length) in enumerate(buffer_info.items()):
            buffer_label = f"{buffer_name}:"
            
            # Format length with appropriate units
            if length >= 1_000_000:
                length_str = f"{length/1_000_000:.1f}M"
            elif length >= 1_000:
                length_str = f"{length/1_000:.1f}K"
            else:
                length_str = str(length)
            
            length_text = f"Size: {length_str}"
            
            # Format the line
            line = f"  {self.c(Color.CYAN, buffer_label)} {self.c(Color.GREEN, length_text)}"
            padding_len = box_width - len(buffer_label) - len(length_text) - 6
            
            print(f"{self.c(Color.CYAN, style['v'])} {line}{' ' * max(0, padding_len)}{self.c(Color.CYAN, style['v'])}")
            
            # Add separator between buffers (except after last)
            if i < len(buffer_info) - 1:
                print(f"{self.c(Color.CYAN, style['v'])}  {self.c(Color.DIM, '·' * (box_width - 6))}{' '* 2}{self.c(Color.CYAN, style['v'])}")
        
        # Footer
        print(f"{self.c(Color.CYAN, style['bl'] + style['h'] * (box_width - 2) + style['br'])}")
        
        # Summary stats
        total_buffers = len(buffer_info)
        total_capacity = sum(buffer_info.values())
        
        if total_capacity >= 1_000_000:
            total_str = f"{total_capacity/1_000_000:.1f}M"
        elif total_capacity >= 1_000:
            total_str = f"{total_capacity/1_000:.1f}K"
        else:
            total_str = str(total_capacity)
        
        summary = f"Total Buffers: {total_buffers} | Combined Size: {total_str}"
        print(f"{self.c(Color.DIM, summary)}\n")


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
    if verbose:
        print("=" * 55)
        print("SAC Configuration")
        print("=" * 55)
        for param, value in agent.get_info().items():
            print (f"parameter {param} : {value}")
        print("-" * 55)
        print()


    metrics = TrainingMetrics()
    # =========================================
    # WARMUP: Random exploration
    # =========================================
    if verbose:
        print(f"\nWarming up with {config.warmup_steps} random steps...")


    
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

            if verbose:
                info = curriculum.get_info()
                conv = info['convergence_status']
                critic_stat = "✓" if conv.get('critic', False) == True else "✗"
                actor_stat = "✓" if conv.get('actor', False) == True else "✗"
                alpha_stat = "✓" if conv.get('alpha', False) == True else "✗"
                avg_reward = np.mean(metrics.episode_rewards[-config.eval_freq:])
                avg_price = np.mean(metrics.episode_prices[-config.eval_freq:])
                
                avg_opp_price_uniform = np.mean(metrics.episode_opp_uniform_prices[-config.eval_freq:])
                avg_opp_price_new = np.mean(metrics.episode_opp_new_prices[-config.eval_freq:])
                avg_opp_price_old = np.mean(metrics.episode_opp_old_prices[-config.eval_freq:])
                avg_opp_regime = np.mean(metrics.episode_opp_regimes[-config.eval_freq:])
                
                if avg_opp_regime == 1.0:
                    opp_regime = "BBP"
                elif avg_opp_regime == 0.0:
                    opp_regime = "Uniform"
                else:
                    opp_regime = "Mixed"

                avg_share = np.mean(metrics.episode_market_shares[-config.eval_freq:])
                lrs = agent.get_current_lrs()
                print(f"Episode {episode + 1}/{config.num_episodes} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Eval: {eval_reward:.1f} | "
                      f"Price: { avg_price :.2f} | "
                      f"Share: {avg_share:.2f} | "
                      f"Opponent Regime : {opp_regime} | "
                      f"Opponent Price (Uniform): {avg_opp_price_uniform:.2f} | "
                      f"Opponent Price (BBP : New): {avg_opp_price_new:.2f} | "
                      f"Opponent Price (BBP : Old): {avg_opp_price_old:.2f} | "
                      f"α: {agent.alpha:.4f} | " 
                      f"LR: {lrs['actor_lr']:.2e} | Stage: {info['stage_name']} | "
                      f"Convergance: Critic{critic_stat}, Actor{actor_stat}, alpha{alpha_stat} ")
                
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


