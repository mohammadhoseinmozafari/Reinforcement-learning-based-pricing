from abc import abstractmethod
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    List,
    Tuple )

import numpy as np

from config.constants import EPISODE_LENGTH, NUM_CONSUMERS
from env.uniform_pricing_env import UniformPricingEnv

# =============================================================================
# OPPONENT DIFFICULTY LEVELS
# =============================================================================

class OpponentDifficulty(Enum):
    """Difficulty tiers for opponent curriculum."""
    TUTORIAL = 1      # Constant Price
    EASY = 2          # Constant competitive price
    MEDIUM = 3        # Random reactive pricing
    HARD = 4          # Aggressive undercutting
    EXPERT = 5        # Adaptive/learning opponent


@dataclass
class OpponentStage:
    """Definition of a single opponent curriculum stage."""
    name: str
    opponent_type: str
    difficulty: OpponentDifficulty
    description: str
    opponent_types : Optional[List[str]] = None ## for mixed stages
    
    
    # How many episodes to spend in this stage (if not using performance-based)
    min_episodes: int = 100
    max_episodes: Optional[int] = None  # None = unlimited until mastered

    
 
class Curriculum : 
    def __init__(self) -> None:
        self.opponent_sequence : List[OpponentStage] = []
    @abstractmethod
    def get_sequence(self):
        raise NotImplementedError
    

# =============================================================================
# CURRICULUM SCHEDULER
# =============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for opponent curriculum learning."""
    curriculum : Curriculum
    # Curriculum stages
    stages: List[OpponentStage] = field(default_factory=list)
    

    # Loss stabilization detection
    window_size : int = 30
    change_threshold : float = 0.02

    monitor_critic : bool = True
    monitor_actor : bool = True
    monitor_alpha : bool = False
    
    # Safety nets
    min_episodes_per_stage: int = 30            # Never advance before this many episodes
    max_episodes_per_stage: Optional[int] = 150  # Force advance after this many (None = no limit)
    
    #mixed stage training
    mixed_stage : bool = True
    mixed_stage_episodes : int = 200

    # Environment settings
    num_consumers: int = NUM_CONSUMERS
    episode_length: int = EPISODE_LENGTH
    
    # Logging
    verbose: bool = True

class OpponentCurriculumScheduler:
    """
    Manages progression through opponent difficulty stages.
    
    Tracks agent performance and determines when to advance
    to the next opponent difficulty level.
    """
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum configuration
            optimal_profit: Theoretical optimal profit for performance comparison
        """
        self.config = config
        
        
        # State tracking
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0
        
        #Performance tracking
        self.critic_losses = deque(maxlen=config.window_size)
        self.actor_losses = deque(maxlen=config.window_size)
        self.alphas = deque(maxlen=config.window_size)

        # History
        self.advancement_records: List[Dict]  = []
    
    @property
    def current_stage(self) -> OpponentStage:
        """Get current curriculum stage."""
        return self.config.stages[self.current_stage_idx]
    
    @property
    def is_last_stage(self) -> bool:
        """Check if currently on the last stage."""
        return self.current_stage_idx >= len(self.config.stages) - 1
    
    @property
    def progress(self) -> float:
        """Overall curriculum progress [0, 1]."""
        return self.current_stage_idx / max(len(self.config.stages) - 1, 1)
    @property
    def current_opponent(self) -> str:
        """Get opponent type for current stage."""
        return self.current_stage.opponent_type
    
    
    def step(self, critic_loss: float, actor_loss: float, alpha: float)-> None:
        """
        Advance curriculum by one episode.
        
        Args:
            critic_loss: critic loss
            actor_loss: actor loss
            alpha: entropy coefficient
        """
        self.total_episodes += 1
        self.episodes_in_stage += 1
        
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.alphas.append(alpha)

    def _is_metric_stable(self, values: deque, name: str= '') -> Tuple[bool, float]:
        """
        Check if a single metric has converged.
        Returns:
            (is_stable, relative_change)
        """   
        if len(values)< self.config.window_size:
            return False,1.0
        
        vals = np.array(values)
        half = len(vals) // 2
        first_half_mean = np.mean(vals[:half])
        second_half_mean = np.mean(vals[half:])
        if second_half_mean> first_half_mean:
            return False, 1.0
        relative_change = abs(second_half_mean - first_half_mean)/ abs(first_half_mean)
        is_stable = relative_change <= self.config.change_threshold

        return bool(is_stable), float(relative_change)
    
    def _is_all_stable(self) -> Tuple[bool , Dict[bool, float]]:
        """
        Check all monitored metrics for convergence.
        """
        status = {}

        if self.config.monitor_actor:
            stable, change = self._is_metric_stable(self.actor_losses, 'actor')
            status['actor']= (stable, change)

        if self.config.monitor_critic:
            stable, change = self._is_metric_stable(self.critic_losses, 'critic')
            status['critic']= (stable, change)
        
        if self.config.monitor_alpha:
            stable, change = self._is_metric_stable(self.alphas, 'alpha')
            status['alpha']= (stable, change)

        all_stable = all(stable for stable,_ in status.values()) if status else True

        return all_stable , status

    def _format_status(self, status: Dict) -> str:
        parts = []
        for name, (stable, change) in status.items():
            symbol = "✓" if stable else "✗"
            parts.append(f"{name}={symbol}({change:.3f})")
        return " | ".join(parts)

    
    def should_advance(self) -> Tuple[bool, str]:
        """
        Determine if agent should advance to next stage.
        
        Returns:
            (should_advance, reason)
        """
        # Can't advance from last stage
        if self.is_last_stage:
            return False, "Already at final stage"
        
        # Check minimum episodes
        if self.episodes_in_stage < self.config.min_episodes_per_stage:
            return False, f"Min episodes not met ({self.episodes_in_stage}/{self.config.min_episodes_per_stage})"
        
        # Check maximum episodes (force advance)
        if (self.config.max_episodes_per_stage is not None and 
            self.episodes_in_stage >= self.config.max_episodes_per_stage):
            return True, f"Max episodes reached ({self.episodes_in_stage}/{self.config.max_episodes_per_stage})"
        
        all_stable , status = self._is_all_stable()
        if all_stable:
            return True, f"All converged: {self._format_status(status)}"
        else:
            return False, f"Not converged: {self._format_status(status)}"
        
        
    
    def advance(self) -> Optional[OpponentStage]:
        """
        Advance to next curriculum stage.
        
        Returns:
            New stage, or None if cannot advance
        """
        should_adv, reason = self.should_advance()
        
        if not should_adv:
            return None
        
        # Record current stage completion
        completion_record = {
            "stage": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
            "episodes_spent": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "reason": reason,
        }
        
        self.advancement_records.append(completion_record)
        
        # Advance
        self.current_stage_idx += 1
        self.episodes_in_stage = 0
        
        
        new_stage = self.current_stage
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"🎓 CURRICULUM ADVANCEMENT: Stage {self.current_stage_idx}")
            print("=" * 60)
            print(f"  Completed: {completion_record['stage']}")
            print(f"  Episodes spent: {completion_record['episodes_spent']}")
            print(f"  Reason: {reason}")
            print(f"  New stage: {new_stage.name}")
            print(f"  New opponent: {new_stage.opponent_type}")
            print(f"  Difficulty: {new_stage.difficulty.name}")
            print(f"  Description: {new_stage.description}")
            print("=" * 60 + "\n")
        
        return new_stage
    
    def create_environment(self, seed: Optional[int] = None)-> UniformPricingEnv:
        """
        Create environment for the current curriculum stage.
        
        Args:
            seed: Random seed
            
        Returns:
            Gym environment configured for current stage
        """
        from env.uniform_pricing_env import make_uniform_pricing_env
        
        # Create environment with current opponent
        env = make_uniform_pricing_env(
            opponent=self.current_stage.opponent_type,
            num_consumers=self.config.num_consumers,
            episode_length=self.config.episode_length,
            seed=seed,
        )
        
      
        self.current_env = env
        self.current_opponent_type = self.current_stage.opponent_type
        
        return env
    
    def get_info(self) -> Dict:
        """Get current curriculum state for logging."""
        all_stable, status = self._is_all_stable() if len(self.critic_losses) >= self.config.window_size else (False, {})
        return {
            "stage_idx": self.current_stage_idx,
            "stage_name": self.current_stage.name,
            "opponent_type": self.current_stage.opponent_type,
            "difficulty": self.current_stage.difficulty.name,
            "difficulty_value": self.current_stage.difficulty.value,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "progress": self.progress,
            "is_last_stage": self.is_last_stage,
            "convergence_status": {k: v for k, v in status.items()},
        }
    
    def get_summary(self) -> Dict:
        """Get curriculum summary after training."""
        return {
            "total_stages_completed": self.current_stage_idx,
            "total_stages": len(self.config.stages),
            "final_stage": self.current_stage.name,
            "final_opponent": self.current_stage.opponent_type,
        }
