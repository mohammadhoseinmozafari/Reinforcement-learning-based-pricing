from abc import ABC, abstractmethod
from collections import defaultdict
from unittest.mock import Base
import numpy as np
from collections import deque
import random
from typing import Dict, Optional, Tuple

from train.uniform_training.curriculum import Curriculum
class BaseReplayBuffer(ABC):
    @abstractmethod
    def push(self , state, action , reward, next_state, done):
        raise NotImplementedError
    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
  
    
    def push(self, state, action, reward, next_state, done) -> None:
        """Store a transition in the buffer."""     
        self.buffer.append((state, action, reward, next_state, done))

    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        
        return self._sample_uniform(batch_size)
        
        
    
    def _sample_uniform(self, batch_size) :
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
        
            return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def clear(self, retain_fraction: Optional[float] = 0.0) -> None :
        if not retain_fraction or retain_fraction <= 0.0:
            self.buffer.clear()
            return
        keep_count = int(len(self.buffer)*retain_fraction)
        retained = list (self.buffer)[-keep_count:]
        self.buffer.clear()
        self.buffer.extend(retained)
        
    def __len__(self):
        return len(self.buffer)


class CurriculumReplayBuffer(BaseReplayBuffer):
    """
    Maintains one replay buffer per curriculum stage and
    performs mixed sampling to prevent catastrophic forgetting.
    """

    def __init__(self, capacity: int, batch_size: int, curriculum : Curriculum):

        self.batch_size = batch_size
        self.capacity = capacity
        self.buffers = self._create_buffers(curriculum)
        self.sampling_weights = self._create_sampling_weights(curriculum)
       
        self.current_stage = curriculum.opponent_sequence[0].opponent_type

        # self.sampling_weights = {
        #     "premium_uniform": {
        #         "premium_uniform": 1.0,
        #     },
        #     "passive_uniform": {
        #         "premium_uniform": 0.20,
        #         "passive_uniform": 0.80,
        #     },
        #     "aggressive_uniform": {
        #         "premium_uniform": 0.15,
        #         "passive_uniform": 0.15,
        #         "aggressive_uniform": 0.70,
        #     }
        # }
    
    def _create_buffers(self, curriculum: Curriculum)-> Dict[str, ReplayBuffer]:
        stages = curriculum.opponent_sequence
        buffers = {}
        for stage in stages : 
            buffers[stage.opponent_type] = ReplayBuffer(self.capacity)

        return buffers

    def _create_sampling_weights(self, curriculum: Curriculum, primal_weight : float = 0.75)-> Dict[str,Dict[str, float]]:
        
        assert 0.0 < primal_weight <=1.0 , 'primal_weight must be in (0,1]'

        stages = curriculum.opponent_sequence
        
        weights = {}
        
        for current_idx , current_stage in enumerate(stages):
            current_name = current_stage.opponent_type

            if current_idx == 0 :
                weights[current_name] = {
                    current_name: 1.0
                } 
                continue
            current_weights = {}

            current_weights[current_name] = primal_weight
            prev_weight = (1.0 - primal_weight) / current_idx
            for prev_idx in range(current_idx):
                prev_name = stages[prev_idx].opponent_type
                current_weights[prev_name]= prev_weight
            
            weights[current_name] = current_weights
        
        return weights



    def set_stage(self, stage_name: str):
        self.current_stage = stage_name

    def push(self, state, action, reward, next_state, done):
        self.buffers[self.current_stage].push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def __len__(self):
        return sum(len(buf) for buf in self.buffers.values())

    def stage_size(self, stage_name):
        return len(self.buffers[stage_name])

    def sample(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        weights = self.sampling_weights[self.current_stage]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        collected = 0

        for stage_name, weight in weights.items():

            buffer = self.buffers[stage_name]

            if len(buffer) == 0:
                continue

            n_samples = int(batch_size * weight)

            n_samples = min(
                n_samples,
                len(buffer)
            )

            if n_samples == 0:
                continue

            batch = buffer.sample(n_samples)

            states.extend(batch[0])
            actions.extend(batch[1])
            rewards.extend(batch[2])
            next_states.extend(batch[3])
            dones.extend(batch[4])

            collected += n_samples

        # Fill remainder from current stage
        remaining = batch_size - collected

        if remaining > 0:

            current_buffer = self.buffers[self.current_stage]

            if len(current_buffer) >= remaining:

                batch = current_buffer.sample(remaining)

                states.extend(batch[0])
                actions.extend(batch[1])
                rewards.extend(batch[2])
                next_states.extend(batch[3])
                dones.extend(batch[4])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    


class RecencyBiasReplayBuffer(BaseReplayBuffer):
    """Experience Replay Buffer which cares more about the recent experiences."""
    
    def __init__(self, capacity: int, recent_bias: float = 0.3):
        self.buffer = deque(maxlen=capacity)
        self.insertion_order = deque(maxlen=capacity)  # Track insertion time
        self.total_insertions = 0
        self.recent_bias = recent_bias  
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""     
        self.buffer.append((state, action, reward, next_state, done))
        self.insertion_order.append(self.total_insertions)
        self.total_insertions+=1
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        if len(self.buffer)< batch_size:
            return self._sample_uniform(batch_size)
        
        insertion_times = np.array(self.insertion_order)
        max_time = insertion_times.max()
        time_diffs = max_time- insertion_times
        weights = np.exp(-self.recent_bias*time_diffs / len(self.buffer))
        weights = weights/weights.sum()

        indices = np.random.choice(len(self.buffer),
                                   size = batch_size,
                                   p = weights,
                                   replace=False)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def _sample_uniform(self, batch_size) :
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
        
            return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)