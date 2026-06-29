from typing import Any, Dict
from evaluation.config import EvaluationConfig
from evaluation.metrics import EvaluationMetrics
from evaluation.results import EvaluationResult





class Evaluator :

    def __init__(self, config: EvaluationConfig) -> None:
        
        self.config = config
        self.evaluation_metrics = EvaluationMetrics()


    def evaluate(self, agent: Any, env) -> Dict:
        """
        Run the agent deterministically for several episodes.
        Returns average stats.
        """
        

        for episode in range(self.config.num_episodes):
            state, _ = env.reset()
            self.evaluation_metrics.reset_episode()
            episode_reward = 0.0

            for step in range(self.config.episode_length):
                action = agent.select_action(state, deterministic=True)
                

                next_state, reward, terminated, truncated, info = env.step(action)
                
                self.evaluation_metrics.record_step(info)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break
            
            self.evaluation_metrics.end_episode(episode_reward)


        return self.evaluation_metrics.collect_steps()
