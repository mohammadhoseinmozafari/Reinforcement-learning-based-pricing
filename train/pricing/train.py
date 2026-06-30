from train import CurriculumConfig, CurriculumTrainer
from train.config import TrainingConfig
from train.pricing import PricingCurriculum
import numpy as np

training_config = TrainingConfig(
    num_episodes= 800,
    warmup_steps= 5000,
    eval_freq= 10,
    save_freq= 100
)

curriculum = PricingCurriculum ()

curriculum_config = CurriculumConfig(

    curriculum= curriculum,
    stages = curriculum.opponent_sequence,
    window_size= 50,
    change_threshold= 0.05,
    min_episodes_per_stage= 50,
    max_episodes_per_stage= 100
)


trainer  = CurriculumTrainer(
    training_config,
    curriculum_config
)

agent , metrics = trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
    
print(f"\nFinal 50 episodes:")
print(f"  Avg Reward: {np.mean(metrics.episode_rewards[-50:]):.2f}")
print(f"  Avg Price: {np.mean(metrics.episode_prices[-50:]):.2f}")
print(f"  Avg Market Share: {np.mean(metrics.episode_market_shares[-50:]):.2f}")
    
print(f"\nResults saved to: {training_config.save_dir}")

