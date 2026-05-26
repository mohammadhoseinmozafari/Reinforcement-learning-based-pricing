import numpy as np
from train.curriculum import CurriculumConfig
from train.uniform_training.curriculum import UniformPricingCurriculum
from train.uniform_training.uniform_curriculum_training import train_with_curriculum
from train.uniform_training.uniform_training import TrainingConfig, plot_training_results, train_uniform_pricing, plot_training_results
# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 2.1 training."""
    # Create configuration
    config = TrainingConfig(
        num_episodes=800,
        warmup_steps=300,
        eval_freq=10,
        save_freq=100,
        opponent_type="premium_uniform",  
        seed=63,
    )
    curr_config = CurriculumConfig(
    stages= UniformPricingCurriculum().OPPONENT_SEQUENCE,
    monitor_critic=False,
    monitor_actor=False,
    window_size=20,
    change_threshold=0.03,
    min_episodes_per_stage=50,
    
)
    
    # Train agent
    agent, metrics = train_with_curriculum(config=config, curriculum_config=curr_config, verbose=True)

    
    # Plot results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Final statistics
    print(f"\nFinal 50 episodes:")
    print(f"  Avg Reward: {np.mean(metrics.episode_rewards[-50:]):.2f}")
    print(f"  Avg Price: {np.mean(metrics.episode_prices[-50:]):.2f}")
    print(f"  Avg Market Share: {np.mean(metrics.episode_market_shares[-50:]):.2f}")
    
    # Plot
    plot_training_results(metrics, config)
    
    print(f"\nResults saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
