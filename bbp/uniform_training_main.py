import numpy as np
from train.uniform_training.uniform_training import TrainingConfig, plot_training_results, train_uniform_pricing, plot_training_results
# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase 2.1 training."""
    # Create configuration
    config = TrainingConfig(
        num_episodes=1000,
        warmup_steps=1000,
        eval_freq=10,
        save_freq=100,
        opponent_type="premium_uniform",  
        seed=42,
    )
    
    # Train agent
    agent, metrics = train_uniform_pricing(config, verbose=True)
    
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
