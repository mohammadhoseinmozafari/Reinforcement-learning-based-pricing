"""
Training example for PettingZoo Hotelling hierarchical environment.

Demonstrates hierarchical RL setup compatible with Stable-Baselines3.
Shows how to use the environment with the hierarchical action/observation structure.
"""

import numpy as np
from env.hotelling_env import HotellingDuopolyEnv
from config.constants import AGENT_IDS, EPISODE_LENGTH, NUM_CONSUMERS


def test_environment():
    """Test environment with random actions."""
    print("=" * 80)
    print("Testing PettingZoo Hotelling Duopoly Environment (Hierarchical RL)")
    print("=" * 80)

    # Create environment
    env = HotellingDuopolyEnv(
        num_consumers=NUM_CONSUMERS,
        episode_length=EPISODE_LENGTH,
        strategy_cycle=10,
        seed=42,
    )

    print(f"\nEnvironment Configuration:")
    print(f"  Agents: {AGENT_IDS}")
    print(f"  Consumers: {NUM_CONSUMERS}")
    print(f"  Episode Length: {EPISODE_LENGTH}")
    print(f"  Strategy Cycle: 10 steps")

    # Print action and observation spaces
    print(f"\nAction Spaces:")
    for agent in AGENT_IDS:
        print(f"  {agent}: {env.action_spaces[agent]}")

    print(f"\nObservation Spaces:")
    for agent in AGENT_IDS:
        print(f"  {agent}:")
        obs_space = env.observation_spaces[agent]
        for key, space in obs_space.spaces.items():
            print(f"    {key}: {space}")

    # Reset environment
    observations, infos = env.reset(seed=42)

    print(f"\n" + "=" * 80)
    print("Running Episode with Random Actions")
    print("=" * 80)

    episode_rewards = {agent: 0.0 for agent in AGENT_IDS}
    episode_steps = 0

    done = False
    while not done:
        episode_steps += 1

        # Random actions for each agent
        actions = {}
        for agent in AGENT_IDS:
            actions[agent] = {
                "strategy": env.action_spaces[agent]["strategy"].sample(),
                "pricing": env.action_spaces[agent]["pricing"].sample(),
            }

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent in AGENT_IDS:
            episode_rewards[agent] += rewards[agent]

        # Check if episode is done
        done = any(truncations.values())

        # Print progress
        if episode_steps % 50 == 0:
            print(f"\nStep {episode_steps}:")
            for agent in AGENT_IDS:
                print(
                    f"  {agent}: Reward={rewards[agent]:.4f}, "
                    f"Market Share={infos[agent]['market_share']:.4f}"
                )

    print(f"\n" + "=" * 80)
    print("Episode Summary")
    print("=" * 80)
    print(f"Total Steps: {episode_steps}")
    for agent in AGENT_IDS:
        avg_reward = episode_rewards[agent] / episode_steps
        total_reward = episode_rewards[agent]
        print(f"{agent}:")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")

    env.close()
    print(f"\n✓ Test completed successfully")


def example_hierarchical_structure():
    """Show the hierarchical observation/action structure."""
    print("\n" + "=" * 80)
    print("Hierarchical Structure Example")
    print("=" * 80)

    env = HotellingDuopolyEnv(
        num_consumers=NUM_CONSUMERS,
        episode_length=20,
        strategy_cycle=10,
        seed=42,
    )

    observations, _ = env.reset(seed=42)

    print("\nHierarchical Action Structure:")
    print("  actions[agent_id] = {")
    print("    'strategy': int (0 or 1),          # Uniform vs BBP")
    print("    'pricing': np.array([u, p_n, p_o]) # Normalized prices [0,1]")
    print("  }")

    print("\nHierarchical Observation Structure:")
    obs = observations["firm_0"]
    print("  observations[agent_id] = {")
    print("    'strategy_controller': {  # For regime selection")
    for key, val in obs["strategy_controller"].items():
        if isinstance(val, np.ndarray):
            print(f"      '{key}': shape {val.shape}")
        else:
            print(f"      '{key}': int")
    print("    },")
    print("    'pricing_controller': {   # For price optimization")
    for key, val in obs["pricing_controller"].items():
        if isinstance(val, np.ndarray):
            print(f"      '{key}': shape {val.shape}")
        else:
            print(f"      '{key}': int")
    print("    }")
    print("  }")

    env.close()


def example_regime_switching():
    """Demonstrate regime switching over time."""
    print("\n" + "=" * 80)
    print("Regime Switching Example (First 30 Steps)")
    print("=" * 80)

    env = HotellingDuopolyEnv(
        num_consumers=30,
        episode_length=100,
        strategy_cycle=10,  # Strategy acts every 10 steps
        seed=42,
    )

    observations, _ = env.reset(seed=42)

    print("\nStep | Regime 0 | Regime 1 | Profit 0 | Profit 1")
    print("-" * 50)

    for step in range(30):
        actions = {
            agent: {
                "strategy": 1 if step < 10 else 0,  # Firm 0: BBP for first 10 steps
                "pricing": np.array([0.5, 0.4, 0.6], dtype=np.float32),
            }
            for agent in AGENT_IDS
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)

        regime_0 = env.regimes["firm_0"]
        regime_1 = env.regimes["firm_1"]
        
        regime_names = {0: "UNIFORM", 1: "BBP"}

        if (step + 1) % 5 == 0 or step == 0:
            print(f"{step+1:4d} | {regime_names[regime_0]:8s} | {regime_names[regime_1]:8s} | "
                  f"{rewards['firm_0']:8.2f} | {rewards['firm_1']:8.2f}")

    env.close()


def example_observation_details():
    """Show detailed observation values."""
    print("\n" + "=" * 80)
    print("Observation Details Example")
    print("=" * 80)

    env = HotellingDuopolyEnv(
        num_consumers=50,
        episode_length=100,
        strategy_cycle=10,
        seed=42,
    )

    observations, _ = env.reset(seed=42)

    # Take a few steps
    for _ in range(5):
        actions = {
            agent: {
                "strategy": np.random.randint(0, 2),
                "pricing": np.random.rand(3).astype(np.float32),
            }
            for agent in AGENT_IDS
        }
        observations, _, _, _, _ = env.step(actions)

    # Display detailed obs
    agent = "firm_0"
    obs = observations[agent]

    print(f"\nDetailed Observations for {agent}:")
    print("\n--- Strategy Controller Observations ---")
    for key, val in obs["strategy_controller"].items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.flatten()} (shape: {val.shape})")
        else:
            print(f"  {key}: {val}")

    print("\n--- Pricing Controller Observations ---")
    for key, val in obs["pricing_controller"].items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.flatten()} (shape: {val.shape})")
        else:
            print(f"  {key}: {val}")

    env.close()


if __name__ == "__main__":
    test_environment()
    example_hierarchical_structure()
    example_regime_switching()
    example_observation_details()

    print("\n" + "=" * 80)
    print("✓ All examples completed successfully")
    print("=" * 80)
