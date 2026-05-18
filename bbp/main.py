from env import make_uniform_pricing_env

if __name__ == "__main__":
    # Quick test
    print("Testing UniformPricingEnv...")
    
    env = make_uniform_pricing_env("reactive_uniform")
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: price={info['price']:.2f}, profit={reward:.2f}, share={info['market_share']:.2f}")
    
    print(f"\nTotal reward (10 steps): {total_reward:.2f}")
    print("Test passed!")