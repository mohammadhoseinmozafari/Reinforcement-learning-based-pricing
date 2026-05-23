"""
Phase 2.1 — Test Trained Model Against Different Opponent Prices

Load a trained SAC agent and evaluate it against opponents
with prices NOT seen during training (e.g. 1.5, 3.0, 4.5, 5.0).
"""



import numpy as np
import matplotlib.pyplot as plt

from env.uniform_pricing_env import UniformPricingEnv
from env.opponent_policies import ConstantOpponentPolicy
from models.SAC import SAC
from config.constants import EPISODE_LENGTH, PRICE_UNIFORM_MIN, PRICE_UNIFORM_MAX


# ── Settings ──────────────────────────────────────────────────
MODEL_PATH = "experiments/phase2/phase2_1_uniform_training_results/sac_uniform_final.pt"
OPPONENT_PRICES = [1.5, 3.0, 4.5, 5.0]   # prices to test against
NUM_EPISODES = 10                          # episodes per opponent price
EPISODE_LEN = EPISODE_LENGTH
NUM_CONSUMERS = 50
SEED = 42


def make_test_env(opponent_price: float) -> UniformPricingEnv:
    """Create an environment with a fixed-price opponent."""
    opp = ConstantOpponentPolicy(uniform_price=opponent_price, regime=0)
    return UniformPricingEnv(
        opponent_policy=opp,
        num_consumers=NUM_CONSUMERS,
        episode_length=EPISODE_LEN,
        seed=SEED,
    )


def load_agent(model_path: str) -> SAC:
    """Load a trained SAC agent."""
    agent = SAC(
        state_dim=7,
        action_dim=1,
        action_scale=1.0,
        hidden_dim=256,
    )
    agent.load(model_path)
    return agent


def test_agent(agent: SAC, env: UniformPricingEnv, num_episodes: int):
    """
    Run the agent deterministically for several episodes.
    Returns per-step records for the last episode and average stats.
    """
    all_rewards, all_profits, all_prices, all_shares = [], [], [], []

    last_ep_steps = []  # detailed step log of the last episode

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_prices, ep_shares = [], []
        step_log = []

        for t in range(EPISODE_LEN):
            action = agent.select_action(state, deterministic=True)
            action = np.clip(action, 0.0, 1.0)

            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_prices.append(info["price"])
            ep_shares.append(info["market_share"])

            step_log.append({
                "t": t,
                "price": info["price"],
                "profit": info["profit"],
                "market_share": info["market_share"],
                "opp_price": info["opponent_price"],
            })

            state = next_state
            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_profits.append(sum(s["profit"] for s in step_log))
        all_prices.append(np.mean(ep_prices))
        all_shares.append(np.mean(ep_shares))
        last_ep_steps = step_log

    stats = {
        "avg_reward": np.mean(all_rewards),
        "avg_profit": np.mean(all_profits),
        "avg_price": np.mean(all_prices),
        "avg_share": np.mean(all_shares),
    }
    return stats, last_ep_steps


# ── Main ──────────────────────────────────────────────────────
def main():
    agent = load_agent(MODEL_PATH)

    results = {}
    all_step_logs = {}

    print("=" * 65)
    print("PHASE 2.1 — TEST AGAINST UNSEEN OPPONENT PRICES")
    print("=" * 65)

    for opp_price in OPPONENT_PRICES:
        env = make_test_env(opp_price)
        stats, steps = test_agent(agent, env, NUM_EPISODES)
        results[opp_price] = stats
        all_step_logs[opp_price] = steps
        env.close()

        print(f"\nOpponent price = {opp_price:.1f}")
        print(f"  Avg Reward       : {stats['avg_reward']:.2f}")
        print(f"  Avg Total Profit : {stats['avg_profit']:.2f}")
        print(f"  Avg Agent Price  : {stats['avg_price']:.2f}")
        print(f"  Avg Market Share : {stats['avg_share']:.2f}")

    # ── Summary table ────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Opp Price':>10} {'Reward':>10} {'Profit':>10} {'Agent Price':>12} {'Mkt Share':>10}")
    print("-" * 65)
    for opp_p, s in results.items():
        print(f"{opp_p:>10.1f} {s['avg_reward']:>10.2f} {s['avg_profit']:>10.2f} "
              f"{s['avg_price']:>12.2f} {s['avg_share']:>10.2f}")
    print("=" * 65)

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    opp_prices_list = list(results.keys())
    agent_prices = [results[p]["avg_price"] for p in opp_prices_list]
    profits = [results[p]["avg_profit"] for p in opp_prices_list]
    shares = [results[p]["avg_share"] for p in opp_prices_list]

    # 1) Agent price vs Opponent price
    ax = axes[0]
    ax.bar(range(len(opp_prices_list)), agent_prices, tick_label=[str(p) for p in opp_prices_list])
    ax.set_xlabel("Opponent Price")
    ax.set_ylabel("Agent Avg Price")
    ax.set_title("Agent Price vs Opponent Price")
    ax.grid(axis="y", alpha=0.3)

    # 2) Profit vs Opponent price
    ax = axes[1]
    ax.bar(range(len(opp_prices_list)), profits, tick_label=[str(p) for p in opp_prices_list], color="green")
    ax.set_xlabel("Opponent Price")
    ax.set_ylabel("Avg Total Profit")
    ax.set_title("Profit vs Opponent Price")
    ax.grid(axis="y", alpha=0.3)

    # 3) Market share vs Opponent price
    ax = axes[2]
    ax.bar(range(len(opp_prices_list)), shares, tick_label=[str(p) for p in opp_prices_list], color="orange")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Opponent Price")
    ax.set_ylabel("Avg Market Share")
    ax.set_title("Market Share vs Opponent Price")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Trained Agent vs Unseen Opponent Prices", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("experiments/phase2/phase2_1_test_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to experiments/phase2/phase2_1_test_results.png")


if __name__ == "__main__":
    main()
