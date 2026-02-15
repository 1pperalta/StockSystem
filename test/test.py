import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from stock_env import StockTradingEnv


def load_stock_data(filepath: str):
    df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
    return df["Close"]["AAPL"].dropna().to_numpy()


def main():
    stock_data = load_stock_data(os.path.join(os.path.dirname(__file__), "..", "data", "AAPL.csv"))
    print(f"Loaded {len(stock_data)} days of AAPL data")
    print(f"Price range: ${stock_data.min():.2f} - ${stock_data.max():.2f}")
    print()

    env = StockTradingEnv(stock_data=stock_data, render_mode="human")
    obs, info = env.reset(seed=42)

    action_names = ["Sell all", "Sell half", "Hold", "Buy half", "Buy all"]

    print(f"Observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print("-" * 110)

    total_reward = 0.0

    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"  Action: {action_names[action]:<15} | Reward: {reward:>+12.2f}")
        print()

        if terminated or truncated:
            print("Episode finished.")
            break

    print("-" * 110)
    print(f"Total reward over {step + 1} steps: {total_reward:>+.2f}")
    print(f"Final profit: {info['profit']:>+.2f}")


if __name__ == "__main__":
    main()