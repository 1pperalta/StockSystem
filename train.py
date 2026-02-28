import os
import sys
import pandas as pd
import numpy as np
import torch
from stock_env import StockTradingEnv
from agent import DQNAgent

def load_stock_data(filepath):
    df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
    return df["Close"]["AAPL"].dropna().to_numpy()

def train(episodes=500):
    stock_data = load_stock_data("data/AAPL.csv")
    env = StockTradingEnv(stock_data=stock_data)
    agent = DQNAgent(state_size=8, action_size=5)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store(state, action, reward, next_state, terminated or truncated)
            agent.learn()

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        print(
            f"Episode {episode + 1:>4d} | "
            f"Profit: {info['profit']:>+12.2f} | "
            f"Reward: {total_reward:>+12.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    agent.save("model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()