import pandas as pd
import numpy as np
from stock_env import StockTradingEnv
from agent import DQNAgent

TICKERS = ["AAPL", "BTC-USD", "ETH-USD", "SOL-USD", "SCHD"]


def load_stock_data(ticker: str) -> np.ndarray:
    df = pd.read_csv(f"data/{ticker}.csv", header=[0, 1], index_col=0, parse_dates=True)
    return df["Close"][ticker].dropna().to_numpy()


def train(ticker: str, episodes: int = 2000):
    stock_data = load_stock_data(ticker)
    env = StockTradingEnv(stock_data=stock_data)
    agent = DQNAgent(state_size=7, action_size=5)

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
            f"[{ticker}] Episode {episode + 1:>4d} | "
            f"Profit: {info['profit']:>+12.2f} | "
            f"Reward: {total_reward:>+12.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    model_path = f"model_{ticker}.pth"
    agent.save(model_path)
    print(f"Training complete. Model saved to {model_path}\n")


if __name__ == "__main__":
    for ticker in TICKERS:
        train(ticker)