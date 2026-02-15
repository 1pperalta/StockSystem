import numpy as np
from stock_env import StockTradingEnv

# Fake stock data: 100 days of prices between 200 and 300
fake_data = np.cumsum(np.random.randn(100)) + 250
fake_data = np.abs(fake_data)

env = StockTradingEnv(stock_data=fake_data, render_mode="human")
obs, info = env.reset(seed=42)

print("Initial observation:", obs)
print("Initial info:", info)
print("-" * 100)

for _ in range(20):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished.")
        break