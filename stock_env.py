from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StockTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        stock_data: np.ndarray,
        initial_capital: float = 1_000_000.0,
        window_size: int = 6,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.stock_data = stock_data
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.window_size + 2,),
            dtype=np.float64,
        )

        self.capital = 0.0
        self.num_shares = 0.0
        self.current_step = 0
        self.previous_total_value = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        max_start = len(self.stock_data) - self.window_size - 1
        self.current_step = self.np_random.integers(self.window_size - 1, max_start)

        self.capital = self.initial_capital
        self.num_shares = 0.0
        self.previous_total_value = self.initial_capital

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        current_price = self.stock_data[self.current_step]

        self._execute_action(action, current_price)

        self.current_step += 1
        new_price = self.stock_data[self.current_step]

        invested_value = self.num_shares * new_price
        total_value = self.capital + invested_value

        reward = total_value - self.previous_total_value
        self.previous_total_value = total_value

        terminated = self.current_step >= len(self.stock_data) - 1
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _execute_action(self, action: int, price: float):
        if action == 0:
            self.capital += self.num_shares * price
            self.num_shares = 0.0

        elif action == 1:
            shares_to_sell = self.num_shares / 2.0
            self.capital += shares_to_sell * price
            self.num_shares -= shares_to_sell

        elif action == 2:
            pass

        elif action == 3:
            amount = self.capital / 2.0
            self.num_shares += amount / price
            self.capital -= amount

        elif action == 4:
            self.num_shares += self.capital / price
            self.capital = 0.0

    def render(self):
        price = self.stock_data[self.current_step]
        invested = self.num_shares * price
        total = self.capital + invested
        profit = total - self.initial_capital

        print(
            f"Day {self.current_step:>4d} | "
            f"Price: {price:>10.2f} | "
            f"Shares: {self.num_shares:>10.2f} | "
            f"Invested: {invested:>12.2f} | "
            f"Capital: {self.capital:>12.2f} | "
            f"Total: {total:>12.2f} | "
            f"Profit: {profit:>+12.2f}"
        )

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        prices = self.stock_data[start:end]

        invested_value = self.num_shares * self.stock_data[self.current_step]

        return np.array(
            [*prices, invested_value, self.capital],
            dtype=np.float64,
        )

    def _get_info(self) -> dict:
        price = self.stock_data[self.current_step]
        invested = self.num_shares * price
        total = self.capital + invested

        return {
            "total_value": total,
            "profit": total - self.initial_capital,
            "num_shares": self.num_shares,
            "stock_price": price,
        }