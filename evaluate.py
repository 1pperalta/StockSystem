import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_env import StockTradingEnv
from agent import DQNAgent

TICKERS = ["AAPL", "BTC-USD", "ETH-USD", "SOL-USD", "SCHD"]


def load_stock_data(ticker: str) -> np.ndarray:
    df = pd.read_csv(f"data/{ticker}.csv", header=[0, 1], index_col=0, parse_dates=True)
    return df["Close"][ticker].dropna().to_numpy()


def buy_and_hold(stock_data: np.ndarray, start: int, end: int, initial_capital: float) -> np.ndarray:
    shares = initial_capital / stock_data[start]
    return np.array([shares * stock_data[i] for i in range(start, end + 1)])


def profit_factor(portfolio_values: list) -> float:
    daily_returns = np.diff(portfolio_values)
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    if losses == 0:
        return float("inf")
    return gains / losses


def max_drawdown(portfolio_values: list) -> float:
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    return drawdowns.max()


def sharpe_ratio(portfolio_values: list, periods_per_year: int = 252) -> float:
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean = np.mean(daily_returns)
    std = np.std(daily_returns)
    if std == 0:
        return 0.0
    return (mean / std) * np.sqrt(periods_per_year)


def evaluate(ticker: str, seed: int = 0):
    stock_data = load_stock_data(ticker)

    env = StockTradingEnv(stock_data=stock_data, max_episode_steps=180)
    agent = DQNAgent(state_size=7, action_size=5)
    agent.load(f"model_{ticker}.pth")
    agent.epsilon = 0.0

    state, info = env.reset(seed=seed)
    start_step = env.current_step

    prices = []
    portfolio_values = []
    actions_taken = []

    action_names = ["Sell all", "Sell half", "Hold", "Buy half", "Buy all"]

    while True:
        prices.append(stock_data[env.current_step])
        portfolio_values.append(info["total_value"])

        action = agent.select_action(state)
        actions_taken.append(action)

        state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            prices.append(stock_data[env.current_step])
            portfolio_values.append(info["total_value"])
            break

    end_step = env.current_step
    days = list(range(len(prices)))
    baseline = buy_and_hold(stock_data, start_step, end_step, env.initial_capital)

    pf = profit_factor(portfolio_values)
    mdd = max_drawdown(portfolio_values)
    sr = sharpe_ratio(portfolio_values)

    print(f"[{ticker}] Steps traded:    {len(actions_taken)}")
    print(f"[{ticker}] Final profit:    {info['profit']:>+.2f}")
    print(f"[{ticker}] Buy-hold profit: {baseline[-1] - env.initial_capital:>+.2f}")
    print(f"[{ticker}] Profit Factor:   {pf:.2f}")
    print(f"[{ticker}] Max Drawdown:    {mdd * 100:.2f}%")
    print(f"[{ticker}] Sharpe Ratio:    {sr:.2f}")
    print()
    for name in action_names:
        count = sum(1 for a in actions_taken if action_names[a] == name)
        print(f"  {name:<30} {count} times")
    print()

    buy_days = [i for i, a in enumerate(actions_taken) if a == 4]
    sell_days = [i for i, a in enumerate(actions_taken) if a == 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(days, prices, color="steelblue", linewidth=1.2, label=f"{ticker} price")
    ax1.scatter(buy_days, [prices[d] for d in buy_days], marker="^", color="green", zorder=5, label="Buy all")
    ax1.scatter(sell_days, [prices[d] for d in sell_days], marker="v", color="red", zorder=5, label="Sell all")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.set_title(f"Agent Decisions on {ticker}")

    ax2.plot(days, portfolio_values, color="darkorange", linewidth=1.2, label="Agent portfolio")
    ax2.plot(days, baseline, color="gray", linewidth=1.0, linestyle="--", label="Buy and hold")
    ax2.axhline(env.initial_capital, color="black", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Portfolio Value (USD)")
    ax2.set_xlabel("Trading Days")
    ax2.legend()

    plt.tight_layout()
    chart_path = f"images/evaluation_{ticker}.png"
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"Chart saved to {chart_path}\n")


if __name__ == "__main__":
    for ticker in TICKERS:
        evaluate(ticker)