# StockSystem

Reinforcement Learning environment for stock trading. Built as a custom Gymnasium environment where a DQN agent learns to buy, sell, or hold stocks to maximize portfolio profit.

## Setup

```bash
uv sync
```

## Usage

Download stock data:

```bash
uv run python download/download_stock_info.py
```

Test the environment with random actions:

```bash
uv run python test/test.py
```

## Project Structure

```
stock_env.py                  - Custom Gymnasium environment
download/download_stock_info.py - Downloads historical stock data from Yahoo Finance
test/test.py                  - Environment test with random actions
data/                         - Stock price CSV files
```

## Environment

- **State:** 6 historical prices + invested value + available capital (8 floats)
- **Actions:** Sell all, Sell half, Hold, Buy half, Buy all
- **Reward:** Change in total portfolio value per step
