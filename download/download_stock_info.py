import os

import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data",
) -> pd.DataFrame:
    stock = yf.download(ticker, start=start_date, end=end_date)

    if stock.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{ticker}.csv")
    stock.to_csv(filepath)

    print(f"Downloaded {len(stock)} rows for {ticker}")
    print(f"Date range: {stock.index[0].date()} to {stock.index[-1].date()}")
    print(f"Saved to: {filepath}")

    return stock


if __name__ == "__main__":
    tickers = ["AAPL", "BTC-USD", "ETH-USD", "SOL-USD", "SCHD"]

    for ticker in tickers:
        download_stock_data(
            ticker=ticker,
            start_date="2020-01-01",
            end_date="2025-12-31",
        )
        print()