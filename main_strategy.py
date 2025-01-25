"""
main_strategy.py

A Python script demonstrating a regime-based trading strategy using
a Hidden Markov Model (HMM) and Yahoo Finance data.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt


def download_data(symbol="AAPL", start="2015-01-01", end="2023-01-01"):
    """
    Download data from Yahoo Finance for a given symbol and date range.
    Returns a pandas DataFrame.
    """
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    return df


def compute_features(df):
    """
    Compute features to feed into HMM.
    Features could be daily returns, volatility, etc.
    """
    # Daily returns
    df['Return'] = df['Adj Close'].pct_change()
    # Rolling volatility (e.g., 5-day standard deviation)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df.dropna(inplace=True)  # HMM can't handle NaNs
    return df


def fit_hmm(df, n_states=3):
    """
    Fit Hidden Markov Model to the data using selected features.
    Returns the fitted model and the state predictions.
    """
    # Prepare feature matrix
    # We use [Return, Volatility] as features
    X = df[['Return', 'Volatility']].values

    # Build and fit HMM
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)

    # Predict regime states
    hidden_states = model.predict(X)
    df['State'] = hidden_states
    return model, df


def generate_trading_signals(df):
    """
    Map each detected state to a 'Bull', 'Bear', or 'Sideways' regime,
    then generate buy/sell signals based on the regime.

    In a real strategy, you might do this mapping dynamically based on
    state means, or incorporate more advanced logic.
    """
    # Example simple approach:
    # 1) Group by state, look at average return for each state
    state_summary = df.groupby('State')['Return'].mean().sort_values()
    # The state with the lowest mean return is "Bear"
    # The one with the highest is "Bull"
    # The one in the middle is "Sideways"
    sorted_states = state_summary.index.tolist()  # from lowest to highest

    # Create a dictionary to map state -> regime
    regime_map = {}
    regime_map[sorted_states[0]] = 'Bear'
    regime_map[sorted_states[1]] = 'Sideways'
    regime_map[sorted_states[2]] = 'Bull'

    df['Regime'] = df['State'].map(regime_map)

    # Generate signals:
    # - Bull  -> +1 (long)
    # - Bear  -> -1 (short)
    # - Sideways -> 0 (flat)
    df['Signal'] = df['Regime'].map({'Bull': 1, 'Bear': -1, 'Sideways': 0})
    return df


def backtest_strategy(df, initial_capital=10000):
    """
    A simplified backtest:
    1. Assume we can only be fully long, fully short, or flat each day.
    2. The position each day is determined by the 'Signal'.
    3. Calculate the daily PnL (mark-to-market).
    """
    # Use daily return of the stock multiplied by the signal as PnL factor
    df['Strategy_Return'] = df['Signal'] * df['Return']

    # Cumulative returns
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Cum_Buy_and_Hold'] = (1 + df['Return']).cumprod()

    # Final values
    final_strategy_value = initial_capital * df['Cum_Strategy_Return'].iloc[-1]
    final_buyhold_value = initial_capital * df['Cum_Buy_and_Hold'].iloc[-1]

    print(f"Final strategy value: ${final_strategy_value:,.2f}")
    print(f"Final buy-and-hold value: ${final_buyhold_value:,.2f}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cum_Strategy_Return'], label='Strategy')
    plt.plot(df.index, df['Cum_Buy_and_Hold'], label='Buy & Hold')
    plt.title('Strategy Performance vs Buy & Hold')
    plt.legend()
    plt.show()


def main():
    # 1. Download Data
    df = download_data(symbol="AAPL", start="2015-01-01", end="2023-01-01")

    # 2. Compute Features
    df = compute_features(df)

    # 3. Fit HMM and get states
    model, df = fit_hmm(df, n_states=3)

    # 4. Generate Trading Signals
    df = generate_trading_signals(df)

    # 5. Backtest
    backtest_strategy(df, initial_capital=10000)


if __name__ == "__main__":
    main()
