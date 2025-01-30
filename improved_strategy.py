"""
improved_strategy.py

An improved version of a regime-based trading strategy using an HMM,
showing how to:
1) Expand features (add RSI)
2) Dynamically label states based on returns & volatility
3) Split data into train/test for out-of-sample evaluation
4) Use simple volatility-based position sizing
"""

import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt

# ============= FEATURE ENGINEERING UTILITIES =============


def compute_rsi(series, period=14):
    """
    Compute RSI (Relative Strength Index) of a price series.
    This is a manual implementation; 
    you could also use a library like TA-Lib if available.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Avoid division by zero
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    Compute various features: daily returns, volatility, RSI, etc.
    """
    # Daily returns
    df['Return'] = df['Adj Close'].pct_change()

    # Rolling volatility (e.g., 10-day std dev)
    df['Volatility'] = df['Return'].rolling(window=10).std()

    # RSI using Adj Close
    df['RSI'] = compute_rsi(df['Adj Close'], period=14)

    # Drop initial NaNs
    df.dropna(inplace=True)
    return df

# ============= HMM TRAINING & STATE PREDICTION =============


def fit_hmm(train_df, n_states=3):
    """
    Fit Hidden Markov Model to the *training* portion of the data.
    Returns:
        model (fitted HMM)
    """
    # We use Return, Volatility, RSI as features
    X_train = train_df[['Return', 'Volatility', 'RSI']].values

    # Build and fit HMM
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type="full",
                            n_iter=100,
                            random_state=42)
    model.fit(X_train)
    return model


def predict_states(model, test_df):
    """
    Use the trained HMM to predict states on *test* (unseen) data.
    """
    X_test = test_df[['Return', 'Volatility', 'RSI']].values
    hidden_states = model.predict(X_test)
    return hidden_states

# ============= REGIME LABELING LOGIC =============


def label_states(df, hidden_states):
    """
    Dynamically label states as Bull, Bear, or Sideways by analyzing
    average return *and* volatility. More advanced methods might also
    consider state persistence, peak/trough, etc.

    Returns a DataFrame with 'State' (int) and 'Regime' (string).
    """
    df = df.copy()
    df['State'] = hidden_states

    # collect the mean return and mean volatility for each state
    state_stats = df.groupby('State').agg(
        mean_return=('Return', 'mean'),
        mean_vol=('Volatility', 'mean')
    )

    # We define a "score" for each state based on:
    #   higher mean_return => more bullish
    #   higher mean_vol => more uncertain
    #
    # Simplistic, but let's give more weight to returns and lightly penalize high volatility.
    # For example: score = mean_return - 0.5 * mean_vol
    state_stats['score'] = state_stats['mean_return'] - \
        0.5 * state_stats['mean_vol']

    # Sort states by score
    sorted_by_score = state_stats.sort_values('score')
    # The lowest score => "Bear", highest => "Bull", middle => "Sideways"
    # If we were to have more than 3 states, we can expand this logic or cluster them.

    # Just to handle the possibility n_states != 3, let's map them in order:
    sorted_state_ids = sorted_by_score.index.tolist()  # from lowest to highest score

    # We'll assume the worst state is Bear, best is Bull, and everything
    # in between is Sideways. If we have more states, they'd also be "Sideways"
    # except the extremes. This is just one approach.

    regime_map = {}
    if len(sorted_state_ids) == 1:
        # Only one state (unlikely but just in case), call it Bull
        regime_map[sorted_state_ids[0]] = 'Bull'
    elif len(sorted_state_ids) == 2:
        regime_map[sorted_state_ids[0]] = 'Bear'
        regime_map[sorted_state_ids[1]] = 'Bull'
    else:
        # first => Bear, last => Bull, everything else => Sideways
        regime_map[sorted_state_ids[0]] = 'Bear'
        for s in sorted_state_ids[1:-1]:
            regime_map[s] = 'Sideways'
        regime_map[sorted_state_ids[-1]] = 'Bull'

    df['Regime'] = df['State'].map(regime_map)
    return df

# ============= POSITION SIZING & TRADING SIGNALS =============


def generate_signals(df):
    """
    Instead of simply +1/-1/0, incorporate a basic volatility-based position sizing.
    For instance, in a Bull regime, we go long with size = 1 / (Volatility * factor).
    In a Bear regime, we go short with that magnitude.
    In Sideways, we might go flat (0).

    This is a simplistic demonstration of how to scale position by volatility.
    """
    df = df.copy()

    # Avoid division by zero; fill NaNs or zeros with small epsilon
    vol = df['Volatility'].replace(0, 1e-9).fillna(1e-9)

    # Example scaling factor: position_size = 0.01 / vol
    # where 0.01 is a baseline daily risk budget, for instance.
    # Then clamp it so we don't exceed, say, +/- 1.5 leverage.
    # (We can tune these as needed.)

    position_size = 0.01 / vol
    position_size = position_size.clip(lower=0, upper=1.5)

    # Now map regime to direction
    #   Bull => + position_size
    #   Bear => - position_size
    #   Sideways => 0
    df['Signal'] = 0.0
    df.loc[df['Regime'] == 'Bull', 'Signal'] = position_size
    df.loc[df['Regime'] == 'Bear', 'Signal'] = -position_size
    df.loc[df['Regime'] == 'Sideways', 'Signal'] = 0.0

    return df

# ============= BACKTESTING UTILITIES =============


def backtest(df, initial_capital=10000):
    """
    Given a DataFrame with columns 'Return' and 'Signal',
    compute the strategy returns and compare with buy-and-hold.
    """
    df = df.copy()

    # Strategy daily return = signal * daily return
    df['Strategy_Return'] = df['Signal'] * df['Return']

    df['Cum_Strategy'] = (1 + df['Strategy_Return']
                          ).cumprod() * initial_capital
    df['Cum_BuyHold'] = (1 + df['Return']).cumprod() * initial_capital

    final_strategy_value = df['Cum_Strategy'].iloc[-1]
    final_buyhold_value = df['Cum_BuyHold'].iloc[-1]

    print(f"Final strategy value: ${final_strategy_value:,.2f}")
    print(f"Final buy-and-hold value: ${final_buyhold_value:,.2f}")

    return df

# ============= MAIN FLOW WITH TRAIN/TEST SPLIT =============


def main():
    # 1. Download Data
    df = download_data(symbol="AAPL", start="2015-01-01", end="2023-01-01")

    # 2. Compute Features
    df = compute_features(df)

    # 3. Train/Test Split (e.g., 80% train, 20% test)
    #    For a more robust approach, use walk-forward (rolling) retraining.
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # 4. Fit HMM on training data
    model = fit_hmm(train_df, n_states=3)

    # 5. Predict states on test data
    hidden_states_test = predict_states(model, test_df)

    # 6. Label states in the test set
    test_df = label_states(test_df, hidden_states_test)

    # 7. Generate signals in the test set
    test_df = generate_signals(test_df)

    # 8. Backtest on the test set
    results_df = backtest(test_df, initial_capital=10000)

    # 9. Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Cum_Strategy'], label='Strategy')
    plt.plot(results_df.index, results_df['Cum_BuyHold'], label='Buy & Hold')
    plt.title('Out-of-Sample Performance (Test Set)')
    plt.legend()
    plt.show()

def train_final_model():
    """
    Train the HMM model on your chosen 'primary' dataset, e.g., AAPL.
    Then return the fitted model (and any parameters you might need).
    """
    # (1) Load primary dataset used for strategy development
    df = yf.download("AAPL", start="2015-01-01", end="2023-01-01")

    # (2) Compute features
    df = compute_features(df)  

    # (3) Fit HMM on the *entire* dataset (no train/test split here),
    #     because we want a "final" model that we won't retrain on future data
    X = df[['Return', 'Volatility', 'RSI']].values
    model = hmm.GaussianHMM(
        n_components=3, covariance_type="full", n_iter=100, random_state=42
    )
    model.fit(X)

    return model

if __name__ == "__main__":
    main()
