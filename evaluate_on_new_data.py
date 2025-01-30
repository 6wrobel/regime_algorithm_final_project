import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from improved_strategy import (
    train_final_model,
    compute_rsi,
    label_states,
    generate_signals,
    backtest
)

def detect_frequency(file_name: str):
    """
    A quick way to guess if the CSV is weekly or monthly based on file name conventions.
    
    - If "copper" is in the file name, treat it as weekly (since we know this is a special case).
    - If the file name ends with '_w.csv', treat as weekly.
    - If the file name ends with '_m.csv', treat as monthly.
    - Otherwise, default to daily.
    """
    lower_file = file_name.lower()
    
    # Special check for copper
    if "copper" in lower_file:
        return 'weekly'
    elif lower_file.endswith('_w.csv'):
        return 'weekly'
    elif lower_file.endswith('_m.csv'):
        return 'monthly'
    else:
        return 'daily'


def load_csv_as_df(file_path: str):
    file_name = os.path.basename(file_path).lower()
    
    # If it's copper, use dayfirst=True
    if "copper" in file_name:
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    else:
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_features_for_candle_data(df):
    """
    Similar to compute_features() in improved_strategy, but tailored to
    our CSV (which has columns: Date, Open, Max, Min, Close, Volume).
    We'll compute Return (Close-based), rolling Volatility, and RSI on Close.
    """
    df = df.copy()
    # Daily (weekly/monthly) return based on Close
    df['Return'] = df['Close'].pct_change()
    
    # Rolling volatility (window=10 by default)
    df['Volatility'] = df['Return'].rolling(window=10).std()
    
    # RSI on Close
    df['RSI'] = compute_rsi(df['Close'], period=14)
    
    df.dropna(inplace=True)
    return df

def compute_performance_metrics(df, frequency='daily'):
    """
    Compute extra metrics like:
      - 'Alpha' = (Annualized Strategy Return) - (Annualized Underlying Return)
      - Win Rate = % of bars with Strategy_Return > 0
    Different frequencies => different annualization factors:
      daily ~ 252, weekly ~ 52, monthly ~ 12
    """
    # Make a copy
    df = df.copy()
    
    # Strategy & underlying daily (or bar) returns
    strat_returns = df['Strategy_Return']
    underlying_returns = df['Return']
    
    # Choose an annualization factor
    if frequency == 'daily':
        ann_factor = 252
    elif frequency == 'weekly':
        ann_factor = 52
    elif frequency == 'monthly':
        ann_factor = 12
    else:
        # fallback (or raise an error)
        ann_factor = 252  # default to daily
    
    # Cumulative returns over the entire series
    # (1 + r).prod() - 1 => total growth - 1 => net growth
    total_strat_return = (1 + strat_returns).prod() - 1
    total_underlying_return = (1 + underlying_returns).prod() - 1
    
    # Convert total returns to annualized returns
    # Approx years in the dataset:
    num_bars = len(df)
    years_approx = num_bars / ann_factor
    
    # If you prefer a direct approach:
    # annualized_return = (1 + total_return)^(ann_factor/num_bars) - 1
    # but that assumes each bar is '1 day' or '1 week'.
    # We'll keep it simpler here:
    
    strat_annual_return = (1 + total_strat_return) ** (1/years_approx) - 1
    underlying_annual_return = (1 + total_underlying_return) ** (1/years_approx) - 1
    
    # 'Alpha' in a simplistic sense: difference in annualized returns
    alpha = strat_annual_return - underlying_annual_return
    
    # Win rate: fraction of bars with strategy_return > 0
    win_rate = (strat_returns > 0).mean()
    
    return {
        'num_bars': num_bars,
        'frequency': frequency,
        'strategy_annual_return': strat_annual_return,
        'underlying_annual_return': underlying_annual_return,
        'alpha': alpha,
        'win_rate': win_rate
    }

def evaluate_new_datasets(csv_files):
    """
    1. Train the final HMM on your primary dev data (e.g., Apple) 
       using train_final_model() from improved_strategy.
    2. For each CSV in csv_files:
       - load via load_csv_as_df()
       - compute features
       - use the final model to predict states (no retraining!)
       - label states, generate signals
       - backtest
       - compute additional performance metrics
       - show/plot results
    """
    # 1) get final, pre-trained model from your dev dataset
    model = train_final_model()  
    
    results_summary = []

    for file_path in csv_files:
        print(f"\n=== Evaluating {file_path} ===")
        
        # detect weekly vs. monthly for annualization
        freq = detect_frequency(file_path)
        
        # load the CSV
        df_raw = load_csv_as_df(file_path)
        
        # compute features
        df_feat = compute_features_for_candle_data(df_raw)
        
        # predict states using pre-trained model
        X_test = df_feat[['Return', 'Volatility', 'RSI']].values
        hidden_states = model.predict(X_test)
        
        # label states (Bull/Bear/Sideways)
        df_feat = label_states(df_feat, hidden_states)
        
        # generate signals
        df_feat = generate_signals(df_feat)
        
        # backtest
        df_backtest = backtest(df_feat, initial_capital=10000)
        
        # compute additional metrics
        perf_metrics = compute_performance_metrics(df_backtest, frequency=freq)
        
        # final values
        final_strat_val = df_backtest['Cum_Strategy'].iloc[-1]
        final_buyhold_val = df_backtest['Cum_BuyHold'].iloc[-1]
        
        # compile results
        result = {
            'file': file_path,
            'final_strategy_value': final_strat_val,
            'final_buyhold_value': final_buyhold_val
        }
        result.update(perf_metrics)
        
        results_summary.append(result)
        
        # print them
        for k,v in result.items():
            if isinstance(v, float):
                print(f"{k:30s}: {v:.4f}")
            else:
                print(f"{k:30s}: {v}")
        
        # optional: plot
        plt.figure(figsize=(10, 5))
        plt.plot(df_backtest.index, df_backtest['Cum_Strategy'], label='Strategy')
        plt.plot(df_backtest.index, df_backtest['Cum_BuyHold'], label='Buy & Hold')
        plt.title(os.path.basename(file_path))
        plt.legend()
        plt.show()

    return results_summary

def main():
    csv_files = [
        "zw=f_copper.csv",  # daily format is DD.MM.YYYY -> dayfirst=True
        "usdjpy_w.csv",     # weekly data, presumably MM/DD/YYYY
        "ge_us_m.csv",      # monthly data, presumably MM/DD/YYYY
        "btc_v_w.csv",      # weekly
        "wig20_w.csv"       # weekly
    ]
    
    summary = evaluate_new_datasets(csv_files)
    print("\nALL RESULTS SUMMARY:")
    for s in summary:
        print(s)

if __name__ == "__main__":
    main()
