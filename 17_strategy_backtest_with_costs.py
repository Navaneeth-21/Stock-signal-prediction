import pandas as pd
import numpy as np

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/signal_test_ml.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# -------------------------------------------------
# 2. RETURNS
# -------------------------------------------------
df['returns'] = df['close'].pct_change()
df = df.dropna().reset_index(drop=True)

# -------------------------------------------------
# 3. SIGNAL (ALREADY CREATED EARLIER)
# -------------------------------------------------
df['signal'] = df['trade_signal']

# -------------------------------------------------
# 4. POSITION SIZING
# -------------------------------------------------
df['position_size'] = np.clip(
    0.6
    + 0.2 * df['momentum_strength']
    + 0.2 * df['volatility_5'],
    0.5,
    2.0
)

# -------------------------------------------------
# 5. TRANSACTION COSTS
# -------------------------------------------------
COST_PER_TRADE = 0.0015   # 0.15% round-trip

df['trade_cost'] = df['signal'] * COST_PER_TRADE

# -------------------------------------------------
# 6. STRATEGY RETURNS (AFTER COSTS)
# -------------------------------------------------
df['strategy_return'] = (
    df['signal']
    * df['position_size']
    * df['returns']
    - df['trade_cost']
)

# -------------------------------------------------
# 7. EQUITY CURVES
# -------------------------------------------------
df['strategy_equity'] = (1 + df['strategy_return']).cumprod()
df['market_equity'] = (1 + df['returns']).cumprod()

# -------------------------------------------------
# 8. METRICS
# -------------------------------------------------
def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_drawdown(eq):
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return dd.min()

print("\nFINAL STRATEGY BACKTEST (WITH COSTS)")
print("-----------------------------------")
print(f"Trades taken     : {df['signal'].sum()}")
print(f"Strategy Sharpe  : {sharpe(df['strategy_return']):.2f}")
print(f"Strategy Max DD  : {max_drawdown(df['strategy_equity']):.2%}")
print(f"Market Sharpe    : {sharpe(df['returns']):.2f}")
print(f"Market Max DD    : {max_drawdown(df['market_equity']):.2%}")
print(f"Final Strategy Equity : {df['strategy_equity'].iloc[-1]:.2f}")
print(f"Final Market Equity  : {df['market_equity'].iloc[-1]:.2f}")
