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
# 3. ENTRY SIGNAL (RELAXED)
# -------------------------------------------------
entry = (
    (df['momentum_strength'] > 0) |
    (df['close'] > df['ma_5'])
).astype(int)

# -------------------------------------------------
# 4. SOFT REGIME WEIGHT
# trade_signal is now a weight, not a gate
# -------------------------------------------------
regime_weight = np.where(df['trade_signal'] == 1, 1.0, 0.3)

# -------------------------------------------------
# 5. POSITION SIZE (ENTRY × REGIME)
# -------------------------------------------------
base_size = 1.0
position = base_size * entry * regime_weight

# -------------------------------------------------
# 6. TRANSACTION COSTS
# -------------------------------------------------
COST = 0.0015
trade_cost = np.zeros(len(df))

for i in range(1, len(df)):
    if position[i] > 0 and position[i - 1] == 0:
        trade_cost[i] = COST
    elif position[i] == 0 and position[i - 1] > 0:
        trade_cost[i] = COST

# -------------------------------------------------
# 7. STRATEGY RETURNS
# -------------------------------------------------
df['strategy_return'] = position * df['returns'] - trade_cost

# -------------------------------------------------
# 8. EQUITY CURVE
# -------------------------------------------------
df['strategy_equity'] = (1 + df['strategy_return']).cumprod()
df['market_equity'] = (1 + df['returns']).cumprod()

# -------------------------------------------------
# 9. METRICS
# -------------------------------------------------
def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_dd(eq):
    peak = eq.cummax()
    return ((eq - peak) / peak).min()

trades = ((position > 0) & (position.shift(1) == 0)).sum()

print("\nSOFT REGIME STRATEGY BACKTEST")
print("-----------------------------")
print(f"Trades taken     : {trades}")
print(f"Strategy Sharpe  : {sharpe(df['strategy_return']):.2f}")
print(f"Strategy Max DD  : {max_dd(df['strategy_equity']):.2%}")
print(f"Final Strategy Equity : {df['strategy_equity'].iloc[-1]:.2f}")
print(f"Final Market Equity  : {df['market_equity'].iloc[-1]:.2f}")
