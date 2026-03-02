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
# 3. SIGNAL
# -------------------------------------------------
signal = df['trade_signal'].values

# -------------------------------------------------
# 4. POSITION SIZING
# -------------------------------------------------
position_size = np.clip(
    0.6
    + 0.2 * df['momentum_strength']
    + 0.2 * df['volatility_5'],
    0.5,
    2.0
)

# -------------------------------------------------
# 5. MULTI-DAY HOLDING (STATE MACHINE)
# -------------------------------------------------
position = np.zeros(len(df))
in_trade = False
entry_count = 0

for i in range(len(df)):
    if signal[i] == 1 and not in_trade:
        in_trade = True
        entry_count += 1
        position[i] = position_size[i]

    elif signal[i] == 1 and in_trade:
        position[i] = position[i - 1]

    elif signal[i] == 0 and in_trade:
        in_trade = False
        position[i] = 0

    else:
        position[i] = 0

df['position'] = position

# -------------------------------------------------
# 6. TRANSACTION COSTS (ENTRY + EXIT)
# -------------------------------------------------
COST = 0.0015
trade_cost = np.zeros(len(df))

for i in range(1, len(df)):
    if position[i] > 0 and position[i - 1] == 0:
        trade_cost[i] = COST
    elif position[i] == 0 and position[i - 1] > 0:
        trade_cost[i] = COST

df['trade_cost'] = trade_cost

# -------------------------------------------------
# 7. STRATEGY RETURNS
# -------------------------------------------------
df['strategy_return'] = position * df['returns'] - trade_cost

# -------------------------------------------------
# 8. EQUITY CURVES
# -------------------------------------------------
df['strategy_equity'] = (1 + df['strategy_return']).cumprod()
df['market_equity'] = (1 + df['returns']).cumprod()

# -------------------------------------------------
# 9. METRICS
# -------------------------------------------------
def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_drawdown(eq):
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return dd.min()

print("\nMULTI-DAY HOLDING STRATEGY BACKTEST (FIXED)")
print("------------------------------------------")
print(f"Trades taken     : {entry_count}")
print(f"Strategy Sharpe  : {sharpe(df['strategy_return']):.2f}")
print(f"Strategy Max DD  : {max_drawdown(df['strategy_equity']):.2%}")
print(f"Market Sharpe    : {sharpe(df['returns']):.2f}")
print(f"Market Max DD    : {max_drawdown(df['market_equity']):.2%}")
print(f"Final Strategy Equity : {df['strategy_equity'].iloc[-1]:.2f}")
print(f"Final Market Equity  : {df['market_equity'].iloc[-1]:.2f}")
