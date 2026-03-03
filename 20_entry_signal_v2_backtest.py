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
# 3. REGIME FILTER
# -------------------------------------------------
regime = df['trade_signal'].values

# -------------------------------------------------
# 4. ENTRY SIGNAL v2 (RELAXED, CONTINUATION)
# -------------------------------------------------
entry = (
    (df['momentum_strength'] > 0.5) &
    (df['close'] > df['ma_5'])
).astype(int).values

# -------------------------------------------------
# 5. POSITION SIZING
# -------------------------------------------------
position_size = np.clip(
    0.6
    + 0.2 * df['momentum_strength']
    + 0.2 * df['volatility_5'],
    0.5,
    2.0
)

# -------------------------------------------------
# 6. MULTI-DAY HOLDING
# -------------------------------------------------
position = np.zeros(len(df))
in_trade = False
entries = 0

for i in range(len(df)):
    if not in_trade and regime[i] == 1 and entry[i] == 1:
        in_trade = True
        position[i] = position_size[i]
        entries += 1
    elif in_trade and regime[i] == 1:
        position[i] = position[i - 1]
    else:
        in_trade = False
        position[i] = 0

# -------------------------------------------------
# 7. TRANSACTION COSTS
# -------------------------------------------------
COST = 0.0015
trade_cost = np.zeros(len(df))

for i in range(1, len(df)):
    if position[i] > 0 and position[i - 1] == 0:
        trade_cost[i] = COST
    elif position[i] == 0 and position[i - 1] > 0:
        trade_cost[i] = COST

# -------------------------------------------------
# 8. STRATEGY RETURNS
# -------------------------------------------------
df['strategy_return'] = position * df['returns'] - trade_cost

# -------------------------------------------------
# 9. EQUITY CURVE
# -------------------------------------------------
df['strategy_equity'] = (1 + df['strategy_return']).cumprod()
df['market_equity'] = (1 + df['returns']).cumprod()

# -------------------------------------------------
# 10. METRICS
# -------------------------------------------------
def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std()

def max_drawdown(eq):
    peak = eq.cummax()
    return ((eq - peak) / peak).min()

print("\nENTRY v2 (Momentum Continuation) STRATEGY")
print("----------------------------------------")
print(f"Trades taken     : {entries}")
print(f"Strategy Sharpe  : {sharpe(df['strategy_return']):.2f}")
print(f"Strategy Max DD  : {max_drawdown(df['strategy_equity']):.2%}")
print(f"Final Strategy Equity : {df['strategy_equity'].iloc[-1]:.2f}")
print(f"Final Market Equity  : {df['market_equity'].iloc[-1]:.2f}")
