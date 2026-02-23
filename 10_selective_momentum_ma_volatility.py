import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/NIFTY50_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# -----------------------------
# Target
# -----------------------------
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# -----------------------------
# Indicators
# -----------------------------
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_10'] = df['close'].rolling(10).mean()

df['daily_range'] = (df['high'] - df['low']) / df['close']
df['volatility'] = df['daily_range'].rolling(5).mean()

# -----------------------------
# Filters
# -----------------------------
MOMENTUM_TH = 0.5 # percent
VOL_MAX = df['volatility'].quantile(0.7)

momentum = df['change_pct'].abs() > MOMENTUM_TH
trend = df['close'] > df['ma_10']
vol_ok = df['volatility'] < VOL_MAX

df['trade_signal'] = momentum & trend & vol_ok
df['prediction'] = (df['change_pct'] > 0).astype(int)

# -----------------------------
# Clean
# -----------------------------
df = df.dropna()

# -----------------------------
# Split (same as before)
# -----------------------------
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
test = df.iloc[val_end:]

# -----------------------------
# Evaluate
# -----------------------------
trades = test[test['trade_signal']]

accuracy = accuracy_score(trades['target'], trades['prediction'])
coverage = len(trades) / len(test)

print("Selective Momentum + MA + Volatility")
print("-----------------------------------")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Coverage      : {coverage:.2%}")
print(f"Trades taken  : {len(trades)} / {len(test)}")