import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load cleaned data
# -----------------------------
df = pd.read_csv("data/NIFTY50_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# -----------------------------
# 2. Create target
# -----------------------------
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# -----------------------------
# 3. Indicators
# -----------------------------
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_10'] = df['close'].rolling(10).mean()

# -----------------------------
# 4. Selective signal
# -----------------------------
THRESHOLD = 0.5

momentum = df['change_pct'].abs() > THRESHOLD
trend_up = df['close'] > df['ma_10']

df['trade_signal'] = momentum & trend_up
df['prediction'] = (df['change_pct'] > 0).astype(int)

# -----------------------------
# 5. Clean
# -----------------------------
df = df.dropna()

# -----------------------------
# 6. Same split
# -----------------------------
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

test = df.iloc[val_end:]

# -----------------------------
# 7. Evaluate
# -----------------------------
trades = test[test['trade_signal']]

accuracy = accuracy_score(trades['target'], trades['prediction'])
coverage = len(trades) / len(test)

print("Selective Momentum + MA Results")
print("-------------------------------")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Coverage      : {coverage:.2%}")
print(f"Trades taken  : {len(trades)} / {len(test)}")
