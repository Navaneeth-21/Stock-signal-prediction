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
# 2. Create target (next-day direction)
# -----------------------------
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# -----------------------------
# 3. Selective momentum signal
# -----------------------------
THRESHOLD = 0.5  # percent
df['trade_signal'] = df['change_pct'].abs() > THRESHOLD

# Direction prediction = momentum direction
df['prediction'] = (df['change_pct'] > 0).astype(int)

# -----------------------------
# 4. Drop last row (no target)
# -----------------------------
df = df.dropna()

# -----------------------------
# 5. Re-create same split as before
# -----------------------------
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

test = df.iloc[val_end:]

# -----------------------------
# 6. Evaluate ONLY traded days
# -----------------------------
trades = test[test['trade_signal']]

accuracy = accuracy_score(trades['target'], trades['prediction'])
coverage = len(trades) / len(test)

print("Selective Momentum Results")
print("--------------------------")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Coverage      : {coverage:.2%}")
print(f"Trades taken  : {len(trades)} / {len(test)}")
