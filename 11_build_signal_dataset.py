import pandas as pd

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
# Signal rules (LOCKED)
# -----------------------------
momentum = df['change_pct'].abs() > 0.5
trend = df['close'] > df['ma_10']
vol_ok = df['volatility'] < df['volatility'].quantile(0.7)

df['trade_signal'] = momentum & trend & vol_ok

# -----------------------------
# Build signal dataset
# -----------------------------
signal_df = df[df['trade_signal']].dropna()

# -----------------------------
# Save
# -----------------------------
signal_df.to_csv("data/signal_only_dataset.csv", index=False)

print("Signal-only dataset created")
print("Rows :", len(signal_df))
    