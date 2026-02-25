import pandas as pd

# -----------------------------
# Load signal-only dataset
# -----------------------------
df = pd.read_csv("data/signal_only_dataset.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# -----------------------------
# Time-based split
# -----------------------------
n = len(df)

train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# -----------------------------
# Save splits
# -----------------------------
train_df.to_csv("data/signal_train_raw.csv", index=False)
val_df.to_csv("data/signal_val_raw.csv", index=False)
test_df.to_csv("data/signal_test_raw.csv", index=False)

print("Signal dataset split completed")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))
