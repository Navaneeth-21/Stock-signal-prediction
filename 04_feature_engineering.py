import pandas as pd

def add_features(df):
    df = df.sort_values('date').reset_index(drop=True)

    df['return_1d'] = df['close'].pct_change()
    df['return_2d'] = df['close'].pct_change(2)
    df['return_5d'] = df['close'].pct_change(5)

    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_ratio'] = df['ma_5'] / df['ma_10']

    df['volatility_5'] = df['return_1d'].rolling(5).std()

    df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df.dropna().reset_index(drop=True)

# Load splits
train = pd.read_csv("data/NIFTY50_train.csv")
val   = pd.read_csv("data/NIFTY50_val.csv")
test  = pd.read_csv("data/NIFTY50_test.csv")

# Apply features
train = add_features(train)
val   = add_features(val)
test  = add_features(test)

# Save
train.to_csv("data/NIFTY50_train_features.csv", index=False)
val.to_csv("data/NIFTY50_val_features.csv", index=False)
test.to_csv("data/NIFTY50_test_features.csv", index=False)

print("✅ Features added to existing splits")
