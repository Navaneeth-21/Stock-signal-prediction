import pandas as pd

def feature_engineer(input_path, output_path):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)

    df['ma_distance_5'] = (df['close'] - df['ma_5']) / df['close']
    df['ma_distance_10'] = (df['close'] - df['ma_10']) / df['close']

    df['volatility_5'] = df['volatility']
    df['momentum_strength'] = df['change_pct'].abs()

    # -----------------------------
    # Clean
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Save
    # -----------------------------
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} | Rows: {len(df)}")

# -----------------------------
# Run for each split
# -----------------------------
feature_engineer(
    "data/signal_train_raw.csv",
    "data/signal_train_ml.csv"
)

feature_engineer(
    "data/signal_val_raw.csv",
    "data/signal_val_ml.csv"
)

feature_engineer(
    "data/signal_test_raw.csv",
    "data/signal_test_ml.csv"
)
