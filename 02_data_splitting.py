import pandas as pd

df = pd.read_csv("data/NIFTY50_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])

#time-based data splitting
train_df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2021-12-31')]
val_df   = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2023-12-31')]
test_df  = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2025-12-31')]


#sanity checks
print("Train:", train_df['date'].min(), "→", train_df['date'].max(), len(train_df))
print("Val:  ", val_df['date'].min(), "→", val_df['date'].max(), len(val_df))
print("Test: ", test_df['date'].min(), "→", test_df['date'].max(), len(test_df))


#save the splits
train_df.to_csv("data/NIFTY50_train.csv", index=False)
val_df.to_csv("data/NIFTY50_val.csv", index=False)
test_df.to_csv("data/NIFTY50_test.csv", index=False)
