# BASELINE 1: NAIVE PREDICTOR

import pandas as pd
import numpy as np

train_df = pd.read_csv("data/NIFTY50_train.csv")
test_df  = pd.read_csv("data/NIFTY50_test.csv")

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Predict next day’s close as today’s close

# Create Naive Predictions
test_df['naive_pred'] = test_df['close'].shift(1)
test_df = test_df.dropna()

# Evaluate (RMSE)
from sklearn.metrics import mean_squared_error

rmse_naive = np.sqrt(
    mean_squared_error(test_df['close'], test_df['naive_pred'])
)

print("Naive RMSE:", rmse_naive)


# BASELINE 2: MOVING AVERAGE (MA-20)

# Create MA Prediction

test_df['ma20_pred'] = test_df['close'].rolling(window=20).mean().shift(1)
test_df = test_df.dropna()

# Evaluate MA

rmse_ma20 = np.sqrt(
    mean_squared_error(test_df['close'], test_df['ma20_pred'])
)

print("MA-20 RMSE:", rmse_ma20)


# DIRECTION ACCURACY

test_df['actual_dir'] = np.sign(test_df['close'].diff())
test_df['naive_dir']  = np.sign(test_df['naive_pred'].diff())

direction_acc = (test_df['actual_dir'] == test_df['naive_d3ir']).mean()
print("Naive Direction Accuracy:", direction_acc)


