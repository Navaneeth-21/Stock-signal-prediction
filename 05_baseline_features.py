import pandas as pd
from sklearn.metrics import accuracy_score

# Load feature splits
train = pd.read_csv("data/NIFTY50_train_features.csv")
val   = pd.read_csv("data/NIFTY50_val_features.csv")
test  = pd.read_csv("data/NIFTY50_test_features.csv")

# ---------------------------
# Rule 1: Momentum baseline
# ---------------------------
val_pred_mom = (val['return_1d'] > 0).astype(int)
test_pred_mom = (test['return_1d'] > 0).astype(int)

print("Momentum Baseline:")
print("Val Accuracy :", accuracy_score(val['direction'], val_pred_mom))
print("Test Accuracy:", accuracy_score(test['direction'], test_pred_mom))

# ---------------------------
# Rule 2: Trend baseline
# ---------------------------
val_pred_trend = (val['ma_ratio'] > 1).astype(int)
test_pred_trend = (test['ma_ratio'] > 1).astype(int)

print("\nTrend Baseline:")
print("Val Accuracy :", accuracy_score(val['direction'], val_pred_trend))
print("Test Accuracy:", accuracy_score(test['direction'], test_pred_trend))
