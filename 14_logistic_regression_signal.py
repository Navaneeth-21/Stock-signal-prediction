import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load datasets
# -----------------------------
train = pd.read_csv("data/signal_train_ml.csv")
val = pd.read_csv("data/signal_val_ml.csv")
test = pd.read_csv("data/signal_test_ml.csv")

# -----------------------------
# Features & target
# -----------------------------
FEATURES = [
    'return_1d', 'return_3d', 'return_5d',
    'ma_distance_5', 'ma_distance_10',
    'volatility_5', 'momentum_strength'
]

X_train = train[FEATURES]
y_train = train['target']

X_val = val[FEATURES]
y_val = val['target']

X_test = test[FEATURES]
y_test = test['target']

# -----------------------------
# Scale features (fit only on train)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation
# -----------------------------
val_pred = model.predict(X_val_scaled)
test_pred = model.predict(X_test_scaled)

print("Logistic Regression (Signal Days Only)")
print("-------------------------------------")
print(f"Val Accuracy : {accuracy_score(y_val, val_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))
