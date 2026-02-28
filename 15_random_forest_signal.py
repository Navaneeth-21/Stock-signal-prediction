import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# Random Forest (conservative)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Random Forest (Signal Days Only)")
print("--------------------------------")
print(f"Val Accuracy : {accuracy_score(y_val, val_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))
