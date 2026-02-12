import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
train = pd.read_csv("data/NIFTY50_train_features.csv")
val   = pd.read_csv("data/NIFTY50_val_features.csv")
test  = pd.read_csv("data/NIFTY50_test_features.csv")

# Feature set (expanded for trees)
features = [
    'return_1d', 'return_2d', 'return_5d',
    'ma_5', 'ma_10', 'ma_ratio',
    'volatility_5'
]

X_train = train[features]
y_train = train['direction']

X_val = val[features]
y_val = val['direction']

X_test = test[features]
y_test = test['direction']

# No Sacaling needed for trees

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
print("Random Forest Results")
print("Val Accuracy :", accuracy_score(y_val, model.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

print("\nTest Classification Report:")
print(classification_report(y_test, model.predict(X_test)))
