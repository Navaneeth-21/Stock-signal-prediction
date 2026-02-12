import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv("data/NIFTY50_train_features.csv")
val   = pd.read_csv("data/NIFTY50_val_features.csv")
test  = pd.read_csv("data/NIFTY50_test_features.csv")

# Feature columns
features = [
    'return_1d', 'return_2d', 'return_5d',
    'ma_ratio', 'volatility_5'
]

# ma_ratio = ma_5 / ma_10 

X_train = train[features]
y_train = train['direction']

X_val = val[features]
y_val = val['direction']

X_test = test[features]
y_test = test['direction']

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print("Logistic Regression Results")
print("Val Accuracy :", accuracy_score(y_val, model.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

print("\nTest Classification Report:")
print(classification_report(y_test, model.predict(X_test)))
