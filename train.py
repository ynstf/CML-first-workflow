import os
import json
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib

# ---------------------------
# 1) Load dataset
# ---------------------------
df = sns.load_dataset("titanic").dropna(subset=["survived"])
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# Fill missing values
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

# Split features and labels
X = df.drop("survived", axis=1)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 2) Train model
# ---------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# 3) Evaluate
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------
# 4) Save model and metrics
# ---------------------------
# Create folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/titanic_model.pkl")

# Save metrics
metrics = {
    "accuracy": accuracy,
    "test_size": len(X_test)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Accuracy: {accuracy:.4f}")
print("Model saved to 'models/titanic_model.pkl'")
print("Metrics saved to 'metrics.json'")
