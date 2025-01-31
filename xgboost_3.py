import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Sample Data: [Age, Salary] -> 1 = Bought, 0 = Did Not Buy
X = np.array([
    [22, 20000], [25, 25000], [28, 30000], [35, 50000], [40, 80000], 
    [50, 100000], [60, 150000], [30, 40000], [42, 60000], [55, 120000]
])  
y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])  # 1 = Bought, 0 = Did Not Buy

# Visualize Data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs. Salary (Buying Pattern)")
plt.colorbar(label="0 = No Purchase, 1 = Purchase")
plt.savefig("xgboost_data_3.png")

# Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost Model with Optimized Parameters
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    scale_pos_weight= sum(y == 0) / sum(y == 1),  # Handle class imbalance
    eval_metric="logloss"
)
model.fit(X_train_scaled, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print actual vs. predicted probabilities
for i, prob in enumerate(y_probs):
    print(f"Actual: {y_test[i]} | Predicted Probabilities: {prob}")
