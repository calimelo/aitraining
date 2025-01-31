import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data: [Age, Salary] -> 1 = Bought, 0 = Did Not Buy
X = np.array([
    [22, 20000], [25, 25000], [28, 30000], [35, 50000], [40, 80000], 
    [50, 100000], [60, 150000], [30, 40000], [42, 60000], [55, 120000]
])  
y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])  # 1 = Bought, 0 = Did Not Buy

# Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost Model with improved parameters
model = xgb.XGBClassifier(
    n_estimators=100,       # More trees
    learning_rate=0.05,     # Slower learning rate
    max_depth=3,            # Prevent overfitting
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # Get probabilities

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print the probabilities for each prediction
print("Predicted Probabilities:", y_prob)

import matplotlib.pyplot as plt

# Scatter plot of data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs. Salary (Buying Pattern)")
plt.colorbar(label="0 = No Purchase, 1 = Purchase")
plt.savefig("xgboost_data.png")