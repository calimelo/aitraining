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

# Train the XGBoost Model (Fix: Removed 'use_label_encoder=False')
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict if a 45-year-old with a $60,000 salary will buy the product
new_data = np.array([[45, 60000]])
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[0][1]  # Probability of buying
print(f"Prediction: {'Will Buy' if prediction[0] == 1 else 'Will Not Buy'}")

