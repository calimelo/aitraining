import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate 100 random data points for Age (20 to 60 years) and Salary (20k to 150k)
np.random.seed(42)
ages = np.random.randint(20, 60, size=100)
salaries = np.random.randint(20000, 150000, size=100)

# Define buying behavior: More likely to buy if age is 30-50 and salary > 50k
y = np.array([(1 if (30 <= age <= 50 and salary > 50000) else 0) for age, salary in zip(ages, salaries)])

# Convert into feature array
X = np.column_stack((ages, salaries))

# Visualize Data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs. Salary (Buying Pattern)")
plt.colorbar(label="0 = No Purchase, 1 = Purchase")
plt.savefig("xgboost_data_4.png")

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
    scale_pos_weight=sum(y == 0) / sum(y == 1),  # Handle class imbalance
    eval_metric="logloss"
)
model.fit(X_train_scaled, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print some actual vs. predicted probabilities
for i, prob in enumerate(y_probs[:10]):  # Show first 10 predictions
    print(f"Actual: {y_test[i]} | Predicted Probabilities: {prob}")
