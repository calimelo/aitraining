import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a simple apples and oranges dataset
data = {
    'Weight': [150, 160, 170, 180, 140, 130, 120, 190, 200, 210],
    'Texture': [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],  # 1 = Smooth (Apple), 0 = Bumpy (Orange)
    'Label': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  # 0 = Apple, 1 = Orange
}

# Convert to DataFrame
df = pd.DataFrame(data)
X = df[['Weight', 'Texture']]
y = df['Label']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors (KNN) classifier
model = KNeighborsClassifier(n_neighbors=3)

# Train the model (Supervised Learning: learns from labeled data)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Show some predictions
predicted_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(predicted_df.head())

#plot
import matplotlib.pyplot as plt
plt.scatter(df['Weight'], df['Texture'], c=df['Label'])
plt.xlabel('Weight')
plt.ylabel('Texture')
plt.title('Apples and Oranges')
#save the plot
plt.savefig('supervised_apples_oranges_model.png')


# Save the model
import joblib
joblib.dump(model, 'supervised_apples_oranges_model.joblib')

