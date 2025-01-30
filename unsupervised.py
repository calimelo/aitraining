from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create a simple apples, oranges, and bananas dataset
data = {
    'Weight': [150, 160, 170, 180, 140, 130, 120, 190, 200, 210, 110, 115, 105],
    'Texture': [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2]  # 1 = Smooth (Apple), 0 = Bumpy (Orange), 2 = Soft (Banana)
}

# Convert to DataFrame
df = pd.DataFrame(data)
X = df[['Weight', 'Texture']]

# Apply K-Means clustering (Unsupervised Learning: No labels)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Print clustered data
print(df)

# Plot the clusters
plt.scatter(df['Weight'], df['Texture'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Weight')
plt.ylabel('Texture')
plt.title('K-Means Clustering of Apples, Oranges, and Bananas')
#save the plot
plt.savefig('unsupervised_apples_oranges_bananas_kmeans.png')

#print accuracy
accuracy = accuracy_score(df['Texture'], df['Cluster'])
print(f'Model Accuracy: {accuracy * 100:.2f}%')


# Save the model
joblib.dump(kmeans, 'unsupervised_apples_oranges_bananas_kmeans.joblib')
