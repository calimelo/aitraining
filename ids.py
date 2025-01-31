import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Sample Network Traffic Data (Simulated)
data = {
    "Bytes Sent": [500, 600, 550, 580, 100000, 530, 560, 120000, 590, 610],
    "Bytes Received": [700, 720, 680, 690, 200000, 750, 730, 250000, 710, 730],
    "Connection Time (ms)": [100, 110, 105, 108, 5000, 102, 107, 6000, 109, 111],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train Isolation Forest (Anomaly Detection Model)
model = IsolationForest(contamination=0.2)  # Assume 20% of data may be anomalies
model.fit(df)

# Predict Anomalies (1 = Normal, -1 = Anomaly)
df["Anomaly"] = model.predict(df)

# Print the results
print("\n🔹 AI-Based Anomaly Detection Results 🔹")
print(df)

# Visualize the anomalies using a scatter plot
plt.figure(figsize=(8, 5))

# Normal points
normal = df[df["Anomaly"] == 1]
plt.scatter(normal["Bytes Sent"], normal["Bytes Received"], color="green", label="Normal", alpha=0.7)

# Anomalous points
anomalies = df[df["Anomaly"] == -1]
plt.scatter(anomalies["Bytes Sent"], anomalies["Bytes Received"], color="red", label="Anomaly", marker="x", s=100)

plt.xlabel("Bytes Sent")
plt.ylabel("Bytes Received")
plt.title("Network Traffic Anomaly Detection")
plt.legend()
plt.grid()
plt.savefig("anomaly_detection.png")
