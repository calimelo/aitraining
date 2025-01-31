import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulated Digital Forensic Logs (Suspicious IP Activity)
data = {
    "IP Address": ["192.168.1.1", "192.168.1.2", "203.0.113.42", "192.168.1.3", "45.33.128.250"],
    "Login Attempts": [2, 3, 50, 1, 100],   # High login attempts = Possible brute-force attack
    "Data Transferred (MB)": [5, 6, 500, 3, 1200],  # Large data transfer = Possible data exfiltration
    "Failed Logins": [0, 1, 40, 0, 80],   # High failed logins = Possible unauthorized access attempt
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train Isolation Forest for Anomaly Detection
model = IsolationForest(contamination=0.2, random_state=42)  # Assume 20% anomalies
model.fit(df[["Login Attempts", "Data Transferred (MB)", "Failed Logins"]])

# Predict anomalies (-1 = suspicious, 1 = normal)
df["Anomaly"] = model.predict(df[["Login Attempts", "Data Transferred (MB)", "Failed Logins"]])

# Convert anomalies to more readable format
df["Anomaly"] = df["Anomaly"].apply(lambda x: "Suspicious" if x == -1 else "Normal")

# Print Results
print("\n🔹 AI-Based Digital Forensic Log Analysis 🔹")
print(df)

# **Visualizing Anomalies**
plt.figure(figsize=(8, 5))

# Normal Traffic
normal = df[df["Anomaly"] == "Normal"]
plt.scatter(normal["Login Attempts"], normal["Data Transferred (MB)"], color="green", label="Normal", alpha=0.7)

# Suspicious Traffic (Anomalies)
anomalies = df[df["Anomaly"] == "Suspicious"]
plt.scatter(anomalies["Login Attempts"], anomalies["Data Transferred (MB)"], color="red", label="Suspicious", marker="x", s=100)

plt.xlabel("Login Attempts")
plt.ylabel("Data Transferred (MB)")
plt.title("Digital Forensic Anomaly Detection")
plt.legend()
plt.grid()
plt.savefig("digital_forensics_anomalies.png")