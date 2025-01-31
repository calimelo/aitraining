import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample Email Data
data = {
    "Email": ["Win a free iPhone now!", "Hello, how are you?", "Claim your free gift today!", 
              "Meeting at 3 PM", "Congratulations! You won a prize!", "Lunch plans for tomorrow?"],
    "Spam": [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

# Convert Data into a Pandas DataFrame
df = pd.DataFrame(data)

# Convert Text into a Bag of Words (Word Counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])
y = df["Spam"]

# Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict if "Free money available now!" is spam
new_email = ["Free money available now!"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")

# Calculate Accuracy on Test Data
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the Model
import joblib
joblib.dump(model, "spam_classifier.joblib")

