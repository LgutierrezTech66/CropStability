import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Small in-code dataset
data = {
    'rainfall': [100, 200, 150, 80, 300, 120],
    'temperature': [25, 30, 27, 22, 35, 24],
    'soil_ph': [6.5, 7.0, 6.8, 5.5, 7.2, 6.0],
    'crop_type': [1, 2, 1, 3, 2, 1],  # 1: Rice, 2: Wheat, 3: Maize
    'stability': [1, 1, 1, 0, 0, 1]   # 1: Stable, 0: Unstable
}

df = pd.DataFrame(data)

# Features and label
X = df.drop('stability', axis=1)
y = df['stability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('model/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
