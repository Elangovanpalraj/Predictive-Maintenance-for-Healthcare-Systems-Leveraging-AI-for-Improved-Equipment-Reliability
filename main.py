import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

# Increasing the dataset to 250 rows
num_rows = 250

# Creating a larger dataset
data = {
    'equipment_id': np.arange(1, num_rows + 1),
    'usage_hours': np.random.randint(500, 5000, num_rows),
    'temperature': np.random.randint(40, 100, num_rows),
    'vibration_level': np.round(np.random.uniform(0.2, 1.5, num_rows), 2),
    'pressure_level': np.round(np.random.uniform(1.5, 4.0, num_rows), 2),
    'last_maintenance': np.random.randint(50, 600, num_rows),
    'failure': np.random.choice([0, 1], num_rows)  # Binary classification (0 = No failure, 1 = Failure)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as an Excel file
df.to_excel("healthcare_equipment_data.xlsx", index=False)

# Display the first few rows
print(df.head())
print(f"\nDataset shape: {df.shape}")  # Check the size of the dataset



# Load dataset
#df = pd.read_excel("healthcare_equipment_data.xlsx")

# Feature selection (X) and target (y)
X = df.drop(columns=['equipment_id', 'failure'])  # Drop ID and target column
y = df['failure']  # Target column

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
joblib.dump(model, "equipment_failure_model1.pkl")
print("Model saved successfully!")
# ======================================================================================
# Load trained model
model = joblib.load("equipment_failure_model1.pkl")

# Example new data for prediction
# Load trained model
model = joblib.load("equipment_failure_model1.pkl")

# Example new data for prediction
new_data = np.array([[3000, 75, 0.8, 2.5, 200]])  # Example input
prediction = model.predict(new_data)

# Display prediction
if prediction[0] == 1:
    print("Equipment Failure Expected! 🚨")
else:
    print("Equipment is Safe ✅")
