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
