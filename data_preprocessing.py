# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from a CSV file
# Replace 'spring_steel_data.csv' with the path to your actual data file
data = pd.read_csv('spring_steel_data.csv')

# Display the first few rows to understand the dataset structure
print(data.head())

# Separate the features (X) and target variables (y)
# Assume columns are named appropriately
# Alloy elements (example: including C, Si, Mn, Cr, V, Mo, Ni, W, etc.)
alloy_elements = ['C', 'Si', 'Mn', 'Cr', 'V', 'Mo', 'Ni', 'W']
# Heat treatment parameters
heat_treatment_params = ['quench_temp', 'quench_time', 'temper_temp', 'temper_time']
# Mechanical properties
mechanical_properties = ['tensile_strength', 'yield_strength', 'elongation', 'hardness']

# Combine alloy elements and heat treatment parameters as features
features = data[alloy_elements + heat_treatment_params]
# Target variables: mechanical properties
target = data[mechanical_properties]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the first few rows of the scaled training data
print("Scaled training data:\n", X_train_scaled[:5])
print("Scaled test data:\n", X_test_scaled[:5])


# Save the processed data to new CSV files for later use
np.savetxt('X_train_scaled.csv', X_train_scaled, delimiter=',')
np.savetxt('X_test_scaled.csv', X_test_scaled, delimiter=',')
np.savetxt('y_train.csv', y_train.values, delimiter=',')
np.savetxt('y_test.csv', y_test.values, delimiter=',')

# Check the shapes of the processed datasets
print(f'X_train_scaled shape: {X_train_scaled.shape}')
print(f'X_test_scaled shape: {X_test_scaled.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
