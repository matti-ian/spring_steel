# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error

# Load the data from a CSV file
data = pd.read_csv('spring_steel_data.csv')

# Separate the features (X) and target variables (y)
features = data[['C', 'Si', 'Mn', 'Cr', 'V', 'quench_temp', 'quench_time', 'temper_temp', 'temper_time']]
target = data[['tensile_strength', 'elongation']]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create the C2P model
def create_c2p_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),  # First hidden layer with 64 neurons
        Dense(32, activation='relu'),                      # Second hidden layer with 32 neurons
        Dense(2, activation='linear')                      # Output layer with 2 neurons (for tensile_strength and elongation)
    ])
    # Compile the model with Adam optimizer and Mean Squared Error loss function
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create the model with the number of input features
c2p_model = create_c2p_model(X_train_scaled.shape[1])

# Train the C2P model
c2p_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict on the test set
y_pred = c2p_model.predict(X_test_scaled)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'C2P Model MAPE: {mape}')
