import os.path as osp
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load the SML2010 Dataset
# The dataset has 24 columns. We use 'sep="\s+"' to handle multiple spaces.
# We skip the first line (the # comments) and manually assign column names.
file_path = osp.join("temp_data", "NEW-DATA-1.T15.txt")
column_names = [
    'Date', 'Time', 'Temp_Comedor', 'Temp_Habitacion', 'Weather_Temp',
    'CO2_Comedor', 'CO2_Habitacion', 'Hum_Comedor', 'Hum_Habitacion',
    'Light_Comedor', 'Light_Habitacion', 'Precipitation', 'Meteo_Crepusculo',
    'Meteo_Viento', 'Meteo_Sol_Oest', 'Meteo_Sol_Est', 'Meteo_Sol_Sud',
    'Meteo_Piranometro', 'Ext_Entalpic_1', 'Ext_Entalpic_2', 'Ext_Entalpic_turbo',
    'Temp_Exterior', 'Hum_Exterior', 'Day_of_Week'
]

try:
    # Read the file, skipping the comment header. 
    # engine='python' is used to avoid parsing warnings with regex separators.
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=24, names=column_names, engine='python')
    print("Dataset loaded successfully. Rows:", len(df))
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the dataset file is in the same folder.")
    exit()

# 2. Time Series Preparation (Listing 11.7 Logic)
# We want to predict Temp_Comedor using 5 previous values
prev_values_count = 5
target_col = 'Temp_Comedor'

# Shift data to create features (t-5, t-4, t-3, t-2, t-1)
X = pd.DataFrame()
for i in range(prev_values_count, 0, -1):
    X[f't-{i}'] = df[target_col].shift(i)

# The target is the current temperature (t)
y = df[target_col].iloc[prev_values_count:]
X = X.dropna() # Remove the first 5 rows which now have NaNs from shifting

# 3. Data Splitting & Normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization is critical for Neural Networks (Listing 11.7, page 276)
train_mean = X_train.mean()
train_std = X_train.std()

X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

# 4. Build the MLP Regressor (Listing 11.7 architecture)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(prev_values_count,)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1) # Single output for continuous value
])

# 5. Compile and Train
# Learning rate 5e-3 and SGD as per textbook
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
    loss=tf.keras.losses.MeanAbsoluteError()
)

print("Training model...")
history = model.fit(
    X_train_norm, y_train, 
    epochs=1000, # Textbook uses 3000, but 1000 is often enough for convergence
    batch_size=128, 
    verbose=1,
    validation_split=0.1
)

# 6. Evaluation and Plotting
y_pred = model.predict(X_test_norm)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error on Test Set: {mae:.4f} °C")

# Plot first 100 results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Actual Temperature", color='blue')
plt.plot(y_pred[:100], label="Predicted Temperature", color='red', linestyle='--')
plt.title("Temperature Prediction (SML2010 Dataset)")
plt.xlabel("Samples")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

model.save("temperature_prediction_mlp.h5")