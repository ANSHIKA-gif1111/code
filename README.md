# code
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load the CSV file
file_path = 'C:/Users/verma/Downloads/csvFile278.csv'
df = pd.read_csv(file_path)

# Convert Dates to numerical format
df['Dates'] = pd.to_datetime(df['Dates'])
df['Days'] = (df['Dates'] - df['Dates'].min()).dt.days

# Prepare features and target
X = df[['Days']]
y = df['ACTUAL (mm)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model (using Gradient Boosting Regressor)
model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict the target values for existing data
predictions = model.predict(X[['Days']])

# Add variability and simulate inconsistencies
noise = np.random.normal(0, 1.5, size=predictions.shape)  # Increased noise
dropout_chance = np.random.rand(predictions.shape[0])  # Random chance to drop values to 0
predictions_with_inconsistency = np.where(dropout_chance > 0.8, 0, predictions + noise)  # 20% chance to drop to 0
df['Predicted (mm)'] = predictions_with_inconsistency

# Generate future dates from 9/15/2024 to 12/31/2100
future_dates = pd.date_range(start='2024-09-15', end='2100-12-31')
future_df = pd.DataFrame(future_dates, columns=['Dates'])
future_df['Days'] = (future_df['Dates'] - df['Dates'].min()).dt.days

# Predict future values
future_predictions = model.predict(future_df[['Days']])

# Add variability and simulate inconsistencies to future predictions
future_noise = np.random.normal(0, 1.5, size=future_predictions.shape)  # Increased noise
future_dropout_chance = np.random.rand(future_predictions.shape[0])  # Random chance to drop values to 0
future_predictions_with_inconsistency = np.where(future_dropout_chance > 0.8, 0, future_predictions + future_noise)
future_df['Predicted (mm)'] = future_predictions_with_inconsistency

# Combine existing and future predictions
combined_df = pd.concat([df, future_df], ignore_index=True)

# Save the results to a new Excel file
output_file_path = 'C:/Users/verma/Downloads/csvFile278_with_improved_predictions.xlsx'
combined_df.to_excel(output_file_path, index=False)

print(f"Improved predictions added and saved to {output_file_path}")
