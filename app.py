import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Sample data generation
data = {
    'Influent_COD': [320, 310, 330, 300, 340, 325, 315, 310, 320, 330,
                     335, 340, 310, 300, 305, 295, 315, 325, 345, 350,
                     315, 335, 325, 320, 310, 340, 325, 330, 345, 300,
                     315, 310, 340, 320, 325, 315, 330, 350, 340, 335],
    'Influent_NH3_N': [22.3, 21.8, 23.0, 20.5, 23.5, 22.0, 21.2, 22.1, 22.0, 23.0,
                       22.5, 23.8, 21.7, 20.6, 23.2, 21.5, 21.8, 22.6, 22.3, 23.1,
                       21.9, 23.3, 22.1, 22.5, 21.6, 22.4, 23.6, 21.9, 22.5, 23.2,
                       22.0, 21.7, 22.4, 22.1, 23.1, 22.8, 23.5, 21.9, 22.6, 23.7],
    'Influent_TN': [49.0, 48.5, 50.0, 47.5, 51.0, 49.5, 48.8, 49.2, 49.0, 50.0,
                    50.5, 49.8, 48.7, 50.5, 49.2, 47.9, 48.5, 51.2, 50.0, 50.5,
                    48.9, 49.6, 48.7, 49.1, 50.2, 49.9, 48.3, 49.5, 50.7, 49.0,
                    48.8, 49.2, 50.4, 47.9, 48.7, 49.1, 50.5, 49.8, 48.9, 50.1],
    'Influent_TP': [3.45, 3.50, 3.60, 3.40, 3.70, 3.55, 3.45, 3.65, 3.50, 3.60,
                    3.52, 3.65, 3.48, 3.55, 3.62, 3.49, 3.57, 3.68, 3.47, 3.58,
                    3.54, 3.61, 3.48, 3.67, 3.49, 3.60, 3.50, 3.55, 3.62, 3.51,
                    3.58, 3.47, 3.64, 3.55, 3.50, 3.59, 3.61, 3.48, 3.63, 3.60],
    'pH': [7.7, 7.6, 7.8, 7.5, 7.9, 7.6, 7.7, 7.8, 7.6, 7.8,
           7.6, 7.9, 7.5, 7.7, 7.8, 7.6, 7.9, 7.8, 7.7, 7.6,
           7.9, 7.7, 7.5, 7.8, 7.7, 7.6, 7.8, 7.7, 7.5, 7.6,
           7.7, 7.8, 7.6, 7.9, 7.7, 7.8, 7.6, 7.7, 7.9, 7.8],
    'Effluent_COD': [19.5, 18.0, 20.0, 17.5, 21.0, 19.0, 18.5, 19.0, 19.5, 20,
                     18.5, 19.5, 17.0, 18.0, 21.0, 20.5, 19.0, 19.2, 20.8, 19.5,
                     18.8, 19.7, 20.3, 17.5, 19.0, 19.9, 19.1, 20.5, 18.7, 20.0,
                     18.6, 19.5, 19.2, 20.1, 17.8, 18.9, 19.3, 19.8, 19.4, 20.2],
    'Effluent_NH3_N': [0.10, 0.09, 0.11, 0.08, 0.12, 0.10, 0.09, 0.10, 0.10, 0.11,
                       0.11, 0.09, 0.12, 0.10, 0.08, 0.11, 0.09, 0.10, 0.12, 0.11,
                       0.09, 0.10, 0.09, 0.11, 0.10, 0.11, 0.12, 0.09, 0.08, 0.10,
                       0.12, 0.11, 0.10, 0.09, 0.11, 0.08, 0.10, 0.09, 0.12, 0.10],
    'Effluent_TN': [8.67, 8.50, 9.00, 8.30, 9.50, 8.80, 8.60, 9.00, 8.67, 9.00,
                    8.50, 9.10, 8.30, 9.20, 9.40, 8.80, 8.70, 9.00, 8.60, 9.10,
                    9.30, 8.90, 8.70, 9.50, 9.00, 8.40, 9.20, 8.70, 9.10, 9.30,
                    8.50, 8.80, 9.00, 9.20, 8.90, 9.00, 8.70, 9.40, 9.10, 9.20],
    'Effluent_TP': [0.12, 0.11, 0.13, 0.10, 0.14, 0.12, 0.11, 0.12, 0.12, 0.13,
                    0.11, 0.14, 0.10, 0.13, 0.11, 0.12, 0.14, 0.12, 0.10, 0.13,
                    0.11, 0.14, 0.12, 0.11, 0.13, 0.10, 0.12, 0.11, 0.13, 0.12,
                    0.11, 0.14, 0.12, 0.13, 0.12, 0.10, 0.13, 0.11, 0.12, 0.14],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variables
X = df[['Influent_COD', 'Influent_NH3_N', 'Influent_TN', 'Influent_TP', 'pH']]
y = df[['Effluent_COD', 'Effluent_NH3_N', 'Effluent_TN', 'Effluent_TP']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

# Streamlit UI
st.title("Effluent Water Quality Prediction")

# Input sliders for user inputs
influent_cod = st.number_input("Influent COD", min_value=0.0, max_value=500.0, value=320.0)
influent_nh3_n = st.number_input("Influent NH3-N", min_value=0.0, max_value=50.0, value=22.0)
influent_tn = st.number_input("Influent TN", min_value=0.0, max_value=100.0, value=50.0)
influent_tp = st.number_input("Influent TP", min_value=0.0, max_value=10.0, value=3.5)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5)

# Prepare the input for prediction
new_influent_data = np.array([influent_cod, influent_nh3_n, influent_tn, influent_tp, ph]).reshape(1, -1)

# Make prediction
if st.button("Predict Effluent Parameters"):
    prediction = model.predict(new_influent_data)
    st.write(f"Predicted COD: {prediction[0][0]:.2f}")
    st.write(f"Predicted NH3-N: {prediction[0][1]:.2f}")
    st.write(f"Predicted TN: {prediction[0][2]:.2f}")
    st.write(f"Predicted TP: {prediction[0][3]:.2f}")
