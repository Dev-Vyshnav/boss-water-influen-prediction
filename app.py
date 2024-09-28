import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Sample data generation
data = {
    'Influent_COD': [320, 310, 330, 300, 340, 325, 315, 310, 320, 330],
    'Influent_NH3_N': [22.3, 21.8, 23.0, 20.5, 23.5, 22.0, 21.2, 22.1, 22.0, 23.0],
    'Influent_TN': [49.0, 48.5, 50.0, 47.5, 51.0, 49.5, 48.8, 49.2, 49.0, 50.0],
    'Influent_TP': [3.45, 3.50, 3.60, 3.40, 3.70, 3.55, 3.45, 3.65, 3.50, 3.60],
    'pH': [7.7, 7.6, 7.8, 7.5, 7.9, 7.6, 7.7, 7.8, 7.6, 7.8],
    'Effluent_COD': [19.5, 18.0, 20.0, 17.5, 21.0, 19.0, 18.5, 19.0, 19.5, 20],
    'Effluent_NH3_N': [0.10, 0.09, 0.11, 0.08, 0.12, 0.10, 0.09, 0.10, 0.10, 0.11],
    'Effluent_TN': [8.67, 8.50, 9.00, 8.30, 9.50, 8.80, 8.60, 9.00, 8.67, 9.00],
    'Effluent_TP': [0.12, 0.11, 0.13, 0.10, 0.14, 0.12, 0.11, 0.12, 0.12, 0.13],
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
