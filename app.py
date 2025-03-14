import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import streamlit as st

st.title("Wastewater Treatment Plant Effluent Prediction with Membrane Fouling")

# Updated sample data with Membrane Fouling Rate (e.g., in units like mbar/day or %/day)
data = {
    'Influent_COD': [320, 310, 330, 300, 340, 325, 315, 310, 320, 330],
    'Influent_NH3_N': [22.3, 21.8, 23.0, 20.5, 23.5, 22.0, 21.2, 22.1, 22.0, 23.0],
    'Influent_TN': [49.0, 48.5, 50.0, 47.5, 51.0, 49.5, 48.8, 49.2, 49.0, 50.0],
    'Influent_TP': [3.45, 3.50, 3.60, 3.40, 3.70, 3.55, 3.45, 3.65, 3.50, 3.60],
    'pH': [7.7, 7.6, 7.8, 7.5, 7.9, 7.6, 7.7, 7.8, 7.6, 7.8],
    'Influent_BOD': [200, 190, 210, 180, 220, 205, 195, 190, 200, 210],
    'Effluent_COD': [19.5, 18.0, 20.0, 17.5, 21.0, 19.0, 18.5, 19.0, 19.5, 20],
    'Effluent_NH3_N': [0.10, 0.09, 0.11, 0.08, 0.12, 0.10, 0.09, 0.10, 0.10, 0.11],
    'Effluent_TN': [8.67, 8.50, 9.00, 8.30, 9.50, 8.80, 8.60, 9.00, 8.67, 9.00],
    'Effluent_TP': [0.12, 0.11, 0.13, 0.10, 0.14, 0.12, 0.11, 0.12, 0.12, 0.13],
    'Effluent_BOD': [15.0, 14.0, 16.0, 13.5, 17.0, 15.5, 14.5, 15.0, 15.0, 16.0],
    'Membrane_Fouling_Rate': [0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.45, 0.55, 0.5, 0.6]  # Example values (e.g., mbar/day)
}

df = pd.DataFrame(data)
X = df[['Influent_COD', 'Influent_NH3_N', 'Influent_TN', 'Influent_TP', 'pH', 'Influent_BOD']]
y = df[['Effluent_COD', 'Effluent_NH3_N', 'Effluent_TN', 'Effluent_TP', 'Effluent_BOD', 'Membrane_Fouling_Rate']]

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build and train model
@st.cache_resource
def train_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))  # Now outputs 6 values
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.2, verbose=0)
    return model, history

model, history = train_model()

# User Input Section
st.subheader("Enter Influent Parameters")
col1, col2 = st.columns(2)

with col1:
    influent_cod = st.number_input("Influent COD (mg/L)", min_value=0.0, value=320.0)
    influent_nh3_n = st.number_input("Influent NH3-N (mg/L)", min_value=0.0, value=22.0)
    influent_tn = st.number_input("Influent TN (mg/L)", min_value=0.0, value=49.0)

with col2:
    influent_tp = st.number_input("Influent TP (mg/L)", min_value=0.0, value=3.5)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.7)
    influent_bod = st.number_input("Influent BOD (mg/L)", min_value=0.0, value=200.0)

# Predict button
if st.button("Predict Effluent Parameters and Fouling Rate"):
    # Input validation
    if ph < 0 or ph > 14:
        st.error("pH must be between 0 and 14")
    elif any(x < 0 for x in [influent_cod, influent_nh3_n, influent_tn, influent_tp, influent_bod]):
        st.error("Influent parameters cannot be negative")
    else:
        # Prepare and scale input data
        input_data = np.array([[influent_cod, influent_nh3_n, influent_tn, influent_tp, ph, influent_bod]])
        input_data_scaled = scaler_X.transform(input_data)
        
        # Make prediction
        prediction_scaled = model.predict(input_data_scaled, verbose=0)[0]
        prediction = scaler_y.inverse_transform([prediction_scaled])[0]
        
        # Display predictions
        st.subheader("Predicted Effluent Parameters and Membrane Fouling Rate")
        pred_df = pd.DataFrame({
            'Parameter': ['COD (mg/L)', 'NH3-N (mg/L)', 'TN (mg/L)', 'TP (mg/L)', 'BOD (mg/L)', 'Membrane Fouling Rate (mbar/day)'],
            'Predicted Value': prediction
        })
        st.dataframe(pred_df.style.format({'Predicted Value': '{:.2f}'}))

# Plot training and validation loss
st.subheader("Training and Validation Loss")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
st.pyplot(fig)

# Optional: Evaluate on test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Test Loss (MSE): {test_loss:.4f}")
