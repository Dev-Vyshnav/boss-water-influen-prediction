import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import streamlit as st

st.title("Wastewater Treatment Plant Effluent Prediction")

# Sample data
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
    'Effluent_BOD': [15.0, 14.0, 16.0, 13.5, 17.0, 15.5, 14.5, 15.0, 15.0, 16.0]
}

df = pd.DataFrame(data)
X = df[['Influent_COD', 'Influent_NH3_N', 'Influent_TN', 'Influent_TP', 'pH', 'Influent_BOD']]
y = df[['Effluent_COD', 'Effluent_NH3_N', 'Effluent_TN', 'Effluent_TP', 'Effluent_BOD']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Build and train model
@st.cache_resource
def train_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
    return model, history

model, history = train_model()

# Predictions
predictions = model.predict(X_test, verbose=0)

# Comparison DataFrame
comparison_df = pd.DataFrame({
    'Actual COD': y_test[:, 0], 'Predicted COD': predictions[:, 0],
    'Actual NH3-N': y_test[:, 1], 'Predicted NH3-N': predictions[:, 1],
    'Actual TN': y_test[:, 2], 'Predicted TN': predictions[:, 2],
    'Actual TP': y_test[:, 3], 'Predicted TP': predictions[:, 3],
    'Actual BOD': y_test[:, 4], 'Predicted BOD': predictions[:, 4]
})

# Display results
st.subheader("Actual vs Predicted Effluent Parameters")
st.dataframe(comparison_df.style.format("{:.2f}"))

# Plot training loss
st.subheader("Training Loss")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
st.pyplot(fig)
