"""
Example: Generating predictions with a trained LSTM model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------
# Load your trained model and scalers
# -------------------------------------------------------------------
# model = load_model("path/to/your_trained_model.h5")
# scaler_features = joblib.load("scaler_features.pkl")
# scaler_targets  = joblib.load("scaler_targets.pkl")

# -------------------------------------------------------------------
# Example new data (replace with your own dataset)
# -------------------------------------------------------------------
new_data = pd.read_csv("data/new_input.csv", parse_dates=["timestamp"])
timestamps = new_data["timestamp"]

# Use same feature columns as during training
feature_columns = ["feat1_loc1", "feat2_loc1", "feat1_loc2", "feat2_loc2"]
features_scaled = scaler_features.transform(new_data[feature_columns])

# -------------------------------------------------------------------
# Create input sequences
# -------------------------------------------------------------------
def create_sequences(features, seq_len):
    X, ts = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i:i + seq_len])
        ts.append(timestamps.iloc[i + seq_len - 1])
    return np.array(X), np.array(ts)

sequence_length = 2
X_input, ts_input = create_sequences(features_scaled, sequence_length)

# -------------------------------------------------------------------
# Generate predictions
# -------------------------------------------------------------------
preds_scaled = model.predict(X_input)
preds = scaler_targets.inverse_transform(preds_scaled)

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------
results = pd.DataFrame(preds, columns=["pred_u_step1", "pred_v_step1", "pred_u_step2", "pred_v_step2"])
results.insert(0, "timestamp", ts_input)
results.to_csv("results/new_predictions.csv", index=False)

print("Predictions saved to results/new_predictions.csv")
