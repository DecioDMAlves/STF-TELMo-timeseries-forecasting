"""
Example LSTM pipeline for multivariate time series forecasting.
This script demonstrates:
- Data loading & preprocessing
- Sequence generation
- Model definition & training
- Predictions & saving outputs

Note: This example uses placeholders for dataset paths and column names.
Replace them with your own data and adjust as needed.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    TimeDistributed, Concatenate, GlobalMaxPooling1D, Conv1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def create_sequences(features, targets, sequence_length, timestamps):
    """Generate overlapping sequences of features and targets."""
    X, y, ts = [], [], []
    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length - 1])
        ts.append(timestamps[i + sequence_length - 1])
    return np.array(X), np.array(y), np.array(ts)

def save_predictions_csv(predictions, true_values, timestamps, file_name, num_steps):
    """Save predictions and true values to CSV with step-wise formatting."""
    data = {"timestamp": timestamps}
    for step in range(num_steps):
        data[f"pred_u_step{step+1}"] = predictions[:, step * 2]
        data[f"pred_v_step{step+1}"] = predictions[:, step * 2 + 1]
        data[f"true_u_step{step+1}"] = true_values[:, step * 2]
        data[f"true_v_step{step+1}"] = true_values[:, step * 2 + 1]

    pd.DataFrame(data).to_csv(file_name, index=False)

# -------------------------------------------------------------------
# Example: Load and merge multiple CSVs
# Replace file paths and columns with your own
# -------------------------------------------------------------------

file_paths = [
    "data/location1.csv",
    "data/location2.csv",
    "data/location3.csv",
    "data/location4.csv",
]

dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    tag = file_path.split("/")[-1].split(".")[0]
    df = df.rename(columns={col: f"{col}_{tag}" if col != "timestamp" else "timestamp"
                            for col in df.columns})
    dfs.append(df)

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on="timestamp", how="inner")

merged_df.dropna(inplace=True)
merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
merged_df.set_index("timestamp", inplace=True)

# -------------------------------------------------------------------
# Train / validation / test split (placeholder dates)
# -------------------------------------------------------------------
train_df = merged_df.loc["2015":"2020"]
val_df   = merged_df.loc["2021"]
test_df  = merged_df.loc["2022":]

# -------------------------------------------------------------------
# Standardization
# -------------------------------------------------------------------
scaler_features = StandardScaler()
scaler_targets = StandardScaler()

# Example feature/target column lists (replace with actual)
feature_columns = ["feat1_loc1", "feat2_loc1", "feat1_loc2", "feat2_loc2"]
target_columns  = ["target_u_t0", "target_v_t0", "target_u_t1", "target_v_t1"]

train_features = scaler_features.fit_transform(train_df[feature_columns])
val_features   = scaler_features.transform(val_df[feature_columns])
test_features  = scaler_features.transform(test_df[feature_columns])

train_targets  = scaler_targets.fit_transform(train_df[target_columns])
val_targets    = scaler_targets.transform(val_df[target_columns])
test_targets   = scaler_targets.transform(test_df[target_columns])

# -------------------------------------------------------------------
# Model definition
# -------------------------------------------------------------------
def create_and_train_model(seq_len, num_features, num_outputs, X_train, y_train, X_val, y_val):
    """Define, compile, and train the LSTM model."""
    inputs = Input(shape=(seq_len, num_features))

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x1 = Bidirectional(LSTM(num_features * 4, return_sequences=True))(x)
    x1 = TimeDistributed(Dense(num_features // 2))(x1)
    x1 = Dropout(0.2)(x1)

    x2 = Bidirectional(LSTM(num_features * 4, return_sequences=True))(x1)
    x2 = TimeDistributed(Dense(num_features // 2))(x2)
    x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x, x1, x2])
    pooled = GlobalMaxPooling1D()(merged)
    dense  = Dense(num_features // 4)(pooled)
    outputs = Dense(num_outputs)(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50, batch_size=128, verbose=1,
              callbacks=[early_stop])
    return model

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
sequence_length = 2
num_iterations = 3
num_steps = 2  # Example number of forecast steps

for it in range(1, num_iterations + 1):
    X_train, y_train, ts_train = create_sequences(train_features, train_targets, sequence_length, train_df.index)
    X_val, y_val, ts_val = create_sequences(val_features, val_targets, sequence_length, val_df.index)
    X_test, y_test, ts_test = create_sequences(test_features, test_targets, sequence_length, test_df.index)

    model = create_and_train_model(sequence_length, X_train.shape[2], y_train.shape[1],
                                   X_train, y_train, X_val, y_val)

    preds = model.predict(X_test)
    preds_orig = scaler_targets.inverse_transform(preds)
    true_orig  = scaler_targets.inverse_transform(y_test)

    save_predictions_csv(preds_orig, true_orig, ts_test, f"results/predictions_iter{it}.csv", num_steps)

    print(f"Iteration {it} complete. Results saved.")
