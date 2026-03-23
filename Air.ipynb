#!/usr/bin/env python3
"""
Air Quality PM2.5 Prediction using LSTM

This script predicts PM2.5 concentrations based on past 24 hours of data
using a multi‑feature LSTM model. It supports training, evaluation, and
prediction modes.

Dataset: Beijing PM2.5 Data (UCI)
https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------- Configuration --------------------
CONFIG = {
    # Data
    'data_path': Path('C:/Users/Francis Musoke/Downloads/Air Quality.csv'),
    'features': ['pm2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM'],
    'target': 'pm2.5',
    'time_steps': 24,          # look back 24 hours
    'train_split': 0.8,
    'random_seed': 42,

    # Model
    'lstm_units': [50, 50],
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mean_squared_error',

    # Paths
    'model_dir': Path('./models'),
    'model_name': 'best_lstm_model.h5',
    'log_dir': Path('./logs'),
}

# Create directories if they don't exist
CONFIG['model_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# -------------------- Data Loading & Preprocessing --------------------
def load_data(file_path, features):
    """Load CSV, select features, and drop missing values."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Select required columns and drop NaN
    data = data[features].dropna()
    logger.info(f"Loaded {len(data)} records.")
    return data


def create_sequences(data, time_steps):
    """
    Convert multivariate time series into input-output sequences.
    X: sequences of past `time_steps` rows (all features except target)
    y: target value at the next time step
    """
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), :-1])   # all features except target
        y.append(data[i + time_steps, 0])         # target is the first column (pm2.5)
    return np.array(X), np.array(y)


def preprocess_data(data, time_steps, train_split, random_seed):
    """Normalize data, create sequences, and split into train/test."""
    # Normalize all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, time_steps)

    # Split into train/test
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


# -------------------- Model Building --------------------
def build_model(input_shape, lstm_units, dropout_rate, learning_rate):
    """Create and compile the LSTM model."""
    model = Sequential()
    # First LSTM layer with return_sequences=True (stacking)
    model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    # Second LSTM layer (return_sequences=False by default)
    model.add(LSTM(lstm_units[1]))
    model.add(Dropout(dropout_rate))
    # Output layer (single value: PM2.5)
    model.add(Dense(1))

    model.compile(optimizer=learning_rate, loss='mean_squared_error')
    return model


# -------------------- Training --------------------
def train_model(model, X_train, y_train, X_val, y_val, config):
    """Train the model and save the best version."""
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1
    )

    # Save the entire model (not just weights)
    model_path = config['model_dir'] / config['model_name']
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return history


# -------------------- Evaluation --------------------
def evaluate_model(model_path, X_test, y_test, scaler, data):
    """
    Load a saved model, compute predictions, and return metrics and predictions.
    """
    model = load_model(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = y_pred_scaled.flatten()

    # Inverse transform: we need to reconstruct the full feature set to inverse scale.
    # We'll use the last time step features (excluding target) for the scaling trick.
    last_features = X_test[:, -1, 1:]  # shape (n_samples, n_features-1)
    y_test_actual = inverse_transform_target(y_test, last_features, scaler)
    y_pred_actual = inverse_transform_target(y_pred, last_features, scaler)

    # Compute RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    logger.info(f"Test RMSE: {test_rmse:.2f}")

    return y_test_actual, y_pred_actual, test_rmse


def inverse_transform_target(y_scaled, last_features, scaler):
    """
    Inverse transform the target variable using the scaler.
    Since scaler was fit on all features, we need to reconstruct a dummy matrix
    with the target as first column and the other features as the last time step values.
    """
    n_samples = len(y_scaled)
    dummy = np.zeros((n_samples, scaler.n_features_in_))
    dummy[:, 0] = y_scaled                      # target column
    dummy[:, 1:] = last_features                # other features (from the last time step)
    dummy_inv = scaler.inverse_transform(dummy)
    return dummy_inv[:, 0]                     # return only the target


# -------------------- Prediction --------------------
def predict_new_data(model_path, new_data, time_steps, scaler):
    """Predict PM2.5 for a new sequence of data."""
    model = load_model(model_path)
    # Ensure new_data has the same features as training
    # new_data should be a numpy array of shape (time_steps, n_features)
    # Normalize using the same scaler
    scaled = scaler.transform(new_data)
    # Add batch dimension
    X_input = np.expand_dims(scaled, axis=0)
    pred_scaled = model.predict(X_input)[0, 0]
    # Inverse transform using last features trick
    last_features = scaled[-1, 1:]  # exclude target column
    pred_actual = inverse_transform_target(np.array([pred_scaled]), last_features.reshape(1, -1), scaler)[0]
    return pred_actual


# -------------------- Plotting --------------------
def plot_results(history, y_test_actual, y_pred_actual):
    """Plot training curves and prediction vs actual."""
    # Training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Predictions vs actual
    plt.subplot(1, 2, 2)
    plt.plot(y_test_actual, label='Actual')
    plt.plot(y_pred_actual, label='Predicted')
    plt.title('PM2.5 Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Air Quality PM2.5 Prediction with LSTM')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'predict'],
                        help='Mode: train, evaluate, or predict')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for evaluate/predict)')
    parser.add_argument('--input_csv', type=str, default=None,
                        help='CSV file with new data for prediction (must include same columns)')
    args = parser.parse_args()

    # Load and preprocess data
    data = load_data(CONFIG['data_path'], CONFIG['features'])
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, CONFIG['time_steps'], CONFIG['train_split'], CONFIG['random_seed']
    )

    if args.mode == 'train':
        model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=CONFIG['lstm_units'],
            dropout_rate=CONFIG['dropout_rate'],
            learning_rate=CONFIG['learning_rate']
        )
        model.summary()

        history = train_model(model, X_train, y_train, X_test, y_test, CONFIG)

        # Plot training curves
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Evaluate on test set (optional, but good to have)
        model_path = CONFIG['model_dir'] / CONFIG['model_name']
        y_test_actual, y_pred_actual, test_rmse = evaluate_model(
            model_path, X_test, y_test, scaler, data
        )
        plot_results(history, y_test_actual, y_pred_actual)

    elif args.mode == 'evaluate':
        if args.model_path is None:
            logger.error("Please provide --model_path for evaluation mode.")
            sys.exit(1)
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        y_test_actual, y_pred_actual, test_rmse = evaluate_model(
            model_path, X_test, y_test, scaler, data
        )
        # Plot predictions vs actual
        plt.figure(figsize=(10, 4))
        plt.plot(y_test_actual, label='Actual')
        plt.plot(y_pred_actual, label='Predicted')
        plt.title('PM2.5 Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('PM2.5')
        plt.legend()
        plt.show()

    elif args.mode == 'predict':
        if args.model_path is None:
            logger.error("Please provide --model_path for prediction mode.")
            sys.exit(1)
        if args.input_csv is None:
            logger.error("Please provide --input_csv with new data.")
            sys.exit(1)

        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        # Load new data (must have same columns as training)
        new_df = pd.read_csv(args.input_csv)
        # Ensure columns match
        if not all(col in new_df.columns for col in CONFIG['features']):
            logger.error(f"Input CSV must contain columns: {CONFIG['features']}")
            sys.exit(1)
        new_data = new_df[CONFIG['features']].dropna()
        if len(new_data) < CONFIG['time_steps']:
            logger.error(f"Need at least {CONFIG['time_steps']} rows to form a sequence.")
            sys.exit(1)

        # Take the last `time_steps` rows
        sequence = new_data.iloc[-CONFIG['time_steps']:].values
        prediction = predict_new_data(model_path, sequence, CONFIG['time_steps'], scaler)
        print(f"Predicted PM2.5 for next hour: {prediction:.2f}")


if __name__ == '__main__':
    main()
