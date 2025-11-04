import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score
)
from tensorflow.keras.models import load_model
from src.data_loader import generate_synthetic_bids
import matplotlib.pyplot as plt


def evaluate_autoencoder(model_path='models/autoencoder.h5', plot=True):
    # --- Load model a nd data ---
    model = load_model(model_path)
    df = generate_synthetic_bids()
    X = df.drop(columns=['is_anomaly']).values
    y_true = df['is_anomaly'].values

    # --- Scale data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Predict (reconstruct) ---
    X_pred = model.predict(X_scaled)

    # --- Compute reconstruction errors ---
    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

    # --- Threshold selection ---
    # Simple heuristic: use percentile of reconstruction errors on training data
    threshold = np.percentile(reconstruction_error, 95)

    # --- Predicted anomalies ---
    y_pred = (reconstruction_error > threshold).astype(int)

    # --- Metrics ---
    roc = roc_auc_score(y_true, reconstruction_error)
    precision, recall, _ = precision_recall_curve(y_true, reconstruction_error)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)

    print(f'ROC-AUC: {roc:.4f}')
    print(f'PR-AUC: {pr_auc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Threshold (95th percentile): {threshold:.4f}')

    # --- Optional Plots ---
    if plot:
        plt.figure(figsize=(10, 5))
        plt.hist(reconstruction_error[y_true == 0], bins=50, alpha=0.6, label='Normal')
        plt.hist(reconstruction_error[y_true == 1], bins=50, alpha=0.6, label='Anomaly')
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    # --- Return dataframe with results ---
    df['reconstruction_error'] = reconstruction_error
    df['predicted_anomaly'] = y_pred
    return df, threshold


def compute_reconstruction_error(model, X_val, percentile=95):
    """
    Compute reconstruction error for validation data and return a threshold.

    Args:
        model: Trained autoencoder (tf.keras.Model)
        X_val: Validation data (numpy array)
        percentile: Percentile for threshold (defualt 95%)

    Returns:
        threshold: Error above which a sample is considered anomalous
    """

    # Reconstruct Validation Data
    X_val_pred = model.predict(X_val)

    # Compoute MSE per sample
    reconstruction_error = np.mean(np.squaree(X_val - X_val_pred), axis=1)

    # Threshold (e.g., top 5% as anomalies)
    threshold = np.percentile(reconstruction_error, percentile)
    return threshold
