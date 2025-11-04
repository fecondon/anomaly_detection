import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained autoencoder
MODEL_PATH = 'models/autoencoder_model.h5'
autoencoder = load_model(MODEL_PATH)

# Scaler loaded from training
scaler = joblib.load('models/scaler.pkl')


def predict_anomaly(input_features, threshold):
    """
    Args:
        input_feautres: 2D numpy array (n_samples x n_features)
        threshold: anomaly threshold
    
    Returns:
        List of dicts with reconstruction_error and is_anomaly
    """

    X_scaled = scaler.transform(input_features)
    X_pred = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

    results = []
    for err in reconstruction_error:
        results.append({
            'reconstruction_error': float(err),
            'is_anomaly': bool(err > threshold)
        })

    return results
