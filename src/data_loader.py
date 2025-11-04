import numpy as np
import pandas as pd


def generate_synthetic_bids(
        n_samples=10_000,
        n_features=5,
        anomaly_fraction=0.01,
        random_state=0
):
    np.random.seed(random_state)

    # Normal bids
    normal_data = np.random.normal(
        loc=0,
        scale=1,
        size=(int(n_samples * (1 - anomaly_fraction)), n_features)
    )

    # Anomalous bids
    anomalies = np.random.normal(
        loc=5,
        scale=1.5,
        size=(int(n_samples * anomaly_fraction), n_features)
    )

    X = np.vstack([normal_data, anomalies])
    y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalies))])

    df = pd.DataFrame(X, columns=[f'features_{i}' for i in range(n_features)])
    df['is_anomaly'] = y

    return df