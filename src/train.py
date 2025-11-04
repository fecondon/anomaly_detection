import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model import build_autoencoder
from src.data_loader import generate_synthetic_bids


def train_autoencoder():
    df = generate_synthetic_bids()
    X = df.drop(columns=['is_anomaly']).values
    y = df['is_anomaly'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    autoencoder = build_autoencoder(input_dim=X.shape[1])
    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val, X_val),
        verbose=1
    )

    autoencoder.save('models/autoencoder.h5')
    return history
