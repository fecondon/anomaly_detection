import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.keras
from src.model import build_autoencoder
from src.data_loader import generate_synthetic_bids
from src.evaluate import evaluate_autoencoder, compute_reconstruction_error


def train_autoencoder():
    # Load Data
    df = generate_synthetic_bids()
    X = df.drop(columns=['is_anomaly']).values
    y = df['is_anomaly'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    # Set param grid
    param_grid = [
        {
            'encoding_dim': 32,
            'activation': 'relu',
            'optimizer': 'adam',
            'epochs': 50,
            'batch_size': 256
        },
        {
            'encoding_dim': 32,
            'activation': 'leaky_relu',
            'optimizer': 'adam',
            'epochs': 50,
            'batch_size': 256
        },
        {
            'encoding_dim': 16,
            'activation': 'relu',
            'optimizer': 'adam',
            'epochs': 50,
            'batch_size': 128
        },
        {
            'encoding_dim': 64,
            'activation': 'relu',
            'optimizer': 'rmsprop',
            'epochs': 50,
            'batch_size': 256
        }
    ]

    for params in param_grid:
        run_name = f"enc{params['encoding_dim']}_act{params['activation']}_opt{params['optimizer']}_bs{params['batch_size']}"
        # Start MLFlow Tracking
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            # Build Model
            autoencoder = build_autoencoder(
                input_dim=X.shape[1],
                encoding_dim=params['encoding_dim'],
                activation=params['activation'],
                optimizer=params['optimizer']
            )

            # Train Model
            history = autoencoder.fit(
                X_train, X_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(X_val, X_val),
                verbose=1
            )

            # Log Metrics
            for epoch, loss in enumerate(history.history['loss']):
                mlflow.log_metric('train_loss', loss, step=epoch)
            for epoch, val_loss in enumerate(history.history['val_loss']):
                mlflow.log_metric('val_loss', val_loss, step=epoch)

            # Evaluate & Log Threshold
            threshold = compute_reconstruction_error(autoencoder, X_val)
            mlflow.log_metric('anomaly_threshold', threshold)

            # Log Model
            mlflow.keras.log_model(autoencoder, 'autoencoder_model')
            autoencoder.save('models/autoencoder.h5')

    return history


if __name__ == '__main__':
    train_autoencoder()
