import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(input_dim, encoding_dim=8, activation='relu', optimizer='adam'):
    input_layer = layers.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(64, activation=activation)(input_layer)
    encoded = layers.Dense(32, activation=activation)(encoded)
    encoded = layers.Dense(encoding_dim, activation=activation)(encoded)

    # Decoder
    decoded = layers.Dense(32, activation=activation)(encoded)
    decoded = layers.Dense(64, activation=activation)(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder
