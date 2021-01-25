from tensorflow import keras
from tensorflow.keras import layers


def define_model(layer):
    """Create model : embedding and MLP"""
    model = keras.Sequential([
        layer,
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
