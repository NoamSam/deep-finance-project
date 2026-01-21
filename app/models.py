from tensorflow.keras import layers, models, optimizers


def build_lstm(input_shape, learning_rate):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn(input_shape, learning_rate):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn_lstm(input_shape, learning_rate):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_model(model_type, input_shape, learning_rate):
    if model_type == "lstm":
        return build_lstm(input_shape, learning_rate)
    if model_type == "cnn":
        return build_cnn(input_shape, learning_rate)
    if model_type == "cnn_lstm":
        return build_cnn_lstm(input_shape, learning_rate)
    raise ValueError("model_type must be one of: lstm, cnn, cnn_lstm")
