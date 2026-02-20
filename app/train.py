from tensorflow.keras import callbacks


def train_model(model, X_train, y_train, X_val, y_val, config):
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stop],
        verbose=config.get("verbose", 0),
    )
    return history
