from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from app.config import DEFAULT_CONFIG
from app.data import load_price_series
from app.eval import evaluate_model
from app.features import compute_log_returns, make_windows, train_val_test_split
from app.models import build_model
from app.train import train_model


@dataclass
class TrainResult:
    model_type: str
    metrics: dict
    history: object


def run_training(csv_path, model_type, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}

    series = load_price_series(csv_path)
    returns = compute_log_returns(series)
    X, y = make_windows(
        returns, window_size=config["window_size"], horizon=config["horizon"]
    )

    X = X[..., np.newaxis]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, test_size=config["test_size"], val_size=config["val_size"]
    )

    scaler = StandardScaler()
    train_shape = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(train_shape[0], -1)).reshape(
        train_shape
    )
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    model = build_model(
        model_type=model_type,
        input_shape=X_train.shape[1:],
        learning_rate=config["learning_rate"],
    )
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    metrics = evaluate_model(model, X_test, y_test)
    return TrainResult(model_type=model_type, metrics=metrics, history=history)
