from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def _raise_if_no_samples(X, window_size, horizon, source_size):
    if len(X) == 0:
        raise ValueError(
            "Not enough data to create training windows. "
            f"Got {source_size} points, need at least {window_size + horizon}."
        )


def _raise_if_empty_split(X_train, X_val, X_test):
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train/validation/test split produced an empty set. "
            "Use more data or reduce test_size/val_size."
        )


def run_training(csv_path, model_type, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}

    series = load_price_series(csv_path)
    if model_type == "lstm":
        # Follow the reference project training style:
        # MinMax scaling on close prices + sliding windows.
        close_values = series.astype(float).values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(close_values).reshape(-1)
        X, y = make_windows(
            scaled_close,
            window_size=config["window_size"],
            horizon=config["horizon"],
        )
        _raise_if_no_samples(
            X,
            window_size=config["window_size"],
            horizon=config["horizon"],
            source_size=len(scaled_close),
        )
        X = X[..., np.newaxis]
    else:
        returns = compute_log_returns(series)
        X, y = make_windows(
            returns,
            window_size=config["window_size"],
            horizon=config["horizon"],
        )
        _raise_if_no_samples(
            X,
            window_size=config["window_size"],
            horizon=config["horizon"],
            source_size=len(returns),
        )
        X = X[..., np.newaxis]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, test_size=config["test_size"], val_size=config["val_size"]
    )
    _raise_if_empty_split(X_train, X_val, X_test)

    if model_type != "lstm":
        scaler = StandardScaler()
        train_shape = X_train.shape
        X_train = scaler.fit_transform(X_train.reshape(train_shape[0], -1)).reshape(
            train_shape
        )
        X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(
            X_val.shape
        )
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(
            X_test.shape
        )

    model = build_model(
        model_type=model_type,
        input_shape=X_train.shape[1:],
        learning_rate=config["learning_rate"],
    )
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    metrics = evaluate_model(model, X_test, y_test)
    return TrainResult(model_type=model_type, metrics=metrics, history=history)
