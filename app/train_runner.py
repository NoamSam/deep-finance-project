import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tensorflow.keras import callbacks

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.models import build_model

LSTM_TARGET_MODE = "log_return"
US_MULTIFEATURE_REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SPY_Return",
    "QQQ_Return",
    "VIX_Return",
]
US_MULTIFEATURE_FEATURE_COLUMNS = [
    "Asset_Return",
    "Open_To_Close",
    "High_To_Close",
    "Low_To_Close",
    "Log_Volume",
    "SPY_Return",
    "QQQ_Return",
    "VIX_Return",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight training runner")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date, Close")
    parser.add_argument(
        "--model",
        choices=["lstm", "lstm_multifeature", "cnn", "cnn_lstm"],
        required=True,
        help="Model type",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lstm-units1", type=int, default=64)
    parser.add_argument("--lstm-units2", type=int, default=32)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional output path for trained Keras model (.keras)",
    )
    parser.add_argument(
        "--save-scaler",
        default=None,
        help="Optional output path for scaler parameters (.npz)",
    )
    parser.add_argument(
        "--save-eval",
        default=None,
        help="Optional output path for evaluation arrays (.npz)",
    )
    parser.add_argument(
        "--save-forecast",
        default=None,
        help="Optional output path for future forecast value (.npz)",
    )
    parser.add_argument(
        "--save-forecast-path",
        default=None,
        help="Optional output path for recursive forecast trajectory (.npz)",
    )
    return parser.parse_args()


def load_close_series(csv_path):
    closes = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if "Close" not in columns and "Adj Close" not in columns:
            raise ValueError("CSV must contain Close (or Adj Close) column")
        for row in reader:
            raw = row.get("Close") or row.get("Adj Close")
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                closes.append(value)

    if not closes:
        raise ValueError("No valid close values found in CSV")
    return np.asarray(closes, dtype=np.float32)


def load_training_frame(csv_path):
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("CSV is empty")
    if "Date" not in frame.columns:
        raise ValueError("CSV must contain Date column")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    frame = frame.drop_duplicates("Date", keep="last")
    return frame.reset_index(drop=True)


def make_windows(values, window_size, horizon):
    X = []
    y = []
    target_indices = []
    last_start = len(values) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        target_index = end + horizon - 1
        X.append(values[start:end])
        y.append(values[target_index])
        target_indices.append(target_index)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int32),
    )


def make_return_sequence_windows(values, window_size, horizon):
    X = []
    y = []
    target_close_indices = []
    last_start = len(values) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        X.append(values[start:end])
        y.append(values[end : end + horizon])
        target_close_indices.append(end + horizon)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_close_indices, dtype=np.int32),
    )


def make_feature_windows(feature_values, target_values, window_size, horizon):
    X = []
    y = []
    target_indices = []
    last_start = len(target_values) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        target_index = end + horizon - 1
        X.append(feature_values[start:end])
        y.append(target_values[target_index])
        target_indices.append(target_index)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int32),
    )


def split_datasets(X, y, target_indices, test_size, val_size):
    n = len(X)
    test_count = int(n * test_size)
    val_count = int(n * val_size)
    train_end = n - test_count - val_count
    val_end = n - test_count

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    idx_train = target_indices[:train_end]
    idx_val = target_indices[train_end:val_end]
    idx_test = target_indices[val_end:]
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train/validation/test split produced an empty set."
        )
    return (
        (X_train, y_train, idx_train),
        (X_val, y_val, idx_val),
        (X_test, y_test, idx_test),
    )


def scale_minmax(values):
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    scale = max_value - min_value
    if scale <= 0:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = ((values - min_value) / scale).astype(np.float32)
    scaler = {
        "scaler_type": "minmax",
        "min": np.asarray([min_value], dtype=np.float32),
        "max": np.asarray([max_value], dtype=np.float32),
    }
    return scaled, scaler


def scale_minmax_by_train(X_train, X_val, X_test, y_train, y_val, y_test):
    train_values = np.concatenate(
        [X_train.reshape(-1), np.asarray(y_train, dtype=np.float32).reshape(-1)]
    )
    min_value = float(np.min(train_values))
    max_value = float(np.max(train_values))
    scale = max_value - min_value

    def transform(values):
        values = np.asarray(values, dtype=np.float32)
        if scale <= 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_value) / scale).astype(np.float32)

    scaler = {
        "scaler_type": "minmax",
        "min": np.asarray([min_value], dtype=np.float32),
        "max": np.asarray([max_value], dtype=np.float32),
    }
    return (
        transform(X_train),
        transform(X_val),
        transform(X_test),
        transform(y_train),
        transform(y_val),
        transform(y_test),
        scaler,
    )


def scale_standard_by_train(X_train, X_val, X_test):
    flat_train = X_train.reshape(X_train.shape[0], -1)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0

    def transform(values):
        flat = values.reshape(values.shape[0], -1)
        return ((flat - mean) / std).reshape(values.shape)

    scaler = {
        "scaler_type": "standard",
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    return transform(X_train), transform(X_val), transform(X_test), scaler


def scale_multifeature_by_train(X_train, X_val, X_test):
    feature_count = int(X_train.shape[-1])
    flat_train = X_train.reshape(-1, feature_count)
    center = np.zeros(feature_count, dtype=np.float32)
    scale = np.ones(feature_count, dtype=np.float32)
    lower = np.full(feature_count, np.nan, dtype=np.float32)
    upper = np.full(feature_count, np.nan, dtype=np.float32)
    clip = np.full(feature_count, np.nan, dtype=np.float32)

    for feature_index, feature_name in enumerate(US_MULTIFEATURE_FEATURE_COLUMNS):
        train_column = flat_train[:, feature_index].astype(np.float32)
        if feature_name == "Log_Volume":
            median = float(np.median(train_column))
            q1 = float(np.quantile(train_column, 0.25))
            q3 = float(np.quantile(train_column, 0.75))
            iqr = q3 - q1
            center[feature_index] = median
            scale[feature_index] = float(iqr) if iqr > 1e-6 else 1.0
            clip[feature_index] = 5.0
        elif feature_name in {"SPY_Return", "QQQ_Return", "VIX_Return"}:
            low = float(np.quantile(train_column, 0.01))
            high = float(np.quantile(train_column, 0.99))
            clipped_train = np.clip(train_column, low, high)
            std = float(np.std(clipped_train))
            center[feature_index] = float(np.mean(clipped_train))
            scale[feature_index] = std if std > 1e-6 else 1.0
            lower[feature_index] = low
            upper[feature_index] = high
            clip[feature_index] = 6.0
        else:
            mean = float(np.mean(train_column))
            std = float(np.std(train_column))
            center[feature_index] = mean
            scale[feature_index] = std if std > 1e-6 else 1.0

    def transform(values):
        transformed = np.asarray(values, dtype=np.float32).copy()
        for feature_index in range(feature_count):
            column = transformed[..., feature_index]
            if np.isfinite(lower[feature_index]) and np.isfinite(upper[feature_index]):
                column = np.clip(
                    column, lower[feature_index], upper[feature_index]
                )
            column = (column - center[feature_index]) / scale[feature_index]
            if np.isfinite(clip[feature_index]):
                column = np.clip(
                    column, -clip[feature_index], clip[feature_index]
                )
            transformed[..., feature_index] = column
        return transformed.astype(np.float32)

    scaler = {
        "scaler_type": "multifeature_mixed",
        "feature_center": center.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "feature_lower": lower.astype(np.float32),
        "feature_upper": upper.astype(np.float32),
        "feature_clip": clip.astype(np.float32),
    }
    return transform(X_train), transform(X_val), transform(X_test), scaler


def _load_close_values_from_frame(training_frame):
    if "Close" not in training_frame.columns:
        raise ValueError("CSV must contain Close column")
    close_values = pd.to_numeric(training_frame["Close"], errors="coerce")
    close_values = close_values.dropna().to_numpy(dtype=np.float32)
    if close_values.size == 0:
        raise ValueError("No valid close values found in CSV")
    return close_values


def _build_us_multifeature_inputs(training_frame):
    missing_columns = [
        column
        for column in US_MULTIFEATURE_REQUIRED_COLUMNS
        if column not in training_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "CSV must contain US multi-feature columns: "
            + ", ".join(US_MULTIFEATURE_REQUIRED_COLUMNS)
        )

    frame = training_frame[US_MULTIFEATURE_REQUIRED_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SPY_Return",
        "QQQ_Return",
        "VIX_Return",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
    frame["Volume"] = frame["Volume"].fillna(0.0)
    for column in ["SPY_Return", "QQQ_Return", "VIX_Return"]:
        frame[column] = frame[column].fillna(0.0)

    close_values = np.maximum(
        frame["Close"].to_numpy(dtype=np.float32), 1e-8
    ).astype(np.float32)
    if close_values.size < 2:
        raise ValueError("Not enough data to build US multi-feature inputs.")

    open_values = np.maximum(
        frame["Open"].to_numpy(dtype=np.float32), 1e-8
    ).astype(np.float32)
    high_values = np.maximum(
        frame["High"].to_numpy(dtype=np.float32), 1e-8
    ).astype(np.float32)
    low_values = np.maximum(
        frame["Low"].to_numpy(dtype=np.float32), 1e-8
    ).astype(np.float32)
    volume_values = np.maximum(
        frame["Volume"].to_numpy(dtype=np.float32), 0.0
    ).astype(np.float32)

    asset_returns = np.diff(np.log(close_values)).astype(np.float32)
    open_to_close = np.log(close_values[1:] / open_values[1:]).astype(np.float32)
    high_to_close = np.log(high_values[1:] / close_values[1:]).astype(np.float32)
    low_to_close = np.log(low_values[1:] / close_values[1:]).astype(np.float32)
    log_volume = np.log1p(volume_values[1:]).astype(np.float32)

    feature_matrix = np.column_stack(
        [
            asset_returns,
            open_to_close,
            high_to_close,
            low_to_close,
            log_volume,
            frame["SPY_Return"].to_numpy(dtype=np.float32)[1:],
            frame["QQQ_Return"].to_numpy(dtype=np.float32)[1:],
            frame["VIX_Return"].to_numpy(dtype=np.float32)[1:],
        ]
    ).astype(np.float32)
    return close_values, feature_matrix, asset_returns


def build_datasets(training_frame, args):
    close_values = _load_close_values_from_frame(training_frame)
    use_return_target = args.model != "lstm" or LSTM_TARGET_MODE == "log_return"
    context = {
        "close_values": close_values,
        "feature_history": None,
    }
    if args.model == "lstm_multifeature":
        close_values, feature_matrix, returns = _build_us_multifeature_inputs(
            training_frame
        )
        X, y, target_return_indices = make_feature_windows(
            feature_matrix,
            returns,
            args.window_size,
            args.horizon,
        )
        target_close_indices = target_return_indices + 1
        scaler = None
        target_mode = "log_return"
        context["close_values"] = close_values
        context["feature_history"] = feature_matrix
    elif args.model == "lstm" and use_return_target:
        safe_close = np.maximum(close_values, 1e-8)
        returns = np.diff(np.log(safe_close)).astype(np.float32)
        X, y, target_close_indices = make_return_sequence_windows(
            returns, args.window_size, args.horizon
        )
        scaler = None
        target_mode = "log_return_path"
    elif use_return_target:
        safe_close = np.maximum(close_values, 1e-8)
        returns = np.diff(np.log(safe_close))
        X, y, target_return_indices = make_windows(
            returns, args.window_size, args.horizon
        )
        target_close_indices = target_return_indices + 1
        scaler = None
        target_mode = "log_return"
    else:
        X, y, target_close_indices = make_windows(
            close_values, args.window_size, args.horizon
        )
        target_mode = "price_level"

    if len(X) == 0:
        if args.model == "lstm":
            needed = args.window_size + args.horizon + 1
        else:
            needed = args.window_size + args.horizon
        raise ValueError(
            f"Not enough data to create training windows. "
            f"Got {len(close_values)} points, need at least {needed}."
        )

    if X.ndim == 2:
        X = X[..., np.newaxis]
    (
        (X_train, y_train, _),
        (X_val, y_val, _),
        (X_test, y_test, idx_test),
    ) = split_datasets(
        X, y, target_close_indices, args.test_size, args.val_size
    )
    if target_mode == "price_level":
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            scaler,
        ) = scale_minmax_by_train(X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.model == "lstm_multifeature":
        X_train, X_val, X_test, scaler = scale_multifeature_by_train(
            X_train, X_val, X_test
        )
    else:
        X_train, X_val, X_test, scaler = scale_standard_by_train(
            X_train, X_val, X_test
        )
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        idx_test,
        scaler,
        target_mode,
        context,
    )


def save_scaler(path, scaler_values):
    if not path or scaler_values is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **scaler_values)


def save_eval(path, y_true, y_pred, target_close_indices):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        target_close_indices=target_close_indices.astype(np.int32),
    )


def save_forecast(path, forecast_value, forecast_lower=None, forecast_upper=None):
    if not path or forecast_value is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "forecast_value": np.asarray([forecast_value], dtype=np.float32),
    }
    if forecast_lower is not None:
        payload["forecast_lower"] = np.asarray([forecast_lower], dtype=np.float32)
    if forecast_upper is not None:
        payload["forecast_upper"] = np.asarray([forecast_upper], dtype=np.float32)
    np.savez(output_path, **payload)


def save_forecast_path(
    path,
    forecast_values,
    forecast_lower_values=None,
    forecast_upper_values=None,
):
    if not path or forecast_values is None:
        return
    forecast_array = np.asarray(forecast_values, dtype=np.float32).reshape(-1)
    if forecast_array.size == 0:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, forecast_array.size + 1, dtype=np.int32)
    payload = {
        "steps": steps,
        "forecast_values": forecast_array,
    }
    if forecast_lower_values is not None:
        lower_array = np.asarray(forecast_lower_values, dtype=np.float32).reshape(-1)
        if lower_array.size == forecast_array.size:
            payload["forecast_lower_values"] = lower_array
    if forecast_upper_values is not None:
        upper_array = np.asarray(forecast_upper_values, dtype=np.float32).reshape(-1)
        if upper_array.size == forecast_array.size:
            payload["forecast_upper_values"] = upper_array
    np.savez(output_path, **payload)


def _model_predict_array(model, X):
    return np.asarray(model(X, training=False), dtype=np.float32)


def _evaluate_in_batches(model, X, y, batch_size):
    if len(X) == 0:
        return float("nan"), float("nan")
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_loss, batch_mae = model.test_on_batch(X[start:end], y[start:end])
        batch_count = end - start
        total_loss += float(batch_loss) * batch_count
        total_mae += float(batch_mae) * batch_count
        total_count += batch_count
    if total_count == 0:
        return float("nan"), float("nan")
    return total_loss / total_count, total_mae / total_count


def _fit_with_train_on_batch(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    patience=5,
    lr_patience=2,
    lr_factor=0.5,
    min_lr=1e-5,
):
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    best_weights = model.get_weights()
    best_val_loss = float("inf")
    stale_epochs = 0
    stale_lr_epochs = 0

    for _epoch in range(int(epochs)):
        indices = np.random.permutation(len(X_train))
        total_loss = 0.0
        total_mae = 0.0
        total_count = 0
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch_loss, batch_mae = model.train_on_batch(
                X_train[batch_indices], y_train[batch_indices]
            )
            batch_count = len(batch_indices)
            total_loss += float(batch_loss) * batch_count
            total_mae += float(batch_mae) * batch_count
            total_count += batch_count

        train_loss = total_loss / max(total_count, 1)
        train_mae = total_mae / max(total_count, 1)
        val_loss, val_mae = _evaluate_in_batches(
            model, X_val, y_val, int(batch_size)
        )
        history["loss"].append(float(train_loss))
        history["mae"].append(float(train_mae))
        history["val_loss"].append(float(val_loss))
        history["val_mae"].append(float(val_mae))

        if np.isfinite(val_loss) and val_loss < (best_val_loss - 1e-8):
            best_val_loss = float(val_loss)
            best_weights = model.get_weights()
            stale_epochs = 0
            stale_lr_epochs = 0
        else:
            stale_epochs += 1
            stale_lr_epochs += 1

        if stale_lr_epochs >= int(lr_patience):
            current_lr = float(model.optimizer.learning_rate.numpy())
            new_lr = max(current_lr * float(lr_factor), float(min_lr))
            if new_lr < current_lr:
                model.optimizer.learning_rate.assign(new_lr)
            stale_lr_epochs = 0

        if stale_epochs >= int(patience):
            break

    model.set_weights(best_weights)
    return history


def _transform_multifeature_values(values, scaler_values):
    transformed = np.asarray(values, dtype=np.float32).copy()
    center = np.asarray(
        scaler_values.get("feature_center", scaler_values.get("feature_mean")),
        dtype=np.float32,
    ).reshape(-1)
    scale = np.asarray(
        scaler_values.get("feature_scale", scaler_values.get("feature_std")),
        dtype=np.float32,
    ).reshape(-1)
    lower = np.asarray(
        scaler_values.get(
            "feature_lower",
            np.full(center.shape, np.nan, dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1)
    upper = np.asarray(
        scaler_values.get(
            "feature_upper",
            np.full(center.shape, np.nan, dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1)
    clip = np.asarray(
        scaler_values.get(
            "feature_clip",
            np.full(center.shape, np.nan, dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1)
    if center.size == 0 or scale.size == 0 or center.size != transformed.shape[-1]:
        return None
    safe_scale = np.where(scale == 0, 1.0, scale).astype(np.float32)

    for feature_index in range(transformed.shape[-1]):
        column = transformed[..., feature_index]
        if np.isfinite(lower[feature_index]) and np.isfinite(upper[feature_index]):
            column = np.clip(column, lower[feature_index], upper[feature_index])
        column = (column - center[feature_index]) / safe_scale[feature_index]
        if np.isfinite(clip[feature_index]):
            column = np.clip(column, -clip[feature_index], clip[feature_index])
        transformed[..., feature_index] = column
    return transformed.astype(np.float32)


def _recursive_lstm_forecast_path(
    model, close_values, scaler_values, window_size, forecast_steps
):
    if (
        model is None
        or scaler_values is None
        or forecast_steps <= 0
        or len(close_values) < window_size
    ):
        return None

    min_value = float(scaler_values["min"][0])
    max_value = float(scaler_values["max"][0])
    scale = max_value - min_value

    close_array = np.asarray(close_values, dtype=np.float32)
    if scale > 0:
        scaled_values = ((close_array - min_value) / scale).astype(np.float32)
    else:
        scaled_values = np.zeros_like(close_array, dtype=np.float32)

    window = scaled_values[-window_size:].tolist()
    predictions = []
    for _ in range(forecast_steps):
        model_input = np.asarray(window[-window_size:], dtype=np.float32).reshape(
            1, window_size, 1
        )
        next_scaled = float(_model_predict_array(model, model_input)[0])
        window.append(next_scaled)
        if scale > 0:
            next_value = (next_scaled * scale) + min_value
        else:
            next_value = min_value
        predictions.append(float(next_value))
    return np.asarray(predictions, dtype=np.float32)


def _recursive_return_forecast_path(
    model, close_values, scaler_values, window_size, forecast_steps
):
    if (
        model is None
        or scaler_values is None
        or forecast_steps <= 0
        or len(close_values) < (window_size + 1)
    ):
        return None

    close_array = np.asarray(close_values, dtype=np.float32)
    safe_close = np.maximum(close_array, 1e-8)
    returns = np.diff(np.log(safe_close)).astype(np.float32)
    if len(returns) < window_size:
        return None

    mean = np.asarray(scaler_values.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(scaler_values.get("std"), dtype=np.float32).reshape(-1)
    if mean.size != window_size or std.size != window_size:
        return None
    std = np.where(std == 0, 1.0, std).astype(np.float32)

    window = returns[-window_size:].astype(np.float32)
    last_close = float(close_array[-1])
    predictions = []
    for _ in range(forecast_steps):
        scaled_window = ((window - mean) / std).reshape(1, window_size, 1)
        next_log_return = float(_model_predict_array(model, scaled_window).reshape(-1)[0])
        last_close = float(last_close * np.exp(next_log_return))
        predictions.append(last_close)
        window = np.concatenate(
            [window[1:], np.asarray([next_log_return], dtype=np.float32)]
        )
    return np.asarray(predictions, dtype=np.float32)


def _direct_return_forecast_path(
    model, close_values, scaler_values, window_size, forecast_steps
):
    if (
        model is None
        or scaler_values is None
        or forecast_steps <= 0
        or len(close_values) < (window_size + 1)
    ):
        return None

    close_array = np.asarray(close_values, dtype=np.float32)
    safe_close = np.maximum(close_array, 1e-8)
    returns = np.diff(np.log(safe_close)).astype(np.float32)
    if len(returns) < window_size:
        return None

    mean = np.asarray(scaler_values.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(scaler_values.get("std"), dtype=np.float32).reshape(-1)
    if mean.size != window_size or std.size != window_size:
        return None
    std = np.where(std == 0, 1.0, std).astype(np.float32)

    window = returns[-window_size:].astype(np.float32)
    scaled_window = ((window - mean) / std).reshape(1, window_size, 1)
    predicted_returns = _model_predict_array(model, scaled_window).reshape(-1)
    if predicted_returns.size == 0:
        return None
    predicted_returns = predicted_returns[:forecast_steps].astype(np.float32)
    last_close = float(close_array[-1])
    forecast_path = last_close * np.exp(np.cumsum(predicted_returns, axis=0))
    return np.asarray(forecast_path, dtype=np.float32)


def _recursive_multifeature_return_forecast_path(
    model, close_values, feature_history, scaler_values, window_size, forecast_steps
):
    if (
        model is None
        or scaler_values is None
        or forecast_steps <= 0
        or len(close_values) < (window_size + 1)
        or feature_history is None
    ):
        return None

    close_array = np.asarray(close_values, dtype=np.float32)
    feature_array = np.asarray(feature_history, dtype=np.float32)
    if feature_array.ndim != 2 or len(feature_array) < window_size:
        return None

    transformed_probe = _transform_multifeature_values(
        feature_array[-window_size:].reshape(1, window_size, -1),
        scaler_values,
    )
    if transformed_probe is None:
        return None

    last_close = float(close_array[-1])
    last_log_volume = float(feature_array[-1, 4])
    predictions = []
    for _ in range(forecast_steps):
        window = feature_array[-window_size:].astype(np.float32).reshape(
            1, window_size, -1
        )
        scaled_window = _transform_multifeature_values(window, scaler_values)
        if scaled_window is None:
            return None
        next_log_return = float(_model_predict_array(model, scaled_window).reshape(-1)[0])
        last_close = float(last_close * np.exp(next_log_return))
        next_feature_row = np.asarray(
            [
                next_log_return,
                0.0,
                0.0,
                0.0,
                last_log_volume,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        feature_array = np.vstack([feature_array, next_feature_row])
        predictions.append(last_close)
    return np.asarray(predictions, dtype=np.float32)


def _build_confidence_interval_paths(forecast_values, residual_std_price, z_score=1.96):
    if forecast_values is None:
        return None, None, None
    forecast_array = np.asarray(forecast_values, dtype=np.float32).reshape(-1)
    if forecast_array.size == 0 or not np.isfinite(residual_std_price):
        return None, None, None
    steps = np.arange(1, forecast_array.size + 1, dtype=np.float32)
    half_width = float(z_score) * float(residual_std_price) * np.sqrt(steps)
    lower = forecast_array - half_width
    upper = forecast_array + half_width
    return lower.astype(np.float32), upper.astype(np.float32), half_width.astype(np.float32)


def train_and_eval(args):
    training_frame = load_training_frame(args.csv)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        idx_test,
        scaler_values,
        target_mode,
        context,
    ) = build_datasets(training_frame, args)
    close_values = context["close_values"]

    model = build_model(
        model_type=args.model,
        input_shape=X_train.shape[1:],
        learning_rate=args.learning_rate,
        lstm_units1=args.lstm_units1,
        lstm_units2=args.lstm_units2,
        lstm_dropout=args.lstm_dropout,
        output_units=int(y_train.shape[-1]) if y_train.ndim > 1 else 1,
    )
    if args.model in {"lstm", "lstm_multifeature"}:
        history = _fit_with_train_on_batch(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        loss, mae = _evaluate_in_batches(
            model, X_test, y_test, int(args.batch_size)
        )
        y_pred = _model_predict_array(model, X_test)
    else:
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        ).history
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_test, dtype=np.float32)
    mse_norm = float(np.mean((y_pred - y_true) ** 2))
    mae_norm = float(np.mean(np.abs(y_pred - y_true)))

    if target_mode == "price_level" and scaler_values:
        y_true_final = y_true.reshape(-1)
        y_pred_final = y_pred.reshape(-1)
        min_value = float(scaler_values["min"][0])
        max_value = float(scaler_values["max"][0])
        scale = max_value - min_value
        if scale > 0:
            y_true_out = (y_true_final * scale) + min_value
            y_pred_out = (y_pred_final * scale) + min_value
        else:
            y_true_out = np.full_like(y_true_final, min_value, dtype=np.float32)
            y_pred_out = np.full_like(y_pred_final, min_value, dtype=np.float32)
        residuals_price = (y_pred_out - y_true_out).astype(np.float32)
    elif args.model == "lstm":
        y_true_seq = y_true.reshape(len(y_true), -1)
        y_pred_seq = y_pred.reshape(len(y_pred), -1)
        last_known_close = close_values[idx_test - int(args.horizon)].astype(np.float32)
        true_paths = last_known_close[:, None] * np.exp(
            np.cumsum(y_true_seq, axis=1)
        )
        pred_paths = last_known_close[:, None] * np.exp(
            np.cumsum(y_pred_seq, axis=1)
        )
        y_true_out = true_paths[:, -1].astype(np.float32)
        y_pred_out = pred_paths[:, -1].astype(np.float32)
        residuals_price = (pred_paths[:, -1] - true_paths[:, -1]).astype(np.float32)
        residual_std_per_step = np.std(pred_paths - true_paths, axis=0, ddof=0).astype(
            np.float32
        )
    else:
        prev_close = close_values[idx_test - 1].astype(np.float32)
        y_true_final = y_true.reshape(-1)
        y_pred_final = y_pred.reshape(-1)
        y_true_out = (prev_close * np.exp(y_true_final)).astype(np.float32)
        y_pred_out = (prev_close * np.exp(y_pred_final)).astype(np.float32)
        residuals_price = (y_pred_out - y_true_out).astype(np.float32)

    mse_price = float(np.mean((y_pred_out - y_true_out) ** 2))
    mae_price = float(np.mean(np.abs(y_pred_out - y_true_out)))
    rmse_price = float(np.sqrt(mse_price))
    residual_std_price = float(np.std(residuals_price, ddof=0))
    safe_mask = np.abs(y_true_out) > 1e-8
    if np.any(safe_mask):
        mape_pct = float(
            np.mean(
                np.abs(
                    (y_pred_out[safe_mask] - y_true_out[safe_mask])
                    / y_true_out[safe_mask]
                )
            )
            * 100.0
        )
    else:
        mape_pct = float("nan")

    forecast_path = None
    forecast_value = None
    forecast_lower = None
    forecast_upper = None
    forecast_lower_path = None
    forecast_upper_path = None
    forecast_ci_half_width = None
    if target_mode == "price_level" and scaler_values:
        forecast_path = _recursive_lstm_forecast_path(
            model=model,
            close_values=close_values,
            scaler_values=scaler_values,
            window_size=args.window_size,
            forecast_steps=args.horizon,
        )
        if forecast_path is not None and len(forecast_path) > 0:
            forecast_value = float(forecast_path[-1])
    elif args.model == "lstm" and scaler_values:
        forecast_path = _direct_return_forecast_path(
            model=model,
            close_values=close_values,
            scaler_values=scaler_values,
            window_size=args.window_size,
            forecast_steps=args.horizon,
        )
        if forecast_path is not None and len(forecast_path) > 0:
            forecast_value = float(forecast_path[-1])
    elif args.model == "lstm_multifeature" and scaler_values:
        forecast_path = _recursive_multifeature_return_forecast_path(
            model=model,
            close_values=close_values,
            feature_history=context.get("feature_history"),
            scaler_values=scaler_values,
            window_size=args.window_size,
            forecast_steps=args.horizon,
        )
        if forecast_path is not None and len(forecast_path) > 0:
            forecast_value = float(forecast_path[-1])
    elif scaler_values:
        forecast_path = _recursive_return_forecast_path(
            model=model,
            close_values=close_values,
            scaler_values=scaler_values,
            window_size=args.window_size,
            forecast_steps=args.horizon,
        )
        if forecast_path is not None and len(forecast_path) > 0:
            forecast_value = float(forecast_path[-1])

    if forecast_path is not None and len(forecast_path) > 0:
        if args.model == "lstm" and "residual_std_per_step" in locals():
            per_step_std = np.asarray(
                residual_std_per_step, dtype=np.float32
            ).reshape(-1)
            if per_step_std.size >= len(forecast_path):
                forecast_ci_half_width_path = (1.96 * per_step_std[: len(forecast_path)]).astype(
                    np.float32
                )
                forecast_lower_path = (
                    np.asarray(forecast_path, dtype=np.float32)
                    - forecast_ci_half_width_path
                ).astype(np.float32)
                forecast_upper_path = (
                    np.asarray(forecast_path, dtype=np.float32)
                    + forecast_ci_half_width_path
                ).astype(np.float32)
            else:
                (
                    forecast_lower_path,
                    forecast_upper_path,
                    forecast_ci_half_width_path,
                ) = _build_confidence_interval_paths(
                    forecast_values=forecast_path,
                    residual_std_price=residual_std_price,
                )
        else:
            (
                forecast_lower_path,
                forecast_upper_path,
                forecast_ci_half_width_path,
            ) = _build_confidence_interval_paths(
                forecast_values=forecast_path,
                residual_std_price=residual_std_price,
            )
        if forecast_lower_path is not None and len(forecast_lower_path) > 0:
            forecast_lower = float(forecast_lower_path[-1])
            forecast_upper = float(forecast_upper_path[-1])
            forecast_ci_half_width = float(forecast_ci_half_width_path[-1])

    if args.save_model:
        output_path = Path(args.save_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
    save_scaler(args.save_scaler, scaler_values)
    save_eval(args.save_eval, y_true_out, y_pred_out, idx_test)
    save_forecast(
        args.save_forecast,
        forecast_value,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
    )
    save_forecast_path(
        args.save_forecast_path,
        forecast_path,
        forecast_lower_values=forecast_lower_path,
        forecast_upper_values=forecast_upper_path,
    )
    return {
        "metrics_version": 7,
        "target_mode": target_mode,
        "loss_eval": float(loss),
        "mae_eval": float(mae),
        "mse": float(mse_price),
        "mae": float(mae_price),
        "mse_price": float(mse_price),
        "mae_price": float(mae_price),
        "rmse_price": float(rmse_price),
        "mape_pct": float(mape_pct),
        "mse_norm": float(mse_norm),
        "mae_norm": float(mae_norm),
        "forecast_residual_std_price": float(residual_std_price),
        "forecast_ci_95_half_width": float(forecast_ci_half_width)
        if forecast_ci_half_width is not None
        else float("nan"),
        "epochs_trained": int(len(history.get("loss", []))),
        "lstm_units1": int(args.lstm_units1),
        "lstm_units2": int(args.lstm_units2),
        "lstm_dropout": float(args.lstm_dropout),
    }


def main():
    args = parse_args()
    try:
        metrics = train_and_eval(args)
        print(json.dumps({"ok": True, "metrics": metrics}, ensure_ascii=True))
    except Exception as exc:
        # Return a concise machine-readable error for the Streamlit caller.
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
