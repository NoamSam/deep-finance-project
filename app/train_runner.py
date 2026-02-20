import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow.keras import callbacks

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight training runner")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date, Close")
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "cnn_lstm"],
        required=True,
        help="Model type",
    )
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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


def build_datasets(close_values, args):
    if args.model == "lstm":
        scaled, scaler = scale_minmax(close_values)
        X, y, target_close_indices = make_windows(
            scaled, args.window_size, args.horizon
        )
    else:
        safe_close = np.maximum(close_values, 1e-8)
        returns = np.diff(np.log(safe_close))
        X, y, target_return_indices = make_windows(
            returns, args.window_size, args.horizon
        )
        target_close_indices = target_return_indices + 1
        scaler = None

    if len(X) == 0:
        needed = args.window_size + args.horizon
        raise ValueError(
            f"Not enough data to create training windows. "
            f"Got {len(close_values)} points, need at least {needed}."
        )

    X = X[..., np.newaxis]
    (
        (X_train, y_train, _),
        (X_val, y_val, _),
        (X_test, y_test, idx_test),
    ) = split_datasets(
        X, y, target_close_indices, args.test_size, args.val_size
    )
    if args.model != "lstm":
        X_train, X_val, X_test, scaler = scale_standard_by_train(
            X_train, X_val, X_test
        )
    return X_train, y_train, X_val, y_val, X_test, y_test, idx_test, scaler


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


def save_forecast(path, forecast_value):
    if not path or forecast_value is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        forecast_value=np.asarray([forecast_value], dtype=np.float32),
    )


def train_and_eval(args):
    close_values = load_close_series(args.csv)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        idx_test,
        scaler_values,
    ) = build_datasets(close_values, args)

    model = build_model(
        model_type=args.model,
        input_shape=X_train.shape[1:],
        learning_rate=args.learning_rate,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    y_true = y_test.reshape(-1)

    if args.model == "lstm" and scaler_values:
        min_value = float(scaler_values["min"][0])
        max_value = float(scaler_values["max"][0])
        scale = max_value - min_value
        if scale > 0:
            y_true_out = (y_true * scale) + min_value
            y_pred_out = (y_pred * scale) + min_value
        else:
            y_true_out = np.full_like(y_true, min_value, dtype=np.float32)
            y_pred_out = np.full_like(y_pred, min_value, dtype=np.float32)
    else:
        y_true_out = y_true
        y_pred_out = y_pred

    forecast_value = None
    if args.model == "lstm" and scaler_values:
        min_value = float(scaler_values["min"][0])
        max_value = float(scaler_values["max"][0])
        scale = max_value - min_value
        if scale > 0:
            scaled_close = ((close_values - min_value) / scale).astype(np.float32)
        else:
            scaled_close = np.zeros_like(close_values, dtype=np.float32)
        if len(scaled_close) >= args.window_size:
            last_window = scaled_close[-args.window_size:].reshape(
                1, args.window_size, 1
            )
            forecast_scaled = float(model.predict(last_window, verbose=0)[0][0])
            if scale > 0:
                forecast_value = (forecast_scaled * scale) + min_value
            else:
                forecast_value = min_value

    if args.save_model:
        output_path = Path(args.save_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
    save_scaler(args.save_scaler, scaler_values)
    save_eval(args.save_eval, y_true_out, y_pred_out, idx_test)
    save_forecast(args.save_forecast, forecast_value)
    return {
        "mse": float(loss),
        "mae": float(mae),
        "epochs_trained": int(len(history.history.get("loss", []))),
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
