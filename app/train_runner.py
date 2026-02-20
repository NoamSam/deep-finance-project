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
    last_start = len(values) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        target_index = end + horizon - 1
        X.append(values[start:end])
        y.append(values[target_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def split_datasets(X, y, test_size, val_size):
    n = len(X)
    test_count = int(n * test_size)
    val_count = int(n * val_size)
    train_end = n - test_count - val_count
    val_end = n - test_count

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train/validation/test split produced an empty set."
        )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def scale_minmax(values):
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    scale = max_value - min_value
    if scale <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / scale).astype(np.float32)


def scale_standard_by_train(X_train, X_val, X_test):
    flat_train = X_train.reshape(X_train.shape[0], -1)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0

    def transform(values):
        flat = values.reshape(values.shape[0], -1)
        return ((flat - mean) / std).reshape(values.shape)

    return transform(X_train), transform(X_val), transform(X_test)


def build_datasets(close_values, args):
    if args.model == "lstm":
        scaled = scale_minmax(close_values)
        X, y = make_windows(scaled, args.window_size, args.horizon)
    else:
        safe_close = np.maximum(close_values, 1e-8)
        returns = np.diff(np.log(safe_close))
        X, y = make_windows(returns, args.window_size, args.horizon)

    if len(X) == 0:
        needed = args.window_size + args.horizon
        raise ValueError(
            f"Not enough data to create training windows. "
            f"Got {len(close_values)} points, need at least {needed}."
        )

    X = X[..., np.newaxis]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_datasets(
        X, y, args.test_size, args.val_size
    )
    if args.model != "lstm":
        X_train, X_val, X_test = scale_standard_by_train(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_eval(args):
    close_values = load_close_series(args.csv)
    X_train, y_train, X_val, y_val, X_test, y_test = build_datasets(
        close_values, args
    )

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
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return {"mse": float(loss), "mae": float(mae)}


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
