import numpy as np


def compute_log_returns(series):
    values = series.astype(float).values
    returns = np.diff(np.log(values))
    return returns


def make_windows(values, window_size, horizon):
    X = []
    y = []
    last_start = len(values) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        target_index = end + horizon - 1
        X.append(values[start:end])
        y.append(values[target_index])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def train_val_test_split(X, y, test_size, val_size):
    n = len(X)
    test_count = int(n * test_size)
    val_count = int(n * val_size)
    train_end = n - test_count - val_count
    val_end = n - test_count

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
