from datetime import datetime
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

try:
    from app.update_curves import DATA_DIR, fetch_asset, update_assets
except ModuleNotFoundError:
    from update_curves import DATA_DIR, fetch_asset, update_assets

BASE_TICKERS_PATH = Path(__file__).resolve().parent / "base_tickers.txt"
MODEL_TYPE_MAP = {
    "LSTM": "lstm",
    "CNN": "cnn",
    "CNN puis LSTM": "cnn_lstm",
}
MIN_WINDOW_SIZE = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
TRAINING_CACHE_DIR = Path(__file__).resolve().parent / "data" / "training_cache"
MAX_DEFAULT_PLOT_ASSETS = 12
DEFAULT_MAX_PLOT_POINTS = 1200
PLOT_RESAMPLE_RULES = {
    "Journalier": "D",
    "Hebdomadaire": "W-FRI",
    "Mensuel": "M",
}
HORIZON_OPTIONS = {
    "1 jour": 1,
    "3 jours": 3,
    "5 jours": 5,
    "7 jours": 7,
    "2 semaines": 10,
    "1 mois": 21,
    "3 mois": 63,
    "6 mois": 126,
    "1 an": 252,
    "3 ans": 756,
}

ASSET_CATEGORIES = {
    "Beautiful Seven (US)": [
        "AAPL",
        "AMZN",
        "GOOG",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
    ],
    "Tout CAC 40": [
        "ACA.PA",
        "AC.PA",
        "AI.PA",
        "AIR.PA",
        "ALO.PA",
        "BN.PA",
        "BNP.PA",
        "CA.PA",
        "CAP.PA",
        "CS.PA",
        "DG.PA",
        "DSY.PA",
        "EDEN.PA",
        "EN.PA",
        "ENGI.PA",
        "ERF.PA",
        "GLE.PA",
        "HO.PA",
        "KER.PA",
        "LR.PA",
        "MC.PA",
        "MT.PA",
        "ML.PA",
        "OR.PA",
        "PUB.PA",
        "RI.PA",
        "RMS.PA",
        "RNO.PA",
        "RUI.PA",
        "SAF.PA",
        "SAN.PA",
        "SGO.PA",
        "STMPA.PA",
        "SU.PA",
        "TEP.PA",
        "TTE.PA",
        "URW.PA",
        "VIE.PA",
        "VIV.PA",
        "WLN.PA",
    ],
    "Luxe": [
        "MC.PA",
        "RMS.PA",
        "KER.PA",
        "RCO.PA",
        "STMPA.PA",
    ],
    "Industrie": [
        "AI.PA",
        "EN.PA",
        "SU.PA",
        "VIE.PA",
        "DG.PA",
        "RNO.PA",
        "STLAP.PA",
        "FRVIA.PA",
        "AIR.PA",
    ],
    "Defense": [
        "AIR.PA",
        "HO.PA",
        "SAF.PA",
        "EXENS.PA",
        "TE.PA",
    ],
}


def load_base_tickers():
    if not BASE_TICKERS_PATH.exists():
        return []
    with BASE_TICKERS_PATH.open("r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


def format_base_ticker(ticker):
    return ticker.strip().upper()


def normalize_ticker(ticker, base_tickers):
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    if "." not in ticker and f"{ticker}.PA" in base_tickers:
        return f"{ticker}.PA"
    return ticker


def normalize_category_name(name):
    return "".join(
        char.lower() if char.isalnum() else "_" for char in name
    ).strip("_")


def build_category_assets(known_assets):
    known_set = set(known_assets)
    category_assets = {}
    for category, tickers in ASSET_CATEGORIES.items():
        category_assets[category] = [
            ticker for ticker in tickers if ticker in known_set
        ]
    return category_assets


def get_asset_path(ticker):
    safe_name = ticker.replace("/", "_")
    return DATA_DIR / f"{safe_name}.csv"


def get_asset_cache_token(ticker):
    path = get_asset_path(ticker)
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


def _hash_training_frame(frame):
    serialized = frame.copy()
    serialized["Date"] = pd.to_datetime(serialized["Date"]).dt.strftime(
        "%Y-%m-%d"
    )
    serialized["Close"] = pd.to_numeric(
        serialized["Close"], errors="coerce"
    ).round(10)
    payload = serialized[["Date", "Close"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_training_cache_key(ticker, model_code, config, data_hash):
    cache_payload = {
        "ticker": ticker,
        "model": model_code,
        "window_size": int(config["window_size"]),
        "epochs": int(config["epochs"]),
        "horizon": int(config.get("horizon", 1)),
        "test_size": float(config.get("test_size", DEFAULT_TEST_SIZE)),
        "val_size": float(config.get("val_size", DEFAULT_VAL_SIZE)),
        "batch_size": int(config.get("batch_size", 32)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "data_hash": data_hash,
    }
    digest = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _get_training_artifact_paths(ticker, cache_key):
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    ticker_dir = TRAINING_CACHE_DIR / safe_ticker
    return {
        "meta": ticker_dir / f"{cache_key}.json",
        "model": ticker_dir / f"{cache_key}.keras",
        "scaler": ticker_dir / f"{cache_key}_scaler.npz",
        "eval": ticker_dir / f"{cache_key}_eval.npz",
        "forecast": ticker_dir / f"{cache_key}_forecast.npz",
    }


def _load_cached_training_result(paths):
    meta_path = paths["meta"]
    if not meta_path.exists():
        return None
    if (
        not paths["model"].exists()
        or not paths["scaler"].exists()
        or not paths["eval"].exists()
    ):
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = payload.get("metrics", {})
    if "mse" not in metrics or "mae" not in metrics:
        return None
    return payload


def _load_eval_frame(eval_path, frame):
    if not eval_path.exists():
        return None
    try:
        payload = np.load(eval_path)
        target_indices = payload["target_close_indices"].astype(np.int64)
        y_true = payload["y_true"].astype(np.float64)
        y_pred = payload["y_pred"].astype(np.float64)
    except Exception:
        return None

    usable_length = min(len(target_indices), len(y_true), len(y_pred))
    if usable_length <= 0:
        return None

    target_indices = target_indices[:usable_length]
    y_true = y_true[:usable_length]
    y_pred = y_pred[:usable_length]

    valid_mask = (target_indices >= 0) & (target_indices < len(frame))
    if not np.any(valid_mask):
        return None

    target_indices = target_indices[valid_mask]
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    plot_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["Date"].iloc[target_indices].to_numpy()),
            "Reel": y_true,
            "Predit": y_pred,
        }
    ).sort_values("Date")
    return plot_frame


def _save_training_metadata(
    paths,
    ticker,
    model_code,
    config,
    metrics,
    data_hash,
    start_date,
    end_date,
):
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "model": model_code,
        "config": {
            "window_size": int(config["window_size"]),
            "epochs": int(config["epochs"]),
            "horizon": int(config.get("horizon", 1)),
            "test_size": float(config.get("test_size", DEFAULT_TEST_SIZE)),
            "val_size": float(config.get("val_size", DEFAULT_VAL_SIZE)),
            "batch_size": int(config.get("batch_size", 32)),
            "learning_rate": float(config.get("learning_rate", 1e-3)),
        },
        "date_filter": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "data_hash": data_hash,
        "metrics": {
            "mse": float(metrics.get("mse", float("nan"))),
            "mae": float(metrics.get("mae", float("nan"))),
            "epochs_trained": int(metrics.get("epochs_trained", 0)),
        },
        "artifacts": {
            "model": str(paths["model"]),
            "scaler": str(paths["scaler"]),
            "eval": str(paths["eval"]),
            "forecast": str(paths["forecast"]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    paths["meta"].write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


@st.cache_data(show_spinner=False)
def load_asset_frame(ticker, start_date=None, end_date=None, cache_token=None):
    # cache_token is used to invalidate Streamlit cache when CSV files are replaced.
    _ = cache_token
    path = get_asset_path(ticker)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = fetch_asset(ticker, start=start_date, end=end_date)

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "Close"})
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column for {ticker}")

    return df[["Date", "Close"]]


def _build_temp_training_csv(ticker, frame):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    path = (
        Path(tempfile.gettempdir())
        / f"deep_finance_train_{safe_ticker}_{timestamp}.csv"
    )
    frame.to_csv(path, index=False)
    return path


def run_training_subprocess(
    csv_path,
    model_code,
    config,
    model_output_path=None,
    scaler_output_path=None,
    eval_output_path=None,
    forecast_output_path=None,
):
    runner_path = Path(__file__).resolve().parent / "train_runner.py"
    command = [
        sys.executable,
        str(runner_path),
        "--csv",
        str(csv_path),
        "--model",
        model_code,
        "--window-size",
        str(config["window_size"]),
        "--epochs",
        str(config["epochs"]),
        "--horizon",
        str(config.get("horizon", 1)),
        "--test-size",
        str(config.get("test_size", 0.2)),
        "--val-size",
        str(config.get("val_size", 0.1)),
        "--batch-size",
        str(config.get("batch_size", 32)),
        "--learning-rate",
        str(config.get("learning_rate", 1e-3)),
    ]
    if model_output_path:
        command.extend(["--save-model", str(model_output_path)])
    if scaler_output_path:
        command.extend(["--save-scaler", str(scaler_output_path)])
    if eval_output_path:
        command.extend(["--save-eval", str(eval_output_path)])
    if forecast_output_path:
        command.extend(["--save-forecast", str(forecast_output_path)])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        details = completed.stderr.strip()
        raise RuntimeError(details or "Training subprocess produced no output")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unable to parse training metrics: {lines[-1]}"
        ) from exc

    if not payload.get("ok"):
        message = payload.get("error") or "Training subprocess failed"
        raise RuntimeError(message)

    if completed.returncode != 0:
        # Non-zero with a parsed payload is still considered a failure.
        message = payload.get("error") or completed.stderr.strip()
        raise RuntimeError(message or "Training subprocess failed")

    return payload["metrics"]


def _required_min_windows(test_size, val_size):
    n = 1
    while True:
        test_count = int(n * test_size)
        val_count = int(n * val_size)
        train_count = n - test_count - val_count
        if test_count >= 1 and val_count >= 1 and train_count >= 1:
            return n
        n += 1


def _effective_window_size(
    total_points, requested_window, horizon, required_windows
):
    max_window = total_points - horizon - (required_windows - 1)
    if max_window < MIN_WINDOW_SIZE:
        return None
    return min(requested_window, max_window)


def _downsample_plot_frame(frame, max_points):
    if frame.empty or max_points <= 0 or len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    reduced = frame.iloc[::step]
    if reduced.index[-1] != frame.index[-1]:
        reduced = pd.concat([reduced, frame.iloc[[-1]]])
    return reduced[~reduced.index.duplicated(keep="last")]


def _append_future_prediction_row(plot_frame, forecast_date, forecast_value):
    future_row = pd.DataFrame(
        {
            "Date": [pd.to_datetime(forecast_date)],
            "Reel": [np.nan],
            "Predit": [float(forecast_value)],
        }
    )
    if plot_frame is None or plot_frame.empty:
        return future_row
    merged = pd.concat([plot_frame, future_row], ignore_index=True)
    merged = (
        merged.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return merged


def _append_future_prediction_path(plot_frame, future_points):
    if future_points is None or future_points.empty:
        return plot_frame
    output = plot_frame
    for _, row in future_points.iterrows():
        output = _append_future_prediction_row(
            output, row["Date"], row["Prix predit"]
        )
    return output


def _load_forecast_value(forecast_path):
    if not forecast_path.exists():
        return None
    try:
        payload = np.load(forecast_path)
        if "forecast_value" not in payload:
            return None
        values = payload["forecast_value"]
        if len(values) == 0:
            return None
        return float(values[0])
    except Exception:
        return None


def _predict_lstm_recursive_path(
    frame, model_path, scaler_path, window_size, forecast_steps
):
    if not model_path.exists() or not scaler_path.exists() or forecast_steps <= 0:
        return None

    close_values = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close_values) < window_size:
        return None

    try:
        scaler_payload = np.load(scaler_path)
        min_value = float(scaler_payload["min"][0])
        max_value = float(scaler_payload["max"][0])
    except Exception:
        return None

    scale = max_value - min_value
    close_array = close_values.to_numpy(dtype=np.float32)
    if scale > 0:
        scaled_values = ((close_array - min_value) / scale).astype(np.float32)
    else:
        scaled_values = np.zeros_like(close_array, dtype=np.float32)

    window = scaled_values[-window_size:].tolist()

    try:
        from tensorflow.keras import backend as keras_backend
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        predictions = []
        for step in range(1, forecast_steps + 1):
            model_input = np.asarray(
                window[-window_size:], dtype=np.float32
            ).reshape(1, window_size, 1)
            next_scaled = float(model.predict(model_input, verbose=0)[0][0])
            window.append(next_scaled)
            if scale > 0:
                next_value = (next_scaled * scale) + min_value
            else:
                next_value = min_value
            next_date = pd.to_datetime(frame["Date"].max()) + pd.offsets.BDay(step)
            predictions.append(
                {
                    "Etape": step,
                    "Date": next_date,
                    "Prix predit": float(next_value),
                }
            )
        keras_backend.clear_session()
    except Exception:
        return None

    return pd.DataFrame(predictions)


def _ensure_lstm_recursive_forecast(
    ticker,
    frame,
    data_hash,
    asset_config,
    force_retrain,
    start_date,
    end_date,
):
    recursive_config = {**asset_config, "horizon": 1}
    recursive_key = _build_training_cache_key(
        ticker, "lstm", recursive_config, data_hash
    )
    recursive_paths = _get_training_artifact_paths(ticker, recursive_key)

    recursive_cached = None
    if not force_retrain:
        recursive_cached = _load_cached_training_result(recursive_paths)
        if (
            recursive_cached is not None
            and not recursive_paths["forecast"].exists()
        ):
            recursive_cached = None

    if recursive_cached is None:
        temp_csv_path = _build_temp_training_csv(ticker, frame)
        try:
            recursive_metrics = run_training_subprocess(
                temp_csv_path,
                "lstm",
                recursive_config,
                model_output_path=recursive_paths["model"],
                scaler_output_path=recursive_paths["scaler"],
                eval_output_path=recursive_paths["eval"],
                forecast_output_path=recursive_paths["forecast"],
            )
        finally:
            temp_csv_path.unlink(missing_ok=True)
        _save_training_metadata(
            paths=recursive_paths,
            ticker=ticker,
            model_code="lstm",
            config=recursive_config,
            metrics=recursive_metrics,
            data_hash=data_hash,
            start_date=start_date,
            end_date=end_date,
        )

    return _predict_lstm_recursive_path(
        frame=frame,
        model_path=recursive_paths["model"],
        scaler_path=recursive_paths["scaler"],
        window_size=recursive_config["window_size"],
        forecast_steps=asset_config["horizon"],
    )


def train_selected_assets(state):
    model_code = MODEL_TYPE_MAP[state["model_type"]]
    requested_window_size = int(state["window_size"])
    force_retrain = bool(state.get("force_retrain", False))
    horizon_steps = int(state.get("horizon_steps", 1))
    base_config = {
        "epochs": int(state["epochs"]),
        "verbose": 0,
        "horizon": horizon_steps,
        "test_size": DEFAULT_TEST_SIZE,
        "val_size": DEFAULT_VAL_SIZE,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }
    required_windows = _required_min_windows(
        base_config["test_size"], base_config["val_size"]
    )

    results = []
    failures = []
    prediction_plots = {}
    future_forecasts = []
    total = len(state["assets"])
    progress = st.progress(0)
    status_line = st.empty()

    for index, ticker in enumerate(state["assets"], start=1):
        status_line.caption(f"Entrainement: {ticker} ({index}/{total})")
        temp_csv_path = None
        try:
            frame = load_asset_frame(
                ticker,
                state["start_date"],
                state["end_date"],
                get_asset_cache_token(ticker),
            )
            effective_window = _effective_window_size(
                total_points=len(frame),
                requested_window=requested_window_size,
                horizon=base_config["horizon"],
                required_windows=required_windows,
            )
            if effective_window is None:
                required_points = (
                    MIN_WINDOW_SIZE
                    + base_config["horizon"]
                    + (required_windows - 1)
                )
                failures.append(
                    {
                        "ticker": ticker,
                        "error": (
                            "Pas assez de points pour entrainer. "
                            f"Points={len(frame)}, requis>={required_points}."
                        ),
                    }
                )
                continue

            asset_config = {
                **base_config,
                "window_size": int(effective_window),
            }
            data_hash = _hash_training_frame(frame)
            cache_key = _build_training_cache_key(
                ticker, model_code, asset_config, data_hash
            )
            artifact_paths = _get_training_artifact_paths(ticker, cache_key)
            cached_payload = None
            if not force_retrain:
                cached_payload = _load_cached_training_result(artifact_paths)
                if (
                    cached_payload is not None
                    and model_code == "lstm"
                    and not artifact_paths["forecast"].exists()
                ):
                    cached_payload = None
            if cached_payload:
                cached_metrics = cached_payload["metrics"]
                cached_plot_frame = _load_eval_frame(artifact_paths["eval"], frame)
                if model_code == "lstm":
                    recursive_future = _ensure_lstm_recursive_forecast(
                        ticker=ticker,
                        frame=frame,
                        data_hash=data_hash,
                        asset_config=asset_config,
                        force_retrain=force_retrain,
                        start_date=state["start_date"],
                        end_date=state["end_date"],
                    )
                    if recursive_future is not None and not recursive_future.empty:
                        cached_plot_frame = _append_future_prediction_path(
                            cached_plot_frame, recursive_future
                        )
                        for _, row in recursive_future.iterrows():
                            future_forecasts.append(
                                {
                                    "Actif": ticker,
                                    "Etape": int(row["Etape"]),
                                    "Date prevision": pd.to_datetime(
                                        row["Date"]
                                    ).strftime("%Y-%m-%d"),
                                    "Prix predit": float(row["Prix predit"]),
                                    "Horizon": state.get(
                                        "horizon_label", "1 jour"
                                    ),
                                    "Source": "Cache",
                                }
                            )
                    else:
                        forecast_value = _load_forecast_value(
                            artifact_paths["forecast"]
                        )
                        if forecast_value is not None:
                            forecast_date = pd.to_datetime(frame["Date"].max()) + pd.offsets.BDay(
                                int(asset_config["horizon"])
                            )
                            cached_plot_frame = _append_future_prediction_row(
                                cached_plot_frame,
                                forecast_date,
                                forecast_value,
                            )
                            future_forecasts.append(
                                {
                                    "Actif": ticker,
                                    "Etape": int(asset_config["horizon"]),
                                    "Date prevision": forecast_date.strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "Prix predit": float(forecast_value),
                                    "Horizon": state.get(
                                        "horizon_label", "1 jour"
                                    ),
                                    "Source": "Cache",
                                }
                            )
                if cached_plot_frame is not None and not cached_plot_frame.empty:
                    prediction_plots[ticker] = cached_plot_frame
                results.append(
                    {
                        "Actif": ticker,
                        "MSE": float(cached_metrics.get("mse", float("nan"))),
                        "MAE": float(cached_metrics.get("mae", float("nan"))),
                        "Points utilises": int(len(frame)),
                        "Fenetre utilisee": int(effective_window),
                        "Source": "Cache",
                    }
                )
                continue

            temp_csv_path = _build_temp_training_csv(ticker, frame)
            metrics = run_training_subprocess(
                temp_csv_path,
                model_code,
                asset_config,
                model_output_path=artifact_paths["model"],
                scaler_output_path=artifact_paths["scaler"],
                eval_output_path=artifact_paths["eval"],
                forecast_output_path=artifact_paths["forecast"],
            )
            _save_training_metadata(
                paths=artifact_paths,
                ticker=ticker,
                model_code=model_code,
                config=asset_config,
                metrics=metrics,
                data_hash=data_hash,
                start_date=state["start_date"],
                end_date=state["end_date"],
            )
            trained_plot_frame = _load_eval_frame(artifact_paths["eval"], frame)
            if model_code == "lstm":
                recursive_future = _ensure_lstm_recursive_forecast(
                    ticker=ticker,
                    frame=frame,
                    data_hash=data_hash,
                    asset_config=asset_config,
                    force_retrain=force_retrain,
                    start_date=state["start_date"],
                    end_date=state["end_date"],
                )
                if recursive_future is not None and not recursive_future.empty:
                    trained_plot_frame = _append_future_prediction_path(
                        trained_plot_frame, recursive_future
                    )
                    for _, row in recursive_future.iterrows():
                        future_forecasts.append(
                            {
                                "Actif": ticker,
                                "Etape": int(row["Etape"]),
                                "Date prevision": pd.to_datetime(
                                    row["Date"]
                                ).strftime("%Y-%m-%d"),
                                "Prix predit": float(row["Prix predit"]),
                                "Horizon": state.get(
                                    "horizon_label", "1 jour"
                                ),
                                "Source": "Train",
                            }
                        )
                else:
                    forecast_value = _load_forecast_value(artifact_paths["forecast"])
                    if forecast_value is not None:
                        forecast_date = pd.to_datetime(frame["Date"].max()) + pd.offsets.BDay(
                            int(asset_config["horizon"])
                        )
                        trained_plot_frame = _append_future_prediction_row(
                            trained_plot_frame,
                            forecast_date,
                            forecast_value,
                        )
                        future_forecasts.append(
                            {
                                "Actif": ticker,
                                "Etape": int(asset_config["horizon"]),
                                "Date prevision": forecast_date.strftime(
                                    "%Y-%m-%d"
                                ),
                                "Prix predit": float(forecast_value),
                                "Horizon": state.get(
                                    "horizon_label", "1 jour"
                                ),
                                "Source": "Train",
                            }
                        )
            if trained_plot_frame is not None and not trained_plot_frame.empty:
                prediction_plots[ticker] = trained_plot_frame
            results.append(
                {
                    "Actif": ticker,
                    "MSE": float(metrics.get("mse", float("nan"))),
                    "MAE": float(metrics.get("mae", float("nan"))),
                    "Points utilises": int(len(frame)),
                    "Fenetre utilisee": int(effective_window),
                    "Source": "Train",
                }
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
        finally:
            if temp_csv_path and temp_csv_path.exists():
                temp_csv_path.unlink(missing_ok=True)
            progress.progress(index / total)

    progress.empty()
    status_line.empty()
    return results, failures, prediction_plots, future_forecasts


def render_training_section(state):
    st.markdown("### Entrainement du modele")
    requested_window_size = int(state["window_size"])
    horizon_label = state.get("horizon_label", "1 jour")
    horizon_steps = int(state.get("horizon_steps", 1))
    st.caption(
        "Utilise les parametres de la sidebar "
        "(fenetre, epochs, modele, horizon) sur les actifs selectionnes. "
        "Les runs sont mis en cache par actif/configuration."
    )

    if st.button("Entrainer les actifs selectionnes", use_container_width=True):
        if not state["assets"]:
            st.info("Aucun actif selectionne.")
        else:
            with st.spinner("Entrainement en cours..."):
                (
                    results,
                    failures,
                    prediction_plots,
                    future_forecasts,
                ) = train_selected_assets(state)
            st.session_state.training_results = results
            st.session_state.training_failures = failures
            st.session_state.training_prediction_plots = prediction_plots
            st.session_state.training_future_forecasts = future_forecasts
            st.session_state.training_meta = {
                "model_type": state["model_type"],
                "window_size": requested_window_size,
                "epochs": int(state["epochs"]),
                "horizon_label": horizon_label,
                "horizon_steps": horizon_steps,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    results = st.session_state.get("training_results", [])
    failures = st.session_state.get("training_failures", [])
    prediction_plots = st.session_state.get("training_prediction_plots", {})
    future_forecasts = st.session_state.get("training_future_forecasts", [])
    meta = st.session_state.get("training_meta")

    if meta:
        st.caption(
            f"Dernier run ({meta['timestamp']}) | "
            f"Modele: {meta['model_type']} | "
            f"Fenetre: {meta['window_size']} | "
            f"Horizon: {meta.get('horizon_label', '1 jour')} ({meta.get('horizon_steps', 1)} pas) | "
            f"Epochs: {meta['epochs']}"
        )

    if results:
        df_results = pd.DataFrame(results).sort_values("Actif")
        st.dataframe(df_results, use_container_width=True)
    if failures:
        with st.expander("Erreurs d'entrainement"):
            for item in failures:
                st.write(f"{item['ticker']}: {item['error']}")
    if prediction_plots:
        st.markdown("#### Reel vs predit")
        for ticker in sorted(prediction_plots.keys()):
            plot_frame = prediction_plots[ticker]
            if plot_frame is None or plot_frame.empty:
                continue
            with st.expander(f"{ticker} - reel vs predit", expanded=False):
                chart_data = (
                    plot_frame.set_index("Date")[["Reel", "Predit"]]
                    .rename(
                        columns={
                            "Reel": f"{ticker} reel",
                            "Predit": f"{ticker} predit",
                        }
                    )
                )
                st.line_chart(chart_data, use_container_width=True)
    if future_forecasts:
        st.markdown("#### Donnees predites")
        forecast_df = pd.DataFrame(future_forecasts).sort_values(
            ["Actif", "Etape"]
        )
        st.dataframe(forecast_df, use_container_width=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## Deep Learning &\nFinance")
        st.caption("Prediction Quantitative Multi-Actifs")
        st.markdown("---")

        st.markdown("### Actifs selectionnes")
        base_tickers = load_base_tickers()
        base_ticker_set = set(base_tickers)
        known_assets = [format_base_ticker(ticker) for ticker in base_tickers]
        if DATA_DIR.exists():
            for path in DATA_DIR.glob("*.csv"):
                known_assets.append(normalize_ticker(path.stem, base_ticker_set))
        known_assets = sorted(set(filter(None, known_assets)))
        category_assets = build_category_assets(known_assets)
        available_categories = [
            name
            for name, assets in category_assets.items()
            if assets
        ]
        default_categories = [
            name
            for name in ["Beautiful Seven (US)", "Tout CAC 40"]
            if name in available_categories
        ]
        selected_categories = st.multiselect(
            "Categories d'actions",
            options=available_categories,
            default=default_categories,
            key="selected_asset_categories",
            placeholder="Choisissez une ou plusieurs categories",
        )

        if "assets_initialized" not in st.session_state:
            initial_selected = {"AIR.PA", "BNP.PA"}
            if DATA_DIR.exists():
                for path in DATA_DIR.glob("*.csv"):
                    normalized = normalize_ticker(path.stem, base_ticker_set)
                    if normalized:
                        initial_selected.add(normalized)
            st.session_state.assets_initialized = True
        else:
            initial_selected = set()

        for category_name in selected_categories:
            category_tickers = category_assets[category_name]
            if not category_tickers:
                continue
            with st.expander(
                f"{category_name} ({len(category_tickers)} actifs)",
                expanded=False,
            ):
                category_key = normalize_category_name(category_name)
                check_col, uncheck_col = st.columns(2)
                if check_col.button(
                    "Cocher toute la categorie",
                    key=f"check_category_{category_key}",
                    use_container_width=True,
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = True
                if uncheck_col.button(
                    "Decocher toute la categorie",
                    key=f"uncheck_category_{category_key}",
                    use_container_width=True,
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = False

        visible_assets = sorted(
            {
                asset
                for category_name in selected_categories
                for asset in category_assets[category_name]
            }
        )

        if visible_assets:
            select_col, clear_col = st.columns(2)
            if select_col.button(
                "Cocher la selection", use_container_width=True
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = True
            if clear_col.button(
                "Decocher la selection", use_container_width=True
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = False
        else:
            st.info("Choisissez au moins une categorie d'actions.")

        selected_assets = []
        for asset in visible_assets:
            key = f"asset_check_{asset}"
            if key not in st.session_state:
                st.session_state[key] = asset in initial_selected
            if st.checkbox(asset, key=key):
                selected_assets.append(asset)

        stale_keys = [
            key
            for key in st.session_state
            if key.startswith("asset_check_")
            and key[len("asset_check_") :] not in known_assets
        ]
        for key in stale_keys:
            st.session_state.pop(key, None)

        if not selected_assets:
            st.write("Aucun actif selectionne")

        st.markdown("### Date de debut")
        start_date = st.date_input("Date de debut", value=None)

        st.markdown("### Date de fin")
        end_date = st.date_input("Date de fin", value=None)

        st.markdown("### Parametres d'entrainement")
        window_size = st.number_input(
            "Taille de la fenetre",
            min_value=5,
            max_value=365,
            value=60,
            step=1,
        )
        epochs = st.number_input(
            "Nombre d'epochs",
            min_value=1,
            max_value=500,
            value=100,
            step=1,
        )
        model_type = st.selectbox(
            "Modele",
            options=["LSTM", "CNN", "CNN puis LSTM"],
            index=0,
        )
        horizon_label = st.selectbox(
            "Horizon",
            options=list(HORIZON_OPTIONS.keys()),
            index=0,
            help=(
                "Nombre de pas de prediction. "
                "Pour les semaines/mois/annees, conversion approx. en seances de bourse."
            ),
        )
        horizon_steps = HORIZON_OPTIONS[horizon_label]
        force_retrain = st.checkbox(
            "Forcer le re-entrainement",
            value=False,
            help=(
                "Ignore les artefacts en cache et relance un entrainement complet."
            ),
        )
        if "generate_curves" not in st.session_state:
            st.session_state.generate_curves = False
        if st.button("Generer les courbes", use_container_width=True):
            st.session_state.generate_curves = True

        st.markdown("---")
        st.markdown("### Mise a jour des donnees")
        today = datetime.now().strftime("%Y-%m-%d")
        update_key = f"data_updated_{today}"
        force_update = st.checkbox("Ecraser les donnees existantes", value=False)
        if st.button("Mettre a jour toutes les donnees", use_container_width=True):
            if not selected_assets:
                st.info("Aucun actif selectionne.")
            else:
                progress_bar = st.progress(0)
                status_line = st.empty()

                def on_progress(index, total, ticker, status, error=None):
                    if total:
                        progress_bar.progress(index / total)
                    message = f"{status.upper()}: {ticker}"
                    if error:
                        message = f"{message} - {error}"
                    status_line.caption(message)

                with st.spinner("Mise a jour en cours..."):
                    result = update_assets(
                        selected_assets,
                        start=start_date,
                        end=end_date,
                        force=force_update,
                        progress=on_progress,
                    )
                load_asset_frame.clear()
                st.session_state[update_key] = True
                st.success("Mise a jour terminee.")
                st.write(
                    f"Mis a jour: {len(result['updated'])} | "
                    f"Ignores: {len(result['skipped'])} | "
                    f"Echecs: {len(result['failed'])}"
                )
                if result["skipped"]:
                    st.info(
                        f"Actifs ignores (recents): {', '.join(result['skipped'])}"
                    )
                if result["failed"]:
                    with st.expander("Details des erreurs"):
                        for item in result["failed"]:
                            st.write(f"{item['ticker']}: {item['error']}")
        if st.session_state.get(update_key, False):
            st.success(f"Mis a jour aujourd'hui ({today})")
        else:
            st.info("Pas de mise a jour aujourd'hui")

        st.markdown("### Importation des donnees")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            target_path = DATA_DIR / uploaded_file.name
            if target_path.exists() and not force_update:
                st.warning("Fichier deja present. Activez l'option d'ecrasement.")
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                load_asset_frame.clear()
                st.success(f"Fichier enregistre: {target_path.name}")

        st.markdown("### Fichiers disponibles")
        if DATA_DIR.exists():
            files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        else:
            files = []
        if files:
            st.write(", ".join(files))
        else:
            st.caption("Aucun fichier dans data/assets")

    return {
        "uploaded_file": uploaded_file,
        "assets": selected_assets,
        "start_date": start_date,
        "end_date": end_date,
        "window_size": window_size,
        "epochs": epochs,
        "model_type": model_type,
        "horizon_label": horizon_label,
        "horizon_steps": horizon_steps,
        "force_retrain": force_retrain,
        "generate_curves": st.session_state.generate_curves,
    }


def render_predictions_tab(state):
    st.subheader("Analyse Multi-Actifs")
    st.caption("Donnees historiques des prix")
    if not state["assets"]:
        st.info("Aucun actif selectionne.")
        return

    render_training_section(state)

    if not state.get("generate_curves"):
        st.markdown(
            "Selectionnez des actifs et cliquez sur \"Generer les courbes\""
        )
        return

    frames = []
    errors = []
    with st.spinner("Chargement des donnees..."):
        for ticker in state["assets"]:
            try:
                df = load_asset_frame(
                    ticker,
                    state["start_date"],
                    state["end_date"],
                    get_asset_cache_token(ticker),
                )
                df = df.set_index("Date").rename(columns={"Close": ticker})
                frames.append(df)
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")

    if errors:
        st.warning("Certaines donnees n'ont pas pu etre chargees.")
        st.write("\n".join(errors))

    if not frames:
        st.info("Aucune donnee disponible.")
        return

    combined = pd.concat(frames, axis=1).sort_index()
    if not combined.empty:
        min_date = combined.index.min().strftime("%Y-%m-%d")
        max_date = combined.index.max().strftime("%Y-%m-%d")
        st.caption(f"Plage de donnees chargees: {min_date} -> {max_date}")
    available_assets = list(combined.columns)
    if len(available_assets) <= MAX_DEFAULT_PLOT_ASSETS:
        default_plot_assets = available_assets
    else:
        default_plot_assets = available_assets[:MAX_DEFAULT_PLOT_ASSETS]
        st.info(
            f"{len(available_assets)} actifs charges. "
            f"Affichage limite a {MAX_DEFAULT_PLOT_ASSETS} actifs par defaut."
        )

    with st.expander("Options du graphe (performance)"):
        plot_assets = st.multiselect(
            "Actifs affiches sur le graphe",
            options=available_assets,
            default=default_plot_assets,
            key="plot_assets_selection",
        )
        resolution_label = st.selectbox(
            "Resolution",
            options=list(PLOT_RESAMPLE_RULES.keys()),
            index=1,
            key="plot_resolution",
        )
        max_points = st.slider(
            "Nombre max de points traces",
            min_value=200,
            max_value=4000,
            value=DEFAULT_MAX_PLOT_POINTS,
            step=100,
            key="plot_max_points",
        )

    if not plot_assets:
        st.info("Selectionnez au moins un actif pour afficher le graphe.")
    else:
        plot_data = combined[plot_assets]
        resample_rule = PLOT_RESAMPLE_RULES[resolution_label]
        if resample_rule != "D":
            plot_data = plot_data.resample(resample_rule).last()
        plot_data = _downsample_plot_frame(plot_data, max_points)
        st.caption(
            f"Graphe affiche: {plot_data.shape[1]} actifs | "
            f"{len(plot_data)} points"
        )
        st.line_chart(plot_data)
    forecast_rows = st.session_state.get("training_future_forecasts", [])
    if forecast_rows:
        with st.expander("Apercu des donnees predites"):
            forecast_df = pd.DataFrame(forecast_rows).sort_values(
                ["Actif", "Etape"]
            )
            st.dataframe(forecast_df, use_container_width=True)
    else:
        with st.expander("Apercu debut de serie"):
            st.dataframe(combined.head(10), use_container_width=True)
    st.dataframe(combined.tail(10), use_container_width=True)


def render_allocation_tab(state):
    st.subheader("Allocation Optimisee du Portefeuille")
    st.caption("Entrainez un modele pour voir l'allocation optimisee")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Parametres d'allocation")
        st.selectbox(
            "Horizon d'investissement",
            options=["", "3 mois", "6 mois", "1 an", "3 ans"],
            index=0,
        )
    with col2:
        st.markdown("### Poids optimaux")
        st.info("Aucune allocation disponible")


def main():
    st.set_page_config(
        page_title="Deep Learning Finance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state = render_sidebar()

    tab_predictions, tab_allocation = st.tabs(
        ["Visualisation des Previsions", "Allocation du Portefeuille"]
    )

    with tab_predictions:
        render_predictions_tab(state)

    with tab_allocation:
        render_allocation_tab(state)


if __name__ == "__main__":
    main()
