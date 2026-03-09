from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
TRAINING_PIPELINE_VERSION = 5
TRAINING_SUBPROCESS_TIMEOUT_SEC = 900
TRAINING_CACHE_DIR = Path(__file__).resolve().parent / "data" / "training_cache"
DEFAULT_LSTM_UNITS1 = 64
DEFAULT_LSTM_UNITS2 = 32
DEFAULT_LSTM_DROPOUT = 0.2


def _safe_ticker_dir_name(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "_")


def with_training_defaults(config: dict, model_code: str | None = None) -> dict:
    normalized = dict(config)
    normalized.setdefault("test_size", DEFAULT_TEST_SIZE)
    normalized.setdefault("val_size", DEFAULT_VAL_SIZE)
    normalized.setdefault("batch_size", 32)
    normalized.setdefault("learning_rate", 5e-4)
    normalized.setdefault("timeout_sec", TRAINING_SUBPROCESS_TIMEOUT_SEC)
    if model_code in {"lstm", "lstm_multifeature"}:
        normalized.setdefault("lstm_units1", DEFAULT_LSTM_UNITS1)
        normalized.setdefault("lstm_units2", DEFAULT_LSTM_UNITS2)
        normalized.setdefault("lstm_dropout", DEFAULT_LSTM_DROPOUT)
    return normalized


def hash_training_frame(frame: pd.DataFrame) -> str:
    serialized = frame.copy()
    for column in serialized.columns:
        if column == "Date":
            serialized[column] = pd.to_datetime(serialized[column]).dt.strftime(
                "%Y-%m-%d"
            )
        else:
            serialized[column] = pd.to_numeric(
                serialized[column], errors="coerce"
            ).round(10)
    payload = serialized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_training_cache_key(
    ticker: str, model_code: str, config: dict, data_hash: str
) -> str:
    normalized = with_training_defaults(config, model_code=model_code)
    cache_payload = {
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "ticker": ticker,
        "model": model_code,
        "window_size": int(normalized["window_size"]),
        "epochs": int(normalized["epochs"]),
        "horizon": int(normalized.get("horizon", 1)),
        "test_size": float(normalized.get("test_size", DEFAULT_TEST_SIZE)),
        "val_size": float(normalized.get("val_size", DEFAULT_VAL_SIZE)),
        "batch_size": int(normalized.get("batch_size", 32)),
        "learning_rate": float(normalized.get("learning_rate", 5e-4)),
        "data_hash": data_hash,
    }
    if model_code in {"lstm", "lstm_multifeature"}:
        cache_payload.update(
            {
                "lstm_units1": int(normalized.get("lstm_units1", DEFAULT_LSTM_UNITS1)),
                "lstm_units2": int(normalized.get("lstm_units2", DEFAULT_LSTM_UNITS2)),
                "lstm_dropout": float(
                    normalized.get("lstm_dropout", DEFAULT_LSTM_DROPOUT)
                ),
            }
        )
    digest = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def get_training_artifact_paths(ticker: str, cache_key: str) -> dict[str, Path]:
    safe_ticker = _safe_ticker_dir_name(ticker)
    ticker_dir = TRAINING_CACHE_DIR / safe_ticker
    return {
        "meta": ticker_dir / f"{cache_key}.json",
        "model": ticker_dir / f"{cache_key}.keras",
        "scaler": ticker_dir / f"{cache_key}_scaler.npz",
        "eval": ticker_dir / f"{cache_key}_eval.npz",
        "forecast": ticker_dir / f"{cache_key}_forecast.npz",
        "forecast_path": ticker_dir / f"{cache_key}_forecast_path.npz",
    }


def load_cached_training_result(
    paths: dict[str, Path], required_artifacts: tuple[str, ...] = ("model", "scaler", "eval")
) -> dict | None:
    meta_path = paths["meta"]
    if not meta_path.exists():
        return None
    for name in required_artifacts:
        artifact_path = paths.get(name)
        if artifact_path is None or not artifact_path.exists():
            return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = payload.get("metrics", {})
    if "mse" not in metrics or "mae" not in metrics:
        return None
    return payload


def save_training_metadata(
    *,
    paths: dict[str, Path],
    ticker: str,
    model_code: str,
    config: dict,
    metrics: dict,
    data_hash: str,
    training_frame: pd.DataFrame,
    start_date,
    end_date,
) -> None:
    normalized = with_training_defaults(config, model_code=model_code)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "ticker": ticker,
        "model": model_code,
        "config": {
            "window_size": int(normalized["window_size"]),
            "epochs": int(normalized["epochs"]),
            "horizon": int(normalized.get("horizon", 1)),
            "test_size": float(normalized.get("test_size", DEFAULT_TEST_SIZE)),
            "val_size": float(normalized.get("val_size", DEFAULT_VAL_SIZE)),
            "batch_size": int(normalized.get("batch_size", 32)),
            "learning_rate": float(normalized.get("learning_rate", 5e-4)),
            "lstm_units1": int(normalized.get("lstm_units1", DEFAULT_LSTM_UNITS1)),
            "lstm_units2": int(normalized.get("lstm_units2", DEFAULT_LSTM_UNITS2)),
            "lstm_dropout": float(
                normalized.get("lstm_dropout", DEFAULT_LSTM_DROPOUT)
            ),
        },
        "date_filter": {
            "start": start_date.isoformat() if start_date is not None else None,
            "end": end_date.isoformat() if end_date is not None else None,
        },
        "data_hash": data_hash,
        "training_last_date": (
            pd.to_datetime(training_frame["Date"]).max().strftime("%Y-%m-%d")
            if training_frame is not None and not training_frame.empty
            else None
        ),
        "training_rows": int(len(training_frame))
        if training_frame is not None
        else None,
        "metrics": {
            "metrics_version": int(metrics.get("metrics_version", 1)),
            "mse": float(metrics.get("mse", float("nan"))),
            "mae": float(metrics.get("mae", float("nan"))),
            "mse_price": float(metrics.get("mse_price", float("nan"))),
            "mae_price": float(metrics.get("mae_price", float("nan"))),
            "rmse_price": float(metrics.get("rmse_price", float("nan"))),
            "mape_pct": float(metrics.get("mape_pct", float("nan"))),
            "mse_norm": float(metrics.get("mse_norm", float("nan"))),
            "mae_norm": float(metrics.get("mae_norm", float("nan"))),
            "epochs_trained": int(metrics.get("epochs_trained", 0)),
            "lstm_units1": int(
                metrics.get("lstm_units1", normalized.get("lstm_units1", DEFAULT_LSTM_UNITS1))
            ),
            "lstm_units2": int(
                metrics.get("lstm_units2", normalized.get("lstm_units2", DEFAULT_LSTM_UNITS2))
            ),
            "lstm_dropout": float(
                metrics.get(
                    "lstm_dropout",
                    normalized.get("lstm_dropout", DEFAULT_LSTM_DROPOUT),
                )
            ),
        },
        "artifacts": {
            "model": str(paths["model"]),
            "scaler": str(paths["scaler"]),
            "eval": str(paths["eval"]),
            "forecast": str(paths["forecast"]),
            "forecast_path": str(paths["forecast_path"]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    paths["meta"].write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def write_temp_training_csv(ticker: str, training_frame: pd.DataFrame) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    path = (
        Path(tempfile.gettempdir())
        / f"deep_finance_train_{safe_ticker}_{timestamp}.csv"
    )
    training_frame.to_csv(path, index=False)
    return path


def run_training_subprocess(
    *,
    csv_path: Path | str,
    model_code: str,
    config: dict,
    model_output_path: Path | str | None = None,
    scaler_output_path: Path | str | None = None,
    eval_output_path: Path | str | None = None,
    forecast_output_path: Path | str | None = None,
    forecast_path_output_path: Path | str | None = None,
) -> dict:
    runner_path = Path(__file__).resolve().parent / "train_runner.py"
    normalized = with_training_defaults(config, model_code=model_code)
    command = [
        sys.executable,
        str(runner_path),
        "--csv",
        str(csv_path),
        "--model",
        model_code,
        "--window-size",
        str(normalized["window_size"]),
        "--epochs",
        str(normalized["epochs"]),
        "--horizon",
        str(normalized.get("horizon", 1)),
        "--test-size",
        str(normalized.get("test_size", DEFAULT_TEST_SIZE)),
        "--val-size",
        str(normalized.get("val_size", DEFAULT_VAL_SIZE)),
        "--batch-size",
        str(normalized.get("batch_size", 32)),
        "--learning-rate",
        str(normalized.get("learning_rate", 5e-4)),
    ]
    if model_code in {"lstm", "lstm_multifeature"}:
        command.extend(
            [
                "--lstm-units1",
                str(normalized.get("lstm_units1", DEFAULT_LSTM_UNITS1)),
                "--lstm-units2",
                str(normalized.get("lstm_units2", DEFAULT_LSTM_UNITS2)),
                "--lstm-dropout",
                str(normalized.get("lstm_dropout", DEFAULT_LSTM_DROPOUT)),
            ]
        )
    if model_output_path:
        command.extend(["--save-model", str(model_output_path)])
    if scaler_output_path:
        command.extend(["--save-scaler", str(scaler_output_path)])
    if eval_output_path:
        command.extend(["--save-eval", str(eval_output_path)])
    if forecast_output_path:
        command.extend(["--save-forecast", str(forecast_output_path)])
    if forecast_path_output_path:
        command.extend(["--save-forecast-path", str(forecast_path_output_path)])

    timeout_sec = int(normalized.get("timeout_sec", TRAINING_SUBPROCESS_TIMEOUT_SEC))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Training timeout after {timeout_sec}s") from exc

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
        message = payload.get("error") or completed.stderr.strip()
        raise RuntimeError(message or "Training subprocess failed")

    return payload["metrics"]


def load_forecast_artifact(forecast_path: Path | str) -> dict | None:
    forecast_path = Path(forecast_path)
    if not forecast_path.exists():
        return None
    try:
        payload = np.load(forecast_path)
        if "forecast_value" not in payload:
            return None
        values = payload["forecast_value"]
        if len(values) == 0:
            return None
        forecast_lower = None
        forecast_upper = None
        if "forecast_lower" in payload and len(payload["forecast_lower"]) > 0:
            forecast_lower = float(payload["forecast_lower"][0])
        if "forecast_upper" in payload and len(payload["forecast_upper"]) > 0:
            forecast_upper = float(payload["forecast_upper"][0])
        return {
            "forecast_value": float(values[0]),
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
        }
    except Exception:
        return None


def resolve_training_cache(
    *,
    ticker: str,
    model_code: str,
    config: dict,
    training_frame: pd.DataFrame,
    force_retrain: bool = False,
    start_date=None,
    end_date=None,
    required_artifacts: tuple[str, ...] = ("model", "scaler", "eval"),
) -> dict:
    normalized = with_training_defaults(config, model_code=model_code)
    data_hash = hash_training_frame(training_frame)
    cache_key = build_training_cache_key(ticker, model_code, normalized, data_hash)
    artifact_paths = get_training_artifact_paths(ticker, cache_key)

    cached_payload = None
    if not force_retrain:
        cached_payload = load_cached_training_result(
            artifact_paths, required_artifacts=required_artifacts
        )
    if cached_payload is not None:
        return {
            "source": "Cache",
            "metrics": cached_payload.get("metrics", {}),
            "artifact_paths": artifact_paths,
            "cache_key": cache_key,
            "data_hash": data_hash,
            "metadata": cached_payload,
            "config": normalized,
        }

    temp_csv_path = write_temp_training_csv(ticker, training_frame)
    try:
        metrics = run_training_subprocess(
            csv_path=temp_csv_path,
            model_code=model_code,
            config=normalized,
            model_output_path=artifact_paths["model"],
            scaler_output_path=artifact_paths["scaler"],
            eval_output_path=artifact_paths["eval"],
            forecast_output_path=artifact_paths["forecast"],
            forecast_path_output_path=artifact_paths["forecast_path"],
        )
        save_training_metadata(
            paths=artifact_paths,
            ticker=ticker,
            model_code=model_code,
            config=normalized,
            metrics=metrics,
            data_hash=data_hash,
            training_frame=training_frame,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        temp_csv_path.unlink(missing_ok=True)

    metadata = load_cached_training_result(
        artifact_paths, required_artifacts=required_artifacts
    )
    return {
        "source": "Train",
        "metrics": metrics,
        "artifact_paths": artifact_paths,
        "cache_key": cache_key,
        "data_hash": data_hash,
        "metadata": metadata,
        "config": normalized,
    }


def _config_matches(metadata_config: dict, normalized_config: dict, model_code: str) -> bool:
    if not metadata_config:
        return False
    keys = [
        "window_size",
        "epochs",
        "horizon",
        "test_size",
        "val_size",
        "batch_size",
        "learning_rate",
    ]
    if model_code in {"lstm", "lstm_multifeature"}:
        keys.extend(["lstm_units1", "lstm_units2", "lstm_dropout"])
    for key in keys:
        meta_value = metadata_config.get(key)
        config_value = normalized_config.get(key)
        if isinstance(config_value, float):
            if meta_value is None or abs(float(meta_value) - float(config_value)) > 1e-12:
                return False
        else:
            if meta_value is None or int(meta_value) != int(config_value):
                return False
    return True


def load_latest_training_cache(
    *,
    ticker: str,
    model_code: str,
    config: dict,
    required_artifacts: tuple[str, ...] = ("model", "scaler", "eval"),
) -> dict | None:
    ticker_dir = TRAINING_CACHE_DIR / _safe_ticker_dir_name(ticker)
    if not ticker_dir.exists():
        return None

    normalized = with_training_defaults(config, model_code=model_code)
    candidates = []
    for meta_path in sorted(ticker_dir.glob("*.json")):
        cache_key = meta_path.stem
        paths = get_training_artifact_paths(ticker, cache_key)
        payload = load_cached_training_result(paths, required_artifacts=required_artifacts)
        if payload is None:
            continue
        if payload.get("pipeline_version") != TRAINING_PIPELINE_VERSION:
            continue
        if payload.get("model") != model_code:
            continue
        if not _config_matches(payload.get("config", {}), normalized, model_code):
            continue
        created_at = payload.get("created_at", "")
        candidates.append((created_at, meta_path.stat().st_mtime, cache_key, payload, paths))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, cache_key, payload, paths = candidates[0]
    return {
        "source": "Cache precedent",
        "metrics": payload.get("metrics", {}),
        "artifact_paths": paths,
        "cache_key": cache_key,
        "data_hash": payload.get("data_hash"),
        "metadata": payload,
        "config": normalized,
    }


def peek_training_cache(
    *,
    ticker: str,
    model_code: str,
    config: dict,
    training_frame: pd.DataFrame,
    required_artifacts: tuple[str, ...] = ("model", "scaler", "eval"),
) -> dict | None:
    normalized = with_training_defaults(config, model_code=model_code)
    data_hash = hash_training_frame(training_frame)
    cache_key = build_training_cache_key(ticker, model_code, normalized, data_hash)
    artifact_paths = get_training_artifact_paths(ticker, cache_key)
    cached_payload = load_cached_training_result(
        artifact_paths, required_artifacts=required_artifacts
    )
    if cached_payload is None:
        return None
    return {
        "source": "Cache",
        "metrics": cached_payload.get("metrics", {}),
        "artifact_paths": artifact_paths,
        "cache_key": cache_key,
        "data_hash": data_hash,
        "metadata": cached_payload,
        "config": normalized,
    }
