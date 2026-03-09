from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd

try:
    from app.training_cache import TRAINING_PIPELINE_VERSION, hash_training_frame
except ModuleNotFoundError:
    from training_cache import TRAINING_PIPELINE_VERSION, hash_training_frame


BACKTEST_CACHE_VERSION = 1
BACKTEST_CACHE_DIR = Path(__file__).resolve().parent / "data" / "backtest_cache"


def _hash_history_map(histories: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        ticker: hash_training_frame(frame.copy())
        for ticker, frame in sorted(histories.items())
    }


def build_backtest_cache_key(
    *,
    assets: list[str],
    model_code: str,
    config: dict,
    start_date,
    end_date,
    history_lookback: int,
    num_periods: int,
    risk_profile: str,
    portfolio_horizon: str,
    max_weight: float,
    transaction_cost_bps: float,
    benchmark_label: str,
    histories: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame | None,
    factor_histories: dict[str, pd.DataFrame] | None,
) -> tuple[str, dict]:
    payload = {
        "backtest_cache_version": BACKTEST_CACHE_VERSION,
        "training_pipeline_version": TRAINING_PIPELINE_VERSION,
        "assets": sorted(assets),
        "model": model_code,
        "config": {
            "window_size": int(config["window_size"]),
            "epochs": int(config["epochs"]),
            "horizon": int(config["horizon"]),
            "test_size": float(config["test_size"]),
            "val_size": float(config["val_size"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["learning_rate"]),
            "lstm_units1": int(config.get("lstm_units1", 64)),
            "lstm_units2": int(config.get("lstm_units2", 32)),
            "lstm_dropout": float(config.get("lstm_dropout", 0.2)),
        },
        "date_filter": {
            "start": start_date.isoformat() if start_date is not None else None,
            "end": end_date.isoformat() if end_date is not None else None,
        },
        "history_lookback": int(history_lookback),
        "num_periods": int(num_periods),
        "risk_profile": risk_profile,
        "portfolio_horizon": portfolio_horizon,
        "max_weight": float(max_weight),
        "transaction_cost_bps": float(transaction_cost_bps),
        "benchmark_label": benchmark_label,
        "histories_hash": _hash_history_map(histories),
        "benchmark_hash": (
            hash_training_frame(benchmark_history.copy())
            if benchmark_history is not None and not benchmark_history.empty
            else None
        ),
        "factor_hash": (
            _hash_history_map(factor_histories) if factor_histories else None
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24], payload


def get_backtest_cache_paths(cache_key: str) -> dict[str, Path]:
    return {
        "meta": BACKTEST_CACHE_DIR / f"{cache_key}.json",
        "payload": BACKTEST_CACHE_DIR / f"{cache_key}.pkl",
    }


def load_cached_backtest(cache_key: str) -> dict | None:
    paths = get_backtest_cache_paths(cache_key)
    if not paths["meta"].exists() or not paths["payload"].exists():
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        with paths["payload"].open("rb") as handle:
            results = pickle.load(handle)
    except Exception:
        return None
    return {"meta": meta, "results": results, "paths": paths}


def save_cached_backtest(cache_key: str, cache_payload: dict, results: dict) -> dict[str, Path]:
    paths = get_backtest_cache_paths(cache_key)
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "cache_key": cache_key,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_payload": cache_payload,
    }
    paths["meta"].write_text(
        json.dumps(meta_payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    with paths["payload"].open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return paths
