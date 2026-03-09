from __future__ import annotations

import numpy as np
import pandas as pd


US_FACTOR_TICKERS = {
    "SPY_Return": "SPY",
    "QQQ_Return": "QQQ",
    "VIX_Return": "^VIX",
}

US_MULTIFEATURE_MODEL_CODE = "lstm_multifeature"
US_MULTIFEATURE_COLUMNS = [
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


def is_us_equity_ticker(ticker: str) -> bool:
    if not ticker:
        return False
    return (
        not str(ticker).startswith("^")
        and "." not in str(ticker)
        and ":" not in str(ticker)
        and "/" not in str(ticker)
    )


def normalize_history_frame(
    frame: pd.DataFrame, require_ohlcv: bool = False
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Empty history frame")

    df = frame.copy()
    rename_map = {
        "date": "Date",
        "datetime": "Date",
        "adjclose": "Adj Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    if "Date" not in df.columns:
        raise ValueError("Missing Date column")
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        raise ValueError("Missing Close column")
    if require_ohlcv:
        for column in ["Open", "High", "Low"]:
            if column not in df.columns:
                raise ValueError(f"Missing {column} column")
    for column in ["Open", "High", "Low"]:
        if column not in df.columns:
            df[column] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df.drop_duplicates("Date", keep="last")
    df["Volume"] = df["Volume"].fillna(0.0)
    if require_ohlcv:
        df = df.dropna(subset=["Open", "High", "Low"])
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(
        drop=True
    )


def _build_close_return_frame(frame: pd.DataFrame, output_column: str) -> pd.DataFrame:
    normalized = normalize_history_frame(frame, require_ohlcv=False)
    safe_close = np.maximum(normalized["Close"].astype(float).to_numpy(), 1e-8)
    returns = np.diff(np.log(safe_close), prepend=np.log(safe_close[0]))
    return pd.DataFrame(
        {
            "Date": normalized["Date"],
            output_column: returns.astype(np.float32),
        }
    )


def build_us_multifeature_frame(
    asset_frame: pd.DataFrame, factor_histories: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    asset = normalize_history_frame(asset_frame, require_ohlcv=True)
    merged = asset.copy()
    for return_column, ticker in US_FACTOR_TICKERS.items():
        if ticker not in factor_histories:
            raise ValueError(f"Missing factor history for {ticker}")
        factor_returns = _build_close_return_frame(
            factor_histories[ticker], return_column
        )
        merged = merged.merge(factor_returns, on="Date", how="left")
    for column in ["SPY_Return", "QQQ_Return", "VIX_Return"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["Volume"] = pd.to_numeric(merged["Volume"], errors="coerce").fillna(0.0)
    merged = merged.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return merged[US_MULTIFEATURE_COLUMNS].reset_index(drop=True)


def build_training_frame_for_model(
    model_code: str,
    ticker: str,
    asset_frame: pd.DataFrame,
    factor_histories: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if model_code == US_MULTIFEATURE_MODEL_CODE:
        if not is_us_equity_ticker(ticker):
            raise ValueError("LSTM US multi-features est reserve aux actifs US")
        if factor_histories is None:
            raise ValueError("Missing factor histories for US multi-feature model")
        return build_us_multifeature_frame(asset_frame, factor_histories)
    normalized = normalize_history_frame(asset_frame, require_ohlcv=False)
    return normalized[["Date", "Close"]].copy()
