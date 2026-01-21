from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from yahooquery import Ticker
except Exception:  # pragma: no cover - optional dependency/runtime network
    Ticker = None


DATA_DIR = Path(__file__).resolve().parent / "data" / "assets"
RELEVANT_COLUMNS = ["Date", "Adj Close"]


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def is_recent(path, max_age_hours=24):
    if not path.exists():
        return False
    age_hours = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
    return age_hours < max_age_hours


def infer_country(ticker):
    ticker = ticker.upper()
    if ticker.endswith(".PA"):
        return "france"
    return "united states"


def normalize_history(data, ticker):
    df = data.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == ticker.upper()]

    rename_map = {
        "date": "Date",
        "close": "Close",
        "adjclose": "Adj Close",
    }
    df = df.rename(columns=rename_map)
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column for {ticker}")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates("Date")

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Adj Close" not in df.columns:
        raise ValueError(f"Missing Adj Close column for {ticker}")

    df = df[RELEVANT_COLUMNS].rename(columns={"Adj Close": "Close"})
    return df


def fetch_asset(ticker, start=None, end=None):
    if Ticker is None:
        raise RuntimeError("yahooquery is not available")
    country = infer_country(ticker)
    try:
        ticker_obj = Ticker(
            ticker,
            country=country,
            timeout=10,
            retry=1,
            backoff=0.2,
        )
    except TypeError:
        ticker_obj = Ticker(ticker, country=country)
    start_value = pd.Timestamp(start).strftime("%Y-%m-%d") if start else None
    end_value = pd.Timestamp(end).strftime("%Y-%m-%d") if end else None
    try:
        data = ticker_obj.history(start=start_value, end=end_value, interval="1d")
    except KeyError as exc:
        raise ValueError(f"YahooQuery error for {ticker}: {exc}") from exc
    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker}")
    return normalize_history(data, ticker)


def update_assets(tickers, start=None, end=None, force=False, progress=None):
    ensure_data_dir()
    results = {"updated": [], "skipped": [], "failed": []}
    total = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        safe_name = ticker.replace("/", "_")
        output_path = DATA_DIR / f"{safe_name}.csv"

        if not force and is_recent(output_path):
            results["skipped"].append(ticker)
            if progress:
                progress(index, total, ticker, "skipped")
            continue

        try:
            df = fetch_asset(ticker, start=start, end=end)
            df.to_csv(output_path, index=False)
            results["updated"].append(ticker)
            if progress:
                progress(index, total, ticker, "updated")
        except Exception as exc:
            results["failed"].append({"ticker": ticker, "error": str(exc)})
            if progress:
                progress(index, total, ticker, "failed", str(exc))

    return results
