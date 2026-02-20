from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf # type: ignore
except Exception:  # pragma: no cover - optional dependency/runtime network
    yf = None

try:
    from yahooquery import Ticker as YahooQueryTicker
except Exception:  # pragma: no cover - optional dependency/runtime network
    YahooQueryTicker = None


DATA_DIR = Path(__file__).resolve().parent / "data" / "assets"
RELEVANT_COLUMNS = ["Date", "Adj Close"]
DEFAULT_HISTORY_YEARS = 15


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def is_recent(path, max_age_hours=24):
    if not path.exists():
        return False
    age_hours = (
        datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    ).total_seconds() / 3600
    return age_hours < max_age_hours


def _extract_date_column(df):
    if "Date" in df.columns:
        return df
    if "date" in df.columns:
        return df.rename(columns={"date": "Date"})
    if "datetime" in df.columns:
        return df.rename(columns={"datetime": "Date"})
    if "index" in df.columns:
        return df.rename(columns={"index": "Date"})
    raise ValueError("Missing Date column")


def _to_naive_timestamp(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None)
    return ts


def normalize_history(data, ticker):
    df = data.copy().reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == ticker.upper()]

    rename_map = {
        "close": "Close",
        "adjclose": "Adj Close",
    }
    df = df.rename(columns=rename_map)
    df = _extract_date_column(df)
    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    original_non_null = df["Date"].notna().sum()
    parsed_non_null = parsed_dates.notna().sum()

    if parsed_non_null < original_non_null:
        # yahooquery can return mixed tz-aware / tz-naive values.
        parsed_dates = df["Date"].apply(_to_naive_timestamp)
        parsed_dates = pd.to_datetime(parsed_dates, errors="coerce")
    else:
        try:
            if parsed_dates.dt.tz is not None:
                parsed_dates = parsed_dates.dt.tz_localize(None)
        except AttributeError:
            parsed_dates = df["Date"].apply(_to_naive_timestamp)
            parsed_dates = pd.to_datetime(parsed_dates, errors="coerce")

    df["Date"] = parsed_dates
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Adj Close" not in df.columns:
        raise ValueError(f"Missing Adj Close column for {ticker}")

    df = df[RELEVANT_COLUMNS].rename(columns={"Adj Close": "Close"})
    return df


def _fetch_with_yfinance(ticker, start=None, end=None):
    if yf is None:
        raise RuntimeError("yfinance is not available")

    history = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
    )
    if history is None or history.empty:
        raise ValueError(f"No data returned by yfinance for {ticker}")
    return history


def _fetch_with_yahooquery(ticker, start=None, end=None):
    if YahooQueryTicker is None:
        raise RuntimeError("yahooquery is not available")

    try:
        ticker_obj = YahooQueryTicker(
            ticker,
            timeout=10,
            retry=1,
            backoff=0.2,
        )
    except TypeError:
        ticker_obj = YahooQueryTicker(ticker)
    try:
        data = ticker_obj.history(start=start, end=end, interval="1d")
    except KeyError as exc:
        raise ValueError(f"YahooQuery error for {ticker}: {exc}") from exc
    if data is None or data.empty:
        raise ValueError(f"No data returned by yahooquery for {ticker}")
    return data


def resolve_history_bounds(start=None, end=None, default_years=DEFAULT_HISTORY_YEARS):
    end_ts = (
        pd.Timestamp(end).normalize()
        if end is not None
        else pd.Timestamp.today().normalize()
    )
    start_ts = (
        pd.Timestamp(start).normalize()
        if start is not None
        else (end_ts - pd.DateOffset(years=default_years))
    )
    if start_ts > end_ts:
        raise ValueError(
            f"Invalid date range: start ({start_ts.date()}) is after end ({end_ts.date()})"
        )
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def fetch_asset(ticker, start=None, end=None):
    start_value, end_value = resolve_history_bounds(start, end)

    providers = [
        ("yfinance", _fetch_with_yfinance),
        ("yahooquery", _fetch_with_yahooquery),
    ]
    errors = []
    for provider_name, provider_fetch in providers:
        try:
            data = provider_fetch(
                ticker,
                start=start_value,
                end=end_value,
            )
            return normalize_history(data, ticker)
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")

    error_message = " | ".join(errors) if errors else "no providers available"
    raise RuntimeError(
        f"Unable to fetch data for {ticker}. {error_message}"
    )


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
