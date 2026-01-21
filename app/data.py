from pathlib import Path

import pandas as pd


def load_price_series(csv_path, date_col="Date", price_col="Close"):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(
            f"CSV must include columns: {date_col}, {price_col}. "
            f"Found: {', '.join(df.columns)}"
        )

    df = df[[date_col, price_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna().sort_values(date_col)
    return df.set_index(date_col)[price_col]
