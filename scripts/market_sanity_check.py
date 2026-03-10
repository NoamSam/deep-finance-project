import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.portfolio_optimizer import run_portfolio_optimization
from app.us_market_features import US_FACTOR_TICKERS, build_training_frame_for_model
from app.update_curves import fetch_asset


DEFAULT_ASSETS = ["AAPL", "MSFT", "NVDA"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Market sanity check for training forecasts and portfolio allocation"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=DEFAULT_ASSETS,
        help="Assets to evaluate",
    )
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "lstm_multifeature", "cnn", "cnn_lstm"],
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lstm-units1", type=int, default=64)
    parser.add_argument("--lstm-units2", type=int, default=32)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-steps", type=int, default=10)
    parser.add_argument("--history-lookback", type=int, default=252)
    parser.add_argument("--risk-profile", default="equilibre")
    parser.add_argument("--portfolio-horizon", default="court")
    parser.add_argument("--max-weight", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where CSV/JSON outputs will be written",
    )
    return parser.parse_args()


def load_histories(assets, start, end, model_code):
    histories = {}
    for ticker in assets:
        df = fetch_asset(ticker, start=start, end=end)
        if model_code == "lstm_multifeature":
            keep_columns = [
                column
                for column in ["Date", "Open", "High", "Low", "Close", "Volume"]
                if column in df.columns
            ]
            df = df[keep_columns].copy()
        else:
            df = df[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna().sort_values("Date").drop_duplicates("Date")
        histories[ticker] = df.reset_index(drop=True)
    return histories


def load_factor_histories(start, end):
    histories = {}
    for ticker in US_FACTOR_TICKERS.values():
        df = fetch_asset(ticker, start=start, end=end)
        df = df[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna().sort_values("Date").drop_duplicates("Date")
        histories[ticker] = df.reset_index(drop=True)
    return histories


def common_dates(histories):
    common = None
    for df in histories.values():
        dates = set(pd.to_datetime(df["Date"]))
        common = dates if common is None else common & dates
    return sorted(common) if common else []


def choose_cutoff_dates(common_market_dates, history_lookback, horizon, num_cutoffs, spacing_steps):
    usable = common_market_dates[history_lookback - 1 : len(common_market_dates) - horizon]
    if len(usable) < num_cutoffs:
        raise ValueError("Not enough common dates to build requested cutoffs.")

    selected = []
    idx = len(usable) - 1
    while idx >= 0 and len(selected) < num_cutoffs:
        selected.append(usable[idx])
        idx -= spacing_steps
    return list(reversed(selected))


def build_temp_csv(frame):
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    frame.to_csv(tmp_path, index=False)
    return tmp_path


def forecast_naive_price(last_close):
    return float(last_close)


def run_training_forecast(
    csv_path,
    model,
    window_size,
    horizon,
    epochs,
    batch_size,
    timeout_sec,
    learning_rate=5e-4,
    lstm_units1=64,
    lstm_units2=32,
    lstm_dropout=0.2,
):
    runner_path = Path(__file__).resolve().parent.parent / "app" / "train_runner.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        forecast_path = Path(tmpdir) / "forecast.npz"
        command = [
            sys.executable,
            str(runner_path),
            "--csv",
            str(csv_path),
            "--model",
            model,
            "--window-size",
            str(window_size),
            "--horizon",
            str(horizon),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--lstm-units1",
            str(lstm_units1),
            "--lstm-units2",
            str(lstm_units2),
            "--lstm-dropout",
            str(lstm_dropout),
            "--save-forecast",
            str(forecast_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=timeout_sec,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(completed.stderr.strip() or "No output from train_runner")
        payload = json.loads(lines[-1])
        if not payload.get("ok") or completed.returncode != 0:
            raise RuntimeError(payload.get("error") or completed.stderr.strip())
        metrics = payload["metrics"]
        forecast_payload = np.load(forecast_path)
        forecast_value = float(forecast_payload["forecast_value"][0])
        return metrics, forecast_value


def compute_realized_returns(price_matrix, cutoff_date, future_date):
    last_close = price_matrix.loc[cutoff_date]
    future_close = price_matrix.loc[future_date]
    return (future_close / last_close) - 1.0


def compute_portfolio_return(realized_returns, weights):
    aligned = realized_returns.loc[weights.index]
    return float((weights * aligned).sum())


def build_asset_summary(asset_df):
    return (
        asset_df.groupby("ticker")
        .agg(
            mean_model_mae_price=("model_mae_price", "mean"),
            market_abs_error=("model_abs_error", "mean"),
            naive_abs_error=("naive_abs_error", "mean"),
            direction_hit_rate=("direction_ok", "mean"),
            naive_direction_hit_rate=("naive_direction_ok", "mean"),
            model_beat_rate_vs_naive=("model_beats_naive", "mean"),
            mean_predicted_return=("predicted_return", "mean"),
            mean_realized_return=("realized_return", "mean"),
        )
        .round(4)
    )


def build_aggregate_summary(portfolio_df):
    return {
        "mean_model_portfolio_return": float(
            portfolio_df["model_portfolio_return"].mean()
        ),
        "mean_equal_weight_return": float(
            portfolio_df["equal_weight_return"].mean()
        ),
        "model_win_rate_vs_equal_weight": float(
            portfolio_df["model_outperformed_equal_weight"].mean()
        ),
    }


def write_outputs(output_dir, config_payload, asset_df, asset_summary, portfolio_df, aggregate_summary):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    asset_df.to_csv(output_path / "asset_details.csv", index=False)
    asset_summary.to_csv(output_path / "asset_summary.csv")
    portfolio_df.to_csv(output_path / "portfolio_summary.csv", index=False)
    summary_payload = {
        "config": config_payload,
        "aggregate_summary": aggregate_summary,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    histories = load_histories(args.assets, args.start, args.end, args.model)
    factor_histories = (
        load_factor_histories(args.start, args.end)
        if args.model == "lstm_multifeature"
        else None
    )
    market_dates = common_dates(histories)
    cutoffs = choose_cutoff_dates(
        market_dates,
        history_lookback=args.history_lookback,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_steps=args.spacing_steps,
    )

    asset_rows = []
    portfolio_rows = []

    aligned_prices = []
    for ticker, df in histories.items():
        aligned_prices.append(
            df[["Date", "Close"]]
            .copy()
            .set_index("Date")
            .rename(columns={"Close": ticker})
        )
    price_matrix = pd.concat(aligned_prices, axis=1).dropna()

    for cutoff_date in cutoffs:
        future_date = market_dates[market_dates.index(cutoff_date) + args.horizon]
        model_predicted_returns = {}

        for ticker, full_df in histories.items():
            train_df = full_df[full_df["Date"] <= cutoff_date].copy()
            training_frame = build_training_frame_for_model(
                model_code=args.model,
                ticker=ticker,
                asset_frame=train_df,
                factor_histories=factor_histories,
            )
            csv_path = build_temp_csv(training_frame)
            try:
                metrics, forecast_price = run_training_forecast(
                    csv_path=csv_path,
                    model=args.model,
                    window_size=args.window_size,
                    horizon=args.horizon,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    timeout_sec=args.timeout_sec,
                    learning_rate=args.learning_rate,
                    lstm_units1=args.lstm_units1,
                    lstm_units2=args.lstm_units2,
                    lstm_dropout=args.lstm_dropout,
                )
            finally:
                csv_path.unlink(missing_ok=True)

            last_close = float(train_df["Close"].iloc[-1])
            actual_close = float(
                full_df.loc[full_df["Date"] == future_date, "Close"].iloc[0]
            )
            naive_price = forecast_naive_price(last_close)
            predicted_return = (forecast_price / last_close) - 1.0
            naive_return = (naive_price / last_close) - 1.0
            realized_return = (actual_close / last_close) - 1.0
            model_predicted_returns[ticker] = predicted_return

            asset_rows.append(
                {
                    "ticker": ticker,
                    "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                    "future_date": future_date.strftime("%Y-%m-%d"),
                    "last_close": last_close,
                    "forecast_close": forecast_price,
                    "naive_close": naive_price,
                    "actual_close": actual_close,
                    "predicted_return": predicted_return,
                    "naive_return": naive_return,
                    "realized_return": realized_return,
                    "model_mae_price": float(metrics.get("mae_price", np.nan)),
                    "direction_ok": int(np.sign(predicted_return) == np.sign(realized_return)),
                    "naive_direction_ok": int(np.sign(naive_return) == np.sign(realized_return)),
                    "naive_abs_error": abs(last_close - actual_close),
                    "model_abs_error": abs(forecast_price - actual_close),
                    "model_beats_naive": int(
                        abs(forecast_price - actual_close) < abs(last_close - actual_close)
                    ),
                }
            )

        trailing_returns = (
            price_matrix.loc[:cutoff_date]
            .pct_change()
            .dropna()
            .tail(args.history_lookback)
        )
        model_weights = run_portfolio_optimization(
            returns_hist=trailing_returns,
            mu_pred=pd.Series(model_predicted_returns),
            risk_profile=args.risk_profile,
            horizon=args.portfolio_horizon,
            allow_short=False,
            max_weight=args.max_weight,
            covariance_method="ledoit_wolf",
        )
        realized_returns = compute_realized_returns(
            price_matrix=price_matrix,
            cutoff_date=cutoff_date,
            future_date=future_date,
        )
        realized_model_return = compute_portfolio_return(realized_returns, model_weights)
        equal_weight_return = float(realized_returns.mean())
        portfolio_rows.append(
            {
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "future_date": future_date.strftime("%Y-%m-%d"),
                "model_portfolio_return": realized_model_return,
                "equal_weight_return": equal_weight_return,
                "model_outperformed_equal_weight": int(
                    realized_model_return > equal_weight_return
                ),
                "model_max_weight": float(model_weights.max()),
                "model_num_active_positions": int((model_weights > 1e-8).sum()),
                "model_weights": ", ".join(
                    f"{k}={v:.3f}" for k, v in model_weights.items()
                ),
                "realized_returns": ", ".join(
                    f"{k}={v:.4%}" for k, v in realized_returns.sort_values(ascending=False).items()
                ),
            }
        )

    asset_df = pd.DataFrame(asset_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)

    config_payload = {
        "assets": args.assets,
        "model": args.model,
        "start": args.start,
        "end": args.end,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lstm_units1": args.lstm_units1,
        "lstm_units2": args.lstm_units2,
        "lstm_dropout": args.lstm_dropout,
        "cutoffs": [d.strftime("%Y-%m-%d") for d in cutoffs],
    }
    asset_summary = build_asset_summary(asset_df)
    aggregate_summary = build_aggregate_summary(portfolio_df)

    if args.output_dir:
        write_outputs(
            output_dir=args.output_dir,
            config_payload=config_payload,
            asset_df=asset_df,
            asset_summary=asset_summary,
            portfolio_df=portfolio_df,
            aggregate_summary=aggregate_summary,
        )

    print("=== Market Sanity Check ===")
    print("config:", json.dumps(config_payload))
    print()
    print("=== Asset Forecast Summary ===")
    print(asset_summary.to_string())
    print()
    print("=== Portfolio Summary ===")
    print(
        portfolio_df[
            [
                "cutoff_date",
                "future_date",
                "model_portfolio_return",
                "equal_weight_return",
                "model_outperformed_equal_weight",
                "model_max_weight",
                "model_num_active_positions",
                "model_weights",
            ]
        ].to_string(index=False)
    )
    print()
    print("=== Aggregate Portfolio Metrics ===")
    print(json.dumps(aggregate_summary, indent=2))
    if args.output_dir:
        print()
        print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
