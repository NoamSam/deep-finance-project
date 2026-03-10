import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ASSETS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]


def _parse_int_list(raw_value):
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def _parse_float_list(raw_value):
    return [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def _parse_unit_pairs(raw_value):
    pairs = []
    for item in str(raw_value).split(","):
        token = item.strip()
        if not token:
            continue
        units1_raw, units2_raw = token.split(":")
        pairs.append((int(units1_raw), int(units2_raw)))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for single-feature LSTM on Beautiful Seven"
    )
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-03-07")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-cutoffs", type=int, default=1)
    parser.add_argument("--spacing-steps", type=int, default=5)
    parser.add_argument("--history-lookback", type=int, default=252)
    parser.add_argument("--risk-profile", default="equilibre")
    parser.add_argument("--portfolio-horizon", default="court")
    parser.add_argument("--max-weight", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--window-sizes", default="120,252")
    parser.add_argument("--learning-rates", default="0.001,0.0005")
    parser.add_argument("--dropouts", default="0.1,0.2")
    parser.add_argument("--unit-pairs", default="64:32,32:16")
    parser.add_argument(
        "--output-dir",
        default="results/lstm_grid_search",
        help="Directory where per-run outputs and summary.csv will be written",
    )
    return parser.parse_args()


def _config_slug(config):
    dropout_token = str(config["lstm_dropout"]).replace(".", "p")
    lr_token = str(config["learning_rate"]).replace(".", "p")
    return (
        f"w{config['window_size']}"
        f"_lr{lr_token}"
        f"_u{config['lstm_units1']}-{config['lstm_units2']}"
        f"_d{dropout_token}"
    )


def main():
    args = parse_args()
    window_sizes = _parse_int_list(args.window_sizes)
    learning_rates = _parse_float_list(args.learning_rates)
    dropouts = _parse_float_list(args.dropouts)
    unit_pairs = _parse_unit_pairs(args.unit_pairs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve().parent / "market_sanity_check.py"

    rows = []
    config_space = list(
        itertools.product(window_sizes, learning_rates, dropouts, unit_pairs)
    )
    total = len(config_space)

    for index, (window_size, learning_rate, dropout, unit_pair) in enumerate(
        config_space, start=1
    ):
        units1, units2 = unit_pair
        config = {
            "window_size": int(window_size),
            "learning_rate": float(learning_rate),
            "lstm_units1": int(units1),
            "lstm_units2": int(units2),
            "lstm_dropout": float(dropout),
        }
        slug = _config_slug(config)
        run_output_dir = output_dir / slug
        print(f"[{index}/{total}] {slug}", flush=True)
        command = [
            sys.executable,
            str(runner_path),
            "--assets",
            *args.assets,
            "--model",
            "lstm",
            "--start",
            args.start,
            "--end",
            args.end,
            "--window-size",
            str(window_size),
            "--horizon",
            str(args.horizon),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(learning_rate),
            "--lstm-units1",
            str(units1),
            "--lstm-units2",
            str(units2),
            "--lstm-dropout",
            str(dropout),
            "--num-cutoffs",
            str(args.num_cutoffs),
            "--spacing-steps",
            str(args.spacing_steps),
            "--history-lookback",
            str(args.history_lookback),
            "--risk-profile",
            args.risk_profile,
            "--portfolio-horizon",
            args.portfolio_horizon,
            "--max-weight",
            str(args.max_weight),
            "--timeout-sec",
            str(args.timeout_sec),
            "--output-dir",
            str(run_output_dir),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        summary_path = run_output_dir / "summary.json"
        asset_summary_path = run_output_dir / "asset_summary.csv"
        if completed.returncode != 0 or not summary_path.exists():
            rows.append(
                {
                    **config,
                    "status": "failed",
                    "error": (completed.stderr or completed.stdout).strip()[-500:],
                }
            )
            continue

        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregate_summary = summary_payload.get("aggregate_summary", {})
        asset_summary = pd.read_csv(asset_summary_path)
        rows.append(
            {
                **config,
                "status": "ok",
                "mean_model_portfolio_return": float(
                    aggregate_summary.get("mean_model_portfolio_return", float("nan"))
                ),
                "mean_equal_weight_return": float(
                    aggregate_summary.get("mean_equal_weight_return", float("nan"))
                ),
                "model_win_rate_vs_equal_weight": float(
                    aggregate_summary.get(
                        "model_win_rate_vs_equal_weight", float("nan")
                    )
                ),
                "mean_market_abs_error": float(
                    asset_summary["market_abs_error"].mean()
                ),
                "mean_model_mae_price": float(
                    asset_summary["mean_model_mae_price"].mean()
                ),
                "beat_naive_assets": int(
                    (asset_summary["model_beat_rate_vs_naive"] > 0.5).sum()
                ),
                "direction_hit_assets": int(
                    (asset_summary["direction_hit_rate"] > 0.5).sum()
                ),
                "cutoffs": ",".join(summary_payload["config"].get("cutoffs", [])),
            }
        )

    results = pd.DataFrame(rows)
    summary_csv = output_dir / "summary.csv"
    results.to_csv(summary_csv, index=False)
    successful = results[results["status"] == "ok"].copy()
    if not successful.empty:
        successful = successful.sort_values(
            [
                "mean_model_portfolio_return",
                "model_win_rate_vs_equal_weight",
                "mean_market_abs_error",
            ],
            ascending=[False, False, True],
        )
        print()
        print("=== Top configs ===")
        print(
            successful[
                [
                    "window_size",
                    "learning_rate",
                    "lstm_units1",
                    "lstm_units2",
                    "lstm_dropout",
                    "mean_model_portfolio_return",
                    "mean_equal_weight_return",
                    "model_win_rate_vs_equal_weight",
                    "mean_market_abs_error",
                    "beat_naive_assets",
                    "direction_hit_assets",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    print()
    print(f"Summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
