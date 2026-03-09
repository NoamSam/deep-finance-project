import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config import DEFAULT_CONFIG
from app.train_pipeline import run_training


def build_parser():
    parser = argparse.ArgumentParser(description="Deep Finance training CLI")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date, Close")
    parser.add_argument(
        "--model",
        choices=["lstm", "lstm_multifeature", "cnn", "cnn_lstm"],
        default="lstm",
        help="Model type to train",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=f"Sliding window size (default: {DEFAULT_CONFIG['window_size']})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=f"Prediction horizon in steps (default: {DEFAULT_CONFIG['horizon']})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help=f"Test split ratio (default: {DEFAULT_CONFIG['test_size']})",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help=f"Validation split ratio (default: {DEFAULT_CONFIG['val_size']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Subprocess timeout in seconds",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = {
        key: value
        for key, value in {
            "window_size": args.window_size,
            "horizon": args.horizon,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "timeout_sec": args.timeout_sec,
        }.items()
        if value is not None
    }
    try:
        result = run_training(args.csv, args.model, config=config)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    print(f"Model: {result.model_type}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
