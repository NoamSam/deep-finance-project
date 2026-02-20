import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.train_pipeline import run_training


def build_parser():
    parser = argparse.ArgumentParser(description="Deep Finance training CLI")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date, Close")
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "cnn_lstm"],
        default="lstm",
        help="Model type to train",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_training(args.csv, args.model)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    print(f"Model: {result.model_type}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
