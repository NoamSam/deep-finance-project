import argparse

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
    result = run_training(args.csv, args.model)
    print(f"Model: {result.model_type}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
