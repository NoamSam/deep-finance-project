from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the status of an offline backtest run and summarize the result."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing run.log and optional result files.",
    )
    return parser.parse_args()


def tail_text(path: Path, line_count: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def load_strategy_metrics(result_dir: Path) -> pd.DataFrame:
    metrics_path = result_dir / "strategy_metrics.csv"
    summary_path = result_dir / "summary.json"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload.get("strategy_metrics", []))
    return pd.DataFrame()


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def detect_status(result_dir: Path) -> tuple[str, str]:
    summary_path = result_dir / "summary.json"
    metrics_path = result_dir / "strategy_metrics.csv"
    report_path = result_dir / "report.txt"
    log_path = result_dir / "run.log"

    if summary_path.exists() or metrics_path.exists() or report_path.exists():
        return "done", "Result files are available."

    log_tail = tail_text(log_path)
    if not log_tail.strip():
        return "pending", "Run log is empty or not started yet."

    if "Traceback" in log_tail or "RuntimeError" in log_tail or "ATTEMPT_FAILED=3" in log_tail:
        return "failed", "Log contains a terminal error."

    if "DONE" in log_tail:
        return "done", "Run completed according to the log."

    return "running", "Run log shows progress but no final output yet."


def build_verdict(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty or "Strategie" not in metrics.columns:
        return ["- Verdict indisponible: resultats incomplets."]

    metrics = metrics.copy()
    for column in ["Rendement cumule", "Rendement annualise", "Max drawdown", "Turnover moyen", "Periodes"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    if "Modele" not in set(metrics["Strategie"]):
        return ["- Verdict indisponible: strategie Modele absente."]

    model_row = metrics[metrics["Strategie"] == "Modele"].iloc[0]
    references = metrics[metrics["Strategie"] != "Modele"].copy()
    beats = references[references["Rendement cumule"] < model_row["Rendement cumule"]]

    lines = [
        "- Modele: "
        f"cumule {safe_float(model_row.get('Rendement cumule')):.2%} | "
        f"annualise {safe_float(model_row.get('Rendement annualise')):.2%} | "
        f"drawdown {safe_float(model_row.get('Max drawdown')):.2%} | "
        f"turnover {safe_float(model_row.get('Turnover moyen')):.2%} | "
        f"periodes {int(safe_float(model_row.get('Periodes')) or 0)}"
    ]
    lines.append(f"- References battues (rendement cumule): {len(beats)}/{len(references)}")
    for _, row in references.sort_values("Rendement cumule", ascending=False).iterrows():
        lines.append(
            f"- {row['Strategie']}: cumule {safe_float(row.get('Rendement cumule')):.2%} | "
            f"annualise {safe_float(row.get('Rendement annualise')):.2%}"
        )
    return lines


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    log_path = result_dir / "run.log"
    report_path = result_dir / "report.txt"

    status, reason = detect_status(result_dir)
    print(f"status: {status}")
    print(f"reason: {reason}")
    print(f"result_dir: {result_dir}")

    if report_path.exists():
        print(f"report: {report_path}")

    metrics = load_strategy_metrics(result_dir)
    if not metrics.empty:
        print("verdict:")
        for line in build_verdict(metrics):
            print(line)
        return 0

    if log_path.exists():
        print("log_tail:")
        print(tail_text(log_path))

    return 0 if status in {"done", "running", "pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
