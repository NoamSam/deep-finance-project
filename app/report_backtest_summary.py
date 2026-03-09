from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a backtest result directory and produce a concise summary."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing summary.json and/or strategy_metrics.csv.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="How long to wait for the result files if they are not available yet.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Polling interval while waiting for result files.",
    )
    return parser.parse_args()


def wait_for_file(path: Path, wait_seconds: int, poll_seconds: int) -> bool:
    if path.exists():
        return True
    if wait_seconds <= 0:
        return False
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(max(int(poll_seconds), 1))
    return path.exists()


def load_strategy_metrics(result_dir: Path) -> pd.DataFrame:
    metrics_path = result_dir / "strategy_metrics.csv"
    summary_path = result_dir / "summary.json"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary_payload.get("strategy_metrics", [])
        return pd.DataFrame(rows)
    raise FileNotFoundError("No strategy_metrics.csv or summary.json found.")


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_report(metrics: pd.DataFrame, result_dir: Path) -> str:
    if metrics.empty:
        return "Aucune metrique disponible."

    metrics = metrics.copy()
    metrics["Rendement cumule"] = metrics["Rendement cumule"].astype(float)
    metrics["Rendement annualise"] = metrics["Rendement annualise"].astype(float)
    metrics["Max drawdown"] = metrics["Max drawdown"].astype(float)
    metrics["VaR 95%"] = metrics["VaR 95%"].astype(float)
    metrics["Turnover moyen"] = metrics["Turnover moyen"].astype(float)
    metrics["Periodes"] = metrics["Periodes"].astype(int)

    ranking = metrics.sort_values("Rendement cumule", ascending=False).reset_index(drop=True)
    ranking["Rang"] = ranking.index + 1
    model_row = ranking[ranking["Strategie"] == "Modele"]
    if model_row.empty:
        raise ValueError("Modele strategy not found in strategy metrics.")
    model_row = model_row.iloc[0]

    references = ranking[ranking["Strategie"] != "Modele"].copy()
    comparison_lines = []
    for _, row in references.iterrows():
        beat = safe_float(model_row["Rendement cumule"]) > safe_float(row["Rendement cumule"])
        comparison_lines.append(
            f"- {row['Strategie']}: {'bat' if beat else 'ne bat pas'} la reference "
            f"(cumule {safe_float(row['Rendement cumule']):.2%}, annualise {safe_float(row['Rendement annualise']):.2%})"
        )

    warnings = []
    if int(model_row["Periodes"]) < 6:
        warnings.append(
            "Backtest court: les metriques annualisees, Sharpe et Calmar restent fragiles."
        )
    if safe_float(model_row["Turnover moyen"]) >= 0.30:
        warnings.append(
            "Turnover eleve: la performance est sensible aux frais de transaction."
        )

    winner = ranking.iloc[0]
    lines = [
        "Resume backtest",
        f"- Dossier: {result_dir}",
        f"- Strategie en tete: {winner['Strategie']}",
        f"- Rang du modele: {int(model_row['Rang'])}/{len(ranking)}",
        (
            "- Modele: "
            f"cumule {safe_float(model_row['Rendement cumule']):.2%} | "
            f"annualise {safe_float(model_row['Rendement annualise']):.2%} | "
            f"max drawdown {safe_float(model_row['Max drawdown']):.2%} | "
            f"VaR 95% {safe_float(model_row['VaR 95%']):.2%} | "
            f"turnover {safe_float(model_row['Turnover moyen']):.2%} | "
            f"periodes {int(model_row['Periodes'])}"
        ),
        "Comparaison aux references",
        *comparison_lines,
    ]
    if warnings:
        lines.append("Points de vigilance")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    summary_path = result_dir / "summary.json"
    metrics_path = result_dir / "strategy_metrics.csv"

    if not wait_for_file(summary_path, args.wait_seconds, args.poll_seconds) and not wait_for_file(
        metrics_path, args.wait_seconds, args.poll_seconds
    ):
        print("Resultats non disponibles pour le moment.")
        return 2

    metrics = load_strategy_metrics(result_dir)
    report = build_report(metrics, result_dir)
    report_path = result_dir / "report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport ecrit dans {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
