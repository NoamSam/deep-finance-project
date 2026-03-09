from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from app.config import DEFAULT_CONFIG


DEFAULT_TRAIN_TIMEOUT_SEC = 900


@dataclass
class TrainResult:
    model_type: str
    metrics: dict
    history: object


def _build_command(csv_path, model_type, config):
    runner_path = Path(__file__).resolve().parent / "train_runner.py"
    return [
        sys.executable,
        str(runner_path),
        "--csv",
        str(csv_path),
        "--model",
        model_type,
        "--window-size",
        str(config["window_size"]),
        "--horizon",
        str(config["horizon"]),
        "--test-size",
        str(config["test_size"]),
        "--val-size",
        str(config["val_size"]),
        "--batch-size",
        str(config["batch_size"]),
        "--epochs",
        str(config["epochs"]),
        "--learning-rate",
        str(config["learning_rate"]),
    ]


def run_training(csv_path, model_type, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    command = _build_command(csv_path, model_type, config)
    timeout_sec = int(config.get("timeout_sec", DEFAULT_TRAIN_TIMEOUT_SEC))

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Training timeout after {timeout_sec}s"
        ) from exc

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        details = completed.stderr.strip()
        raise RuntimeError(details or "Training subprocess produced no output")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unable to parse training metrics: {lines[-1]}"
        ) from exc

    if not payload.get("ok"):
        message = payload.get("error") or completed.stderr.strip()
        raise RuntimeError(message or "Training subprocess failed")
    if completed.returncode != 0:
        message = payload.get("error") or completed.stderr.strip()
        raise RuntimeError(message or "Training subprocess failed")

    return TrainResult(
        model_type=model_type,
        metrics=payload["metrics"],
        history=None,
    )
