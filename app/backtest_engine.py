from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from app.portfolio_optimizer import run_portfolio_optimization
    from app.training_cache import (
        DEFAULT_TEST_SIZE,
        DEFAULT_VAL_SIZE,
        load_forecast_artifact,
        resolve_training_cache,
        with_training_defaults,
    )
    from app.us_market_features import build_training_frame_for_model
except ModuleNotFoundError:
    from portfolio_optimizer import run_portfolio_optimization
    from training_cache import (
        DEFAULT_TEST_SIZE,
        DEFAULT_VAL_SIZE,
        load_forecast_artifact,
        resolve_training_cache,
        with_training_defaults,
    )
    from us_market_features import build_training_frame_for_model


def _required_min_windows(test_size: float, val_size: float) -> int:
    n = 1
    while True:
        test_count = int(n * test_size)
        val_count = int(n * val_size)
        train_count = n - test_count - val_count
        if test_count >= 1 and val_count >= 1 and train_count >= 1:
            return n
        n += 1


def build_price_matrix(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, frame in histories.items():
        cleaned = frame[["Date", "Close"]].copy()
        cleaned["Date"] = pd.to_datetime(cleaned["Date"])
        cleaned = cleaned.dropna().drop_duplicates("Date").sort_values("Date")
        frames.append(cleaned.set_index("Date").rename(columns={"Close": ticker}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna().sort_index()


def common_dates(histories: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    price_matrix = build_price_matrix(histories)
    return list(price_matrix.index.to_list())


def choose_rebalance_dates(
    common_market_dates: list[pd.Timestamp],
    required_history: int,
    horizon_steps: int,
    num_periods: int,
) -> list[pd.Timestamp]:
    usable = common_market_dates[
        required_history - 1 : len(common_market_dates) - horizon_steps
    ]
    if len(usable) < num_periods:
        raise ValueError("Not enough common dates to build the requested backtest.")

    selected = []
    index = len(usable) - 1
    step = max(int(horizon_steps), 1)
    while index >= 0 and len(selected) < num_periods:
        selected.append(usable[index])
        index -= step
    if len(selected) < num_periods:
        raise ValueError("Not enough non-overlapping periods for the requested backtest.")
    return list(reversed(selected))


def forecast_naive_price(last_close: float) -> float:
    return float(last_close)


def run_training_forecast(
    ticker: str,
    model_code: str,
    config: dict,
    training_frame: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    force_retrain: bool = False,
) -> tuple[dict, float, float | None, float | None, str]:
    cache_result = resolve_training_cache(
        ticker=ticker,
        model_code=model_code,
        config=config,
        training_frame=training_frame,
        force_retrain=force_retrain,
        start_date=pd.to_datetime(training_frame["Date"]).min(),
        end_date=pd.to_datetime(cutoff_date),
        required_artifacts=("forecast",),
    )
    forecast_payload = load_forecast_artifact(cache_result["artifact_paths"]["forecast"])
    if not forecast_payload:
        raise RuntimeError("Missing forecast artifact from training cache")
    return (
        cache_result["metrics"],
        float(forecast_payload["forecast_value"]),
        forecast_payload.get("forecast_lower"),
        forecast_payload.get("forecast_upper"),
        cache_result["source"],
    )


def compute_realized_returns(
    price_matrix: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    future_date: pd.Timestamp,
) -> pd.Series:
    last_close = price_matrix.loc[cutoff_date]
    future_close = price_matrix.loc[future_date]
    return ((future_close / last_close) - 1.0).astype(float)


def compute_turnover(
    new_weights: pd.Series,
    previous_weights: pd.Series | None,
) -> float:
    if previous_weights is None:
        return 0.0
    aligned_previous = previous_weights.reindex(new_weights.index).fillna(0.0)
    return float(0.5 * np.abs(new_weights - aligned_previous).sum())


def compute_drawdown(nav_series: pd.Series) -> pd.Series:
    running_max = nav_series.cummax()
    return (nav_series / running_max) - 1.0


def compute_performance_metrics(
    periodic_returns: pd.Series,
    turnover_series: pd.Series | None,
    periods_per_year: float,
) -> dict:
    clean_returns = periodic_returns.dropna().astype(float)
    if clean_returns.empty:
        return {
            "Rendement cumule": np.nan,
            "Rendement annualise": np.nan,
            "Volatilite annualisee": np.nan,
            "Sharpe": np.nan,
            "Calmar": np.nan,
            "Max drawdown": np.nan,
            "VaR 95%": np.nan,
            "CVaR 95%": np.nan,
            "Turnover moyen": np.nan,
            "Periodes": 0,
        }

    nav = (1.0 + clean_returns).cumprod()
    cumulative_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float(nav.iloc[-1] ** (periods_per_year / len(clean_returns)) - 1.0)
    annualized_vol = float(clean_returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = annualized_return / annualized_vol if annualized_vol > 1e-12 else np.nan
    max_drawdown = float(compute_drawdown(nav).min())
    var_95 = float(np.quantile(clean_returns, 0.05))
    tail_losses = clean_returns[clean_returns <= var_95]
    cvar_95 = float(tail_losses.mean()) if not tail_losses.empty else var_95
    calmar = (
        annualized_return / abs(max_drawdown)
        if abs(max_drawdown) > 1e-12
        else np.nan
    )
    turnover_mean = (
        float(turnover_series.dropna().mean())
        if turnover_series is not None and not turnover_series.dropna().empty
        else np.nan
    )
    return {
        "Rendement cumule": cumulative_return,
        "Rendement annualise": annualized_return,
        "Volatilite annualisee": annualized_vol,
        "Sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "Calmar": float(calmar) if not pd.isna(calmar) else np.nan,
        "Max drawdown": max_drawdown,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Turnover moyen": turnover_mean,
        "Periodes": int(len(clean_returns)),
    }


def _build_strategy_weights(
    trailing_returns: pd.DataFrame,
    mu_pred: pd.Series,
    risk_profile: str,
    portfolio_horizon: str,
    max_weight: float | None,
) -> pd.Series:
    if len(mu_pred.index) == 1:
        return pd.Series([1.0], index=mu_pred.index, name="weight")
    return run_portfolio_optimization(
        returns_hist=trailing_returns,
        mu_pred=mu_pred,
        risk_profile=risk_profile,
        horizon=portfolio_horizon,
        allow_short=False,
        max_weight=max_weight,
        covariance_method="ledoit_wolf",
    )


def _prepare_benchmark_series(
    benchmark_history: pd.DataFrame | None,
    market_index: pd.Index,
) -> pd.Series | None:
    if benchmark_history is None or benchmark_history.empty:
        return None
    benchmark_frame = benchmark_history[["Date", "Close"]].copy()
    benchmark_frame["Date"] = pd.to_datetime(benchmark_frame["Date"])
    benchmark_series = (
        benchmark_frame.dropna()
        .drop_duplicates("Date")
        .sort_values("Date")
        .set_index("Date")["Close"]
        .astype(float)
    )
    benchmark_series = benchmark_series.reindex(market_index).ffill()
    if benchmark_series.dropna().empty:
        return None
    return benchmark_series


def apply_benchmark_to_backtest_results(
    results: dict,
    benchmark_history: pd.DataFrame | None,
    benchmark_label: str,
    horizon_steps: int,
) -> dict:
    if not results or "period_returns" not in results:
        raise ValueError("Backtest results unavailable for benchmark refresh.")

    period_df = results["period_returns"].copy()
    if period_df is None or period_df.empty:
        raise ValueError("No period returns available for benchmark refresh.")

    existing_benchmark_label = results.get("benchmark_label")
    if (
        existing_benchmark_label
        and existing_benchmark_label in period_df.columns
        and existing_benchmark_label not in {"Modele", "Naive", "Equal Weight"}
        and existing_benchmark_label != benchmark_label
    ):
        period_df = period_df.drop(columns=[existing_benchmark_label])

    if benchmark_label and benchmark_history is not None and not benchmark_history.empty:
        benchmark_index = pd.Index(
            sorted(
                set(pd.to_datetime(period_df["Date de rebalance"]))
                | set(pd.to_datetime(period_df["Date realisee"]))
            )
        )
        benchmark_series = _prepare_benchmark_series(benchmark_history, benchmark_index)
    else:
        benchmark_series = None

    benchmark_returns = []
    for _, row in period_df.iterrows():
        cutoff_date = pd.to_datetime(row["Date de rebalance"])
        future_date = pd.to_datetime(row["Date realisee"])
        benchmark_return = np.nan
        if benchmark_series is not None:
            last_benchmark = benchmark_series.loc[cutoff_date]
            future_benchmark = benchmark_series.loc[future_date]
            if (
                pd.notna(last_benchmark)
                and pd.notna(future_benchmark)
                and float(last_benchmark) > 0
            ):
                benchmark_return = float((future_benchmark / last_benchmark) - 1.0)
        benchmark_returns.append(benchmark_return)

    if benchmark_label:
        period_df[benchmark_label] = benchmark_returns

    nav_columns = ["Modele", "Naive", "Equal Weight"]
    benchmark_available = (
        benchmark_label in period_df.columns
        and not period_df[benchmark_label].dropna().empty
    )
    if benchmark_available:
        nav_columns.append(benchmark_label)

    nav_frame = period_df[["Date realisee", *nav_columns]].copy()
    nav_frame["Date realisee"] = pd.to_datetime(nav_frame["Date realisee"])
    nav_frame = nav_frame.rename(columns={"Date realisee": "Date"}).sort_values("Date")
    nav_output = pd.DataFrame({"Date": nav_frame["Date"].to_numpy()})
    for column in nav_columns:
        series = nav_frame[column].astype(float).fillna(0.0).to_numpy()
        nav_output[column] = np.cumprod(1.0 + series)

    drawdown_output = pd.DataFrame({"Date": nav_output["Date"]})
    for column in nav_columns:
        drawdown_output[column] = compute_drawdown(
            nav_output.set_index("Date")[column]
        ).values

    periods_per_year = 252.0 / max(int(horizon_steps), 1)
    turnover_map = {
        "Modele": "Turnover modele",
        "Naive": "Turnover naive",
        "Equal Weight": "Turnover equal",
    }
    if benchmark_available:
        turnover_map[benchmark_label] = None

    metric_rows = []
    for strategy_name in nav_columns:
        turnover_series = (
            period_df[turnover_map[strategy_name]]
            if turnover_map.get(strategy_name) in period_df.columns
            else None
        )
        metrics = compute_performance_metrics(
            periodic_returns=period_df[strategy_name],
            turnover_series=turnover_series,
            periods_per_year=periods_per_year,
        )
        metrics["Strategie"] = strategy_name
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)[
        [
            "Strategie",
            "Rendement cumule",
            "Rendement annualise",
            "Volatilite annualisee",
            "Sharpe",
            "Calmar",
            "Max drawdown",
            "VaR 95%",
            "CVaR 95%",
            "Turnover moyen",
            "Periodes",
        ]
    ]

    updated_results = dict(results)
    updated_results["period_returns"] = period_df
    updated_results["nav"] = nav_output
    updated_results["drawdown"] = drawdown_output
    updated_results["strategy_metrics"] = metrics_df
    updated_results["benchmark_label"] = benchmark_label
    return updated_results


def run_market_backtest(
    histories: dict[str, pd.DataFrame],
    model_code: str,
    config: dict,
    history_lookback: int,
    num_periods: int,
    risk_profile: str,
    portfolio_horizon: str,
    max_weight: float,
    transaction_cost_bps: float = 0.0,
    benchmark_history: pd.DataFrame | None = None,
    benchmark_label: str = "Benchmark",
    factor_histories: dict[str, pd.DataFrame] | None = None,
    force_retrain: bool = False,
    progress_callback=None,
) -> dict:
    config = with_training_defaults(config, model_code=model_code)
    price_matrix = build_price_matrix(histories)
    if price_matrix.empty:
        raise ValueError("No common price history available for backtest.")

    required_points = max(
        int(history_lookback),
        int(config["window_size"])
        + int(config["horizon"])
        + (1 if model_code == "lstm" else 0)
        + _required_min_windows(
            float(config.get("test_size", DEFAULT_TEST_SIZE)),
            float(config.get("val_size", DEFAULT_VAL_SIZE)),
        )
        - 1,
    )
    market_dates = list(price_matrix.index.to_list())
    cutoffs = choose_rebalance_dates(
        market_dates,
        required_history=required_points,
        horizon_steps=int(config["horizon"]),
        num_periods=int(num_periods),
    )
    benchmark_series = _prepare_benchmark_series(benchmark_history, price_matrix.index)

    asset_rows = []
    period_rows = []
    allocation_rows = []
    previous_weights = {
        "Modele": None,
        "Naive": None,
        "Equal Weight": None,
    }
    total_steps = (len(cutoffs) * len(histories)) + len(cutoffs)
    completed_steps = 0

    for period_index, cutoff_date in enumerate(cutoffs, start=1):
        future_date = market_dates[market_dates.index(cutoff_date) + int(config["horizon"])]
        model_predicted_returns = {}
        naive_predicted_returns = {}

        for asset_index, (ticker, full_df) in enumerate(histories.items(), start=1):
            if progress_callback is not None:
                progress_callback(
                    completed_steps,
                    total_steps,
                    (
                        f"Periode {period_index}/{len(cutoffs)} - "
                        f"generation du signal pour {ticker} "
                        f"({asset_index}/{len(histories)})"
                    ),
                )
            normalized_df = full_df.copy()
            normalized_df["Date"] = pd.to_datetime(normalized_df["Date"])
            normalized_df["Close"] = pd.to_numeric(
                normalized_df["Close"], errors="coerce"
            )
            normalized_df = normalized_df.dropna(subset=["Date", "Close"]).sort_values(
                "Date"
            )
            train_df = normalized_df[normalized_df["Date"] <= cutoff_date].copy()
            training_frame = build_training_frame_for_model(
                model_code=model_code,
                ticker=ticker,
                asset_frame=train_df,
                factor_histories=factor_histories,
            )
            (
                metrics,
                forecast_price,
                forecast_lower,
                forecast_upper,
                training_source,
            ) = run_training_forecast(
                ticker=ticker,
                model_code=model_code,
                config=config,
                training_frame=training_frame,
                cutoff_date=cutoff_date,
                force_retrain=force_retrain,
            )
            if progress_callback is not None:
                progress_callback(
                    completed_steps,
                    total_steps,
                    (
                        f"Periode {period_index}/{len(cutoffs)} - "
                        f"generation du signal pour {ticker} "
                        f"({asset_index}/{len(histories)}) - {training_source.lower()}"
                    ),
                )

            last_close = float(train_df["Close"].iloc[-1])
            actual_close = float(
                normalized_df.loc[normalized_df["Date"] == future_date, "Close"].iloc[0]
            )
            naive_price = forecast_naive_price(last_close)
            predicted_return = (forecast_price / last_close) - 1.0
            naive_return = (naive_price / last_close) - 1.0
            realized_return = (actual_close / last_close) - 1.0
            model_predicted_returns[ticker] = predicted_return
            naive_predicted_returns[ticker] = naive_return

            asset_rows.append(
                {
                    "Actif": ticker,
                    "Date de rebalance": cutoff_date.strftime("%Y-%m-%d"),
                    "Date realisee": future_date.strftime("%Y-%m-%d"),
                    "Dernier prix": last_close,
                    "Prix predit": forecast_price,
                    "IC 95% bas": forecast_lower,
                    "IC 95% haut": forecast_upper,
                    "Prix naive": naive_price,
                    "Prix reel": actual_close,
                    "Rendement predit": predicted_return,
                    "Rendement naive": naive_return,
                    "Rendement reel": realized_return,
                    "MAE prix": float(metrics.get("mae_price", np.nan)),
                    "Source training": training_source,
                    "Erreur modele": abs(forecast_price - actual_close),
                    "Erreur naive": abs(naive_price - actual_close),
                    "Largeur IC 95%": (
                        float(forecast_upper - forecast_lower)
                        if forecast_lower is not None and forecast_upper is not None
                        else np.nan
                    ),
                    "Reel dans IC 95%": int(
                        forecast_lower is not None
                        and forecast_upper is not None
                        and forecast_lower <= actual_close <= forecast_upper
                    )
                    if forecast_lower is not None and forecast_upper is not None
                    else np.nan,
                    "Modele bat naive": int(
                        abs(forecast_price - actual_close) < abs(naive_price - actual_close)
                    ),
                }
            )
            completed_steps += 1

        if progress_callback is not None:
            progress_callback(
                completed_steps,
                total_steps,
                f"Periode {period_index}/{len(cutoffs)} - construction des portefeuilles",
            )
        trailing_returns = (
            price_matrix.loc[:cutoff_date]
            .pct_change()
            .dropna()
            .tail(int(history_lookback))
        )
        realized_returns = compute_realized_returns(
            price_matrix=price_matrix,
            cutoff_date=cutoff_date,
            future_date=future_date,
        )

        model_weights = _build_strategy_weights(
            trailing_returns=trailing_returns,
            mu_pred=pd.Series(model_predicted_returns),
            risk_profile=risk_profile,
            portfolio_horizon=portfolio_horizon,
            max_weight=max_weight,
        )
        naive_weights = _build_strategy_weights(
            trailing_returns=trailing_returns,
            mu_pred=pd.Series(naive_predicted_returns),
            risk_profile=risk_profile,
            portfolio_horizon=portfolio_horizon,
            max_weight=max_weight,
        )
        equal_weights = pd.Series(
            1.0 / len(realized_returns),
            index=realized_returns.index,
            name="weight",
        )

        strategy_weights = {
            "Modele": model_weights,
            "Naive": naive_weights,
            "Equal Weight": equal_weights,
        }
        strategy_returns = {}
        turnovers = {}
        costs = {}
        for strategy_name, weights in strategy_weights.items():
            turnover = compute_turnover(weights, previous_weights[strategy_name])
            cost = turnover * (float(transaction_cost_bps) / 10000.0)
            realized = float((weights * realized_returns.loc[weights.index]).sum())
            strategy_returns[strategy_name] = realized - cost
            turnovers[strategy_name] = turnover
            costs[strategy_name] = cost
            previous_weights[strategy_name] = weights
            allocation_rows.append(
                {
                    "Date de rebalance": cutoff_date.strftime("%Y-%m-%d"),
                    "Date realisee": future_date.strftime("%Y-%m-%d"),
                    "Strategie": strategy_name,
                    "Turnover": turnover,
                    "Cout": cost,
                    "Poids": ", ".join(
                        f"{asset}={value:.3f}" for asset, value in weights.items()
                    ),
                }
            )

        benchmark_return = np.nan
        if benchmark_series is not None:
            last_benchmark = benchmark_series.loc[cutoff_date]
            future_benchmark = benchmark_series.loc[future_date]
            if pd.notna(last_benchmark) and pd.notna(future_benchmark) and last_benchmark > 0:
                benchmark_return = float((future_benchmark / last_benchmark) - 1.0)

        period_rows.append(
            {
                "Date de rebalance": cutoff_date.strftime("%Y-%m-%d"),
                "Date realisee": future_date.strftime("%Y-%m-%d"),
                "Modele": strategy_returns["Modele"],
                "Naive": strategy_returns["Naive"],
                "Equal Weight": strategy_returns["Equal Weight"],
                benchmark_label: benchmark_return,
                "Turnover modele": turnovers["Modele"],
                "Turnover naive": turnovers["Naive"],
                "Turnover equal": turnovers["Equal Weight"],
                "Cout modele": costs["Modele"],
                "Cout naive": costs["Naive"],
                "Cout equal": costs["Equal Weight"],
            }
        )
        completed_steps += 1

    asset_details = pd.DataFrame(asset_rows)
    period_df = pd.DataFrame(period_rows).sort_values("Date realisee")
    allocation_df = pd.DataFrame(allocation_rows).sort_values(
        ["Date realisee", "Strategie"]
    )

    nav_columns = ["Modele", "Naive", "Equal Weight"]
    benchmark_available = (
        benchmark_label in period_df.columns
        and not period_df[benchmark_label].dropna().empty
    )
    if benchmark_available:
        nav_columns.append(benchmark_label)

    nav_frame = period_df[["Date realisee", *nav_columns]].copy()
    nav_frame["Date realisee"] = pd.to_datetime(nav_frame["Date realisee"])
    nav_frame = nav_frame.rename(columns={"Date realisee": "Date"})
    nav_frame = nav_frame.sort_values("Date")
    nav_output = pd.DataFrame({"Date": nav_frame["Date"].to_numpy()})
    for column in nav_columns:
        series = nav_frame[column].astype(float).fillna(0.0).to_numpy()
        nav_output[column] = np.cumprod(1.0 + series)

    drawdown_output = pd.DataFrame({"Date": nav_output["Date"]})
    for column in nav_columns:
        drawdown_output[column] = compute_drawdown(
            nav_output.set_index("Date")[column]
        ).values

    periods_per_year = 252.0 / max(int(config["horizon"]), 1)
    metric_rows = []
    turnover_map = {
        "Modele": "Turnover modele",
        "Naive": "Turnover naive",
        "Equal Weight": "Turnover equal",
    }
    if benchmark_available:
        turnover_map[benchmark_label] = None
    for strategy_name in nav_columns:
        turnover_series = (
            period_df[turnover_map[strategy_name]]
            if turnover_map.get(strategy_name) in period_df.columns
            else None
        )
        metrics = compute_performance_metrics(
            periodic_returns=period_df[strategy_name],
            turnover_series=turnover_series,
            periods_per_year=periods_per_year,
        )
        metrics["Strategie"] = strategy_name
        metric_rows.append(metrics)
    metrics_df = pd.DataFrame(metric_rows)[
        [
            "Strategie",
            "Rendement cumule",
            "Rendement annualise",
            "Volatilite annualisee",
            "Sharpe",
            "Calmar",
            "Max drawdown",
            "VaR 95%",
            "CVaR 95%",
            "Turnover moyen",
            "Periodes",
        ]
    ]

    forecast_summary = (
        asset_details.groupby("Actif")
        .agg(
            mae_prix=("MAE prix", "mean"),
            erreur_modele=("Erreur modele", "mean"),
            erreur_naive=("Erreur naive", "mean"),
            ic95_largeur=("Largeur IC 95%", "mean"),
            ic95_couverture=("Reel dans IC 95%", "mean"),
            beat_naive=("Modele bat naive", "mean"),
            rendement_predit=("Rendement predit", "mean"),
            rendement_naive=("Rendement naive", "mean"),
            rendement_reel=("Rendement reel", "mean"),
        )
        .reset_index()
    )

    return {
        "asset_details": asset_details,
        "forecast_summary": forecast_summary,
        "period_returns": period_df,
        "allocation_history": allocation_df,
        "nav": nav_output,
        "drawdown": drawdown_output,
        "strategy_metrics": metrics_df,
        "cutoff_dates": [date.strftime("%Y-%m-%d") for date in cutoffs],
        "benchmark_label": benchmark_label,
    }
