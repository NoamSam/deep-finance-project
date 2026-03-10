import altair as alt
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

try:
    from app.backtest_engine import apply_benchmark_to_backtest_results, run_market_backtest
    from app.backtest_cache import (
        build_backtest_cache_key,
        load_cached_backtest,
        save_cached_backtest,
    )
    from app.portfolio_optimizer import estimate_covariance
    from app.training_cache import (
        DEFAULT_LSTM_DROPOUT,
        DEFAULT_LSTM_UNITS1,
        DEFAULT_LSTM_UNITS2,
        TRAINING_CACHE_DIR,
        TRAINING_PIPELINE_VERSION,
        TRAINING_SUBPROCESS_TIMEOUT_SEC,
        load_latest_training_cache,
        peek_training_cache,
        resolve_training_cache,
    )
    from app.us_market_features import (
        US_FACTOR_TICKERS,
        build_training_frame_for_model,
        normalize_history_frame,
    )
    from app.update_curves import DATA_DIR, fetch_asset, update_assets
except ModuleNotFoundError:
    from backtest_engine import apply_benchmark_to_backtest_results, run_market_backtest
    from backtest_cache import (
        build_backtest_cache_key,
        load_cached_backtest,
        save_cached_backtest,
    )
    from portfolio_optimizer import estimate_covariance
    from training_cache import (
        DEFAULT_LSTM_DROPOUT,
        DEFAULT_LSTM_UNITS1,
        DEFAULT_LSTM_UNITS2,
        TRAINING_CACHE_DIR,
        TRAINING_PIPELINE_VERSION,
        TRAINING_SUBPROCESS_TIMEOUT_SEC,
        load_latest_training_cache,
        peek_training_cache,
        resolve_training_cache,
    )
    from us_market_features import (
        US_FACTOR_TICKERS,
        build_training_frame_for_model,
        normalize_history_frame,
    )
    from update_curves import DATA_DIR, fetch_asset, update_assets

BASE_TICKERS_PATH = Path(__file__).resolve().parent / "base_tickers.txt"
MODEL_TYPE_MAP = {
    "LSTM": "lstm",
    "LSTM US multi-features": "lstm_multifeature",
    "CNN": "cnn",
    "CNN puis LSTM": "cnn_lstm",
}
MIN_WINDOW_SIZE = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
MAX_DEFAULT_PLOT_ASSETS = 12
DEFAULT_MAX_PLOT_POINTS = 1200
PLOT_RESAMPLE_RULES = {
    "Journalier": "D",
    "Hebdomadaire": "W-FRI",
    "Mensuel": "M",
}
MIXED_DISPLAY_DAILY_LIMIT = 21
MIXED_DISPLAY_WEEKLY_STEP = 5
HORIZON_OPTIONS = {
    "1 jour": 1,
    "3 jours": 3,
    "5 jours": 5,
    "7 jours": 7,
    "2 semaines": 10,
    "1 mois": 21,
    "3 mois": 63,
    "6 mois": 126,
    "1 an": 252,
    "3 ans": 756,
}
BACKTEST_BENCHMARKS = {
    "Aucun": None,
    "MAGS (Magnificent Seven ETF)": "MAGS",
    "SPY (ETF S&P 500)": "SPY",
    "CAC 40": "^FCHI",
}
BACKTEST_RISK_PROFILES = ["conservateur", "equilibre", "dynamique"]
BACKTEST_PORTFOLIO_HORIZONS = ["court", "moyen", "long"]
BACKTEST_RISK_PROFILE_LABELS = {
    "conservateur": "Conservateur - priorite a la stabilite et a la reduction du risque",
    "equilibre": "Equilibre - compromis entre potentiel de performance et controle du risque",
    "dynamique": "Dynamique - recherche de rendement avec plus de volatilite acceptee",
}
BACKTEST_PORTFOLIO_HORIZON_LABELS = {
    "court": "Court - prudence elevee (repere 21 jours de bourse)",
    "moyen": "Moyen - prudence intermediaire (repere 63 jours de bourse)",
    "long": "Long - prudence plus faible (repere 252 jours de bourse)",
}
BACKTEST_PERIOD_PRESETS = {
    "Rapide - 4 periodes": 4,
    "Robuste - 10 periodes": 10,
}


def _recommended_risk_horizon(horizon_steps: int) -> str:
    if horizon_steps <= 21:
        return "court"
    if horizon_steps <= 63:
        return "moyen"
    return "long"


def _sync_backtest_risk_horizon_default(horizon_steps: int) -> str:
    recommended = _recommended_risk_horizon(int(horizon_steps))
    previous_recommended = st.session_state.get("_backtest_portfolio_horizon_reco")
    previous_model_horizon = st.session_state.get(
        "_backtest_portfolio_horizon_model_horizon"
    )
    current_value = st.session_state.get("backtest_portfolio_horizon")
    if current_value is None:
        st.session_state.backtest_portfolio_horizon = recommended
    elif int(previous_model_horizon or -1) != int(horizon_steps):
        if current_value == previous_recommended:
            st.session_state.backtest_portfolio_horizon = recommended
    st.session_state._backtest_portfolio_horizon_reco = recommended
    st.session_state._backtest_portfolio_horizon_model_horizon = int(horizon_steps)
    return recommended


def _sync_numeric_widget_with_recommendation(
    widget_key: str, tracker_key: str, recommended_value
):
    previous_recommended = st.session_state.get(tracker_key)
    current_value = st.session_state.get(widget_key)
    if current_value is None or current_value == previous_recommended:
        st.session_state[widget_key] = recommended_value
    st.session_state[tracker_key] = recommended_value


HELP_TEXT = {
    "asset_categories": "Regroupe les actifs par univers pour accelerer la selection.",
    "start_date": "Date minimale des donnees chargees et utilisees pour les graphes, le training et le backtest.",
    "end_date": "Date maximale des donnees chargees. Pratique pour figer un backtest a une date passee.",
    "window_size": "Nombre de points historiques donnes au modele pour produire une prediction.",
    "epochs": "Nombre maximum de passages sur les donnees d'entrainement.",
    "model_type": "Architecture utilisee pour produire le signal de prediction. LSTM = single-feature, LSTM US multi-features = OHLCV + SPY/QQQ/VIX.",
    "horizon": "Nombre de pas de marche vises par la prediction. 1 mois = environ 21 seances.",
    "force_retrain": "Ignore le cache local et relance un entrainement complet pour chaque actif.",
    "generate_curves": "Charge les donnees historiques et active les graphes de prix pour les actifs selectionnes.",
    "update_assets": "Telecharge ou rafraichit les historiques Yahoo Finance des actifs selectionnes.",
    "plot_assets": "Actifs affiches sur le graphe de prix historique.",
    "plot_resolution": "Frequence de re-echantillonnage du graphe pour limiter la charge visuelle.",
    "plot_max_points": "Cap de points traces pour reduire le lag quand beaucoup d'actifs sont affiches.",
    "backtest_periods": "Nombre de periodes de rebalance non chevauchantes utilisees pour le backtest.",
    "backtest_period_preset": "Preset rapide pour choisir entre une lecture courte de demonstration et un backtest plus robuste.",
    "history_lookback": "Historique de rendements utilise pour estimer le risque et la covariance du portefeuille.",
    "risk_profile": "Pilote l'aversion au risque dans l'optimisation portefeuille.",
    "portfolio_horizon": "Cadre de risque portefeuille utilise pour regler la penalisation du risque dans l'allocation. Il ne change pas l'horizon de prediction du modele ni la duree de detention backtestee. Son effet peut rester limite si le signal est deja tres concentre ou si le poids max par actif bloque deja l'optimisation.",
    "max_weight": "Poids maximum autorise sur un actif pour limiter la concentration.",
    "transaction_cost_bps": "Frais appliques a chaque rebalance. 0 bps = 0%, 10 bps = 0,10%.",
    "benchmark": "Indice ou ETF de reference pour comparer la strategie.",
    "run_backtest": "Lance un backtest rolling a partir de la configuration actuelle.",
    "strategy_metrics": "Synthese risque/rendement de chaque strategie sur les periodes testees.",
    "cumulative_perf": "Valeur cumulee de 1 euro investi dans chaque strategie au fil du backtest.",
    "drawdown": "Perte relative par rapport au plus haut historique de la strategie.",
    "forecast_compare": "Compare le signal du modele a la reference Naive, actif par actif.",
    "alloc_recent": "Dernieres allocations calculees pour chaque strategie au fil des rebalances.",
    "period_details": "Rendements, turnovers et couts strategie par strategie pour chaque periode.",
    "executive_summary": "Vue courte pour presenter le verdict du backtest a un jury, un client ou un investisseur.",
    "best_strategy": "Strategie ayant le meilleur rendement annualise sur le backtest courant.",
    "annualized_return": "Performance projetee sur un an a partir des periodes du backtest.",
    "sharpe": "Rendement annualise rapporte a la volatilite. Plus haut est generalement meilleur.",
    "max_drawdown": "Pire baisse observee depuis un plus haut historique.",
    "var_95": "VaR 95%: perte seuil estimee sur une periode. Exemple: -2% signifie qu'environ 5% des periodes font pire que -2%.",
    "cvar_95": "CVaR 95%: perte moyenne observee dans les 5% pires periodes. Mesure plus severe que la VaR.",
    "calmar": "Calmar ratio: rendement annualise divise par le drawdown maximal. Plus haut est generalement meilleur.",
    "turnover": "Part moyenne du portefeuille qui change a chaque rebalance.",
    "beat_rate": "Frequence a laquelle le modele bat une reference sur la mesure indiquee.",
    "allocation_view": "Derniere allocation modele issue du backtest, avec comparaison aux references.",
    "active_positions": "Nombre d'actifs portant un poids non nul dans le portefeuille.",
    "max_weight_metric": "Poids de la position la plus importante du portefeuille.",
    "effective_positions": "Mesure de diversification basee sur 1 / somme des poids au carre.",
    "top3_weight_share": "Part du portefeuille concentree sur les trois plus grosses positions.",
    "concentration_hhi": "Indice de concentration Herfindahl-Hirschman. Plus il est eleve, plus le portefeuille est concentre.",
    "concentration_label": "Lecture synthétique du niveau de concentration du portefeuille.",
    "weight_chart": "Repartition des poids du portefeuille modele sur la derniere date de rebalance.",
    "risk_contribution_chart": "Visualise la part relative du risque portefeuille portee par chaque actif.",
    "overweights": "Actifs les plus surponderes par rapport a un portefeuille equipondere.",
    "underweights": "Actifs les plus sous-ponderes par rapport a un portefeuille equipondere.",
    "download_metrics": "Exporte les metriques de strategie du backtest courant.",
    "download_periods": "Exporte les rendements et couts periode par periode.",
    "download_allocations": "Exporte l'historique des poids de portefeuille.",
    "download_forecast": "Exporte le resume forecast par actif.",
    "download_asset_details": "Exporte les details forecast de tous les actifs et periodes.",
    "download_summary": "Exporte un petit resume JSON du run courant.",
}

TABLE_COLUMN_HELP = {
    "training_results": [
        ("MAE prix", "Erreur absolue moyenne entre prix predit et prix reel. Plus bas est meilleur."),
        ("RMSE prix", "Erreur quadratique moyenne racine. Penalise davantage les grosses erreurs."),
        ("MAPE %", "Erreur moyenne relative en pourcentage du prix reel."),
        ("MAE norm", "Erreur absolue moyenne dans l'espace normalise du modele. Utile surtout pour comparer des runs techniques."),
        ("Points utilises", "Nombre de points historiques effectivement disponibles pour cet actif."),
        ("Fenetre utilisee", "Taille de fenetre finalement retenue pour entrainer le modele sur cet actif."),
        ("Source", "Origine du resultat : cache existant ou nouvel entrainement."),
    ],
    "future_forecasts": [
        ("Etape", "Pas de projection. Avec horizon 5 jours, les etapes vont de 1 a 5."),
        ("Date prevision", "Date de marche visee par la prediction."),
        ("Prix predit", "Prix projete par le modele a cette etape."),
        ("IC 95% bas", "Borne basse de la fourchette d'incertitude empirique."),
        ("IC 95% haut", "Borne haute de la fourchette d'incertitude empirique."),
        ("Source", "Origine du resultat : cache existant ou nouvel entrainement."),
    ],
    "strategy_metrics": [
        ("Rendement annualise", "Performance annualisee estimee a partir des periodes du backtest."),
        ("Rendement cumule", "Performance totale cumulee sur la fenetre de backtest."),
        ("Volatilite annualisee", "Volatilite des rendements, annualisee."),
        ("Sharpe", "Rendement annualise rapporte a la volatilite. Plus haut est meilleur."),
        ("Max drawdown", "Pire baisse observee depuis un plus haut historique."),
        ("VaR 95%", "Perte seuil estimee sur une periode. Environ 5% des periodes font pire."),
        ("CVaR 95%", "Perte moyenne dans les 5% pires periodes. Plus severe que la VaR."),
        ("Calmar", "Rendement annualise divise par le drawdown maximal."),
        ("Turnover moyen", "Part moyenne du portefeuille remplacee a chaque rebalance."),
        ("Periodes", "Nombre de periodes de backtest effectivement utilisees."),
    ],
    "forecast_summary": [
        ("MAE prix", "Erreur absolue moyenne du modele sur le prix."),
        ("Erreur modele", "Erreur absolue moyenne du modele sur les periodes du backtest."),
        ("Erreur naive", "Erreur absolue moyenne de la reference Naive."),
        ("Largeur IC 95%", "Amplitude moyenne de la fourchette d'incertitude du modele."),
        ("Couverture IC 95%", "Frequence a laquelle le prix reel tombe dans la fourchette IC 95%."),
        ("Beat vs naive", "Part des cas ou le modele fait mieux que la reference Naive."),
        ("Rendement modele", "Rendement moyen projete par le modele."),
        ("Rendement naive", "Rendement moyen projete par la reference Naive."),
        ("Rendement reel", "Rendement moyen observe sur la periode cible."),
    ],
    "allocation_main": [
        ("Poids modele", "Poids recommande par la strategie Modele."),
        ("Poids equal-weight", "Poids equipondere de reference."),
        ("Ecart vs equal-weight", "Sur- ou sous-ponderation du modele versus equipondere."),
        ("Poids naive", "Poids recommends par la reference Naive."),
        ("Ecart vs naive", "Ecart de poids entre Modele et Naive."),
        ("Signal modele", "Rendement projete par le modele pour l'actif."),
        ("Signal naive", "Rendement projete par la reference Naive."),
        ("Rendement realise", "Rendement effectivement observe sur la periode realisee."),
        ("Contribution risque", "Part relative du risque portefeuille portee par l'actif."),
        ("Erreur modele", "Erreur absolue du modele sur cet actif pour la derniere periode."),
    ],
    "period_details": [
        ("Modele", "Rendement portefeuille de la strategie Modele sur la periode."),
        ("Naive", "Rendement portefeuille de la reference Naive sur la periode."),
        ("Equal Weight", "Rendement du portefeuille equipondere sur la periode."),
        ("Turnover modele", "Part du portefeuille Modele remplacee a la rebalance."),
        ("Turnover naive", "Part du portefeuille Naive remplacee a la rebalance."),
        ("Turnover equal", "Part du portefeuille Equal Weight remplacee a la rebalance."),
        ("Cout modele", "Impact estime des frais de transaction du portefeuille Modele."),
        ("Cout naive", "Impact estime des frais de transaction du portefeuille Naive."),
        ("Cout equal", "Impact estime des frais de transaction du portefeuille Equal Weight."),
    ],
}

ASSET_CATEGORIES = {
    "Beautiful Seven (US)": [
        "AAPL",
        "AMZN",
        "GOOG",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
    ],
    "S&P 500 - Top 50": [
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "GOOG",
        "META",
        "AVGO",
        "TSLA",
        "BRK-B",
        "WMT",
        "LLY",
        "JPM",
        "XOM",
        "V",
        "JNJ",
        "MA",
        "COST",
        "ORCL",
        "NFLX",
        "MU",
        "ABBV",
        "CVX",
        "PLTR",
        "PG",
        "HD",
        "BAC",
        "GE",
        "KO",
        "CAT",
        "AMD",
        "CSCO",
        "MRK",
        "RTX",
        "PM",
        "UNH",
        "AMAT",
        "MS",
        "LRCX",
        "WFC",
        "GS",
        "TMUS",
        "IBM",
        "MCD",
        "LIN",
        "PEP",
        "INTC",
        "VZ",
        "GEV",
        "AXP",
    ],
    "Tout CAC 40": [
        "ACA.PA",
        "AC.PA",
        "AI.PA",
        "AIR.PA",
        "ALO.PA",
        "BN.PA",
        "BNP.PA",
        "CA.PA",
        "CAP.PA",
        "CS.PA",
        "DG.PA",
        "DSY.PA",
        "EDEN.PA",
        "EN.PA",
        "ENGI.PA",
        "ERF.PA",
        "GLE.PA",
        "HO.PA",
        "KER.PA",
        "LR.PA",
        "MC.PA",
        "MT.PA",
        "ML.PA",
        "OR.PA",
        "PUB.PA",
        "RI.PA",
        "RMS.PA",
        "RNO.PA",
        "RUI.PA",
        "SAF.PA",
        "SAN.PA",
        "SGO.PA",
        "STMPA.PA",
        "SU.PA",
        "TEP.PA",
        "TTE.PA",
        "URW.PA",
        "VIE.PA",
        "VIV.PA",
        "WLN.PA",
    ],
    "Luxe": [
        "MC.PA",
        "RMS.PA",
        "KER.PA",
        "RCO.PA",
        "STMPA.PA",
    ],
    "Industrie": [
        "AI.PA",
        "EN.PA",
        "SU.PA",
        "VIE.PA",
        "DG.PA",
        "RNO.PA",
        "STLAP.PA",
        "FRVIA.PA",
        "AIR.PA",
    ],
    "Defense": [
        "AIR.PA",
        "HO.PA",
        "SAF.PA",
        "EXENS.PA",
        "TE.PA",
    ],
}


def load_base_tickers():
    if not BASE_TICKERS_PATH.exists():
        return []
    with BASE_TICKERS_PATH.open("r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


def format_base_ticker(ticker):
    return ticker.strip().upper()


def normalize_ticker(ticker, base_tickers):
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    if "." not in ticker and f"{ticker}.PA" in base_tickers:
        return f"{ticker}.PA"
    return ticker


def normalize_category_name(name):
    return "".join(
        char.lower() if char.isalnum() else "_" for char in name
    ).strip("_")


def build_category_assets(known_assets):
    known_set = set(known_assets)
    category_assets = {}
    for category, tickers in ASSET_CATEGORIES.items():
        category_assets[category] = [
            ticker for ticker in tickers if ticker in known_set
        ]
    return category_assets


def get_asset_path(ticker):
    safe_name = ticker.replace("/", "_")
    return DATA_DIR / f"{safe_name}.csv"


def get_asset_cache_token(ticker):
    path = get_asset_path(ticker)
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


def _load_or_fetch_asset_history(ticker, start_date=None, end_date=None, require_ohlcv=False):
    path = get_asset_path(ticker)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = fetch_asset(ticker, start=start_date, end=end_date)

    missing_ohlcv = require_ohlcv and not {"Open", "High", "Low", "Volume"}.issubset(
        set(df.columns)
    )
    if missing_ohlcv:
        df = fetch_asset(ticker, start=start_date, end=end_date)
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass

    normalized = normalize_history_frame(df, require_ohlcv=require_ohlcv)
    if start_date:
        normalized = normalized[normalized["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        normalized = normalized[normalized["Date"] <= pd.to_datetime(end_date)]
    return normalized.reset_index(drop=True)


def _load_eval_frame(eval_path, frame, horizon_steps):
    if not eval_path.exists():
        return None
    try:
        payload = np.load(eval_path)
        target_indices = payload["target_close_indices"].astype(np.int64)
        y_true = payload["y_true"].astype(np.float64)
        y_pred = payload["y_pred"].astype(np.float64)
    except Exception:
        return None

    usable_length = min(len(target_indices), len(y_true), len(y_pred))
    if usable_length <= 0:
        return None

    target_indices = target_indices[:usable_length]
    y_true = y_true[:usable_length]
    y_pred = y_pred[:usable_length]

    valid_mask = (target_indices >= 0) & (target_indices < len(frame))
    if not np.any(valid_mask):
        return None

    target_indices = target_indices[valid_mask]
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    decision_indices = target_indices - int(horizon_steps)
    decision_dates = np.full(len(target_indices), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_decision = (decision_indices >= 0) & (decision_indices < len(frame))
    if np.any(valid_decision):
        decision_dates[valid_decision] = pd.to_datetime(
            frame["Date"].iloc[decision_indices[valid_decision]].to_numpy()
        ).to_numpy()

    plot_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["Date"].iloc[target_indices].to_numpy()),
            "Date cible": pd.to_datetime(frame["Date"].iloc[target_indices].to_numpy()),
            "Date decision": pd.to_datetime(decision_dates),
            "Reel": y_true,
            "Predit": y_pred,
        }
    ).sort_values("Date")
    return plot_frame


@st.cache_data(show_spinner=False)
def load_asset_frame(ticker, start_date=None, end_date=None, cache_token=None):
    # cache_token is used to invalidate Streamlit cache when CSV files are replaced.
    _ = cache_token
    df = _load_or_fetch_asset_history(
        ticker,
        start_date=start_date,
        end_date=end_date,
        require_ohlcv=False,
    )
    return df[["Date", "Close"]]


@st.cache_data(show_spinner=False)
def load_asset_ohlcv_frame(ticker, start_date=None, end_date=None, cache_token=None):
    _ = cache_token
    return _load_or_fetch_asset_history(
        ticker,
        start_date=start_date,
        end_date=end_date,
        require_ohlcv=True,
    )


@st.cache_data(show_spinner=False)
def load_us_factor_histories(start_date=None, end_date=None, cache_token=None):
    _ = cache_token
    histories = {}
    for ticker in US_FACTOR_TICKERS.values():
        histories[ticker] = load_asset_frame(
            ticker,
            start_date,
            end_date,
            get_asset_cache_token(ticker),
        )
    return histories


def _required_min_windows(test_size, val_size):
    n = 1
    while True:
        test_count = int(n * test_size)
        val_count = int(n * val_size)
        train_count = n - test_count - val_count
        if test_count >= 1 and val_count >= 1 and train_count >= 1:
            return n
        n += 1


def _effective_window_size(
    total_points, requested_window, horizon, required_windows, extra_points_needed=0
):
    max_window = (
        total_points - horizon - (required_windows - 1) - int(extra_points_needed)
    )
    if max_window < MIN_WINDOW_SIZE:
        return None
    return min(requested_window, max_window)


def _downsample_plot_frame(frame, max_points):
    if frame.empty or max_points <= 0 or len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    reduced = frame.iloc[::step]
    if reduced.index[-1] != frame.index[-1]:
        reduced = pd.concat([reduced, frame.iloc[[-1]]])
    return reduced[~reduced.index.duplicated(keep="last")]


def _append_future_prediction_row(
    plot_frame,
    forecast_date,
    forecast_value,
    forecast_lower=None,
    forecast_upper=None,
):
    payload = {
        "Date": [pd.to_datetime(forecast_date)],
        "Reel": [np.nan],
        "Predit": [float(forecast_value)],
    }
    if forecast_lower is not None:
        payload["IC 95% bas"] = [float(forecast_lower)]
    if forecast_upper is not None:
        payload["IC 95% haut"] = [float(forecast_upper)]
    future_row = pd.DataFrame(payload)
    if plot_frame is None or plot_frame.empty:
        return future_row
    merged = pd.concat([plot_frame, future_row], ignore_index=True)
    merged = (
        merged.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return merged


def _append_future_prediction_path(plot_frame, future_points):
    if future_points is None or future_points.empty:
        return plot_frame
    output = plot_frame
    for _, row in future_points.iterrows():
        output = _append_future_prediction_row(
            output,
            row["Date"],
            row["Prix predit"],
            forecast_lower=row["IC 95% bas"]
            if "IC 95% bas" in row and pd.notna(row["IC 95% bas"])
            else None,
            forecast_upper=row["IC 95% haut"]
            if "IC 95% haut" in row and pd.notna(row["IC 95% haut"])
            else None,
        )
    return output


def _load_forecast_artifact(forecast_path):
    if not forecast_path.exists():
        return None
    try:
        payload = np.load(forecast_path)
        if "forecast_value" not in payload:
            return None
        values = payload["forecast_value"]
        if len(values) == 0:
            return None
        forecast_lower = None
        forecast_upper = None
        if "forecast_lower" in payload and len(payload["forecast_lower"]) > 0:
            forecast_lower = float(payload["forecast_lower"][0])
        if "forecast_upper" in payload and len(payload["forecast_upper"]) > 0:
            forecast_upper = float(payload["forecast_upper"][0])
        return {
            "forecast_value": float(values[0]),
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
        }
    except Exception:
        return None


def _load_forecast_path_frame(forecast_path, frame, horizon_steps):
    if not forecast_path.exists():
        return None
    try:
        payload = np.load(forecast_path)
        if "steps" not in payload or "forecast_values" not in payload:
            return None
        steps = np.asarray(payload["steps"], dtype=np.int32).reshape(-1)
        values = np.asarray(payload["forecast_values"], dtype=np.float64).reshape(
            -1
        )
        lower_values = (
            np.asarray(payload["forecast_lower_values"], dtype=np.float64).reshape(-1)
            if "forecast_lower_values" in payload
            else None
        )
        upper_values = (
            np.asarray(payload["forecast_upper_values"], dtype=np.float64).reshape(-1)
            if "forecast_upper_values" in payload
            else None
        )
    except Exception:
        return None

    usable_length = min(len(steps), len(values))
    if lower_values is not None:
        usable_length = min(usable_length, len(lower_values))
    if upper_values is not None:
        usable_length = min(usable_length, len(upper_values))
    if usable_length <= 0:
        return None

    steps = steps[:usable_length]
    values = values[:usable_length]
    if lower_values is not None:
        lower_values = lower_values[:usable_length]
    if upper_values is not None:
        upper_values = upper_values[:usable_length]
    valid = steps > 0
    if horizon_steps > 0:
        valid = valid & (steps <= int(horizon_steps))
    if not np.any(valid):
        return None

    steps = steps[valid]
    values = values[valid]
    if lower_values is not None:
        lower_values = lower_values[valid]
    if upper_values is not None:
        upper_values = upper_values[valid]
    base_date = pd.to_datetime(frame["Date"].max())
    forecast_dates = [base_date + pd.offsets.BDay(int(step)) for step in steps]
    payload = {
        "Etape": steps.astype(int),
        "Date": forecast_dates,
        "Prix predit": values.astype(float),
    }
    if lower_values is not None:
        payload["IC 95% bas"] = lower_values.astype(float)
    if upper_values is not None:
        payload["IC 95% haut"] = upper_values.astype(float)
    return pd.DataFrame(payload).sort_values("Etape")


def _build_single_step_forecast_frame(frame, horizon_steps, forecast_artifact):
    if not forecast_artifact:
        return None
    forecast_date = pd.to_datetime(frame["Date"].max()) + pd.offsets.BDay(
        int(horizon_steps)
    )
    payload = {
        "Etape": [int(horizon_steps)],
        "Date": [forecast_date],
        "Prix predit": [float(forecast_artifact["forecast_value"])],
    }
    if forecast_artifact.get("forecast_lower") is not None:
        payload["IC 95% bas"] = [float(forecast_artifact["forecast_lower"])]
    if forecast_artifact.get("forecast_upper") is not None:
        payload["IC 95% haut"] = [float(forecast_artifact["forecast_upper"])]
    return pd.DataFrame(payload)


def _frame_for_cached_artifacts(frame, metadata):
    if frame is None or frame.empty or not metadata:
        return frame
    display_frame = frame.copy()
    date_filter = metadata.get("date_filter", {}) or {}
    if date_filter.get("start"):
        display_frame = display_frame[
            pd.to_datetime(display_frame["Date"]) >= pd.to_datetime(date_filter["start"])
        ].copy()
    training_last_date = metadata.get("training_last_date")
    if training_last_date:
        display_frame = display_frame[
            pd.to_datetime(display_frame["Date"]) <= pd.to_datetime(training_last_date)
        ].copy()
    elif date_filter.get("end"):
        display_frame = display_frame[
            pd.to_datetime(display_frame["Date"]) <= pd.to_datetime(date_filter["end"])
        ].copy()
    return display_frame.reset_index(drop=True)


def _filter_future_points_for_display(future_points, horizon_steps):
    if future_points is None or future_points.empty:
        return future_points
    points = (
        future_points.sort_values("Etape")
        .drop_duplicates(subset=["Etape"], keep="last")
        .copy()
    )
    if int(horizon_steps) <= MIXED_DISPLAY_DAILY_LIMIT:
        return points
    last_step = int(points["Etape"].max())
    weekly_mask = (points["Etape"] % MIXED_DISPLAY_WEEKLY_STEP) == 0
    keep_mask = weekly_mask | (points["Etape"] == last_step)
    return points.loc[keep_mask].copy()


def _future_rows_for_table(ticker, future_points, horizon_label, source):
    if future_points is None or future_points.empty:
        return []
    rows = []
    for _, row in future_points.iterrows():
        rows.append(
            {
                "Actif": ticker,
                "Etape": int(row["Etape"]),
                "Date prevision": pd.to_datetime(row["Date"]).strftime(
                    "%Y-%m-%d"
                ),
                "Prix predit": float(row["Prix predit"]),
                "IC 95% bas": float(row["IC 95% bas"])
                if "IC 95% bas" in row and pd.notna(row["IC 95% bas"])
                else np.nan,
                "IC 95% haut": float(row["IC 95% haut"])
                if "IC 95% haut" in row and pd.notna(row["IC 95% haut"])
                else np.nan,
                "Horizon": horizon_label,
                "Source": source,
            }
        )
    return rows


def _metric_value(metrics, key):
    value = metrics.get(key)
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _format_percent_columns(frame, columns):
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    return output


def _format_number_columns(frame, columns, digits=2, suffix=""):
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).map(
                lambda value: f"{value:.{digits}f}{suffix}"
                if pd.notna(value)
                else ""
            )
    return output


def _prepare_table_display(
    frame,
    rename_map=None,
    order=None,
    percent_columns=None,
    number_columns=None,
):
    output = frame.copy()
    if rename_map:
        output = output.rename(columns=rename_map)
    if percent_columns:
        output = _format_percent_columns(output, percent_columns)
    if number_columns:
        for columns, digits, suffix in number_columns:
            output = _format_number_columns(
                output,
                columns=columns,
                digits=digits,
                suffix=suffix,
            )
    if order:
        ordered = [column for column in order if column in output.columns]
        remaining = [column for column in output.columns if column not in ordered]
        output = output[ordered + remaining]
    return output


def _df_to_csv_bytes(frame):
    if frame is None:
        frame = pd.DataFrame()
    return frame.to_csv(index=False).encode("utf-8")


def _json_to_bytes(payload):
    return json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")


def _render_compact_legend(items):
    parts = [f"`{label}` = {description}" for label, description in items]
    st.caption("Lecture rapide: " + " | ".join(parts))


def _render_legend_expander(items):
    with st.expander("Legende", expanded=False):
        _render_compact_legend(items)


def _render_recommended_settings_expander(section="general"):
    content_map = {
        "predictions": (
            "Recommandations pour generer les signaux sur les actifs US "
            "(Beautiful Seven).",
            "- Modele recommande: `LSTM`\n"
            "- Fenetre recommandee: `120`\n"
            "- Epochs recommandes: `10`\n"
            "- Horizon recommande: `5 jours`\n"
            "- `10 jours` et `21 jours` restent plus instables",
        ),
        "backtest": (
            "Configuration recommandee pour comparer les strategies sans alourdir le run.",
            "- Univers recommande: `Beautiful Seven`\n"
            "- Modele recommande: `LSTM`\n"
            "- Fenetre `120`, epochs `10`, horizon `5 jours`\n"
            "- `2 a 4` periodes pour un run rapide, puis `6+` pour consolider",
        ),
        "summary": (
            "Message simple a retenir pour la synthese et la demo.",
            "- Le setup le plus defendable aujourd'hui est `LSTM + fenetre 120 + 10 epochs + horizon 5 jours`\n"
            "- Le modele sert surtout a comparer des signaux et guider l'allocation\n"
            "- Evitez de vendre `10 jours` et `21 jours` comme settings de reference",
        ),
        "allocation": (
            "Reglages conseilles pour obtenir une allocation lisible et peu concentree.",
            "- Profil risque recommande: `equilibre`\n"
            "- Cadre de risque portefeuille recommande: `court`\n"
            "- Poids max recommande: `35%`\n"
            "- Conservez `0 bps` pour les tests, puis ajoutez des frais pour une lecture plus realiste",
        ),
        "general": (
            "Recommandations actuelles issues des benchmarks menes sur les actifs US "
            "(Beautiful Seven).",
            "- Modele recommande: `LSTM`\n"
            "- Fenetre recommandee: `120`\n"
            "- Epochs recommandes: `10`\n"
            "- Horizon recommande: `5 jours`\n"
            "- `10 jours` et `21 jours` restent exploratoires et moins robustes",
        ),
    }
    caption_text, body_text = content_map.get(section, content_map["general"])
    with st.expander("Parametres recommandes", expanded=False):
        st.caption(caption_text)
        st.write(body_text)


def _render_precise_nav_chart(nav_frame):
    if nav_frame is None or nav_frame.empty:
        st.info("Aucune serie disponible.")
        return

    chart_frame = nav_frame.copy()
    chart_frame["Date"] = pd.to_datetime(chart_frame["Date"])
    value_columns = [column for column in chart_frame.columns if column != "Date"]
    for column in value_columns:
        chart_frame[column] = pd.to_numeric(chart_frame[column], errors="coerce")

    value_array = chart_frame[value_columns].to_numpy(dtype=float)
    finite_values = value_array[np.isfinite(value_array)]
    if finite_values.size == 0:
        st.line_chart(chart_frame.set_index("Date"), width="stretch")
        return

    y_min = float(finite_values.min())
    y_max = float(finite_values.max())
    spread = y_max - y_min
    padding = max(spread * 0.12, max(abs(y_max), 1.0) * 0.002)
    y_domain = [y_min - padding, y_max + padding]
    axis_format = ".4f" if spread < 0.05 else ".3f"

    melted = chart_frame.melt(
        id_vars="Date",
        value_vars=value_columns,
        var_name="Strategie",
        value_name="Valeur",
    ).dropna(subset=["Valeur"])

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Valeur:Q",
                title="Valeur",
                scale=alt.Scale(domain=y_domain, nice=False, zero=False),
                axis=alt.Axis(format=axis_format),
            ),
            color=alt.Color("Strategie:N", title=None),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Strategie:N", title="Strategie"),
                alt.Tooltip("Valeur:Q", title="Valeur", format=axis_format),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def _render_period_return_chart(period_returns_frame, benchmark_label):
    if period_returns_frame is None or period_returns_frame.empty:
        st.info("Aucune periode disponible.")
        return

    base_columns = ["Modele", "Naive", "Equal Weight"]
    if benchmark_label in period_returns_frame.columns:
        base_columns.append(benchmark_label)

    available_columns = [
        column for column in base_columns if column in period_returns_frame.columns
    ]
    if not available_columns:
        st.info("Aucune serie de rendement disponible.")
        return

    chart_frame = period_returns_frame[["Date realisee", *available_columns]].copy()
    chart_frame["Date realisee"] = pd.to_datetime(chart_frame["Date realisee"])
    melted = chart_frame.melt(
        id_vars="Date realisee",
        value_vars=available_columns,
        var_name="Strategie",
        value_name="Rendement",
    ).dropna(subset=["Rendement"])

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Date realisee:T", title="Date realisee"),
            y=alt.Y("Rendement:Q", title="Rendement par periode", axis=alt.Axis(format=".1%")),
            color=alt.Color("Strategie:N", title="Strategie"),
            xOffset="Strategie:N",
            tooltip=[
                alt.Tooltip("Date realisee:T", title="Date"),
                alt.Tooltip("Strategie:N", title="Strategie"),
                alt.Tooltip("Rendement:Q", title="Rendement", format=".2%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _render_table_help(help_key, title="Definitions des colonnes"):
    items = TABLE_COLUMN_HELP.get(help_key, [])
    if not items:
        return
    with st.expander(title):
        for label, description in items:
            st.markdown(f"`{label}` : {description}")


def _strategy_metric_lookup(strategy_metrics):
    if strategy_metrics is None or strategy_metrics.empty:
        return {}
    return strategy_metrics.set_index("Strategie").to_dict(orient="index")


def _latest_weights_for_strategy(allocation_history, strategy_name):
    latest_rows = _latest_allocation_rows(allocation_history, strategy_name)
    if latest_rows.empty:
        return pd.Series(dtype=float)
    return _parse_weights_text(latest_rows.iloc[-1]["Poids"])


def _build_executive_summary_payload(results, meta, asset_count):
    strategy_metrics = results.get("strategy_metrics", pd.DataFrame())
    metric_lookup = _strategy_metric_lookup(strategy_metrics)
    model_metrics = metric_lookup.get("Modele", {})
    benchmark_label = results.get("benchmark_label", meta.get("benchmark_label"))

    ranking = strategy_metrics.copy()
    if not ranking.empty:
        ranking = ranking.sort_values(
            ["Rendement annualise", "Sharpe"],
            ascending=[False, False],
        ).reset_index(drop=True)
    best_strategy = (
        ranking.iloc[0]["Strategie"] if not ranking.empty else "Indisponible"
    )

    comparisons = ["Naive", "Equal Weight"]
    if benchmark_label and benchmark_label in metric_lookup:
        comparisons.append(benchmark_label)

    model_return = _safe_float(model_metrics.get("Rendement annualise"))
    comparison_results = []
    for strategy_name in comparisons:
        other_return = _safe_float(
            metric_lookup.get(strategy_name, {}).get("Rendement annualise")
        )
        if pd.isna(model_return) or pd.isna(other_return):
            verdict = None
        else:
            verdict = bool(model_return > other_return)
        comparison_results.append(
            {
                "Strategie": strategy_name,
                "Modele bat": verdict,
                "Rendement annualise": other_return,
            }
        )

    valid_verdicts = [
        item["Modele bat"]
        for item in comparison_results
        if item["Modele bat"] is not None
    ]
    model_beats_count = int(sum(valid_verdicts))
    comparison_count = int(len(valid_verdicts))
    available_reference_returns = [
        item
        for item in comparison_results
        if pd.notna(item["Rendement annualise"])
    ]
    best_reference = None
    if available_reference_returns:
        best_reference = max(
            available_reference_returns,
            key=lambda item: float(item["Rendement annualise"]),
        )
    best_reference_name = (
        best_reference["Strategie"] if best_reference is not None else "Indisponible"
    )
    best_reference_return = (
        _safe_float(best_reference["Rendement annualise"])
        if best_reference is not None
        else np.nan
    )
    if pd.notna(model_return) and pd.notna(best_reference_return):
        return_gap_vs_best_reference = model_return - best_reference_return
    else:
        return_gap_vs_best_reference = np.nan

    forecast_summary = results.get("forecast_summary", pd.DataFrame())
    beat_naive = (
        _safe_float(forecast_summary["beat_naive"].mean())
        if "beat_naive" in forecast_summary
        else np.nan
    )

    allocation_history = results.get("allocation_history", pd.DataFrame())
    latest_model_weights = _latest_weights_for_strategy(
        allocation_history, "Modele"
    )
    top_weights = (
        ", ".join(
            f"{ticker} {weight:.0%}"
            for ticker, weight in latest_model_weights.head(3).items()
        )
        if not latest_model_weights.empty
        else "Indisponible"
    )

    verdict_level = "warning"
    verdict_label = "Sous-performance"
    if comparison_count > 0 and model_beats_count == comparison_count:
        verdict_level = "success"
        verdict_label = "Surperformance"
    elif model_beats_count > 0:
        verdict_level = "info"
        verdict_label = "Mitige"

    comparison_text = (
        f"Le modele bat {model_beats_count} reference(s) sur {comparison_count} "
        f"en rendement annualise."
        if comparison_count > 0
        else "Comparaison references indisponible."
    )
    period_count = int(meta.get("num_periods", 0))
    model_turnover = _safe_float(model_metrics.get("Turnover moyen"))
    max_weight = (
        float(latest_model_weights.max()) if not latest_model_weights.empty else np.nan
    )
    if best_strategy == "Modele" and period_count >= 6:
        headline = "Le modele domine les references sur un backtest deja plus consistant."
    elif best_strategy == "Modele":
        headline = "Le modele est en tete, mais le backtest reste court."
    else:
        headline = f"La meilleure strategie actuelle est {best_strategy}."

    if best_strategy == "Modele" and pd.notna(model_turnover) and model_turnover >= 0.30:
        comparison_text += " La surperformance reste sensible aux frais car le turnover est eleve."
    if pd.notna(max_weight) and max_weight >= 0.45:
        comparison_text += " Le portefeuille reste concentre sur peu de lignes."
    if period_count < 6:
        comparison_text += " Les ratios annualises doivent etre lus avec prudence."

    return {
        "headline": headline,
        "comparison_text": comparison_text,
        "verdict_level": verdict_level,
        "verdict_label": verdict_label,
        "best_strategy": best_strategy,
        "asset_count": int(asset_count),
        "period_count": period_count,
        "model_beats_count": model_beats_count,
        "comparison_count": comparison_count,
        "model_cumulative_return": _safe_float(model_metrics.get("Rendement cumule")),
        "model_annualized_return": model_return,
        "model_sharpe": _safe_float(model_metrics.get("Sharpe")),
        "model_max_drawdown": _safe_float(model_metrics.get("Max drawdown")),
        "model_var_95": _safe_float(model_metrics.get("VaR 95%")),
        "model_cvar_95": _safe_float(model_metrics.get("CVaR 95%")),
        "model_calmar": _safe_float(model_metrics.get("Calmar")),
        "model_turnover": _safe_float(model_metrics.get("Turnover moyen")),
        "best_reference_name": best_reference_name,
        "best_reference_return": best_reference_return,
        "return_gap_vs_best_reference": return_gap_vs_best_reference,
        "beat_naive_rate": beat_naive,
        "top_weights": top_weights,
        "comparison_results": pd.DataFrame(comparison_results),
        "ranking": ranking,
    }


def _format_strategy_comparison_value(value, kind):
    if pd.isna(value):
        return "n/a"
    if kind == "pct":
        return f"{value:.2%}"
    return f"{value:.2f}"


def _build_strategy_comparison_frame(metric_lookup, strategy_a, strategy_b):
    metric_specs = [
        ("Rendement cumule", "pct"),
        ("Rendement annualise", "pct"),
        ("Volatilite annualisee", "pct"),
        ("Sharpe", "num"),
        ("Max drawdown", "pct"),
        ("VaR 95%", "pct"),
        ("CVaR 95%", "pct"),
        ("Calmar", "num"),
        ("Turnover moyen", "pct"),
    ]
    rows = []
    strategy_a_metrics = metric_lookup.get(strategy_a, {})
    strategy_b_metrics = metric_lookup.get(strategy_b, {})
    for metric_name, kind in metric_specs:
        value_a = _safe_float(strategy_a_metrics.get(metric_name))
        value_b = _safe_float(strategy_b_metrics.get(metric_name))
        delta = value_a - value_b if pd.notna(value_a) and pd.notna(value_b) else np.nan
        rows.append(
            {
                "Metrique": metric_name,
                strategy_a: _format_strategy_comparison_value(value_a, kind),
                strategy_b: _format_strategy_comparison_value(value_b, kind),
                "Ecart (A-B)": _format_strategy_comparison_value(delta, kind),
            }
        )
    return pd.DataFrame(rows)


def _parse_weights_text(weights_text):
    if not weights_text:
        return pd.Series(dtype=float)
    entries = {}
    for part in str(weights_text).split(","):
        if "=" not in part:
            continue
        asset, value = part.split("=", 1)
        asset = asset.strip()
        try:
            entries[asset] = float(value.strip())
        except Exception:
            continue
    if not entries:
        return pd.Series(dtype=float)
    return pd.Series(entries, dtype=float).sort_values(ascending=False)


def _effective_position_count(weights):
    if weights.empty:
        return 0.0
    denom = float((weights**2).sum())
    if denom <= 1e-12:
        return 0.0
    return float(1.0 / denom)


def _top_weight_share(weights, top_n=3):
    if weights.empty:
        return 0.0
    return float(weights.sort_values(ascending=False).head(int(top_n)).sum())


def _concentration_hhi(weights):
    if weights.empty:
        return 0.0
    return float((weights**2).sum())


def _concentration_label(weights):
    if weights.empty:
        return "n/a"
    max_weight = float(weights.max())
    top3_share = _top_weight_share(weights, top_n=3)
    effective_positions = _effective_position_count(weights)
    hhi = _concentration_hhi(weights)
    if (
        max_weight >= 0.45
        or top3_share >= 0.80
        or effective_positions < 2.5
        or hhi >= 0.30
    ):
        return "Elevee"
    if (
        max_weight >= 0.35
        or top3_share >= 0.65
        or effective_positions < 4.0
        or hhi >= 0.20
    ):
        return "Moyenne"
    return "Faible"


def _latest_allocation_rows(allocation_history, strategy_name):
    if allocation_history is None or allocation_history.empty:
        return pd.DataFrame()
    rows = allocation_history[
        allocation_history["Strategie"] == strategy_name
    ].copy()
    if rows.empty:
        return rows
    rows["Date de rebalance"] = pd.to_datetime(rows["Date de rebalance"])
    latest_date = rows["Date de rebalance"].max()
    return rows[rows["Date de rebalance"] == latest_date].copy()


def _load_trailing_returns_for_assets(state, tickers, cutoff_date, history_lookback):
    frames = []
    for ticker in tickers:
        frame = load_asset_frame(
            ticker,
            state["start_date"],
            state["end_date"],
            get_asset_cache_token(ticker),
        )
        frames.append(
            frame.set_index("Date")
            .rename(columns={"Close": ticker})
            .sort_index()
        )
    if not frames:
        return pd.DataFrame()
    price_matrix = pd.concat(frames, axis=1).dropna().sort_index()
    cutoff_ts = pd.to_datetime(cutoff_date)
    trailing_returns = (
        price_matrix.loc[:cutoff_ts]
        .pct_change()
        .dropna()
        .tail(int(history_lookback))
    )
    return trailing_returns


def _compute_risk_contribution(weights, trailing_returns):
    if weights.empty or trailing_returns.empty:
        return pd.Series(dtype=float)
    aligned_returns = trailing_returns[weights.index].dropna()
    if aligned_returns.shape[0] < 5 or aligned_returns.shape[1] < 2:
        return pd.Series(dtype=float)
    cov = estimate_covariance(aligned_returns, method="ledoit_wolf")
    w = weights.to_numpy(dtype=float)
    portfolio_var = float(w.T @ cov @ w)
    if portfolio_var <= 1e-12:
        return pd.Series(dtype=float)
    marginal = cov @ w
    contribution = (w * marginal) / portfolio_var
    return pd.Series(contribution, index=weights.index, dtype=float)


def _build_result_row(ticker, metrics, points_count, effective_window, source):
    version = int(metrics.get("metrics_version", 1))
    if version >= 2:
        mae_price = _metric_value(metrics, "mae_price")
        rmse_price = _metric_value(metrics, "rmse_price")
        mape_pct = _metric_value(metrics, "mape_pct")
        mae_norm = _metric_value(metrics, "mae_norm")
    else:
        # Legacy cache: only normalized metrics available.
        mae_price = np.nan
        rmse_price = np.nan
        mape_pct = np.nan
        mae_norm = _metric_value(metrics, "mae")
    return {
        "Actif": ticker,
        "MAE prix": mae_price,
        "RMSE prix": rmse_price,
        "MAPE %": mape_pct,
        "MAE norm": mae_norm,
        "Points utilises": int(points_count),
        "Fenetre utilisee": int(effective_window),
        "Source": source,
    }


def _backtest_run_quality(results, meta):
    period_count = int(meta.get("num_periods", 0))
    allocation_history = results.get("allocation_history", pd.DataFrame())
    asset_details = results.get("asset_details", pd.DataFrame())
    source_counts = {}
    if not asset_details.empty and "Source training" in asset_details.columns:
        source_counts = (
            asset_details["Source training"].value_counts(dropna=False).to_dict()
        )

    latest_model_weights = _latest_weights_for_strategy(allocation_history, "Modele")
    max_weight = float(latest_model_weights.max()) if not latest_model_weights.empty else np.nan
    effective_positions = (
        _effective_position_count(latest_model_weights)
        if not latest_model_weights.empty
        else np.nan
    )
    top3_share = (
        _top_weight_share(latest_model_weights, top_n=3)
        if not latest_model_weights.empty
        else np.nan
    )
    concentration_hhi = (
        _concentration_hhi(latest_model_weights)
        if not latest_model_weights.empty
        else np.nan
    )
    concentration_label = (
        _concentration_label(latest_model_weights)
        if not latest_model_weights.empty
        else "n/a"
    )
    metric_lookup = _strategy_metric_lookup(results.get("strategy_metrics", pd.DataFrame()))
    model_metrics = metric_lookup.get("Modele", {})
    turnover = _safe_float(model_metrics.get("Turnover moyen"))

    warnings = []
    if period_count < 6:
        warnings.append(
            "Run fragile: moins de 6 periodes, les metriques annualisees restent volatiles."
        )
    if pd.notna(turnover) and turnover >= 0.30:
        warnings.append(
            "Turnover eleve: les performances sont plus sensibles aux frais de transaction."
        )
    if pd.notna(max_weight) and max_weight >= 0.45:
        warnings.append(
            "Portefeuille concentre: une position porte une part importante du risque."
        )
    elif pd.notna(top3_share) and top3_share >= 0.75:
        warnings.append(
            "Portefeuille concentre: les trois premieres positions portent l'essentiel de l'allocation."
        )
    elif pd.notna(effective_positions) and effective_positions < 3:
        warnings.append(
            "Diversification limitee: le nombre effectif de positions reste faible."
        )

    return {
        "period_count": period_count,
        "source_counts": source_counts,
        "max_weight": max_weight,
        "effective_positions": effective_positions,
        "top3_share": top3_share,
        "concentration_hhi": concentration_hhi,
        "concentration_label": concentration_label,
        "turnover": turnover,
        "warnings": warnings,
    }


def _select_plot_frame_for_mode(plot_frame, mode_label):
    if plot_frame is None or plot_frame.empty:
        return None
    if mode_label == "Backtest":
        filtered = plot_frame[plot_frame["Reel"].notna()].copy()
    elif mode_label == "Futur":
        filtered = plot_frame[plot_frame["Reel"].isna()].copy()
    else:
        filtered = plot_frame.copy()
    if filtered.empty:
        return None
    return filtered


def _build_signal_chart_data(plot_frame, ticker, split_future_projection=False):
    if plot_frame is None or plot_frame.empty:
        return None
    chart_frame = plot_frame.copy()
    chart_columns = []
    rename_map = {}

    series_to_render = [
        ("IC 95% bas", f"{ticker} IC 95% bas"),
        ("IC 95% haut", f"{ticker} IC 95% haut"),
        ("Predit", f"{ticker} predit"),
        ("Reel", f"{ticker} reel"),
    ]

    for column, label in series_to_render:
        if column in chart_frame.columns and chart_frame[column].notna().any():
            chart_columns.append(column)
            rename_map[column] = label
    if not chart_columns:
        return None
    return chart_frame.set_index("Date")[chart_columns].rename(columns=rename_map)


def _apply_backtest_alignment(plot_frame, alignment_mode):
    if plot_frame is None or plot_frame.empty:
        return plot_frame
    if alignment_mode != "Date de decision":
        return plot_frame
    if "Date decision" not in plot_frame.columns:
        return plot_frame
    parts = []

    if "Reel" in plot_frame.columns and plot_frame["Reel"].notna().any():
        real_date_col = "Date cible" if "Date cible" in plot_frame.columns else "Date"
        real_part = (
            plot_frame.loc[plot_frame["Reel"].notna(), [real_date_col, "Reel"]]
            .rename(columns={real_date_col: "Date"})
            .dropna(subset=["Date"])
            .groupby("Date", as_index=False)
            .last()
        )
        parts.append(real_part)

    pred_columns = ["Predit"]
    if "IC 95% bas" in plot_frame.columns:
        pred_columns.append("IC 95% bas")
    if "IC 95% haut" in plot_frame.columns:
        pred_columns.append("IC 95% haut")
    if any(
        column in plot_frame.columns and plot_frame[column].notna().any()
        for column in pred_columns
    ):
        pred_part = plot_frame[["Date", "Date decision", *pred_columns]].copy()
        pred_part["Date"] = pred_part["Date decision"].where(
            pred_part["Date decision"].notna(), pred_part["Date"]
        )
        pred_part = (
            pred_part.drop(columns=["Date decision"])
            .dropna(subset=["Date"])
            .groupby("Date", as_index=False)
            .last()
        )
        parts.append(pred_part)

    if not parts:
        return plot_frame

    aligned = parts[0]
    for part in parts[1:]:
        aligned = aligned.merge(part, on="Date", how="outer")
    return aligned.sort_values("Date")


def train_selected_assets(state, cache_only=False):
    model_code = MODEL_TYPE_MAP[state["model_type"]]
    use_us_multifeature = model_code == "lstm_multifeature"
    requested_window_size = int(state["window_size"])
    force_retrain = bool(state.get("force_retrain", False))
    horizon_steps = int(state.get("horizon_steps", 1))
    base_config = {
        "epochs": int(state["epochs"]),
        "verbose": 0,
        "horizon": horizon_steps,
        "test_size": DEFAULT_TEST_SIZE,
        "val_size": DEFAULT_VAL_SIZE,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "lstm_units1": DEFAULT_LSTM_UNITS1,
        "lstm_units2": DEFAULT_LSTM_UNITS2,
        "lstm_dropout": DEFAULT_LSTM_DROPOUT,
    }
    required_windows = _required_min_windows(
        base_config["test_size"], base_config["val_size"]
    )

    results = []
    failures = []
    prediction_plots = {}
    future_forecasts = []
    total = len(state["assets"])
    progress = st.progress(0)
    status_line = st.empty()
    legacy_path_assets = []
    factor_histories = None

    if use_us_multifeature:
        factor_cache_token = tuple(
            get_asset_cache_token(ticker) for ticker in US_FACTOR_TICKERS.values()
        )
        factor_histories = load_us_factor_histories(
            state["start_date"],
            state["end_date"],
            factor_cache_token,
        )

    for index, ticker in enumerate(state["assets"], start=1):
        status_prefix = f"{ticker} ({index}/{total})"
        try:
            status_line.caption(f"{status_prefix} - chargement")
            if use_us_multifeature:
                frame = load_asset_ohlcv_frame(
                    ticker,
                    state["start_date"],
                    state["end_date"],
                    get_asset_cache_token(ticker),
                )
            else:
                frame = load_asset_frame(
                    ticker,
                    state["start_date"],
                    state["end_date"],
                    get_asset_cache_token(ticker),
                )
            effective_window = _effective_window_size(
                total_points=len(frame),
                requested_window=requested_window_size,
                horizon=base_config["horizon"],
                required_windows=required_windows,
                extra_points_needed=1 if model_code == "lstm" else 0,
            )
            if effective_window is None:
                required_points = (
                    MIN_WINDOW_SIZE
                    + base_config["horizon"]
                    + (required_windows - 1)
                    + (1 if model_code == "lstm" else 0)
                )
                failures.append(
                    {
                        "ticker": ticker,
                        "error": (
                            "Pas assez de points pour entrainer. "
                            f"Points={len(frame)}, requis>={required_points}."
                        ),
                    }
                )
                continue

            asset_config = {
                **base_config,
                "window_size": int(effective_window),
            }
            training_frame = build_training_frame_for_model(
                model_code=model_code,
                ticker=ticker,
                asset_frame=frame,
                factor_histories=factor_histories,
            )
            if cache_only:
                cache_result = peek_training_cache(
                    ticker=ticker,
                    model_code=model_code,
                    config=asset_config,
                    training_frame=training_frame,
                    required_artifacts=("model", "scaler", "eval", "forecast"),
                )
                if cache_result is None:
                    cache_result = load_latest_training_cache(
                        ticker=ticker,
                        model_code=model_code,
                        config=asset_config,
                        required_artifacts=("model", "scaler", "eval", "forecast"),
                    )
                if cache_result is None:
                    failures.append(
                        {
                            "ticker": ticker,
                            "error": (
                                "Aucun cache correspondant a la configuration courante "
                                "ou au dernier entrainement compatible."
                            ),
                        }
                    )
                    continue
            else:
                cache_result = resolve_training_cache(
                    ticker=ticker,
                    model_code=model_code,
                    config=asset_config,
                    training_frame=training_frame,
                    force_retrain=force_retrain,
                    start_date=state["start_date"],
                    end_date=state["end_date"],
                    required_artifacts=("model", "scaler", "eval", "forecast"),
                )
            artifact_paths = cache_result["artifact_paths"]
            metrics = cache_result["metrics"]
            source = cache_result["source"]
            metadata = cache_result.get("metadata") or {}
            status_line.caption(f"{status_prefix} - {source.lower()}")
            artifact_frame = _frame_for_cached_artifacts(frame, metadata)
            trained_plot_frame = _load_eval_frame(
                artifact_paths["eval"], artifact_frame, int(asset_config["horizon"])
            )
            if model_code == "lstm":
                status_line.caption(f"{status_prefix} - chargement forecast path")
                future_points = _load_forecast_path_frame(
                    artifact_paths["forecast_path"],
                    artifact_frame,
                    int(asset_config["horizon"]),
                )
                if future_points is None or future_points.empty:
                    forecast_artifact = _load_forecast_artifact(
                        artifact_paths["forecast"]
                    )
                    future_points = _build_single_step_forecast_frame(
                        frame=artifact_frame,
                        horizon_steps=int(asset_config["horizon"]),
                        forecast_artifact=forecast_artifact,
                    )
                    if (
                        source == "Cache"
                        and int(asset_config["horizon"]) > 1
                        and artifact_paths["forecast"].exists()
                        and not artifact_paths["forecast_path"].exists()
                    ):
                        legacy_path_assets.append(ticker)
                future_points = _filter_future_points_for_display(
                    future_points, int(asset_config["horizon"])
                )
                if future_points is not None and not future_points.empty:
                    trained_plot_frame = _append_future_prediction_path(
                        trained_plot_frame, future_points
                    )
                    future_forecasts.extend(
                        _future_rows_for_table(
                            ticker=ticker,
                            future_points=future_points,
                            horizon_label=state.get("horizon_label", "1 jour"),
                            source=source,
                        )
                    )
            if trained_plot_frame is not None and not trained_plot_frame.empty:
                prediction_plots[ticker] = trained_plot_frame
            results.append(
                _build_result_row(
                    ticker=ticker,
                    metrics=metrics,
                    points_count=len(frame),
                    effective_window=effective_window,
                    source=source,
                )
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
        finally:
            progress.progress(index / total)

    progress.empty()
    status_line.empty()
    if legacy_path_assets:
        st.caption(
            "Forecast path indisponible pour certains caches legacy: "
            + ", ".join(sorted(set(legacy_path_assets)))
        )
    return results, failures, prediction_plots, future_forecasts


def render_training_section(state):
    st.markdown("### Generation du signal")
    requested_window_size = int(state["window_size"])
    horizon_label = state.get("horizon_label", "1 jour")
    horizon_steps = int(state.get("horizon_steps", 1))
    st.caption(
        "Construit un signal quantitatif sur les actifs selectionnes a partir "
        "du modele, de la fenetre, du nombre d'epochs et de l'horizon choisis. "
        "Les runs sont mis en cache par actif, configuration et historique utilise. "
        "Le backtest peut les reutiliser quand la date de cutoff correspond."
    )
    st.caption(
        "Vous pouvez aussi recharger le dernier cache correspondant a la configuration courante, sans relancer l'entrainement."
    )
    asset_count = len(state.get("assets", []))
    st.caption(f"Univers analyse: {asset_count} actif(s)")
    if asset_count >= 15:
        st.warning(
            "Beaucoup d'actifs selectionnes: le run peut etre long. "
            "Reduisez la selection ou les epochs pour un test rapide."
        )

    action_cols = st.columns(2)
    load_cached_clicked = action_cols[0].button(
        "Afficher le cache precedent",
        width="stretch",
        help="Recharge uniquement les resultats deja en cache pour la configuration courante, sans relancer l'entrainement.",
    )
    generate_signals_clicked = action_cols[1].button(
        "Generer les signaux",
        width="stretch",
        help="Entraine le modele sur les actifs coches avec les parametres de la sidebar.",
    )

    if load_cached_clicked or generate_signals_clicked:
        if not state["assets"]:
            st.info("Aucun actif selectionne.")
        else:
            spinner_label = (
                "Chargement du cache en cours..."
                if load_cached_clicked
                else "Entrainement en cours..."
            )
            with st.spinner(spinner_label):
                (
                    results,
                    failures,
                    prediction_plots,
                    future_forecasts,
                ) = train_selected_assets(
                    state,
                    cache_only=bool(load_cached_clicked),
                )
            st.session_state.training_results = results
            st.session_state.training_failures = failures
            st.session_state.training_prediction_plots = prediction_plots
            st.session_state.training_future_forecasts = future_forecasts
            st.session_state.training_meta = {
                "model_type": state["model_type"],
                "window_size": requested_window_size,
                "epochs": int(state["epochs"]),
                "horizon_label": horizon_label,
                "horizon_steps": horizon_steps,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result_source": "Cache" if load_cached_clicked else "Train/Cache",
            }

    results = st.session_state.get("training_results", [])
    failures = st.session_state.get("training_failures", [])
    prediction_plots = st.session_state.get("training_prediction_plots", {})
    future_forecasts = st.session_state.get("training_future_forecasts", [])
    meta = st.session_state.get("training_meta")

    if meta:
        st.caption(
            f"Derniere generation ({meta['timestamp']}) | "
            f"Modele: {meta['model_type']} | "
            f"Fenetre: {meta['window_size']} | "
            f"Horizon: {meta.get('horizon_label', '1 jour')} ({meta.get('horizon_steps', 1)} pas) | "
            f"Epochs: {meta['epochs']} | "
            f"Source: {meta.get('result_source', 'Train/Cache')}"
        )

    if results:
        df_results = pd.DataFrame(results).sort_values("Actif")
        df_results_display = _prepare_table_display(
            df_results,
            order=[
                "Actif",
                "MAE prix",
                "RMSE prix",
                "MAPE %",
                "MAE norm",
                "Points utilises",
                "Fenetre utilisee",
                "Source",
            ],
            number_columns=[
                (["MAE prix", "RMSE prix"], 2, ""),
                (["MAPE %"], 1, "%"),
                (["MAE norm"], 4, ""),
            ],
        )
        st.dataframe(df_results_display, width="stretch", hide_index=True)
        _render_table_help("training_results")
        if df_results["MAE prix"].isna().any():
            st.caption(
                "Certaines lignes proviennent d'un cache legacy: "
                "relancez la generation pour obtenir les metriques en prix."
            )
    if failures:
        with st.expander("Incidents de generation"):
            for item in failures:
                st.write(f"{item['ticker']}: {item['error']}")
    if prediction_plots:
        st.markdown("#### Lecture du signal")
        plot_mode = st.radio(
            "Affichage des courbes",
            options=["Mixte", "Backtest", "Futur"],
            horizontal=True,
            key="training_plot_mode",
            help=(
                "Mixte = backtest + futur, Backtest = historique test seulement, "
                "Futur = projection future seulement."
            ),
        )
        backtest_alignment = st.radio(
            "Alignement du backtest",
            options=["Date cible", "Date de decision"],
            horizontal=True,
            key="training_plot_alignment",
            help=(
                "Date cible = aligne la prediction sur la date qu'elle vise. "
                "Date de decision = aligne la prediction sur la date ou elle est emise."
            ),
        )
        for ticker in sorted(prediction_plots.keys()):
            plot_frame = _select_plot_frame_for_mode(
                prediction_plots[ticker], plot_mode
            )
            if plot_frame is None or plot_frame.empty:
                continue
            with st.expander(f"{ticker} - reel vs predit", expanded=False):
                display_frame = plot_frame
                if plot_mode != "Futur":
                    display_frame = _apply_backtest_alignment(
                        display_frame, backtest_alignment
                    )
                if (
                    plot_mode == "Mixte"
                    and "Reel" in display_frame.columns
                    and "Predit" in display_frame.columns
                ):
                    backtest_mask = display_frame["Reel"].notna()
                    future_mask = display_frame["Reel"].isna() & display_frame[
                        "Predit"
                    ].notna()
                    if backtest_mask.any() and future_mask.any():
                        backtest_end = pd.to_datetime(
                            display_frame.loc[backtest_mask, "Date"].max()
                        )
                        future_start = pd.to_datetime(
                            display_frame.loc[future_mask, "Date"].min()
                        )
                        st.caption(
                            "Frontiere backtest / futur : "
                            f"dernier point backtest = {backtest_end.strftime('%Y-%m-%d')} | "
                            f"premiere projection future = {future_start.strftime('%Y-%m-%d')}"
                        )
                chart_data = _build_signal_chart_data(
                    display_frame,
                    ticker,
                    split_future_projection=False,
                )
                if chart_data is not None:
                    st.line_chart(chart_data, width="stretch")
    if future_forecasts:
        st.markdown("#### Projections futures")
        st.caption(
            "IC 95%: borne basse/haute estimee a partir de l'erreur hors echantillon du modele."
        )
        forecast_df = pd.DataFrame(future_forecasts).sort_values(
            ["Actif", "Etape"]
        )
        forecast_df_display = _prepare_table_display(
            forecast_df,
            order=[
                "Actif",
                "Etape",
                "Date prevision",
                "Prix predit",
                "IC 95% bas",
                "IC 95% haut",
                "Horizon",
                "Source",
            ],
            number_columns=[
                (["Prix predit", "IC 95% bas", "IC 95% haut"], 2, "")
            ],
        )
        st.dataframe(forecast_df_display, width="stretch", hide_index=True)
        _render_table_help("future_forecasts")


def render_sidebar():
    with st.sidebar:
        st.markdown("## Cockpit\nQuantitatif")
        st.caption("Analyse multi-actifs, backtests et allocation")
        st.markdown("---")

        st.markdown("### Univers d'investissement")
        base_tickers = load_base_tickers()
        base_ticker_set = set(base_tickers)
        known_assets = [format_base_ticker(ticker) for ticker in base_tickers]
        if DATA_DIR.exists():
            for path in DATA_DIR.glob("*.csv"):
                known_assets.append(normalize_ticker(path.stem, base_ticker_set))
        known_assets = sorted(set(filter(None, known_assets)))
        category_assets = build_category_assets(known_assets)
        available_categories = [
            name
            for name, assets in category_assets.items()
            if assets
        ]
        default_categories = [
            name
            for name in ["Beautiful Seven (US)"]
            if name in available_categories
        ]
        selected_categories = st.multiselect(
            "Categories d'actions",
            options=available_categories,
            default=default_categories,
            key="selected_asset_categories",
            help=HELP_TEXT["asset_categories"],
            placeholder="Choisissez une ou plusieurs categories",
        )

        if "assets_initialized" not in st.session_state:
            initial_selected = set(category_assets.get("Beautiful Seven (US)", []))
            st.session_state.assets_initialized = True
        else:
            initial_selected = set()

        for category_name in selected_categories:
            category_tickers = category_assets[category_name]
            if not category_tickers:
                continue
            with st.expander(
                f"{category_name} ({len(category_tickers)} actifs)",
                expanded=False,
            ):
                category_key = normalize_category_name(category_name)
                check_col, uncheck_col = st.columns(2)
                if check_col.button(
                    "Cocher toute la categorie",
                    key=f"check_category_{category_key}",
                    width="stretch",
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = True
                if uncheck_col.button(
                    "Decocher toute la categorie",
                    key=f"uncheck_category_{category_key}",
                    width="stretch",
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = False

        visible_assets = sorted(
            {
                asset
                for category_name in selected_categories
                for asset in category_assets[category_name]
            }
        )

        if visible_assets:
            select_col, clear_col = st.columns(2)
            if select_col.button(
                "Cocher la selection", width="stretch"
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = True
            if clear_col.button(
                "Decocher la selection", width="stretch"
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = False
        else:
            st.info("Choisissez au moins une categorie d'actions.")

        selected_assets = []
        for asset in visible_assets:
            key = f"asset_check_{asset}"
            if key not in st.session_state:
                st.session_state[key] = asset in initial_selected
            if st.checkbox(asset, key=key):
                selected_assets.append(asset)

        stale_keys = [
            key
            for key in st.session_state
            if key.startswith("asset_check_")
            and key[len("asset_check_") :] not in known_assets
        ]
        for key in stale_keys:
            st.session_state.pop(key, None)

        if not selected_assets:
            st.write("Aucun actif selectionne")

        st.markdown("### Date de debut")
        start_date = st.date_input(
            "Date de debut",
            value=None,
            help=HELP_TEXT["start_date"],
        )

        st.markdown("### Date de fin")
        end_date = st.date_input(
            "Date de fin",
            value=None,
            help=HELP_TEXT["end_date"],
        )

        st.markdown("### Parametres du modele")
        window_size = st.number_input(
            "Taille de la fenetre",
            min_value=5,
            max_value=365,
            value=120,
            step=1,
            help=HELP_TEXT["window_size"],
        )
        epochs = st.number_input(
            "Nombre d'epochs",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            help=HELP_TEXT["epochs"],
        )
        model_type = st.selectbox(
            "Modele",
            options=["LSTM", "LSTM multi-features", "CNN", "CNN puis LSTM"],
            index=0,
            help=HELP_TEXT["model_type"],
        )
        horizon_options = list(HORIZON_OPTIONS.keys())
        horizon_label = st.selectbox(
            "Horizon",
            options=horizon_options,
            index=horizon_options.index("5 jours"),
            help=HELP_TEXT["horizon"],
        )
        horizon_steps = HORIZON_OPTIONS[horizon_label]
        force_retrain = st.checkbox(
            "Forcer le re-entrainement",
            value=False,
            help=HELP_TEXT["force_retrain"],
        )
        if "generate_curves" not in st.session_state:
            st.session_state.generate_curves = False
        if st.button(
            "Charger le marche",
            width="stretch",
            help=HELP_TEXT["generate_curves"],
        ):
            st.session_state.generate_curves = True

        st.markdown("---")
        st.markdown("### Synchronisation des donnees")
        today = datetime.now().strftime("%Y-%m-%d")
        update_key = f"data_updated_{today}"
        force_update = st.checkbox("Ecraser les donnees existantes", value=False)
        if st.button(
            "Rafraichir les historiques",
            width="stretch",
            help=HELP_TEXT["update_assets"],
        ):
            if not selected_assets:
                st.info("Aucun actif selectionne.")
            else:
                progress_bar = st.progress(0)
                status_line = st.empty()

                def on_progress(index, total, ticker, status, error=None):
                    if total:
                        progress_bar.progress(index / total)
                    message = f"{status.upper()}: {ticker}"
                    if error:
                        message = f"{message} - {error}"
                    status_line.caption(message)

                with st.spinner("Mise a jour en cours..."):
                    result = update_assets(
                        selected_assets,
                        start=start_date,
                        end=end_date,
                        force=force_update,
                        progress=on_progress,
                    )
                load_asset_frame.clear()
                st.session_state[update_key] = True
                st.success("Mise a jour terminee.")
                st.write(
                    f"Mis a jour: {len(result['updated'])} | "
                    f"Ignores: {len(result['skipped'])} | "
                    f"Echecs: {len(result['failed'])}"
                )
                if result["skipped"]:
                    st.info(
                        f"Actifs ignores (recents): {', '.join(result['skipped'])}"
                    )
                if result["failed"]:
                    with st.expander("Details des erreurs"):
                        for item in result["failed"]:
                            st.write(f"{item['ticker']}: {item['error']}")
        if st.session_state.get(update_key, False):
            st.success(f"Mis a jour aujourd'hui ({today})")
        else:
            st.info("Pas de mise a jour aujourd'hui")

        st.markdown("### Import manuel")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            target_path = DATA_DIR / uploaded_file.name
            if target_path.exists() and not force_update:
                st.warning("Fichier deja present. Activez l'option d'ecrasement.")
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                load_asset_frame.clear()
                st.success(f"Fichier enregistre: {target_path.name}")

        st.markdown("### Historique local")
        if DATA_DIR.exists():
            files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        else:
            files = []
        if files:
            st.write(", ".join(files))
        else:
            st.caption("Aucun fichier dans data/assets")

    return {
        "uploaded_file": uploaded_file,
        "assets": selected_assets,
        "start_date": start_date,
        "end_date": end_date,
        "window_size": window_size,
        "epochs": epochs,
        "model_type": model_type,
        "horizon_label": horizon_label,
        "horizon_steps": horizon_steps,
        "force_retrain": force_retrain,
        "generate_curves": st.session_state.generate_curves,
    }


def render_predictions_tab(state):
    st.subheader(
        "Marche et Signaux",
        help="Vue historique des prix et sorties du modele pour les actifs selectionnes.",
    )
    st.caption(
        "Visualisez l'historique de marche, les sorties du modele et les projections futures."
    )
    if not state["assets"]:
        st.info("Aucun actif selectionne.")
        return

    render_training_section(state)

    if not state.get("generate_curves"):
        st.markdown(
            "Selectionnez des actifs et cliquez sur \"Charger le marche\""
        )
        return

    frames = []
    errors = []
    with st.spinner("Chargement des donnees..."):
        for ticker in state["assets"]:
            try:
                df = load_asset_frame(
                    ticker,
                    state["start_date"],
                    state["end_date"],
                    get_asset_cache_token(ticker),
                )
                df = df.set_index("Date").rename(columns={"Close": ticker})
                frames.append(df)
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")

    if errors:
        st.warning("Certaines donnees n'ont pas pu etre chargees.")
        st.write("\n".join(errors))

    if not frames:
        st.info("Aucune donnee disponible.")
        return

    combined = pd.concat(frames, axis=1).sort_index()
    if not combined.empty:
        min_date = combined.index.min().strftime("%Y-%m-%d")
        max_date = combined.index.max().strftime("%Y-%m-%d")
        st.caption(f"Plage de donnees chargees: {min_date} -> {max_date}")
    available_assets = list(combined.columns)
    if len(available_assets) <= MAX_DEFAULT_PLOT_ASSETS:
        default_plot_assets = available_assets
    else:
        default_plot_assets = available_assets[:MAX_DEFAULT_PLOT_ASSETS]
        st.info(
            f"{len(available_assets)} actifs charges. "
            f"Affichage limite a {MAX_DEFAULT_PLOT_ASSETS} actifs par defaut."
        )

    with st.expander("Options du graphe (performance)"):
        plot_assets = st.multiselect(
            "Actifs affiches sur le graphe",
            options=available_assets,
            default=default_plot_assets,
            key="plot_assets_selection",
            help=HELP_TEXT["plot_assets"],
        )
        resolution_label = st.selectbox(
            "Resolution",
            options=list(PLOT_RESAMPLE_RULES.keys()),
            index=1,
            key="plot_resolution",
            help=HELP_TEXT["plot_resolution"],
        )
        max_points = st.slider(
            "Nombre max de points traces",
            min_value=200,
            max_value=4000,
            value=DEFAULT_MAX_PLOT_POINTS,
            step=100,
            key="plot_max_points",
            help=HELP_TEXT["plot_max_points"],
        )

    if not plot_assets:
        st.info("Selectionnez au moins un actif pour afficher le graphe.")
    else:
        plot_data = combined[plot_assets]
        resample_rule = PLOT_RESAMPLE_RULES[resolution_label]
        if resample_rule != "D":
            plot_data = plot_data.resample(resample_rule).last()
        plot_data = _downsample_plot_frame(plot_data, max_points)
        st.caption(
            f"Graphe affiche: {plot_data.shape[1]} actifs | "
            f"{len(plot_data)} points"
        )
        st.line_chart(plot_data)
    forecast_rows = st.session_state.get("training_future_forecasts", [])
    if forecast_rows:
        with st.expander("Apercu des projections"):
            forecast_df = pd.DataFrame(forecast_rows).sort_values(
                ["Actif", "Etape"]
            )
            forecast_df_display = _prepare_table_display(
                forecast_df,
                number_columns=[(["Prix predit"], 2, "")],
            )
            st.dataframe(forecast_df_display, width="stretch", hide_index=True)
    else:
        with st.expander("Apercu des premieres lignes"):
            st.dataframe(combined.head(10), width="stretch", hide_index=False)
    st.dataframe(combined.tail(10), width="stretch", hide_index=False)


def render_backtest_tab(state):
    st.subheader(
        "Backtests et Benchmarks",
        help=HELP_TEXT["strategy_metrics"],
    )
    st.caption(
        "Compare la strategie pilotee par le modele a des references simples "
        "(Naive, Equal Weight et benchmark de marche)."
    )
    if not state["assets"]:
        st.info("Aucun actif selectionne.")
        return

    st.caption(
        f"Configuration analysee: {state['model_type']} | "
        f"Fenetre {int(state['window_size'])} | "
        f"Horizon {state.get('horizon_label', '1 jour')} | "
        f"Epochs {int(state['epochs'])}"
    )
    st.caption(
        "Le backtest reutilise le cache d'entrainement par actif/config/date de cutoff "
        "quand il existe deja."
    )
    recommended_risk_horizon = _sync_backtest_risk_horizon_default(
        int(state.get("horizon_steps", 1))
    )

    with st.expander("Parametres du backtest", expanded=True):
        preset_cols = st.columns(1)
        with preset_cols[0]:
            period_preset_label = st.selectbox(
                "Profondeur du backtest",
                options=list(BACKTEST_PERIOD_PRESETS.keys()),
                index=0,
                key="backtest_period_preset",
                help=HELP_TEXT["backtest_period_preset"],
            )
            _sync_numeric_widget_with_recommendation(
                "backtest_num_periods",
                "_backtest_num_periods_reco",
                BACKTEST_PERIOD_PRESETS[period_preset_label],
            )
        form_cols = st.columns(2)
        with form_cols[0]:
            num_periods = int(st.session_state.get("backtest_num_periods", 4))
            history_lookback = st.selectbox(
                "Historique pour le risque",
                options=[63, 126, 252],
                index=2,
                key="backtest_history_lookback",
                help=HELP_TEXT["history_lookback"],
            )
            transaction_cost_bps = st.number_input(
                "Frais de transaction (bps)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="backtest_transaction_cost_bps",
                help=HELP_TEXT["transaction_cost_bps"],
            )
            max_weight = st.slider(
                "Poids max par actif",
                min_value=0.20,
                max_value=1.00,
                value=0.35,
                step=0.05,
                key="backtest_max_weight",
                help=HELP_TEXT["max_weight"],
            )
        with form_cols[1]:
            risk_profile = st.selectbox(
                "Profil risque",
                options=BACKTEST_RISK_PROFILES,
                index=1,
                key="backtest_risk_profile",
                format_func=lambda value: BACKTEST_RISK_PROFILE_LABELS.get(
                    value, value
                ),
                help=HELP_TEXT["risk_profile"],
            )
            portfolio_horizon = st.selectbox(
                "Cadre de risque portefeuille",
                options=BACKTEST_PORTFOLIO_HORIZONS,
                index=BACKTEST_PORTFOLIO_HORIZONS.index(recommended_risk_horizon),
                key="backtest_portfolio_horizon",
                format_func=lambda value: BACKTEST_PORTFOLIO_HORIZON_LABELS.get(
                    value, value
                ),
                help=HELP_TEXT["portfolio_horizon"],
            )
            benchmark_label = st.selectbox(
                "Benchmark",
                options=list(BACKTEST_BENCHMARKS.keys()),
                index=1,
                key="backtest_benchmark_label",
                help=HELP_TEXT["benchmark"],
            )
    st.caption(
        "Chaque periode correspond a une decision d'investissement historique simulee "
        "(une date de rebalance): on entraine sur l'historique disponible a cette date, "
        "puis on compare la projection au marche reel."
    )
    if int(num_periods) < 6:
        st.warning(
            "Moins de 6 periodes: les metriques annualisees et ratios "
            "(`Rendement annualise`, `Sharpe`, `Calmar`) restent fragiles."
        )

    run_cols = st.columns(2)
    run_backtest_clicked = run_cols[0].button(
        "Lancer la comparaison des strategies",
        width="stretch",
        help=HELP_TEXT["run_backtest"],
    )
    force_backtest_clicked = run_cols[1].button(
        "Relancer sans cache",
        width="stretch",
        help="Ignore le cache complet du backtest et le cache de training pour recalculer chaque cutoff.",
    )
    if run_backtest_clicked or force_backtest_clicked:
        force_retrain_backtest = bool(force_backtest_clicked)
        model_code = MODEL_TYPE_MAP[state["model_type"]]
        use_us_multifeature = model_code == "lstm_multifeature"
        histories = {}
        errors = []
        progress_bar = st.progress(0)
        status_line = st.empty()
        final_status_message = "Backtest en attente"
        factor_histories = None

        def on_backtest_progress(current_step, total_steps, message):
            if total_steps > 0:
                progress_bar.progress(min(current_step / total_steps, 1.0))
            status_line.caption(message)

        with st.spinner("Backtest en cours..."):
            try:
                status_line.caption("Chargement des historiques de marche")
                if use_us_multifeature:
                    factor_cache_token = tuple(
                        get_asset_cache_token(ticker)
                        for ticker in US_FACTOR_TICKERS.values()
                    )
                    factor_histories = load_us_factor_histories(
                        state["start_date"],
                        state["end_date"],
                        factor_cache_token,
                    )
                for ticker in state["assets"]:
                    try:
                        if use_us_multifeature:
                            histories[ticker] = load_asset_ohlcv_frame(
                                ticker,
                                state["start_date"],
                                state["end_date"],
                                get_asset_cache_token(ticker),
                            )
                        else:
                            histories[ticker] = load_asset_frame(
                                ticker,
                                state["start_date"],
                                state["end_date"],
                                get_asset_cache_token(ticker),
                            )
                    except Exception as exc:
                        errors.append(f"{ticker}: {exc}")

                benchmark_history = None
                benchmark_ticker = BACKTEST_BENCHMARKS[benchmark_label]
                if benchmark_ticker:
                    status_line.caption(f"Chargement du benchmark {benchmark_ticker}")
                    try:
                        benchmark_history = load_asset_frame(
                            benchmark_ticker,
                            state["start_date"],
                            state["end_date"],
                            get_asset_cache_token(benchmark_ticker),
                        )
                    except Exception as exc:
                        errors.append(f"{benchmark_ticker}: {exc}")

                if errors:
                    final_status_message = "Backtest interrompu: erreurs de chargement"
                    st.session_state.backtest_errors = errors
                else:
                    config = {
                        "window_size": int(state["window_size"]),
                        "epochs": int(state["epochs"]),
                        "horizon": int(state.get("horizon_steps", 1)),
                        "test_size": DEFAULT_TEST_SIZE,
                        "val_size": DEFAULT_VAL_SIZE,
                        "batch_size": 128,
                        "learning_rate": 5e-4,
                        "lstm_units1": DEFAULT_LSTM_UNITS1,
                        "lstm_units2": DEFAULT_LSTM_UNITS2,
                        "lstm_dropout": DEFAULT_LSTM_DROPOUT,
                        "timeout_sec": TRAINING_SUBPROCESS_TIMEOUT_SEC,
                    }
                    cache_key, cache_payload = build_backtest_cache_key(
                        assets=list(state["assets"]),
                        model_code=model_code,
                        config=config,
                        start_date=state["start_date"],
                        end_date=state["end_date"],
                        history_lookback=int(history_lookback),
                        num_periods=int(num_periods),
                        risk_profile=risk_profile,
                        portfolio_horizon=portfolio_horizon,
                        max_weight=float(max_weight),
                        transaction_cost_bps=float(transaction_cost_bps),
                        benchmark_label=benchmark_label,
                        histories=histories,
                        benchmark_history=benchmark_history,
                        factor_histories=factor_histories,
                    )
                    cached_backtest = None
                    if not force_retrain_backtest:
                        cached_backtest = load_cached_backtest(cache_key)
                    benchmark_refresh_from_session = False
                    if cached_backtest is None and not force_retrain_backtest:
                        previous_results = st.session_state.get("backtest_results")
                        previous_meta = st.session_state.get("backtest_meta") or {}
                        previous_assets = previous_meta.get("assets")
                        if previous_assets is None and previous_results:
                            asset_details = previous_results.get("asset_details", pd.DataFrame())
                            if asset_details is not None and not asset_details.empty and "Actif" in asset_details.columns:
                                previous_assets = sorted(asset_details["Actif"].dropna().unique().tolist())
                        current_start = (
                            state["start_date"].isoformat()
                            if state["start_date"] is not None
                            else None
                        )
                        current_end = (
                            state["end_date"].isoformat()
                            if state["end_date"] is not None
                            else None
                        )
                        previous_start = previous_meta.get("start_date")
                        previous_end = previous_meta.get("end_date")
                        same_date_filter = (
                            previous_start == current_start and previous_end == current_end
                            if previous_start is not None or previous_end is not None
                            else current_start is None and current_end is None
                        )
                        same_core_backtest = (
                            previous_results is not None
                            and previous_meta.get("model_type") == state["model_type"]
                            and int(previous_meta.get("window_size", -1)) == int(state["window_size"])
                            and int(previous_meta.get("epochs", -1)) == int(state["epochs"])
                            and int(previous_meta.get("horizon_steps", -1)) == int(state.get("horizon_steps", 1))
                            and int(previous_meta.get("num_periods", -1)) == int(num_periods)
                            and int(previous_meta.get("history_lookback", -1)) == int(history_lookback)
                            and previous_meta.get("risk_profile") == risk_profile
                            and previous_meta.get("portfolio_horizon") == portfolio_horizon
                            and abs(float(previous_meta.get("max_weight", np.nan)) - float(max_weight)) < 1e-12
                            and abs(
                                float(previous_meta.get("transaction_cost_bps", np.nan))
                                - float(transaction_cost_bps)
                            )
                            < 1e-12
                            and sorted(previous_assets or []) == sorted(state["assets"])
                            and same_date_filter
                            and previous_meta.get("benchmark_label") != benchmark_label
                        )
                        if same_core_backtest:
                            try:
                                backtest_results = apply_benchmark_to_backtest_results(
                                    previous_results,
                                    benchmark_history=benchmark_history,
                                    benchmark_label=benchmark_label,
                                    horizon_steps=int(config["horizon"]),
                                )
                                save_cached_backtest(
                                    cache_key=cache_key,
                                    cache_payload=cache_payload,
                                    results=backtest_results,
                                )
                                cached_backtest = {"results": backtest_results}
                                benchmark_refresh_from_session = True
                                progress_bar.progress(1.0)
                                final_status_message = (
                                    "Benchmark recalcule sans relancer l'entrainement"
                                )
                                status_line.caption(final_status_message)
                            except Exception:
                                benchmark_refresh_from_session = False
                    if cached_backtest is not None:
                        progress_bar.progress(1.0)
                        if not benchmark_refresh_from_session:
                            final_status_message = "Backtest charge depuis le cache"
                        status_line.caption(final_status_message)
                        backtest_results = cached_backtest["results"]
                    try:
                        if cached_backtest is None:
                            backtest_results = run_market_backtest(
                                histories=histories,
                                model_code=model_code,
                                config=config,
                                history_lookback=int(history_lookback),
                                num_periods=int(num_periods),
                                risk_profile=risk_profile,
                                portfolio_horizon=portfolio_horizon,
                                max_weight=float(max_weight),
                                transaction_cost_bps=float(transaction_cost_bps),
                                benchmark_history=benchmark_history,
                                benchmark_label=benchmark_label,
                                factor_histories=factor_histories,
                                force_retrain=force_retrain_backtest,
                                progress_callback=on_backtest_progress,
                            )
                            save_cached_backtest(
                                cache_key=cache_key,
                                cache_payload=cache_payload,
                                results=backtest_results,
                            )
                    except Exception as exc:
                        final_status_message = "Backtest interrompu: erreur d'execution"
                        st.session_state.backtest_errors = [str(exc)]
                    else:
                        progress_bar.progress(1.0)
                        if cached_backtest is None:
                            final_status_message = "Backtest termine"
                        status_line.caption(final_status_message)
                        st.session_state.backtest_results = backtest_results
                        st.session_state.backtest_errors = []
                        st.session_state.backtest_meta = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model_type": state["model_type"],
                            "window_size": int(state["window_size"]),
                            "epochs": int(state["epochs"]),
                            "horizon_label": state.get("horizon_label", "1 jour"),
                            "horizon_steps": int(state.get("horizon_steps", 1)),
                            "num_periods": int(num_periods),
                            "history_lookback": int(history_lookback),
                            "risk_profile": risk_profile,
                            "portfolio_horizon": portfolio_horizon,
                            "max_weight": float(max_weight),
                            "transaction_cost_bps": float(transaction_cost_bps),
                            "benchmark_label": benchmark_label,
                            "assets": list(state["assets"]),
                            "start_date": state["start_date"].isoformat()
                            if state["start_date"] is not None
                            else None,
                            "end_date": state["end_date"].isoformat()
                            if state["end_date"] is not None
                            else None,
                            "cutoff_dates": backtest_results.get("cutoff_dates", []),
                            "result_source": (
                                "Benchmark"
                                if benchmark_refresh_from_session
                                else ("Cache" if cached_backtest is not None else "Train")
                            ),
                            "backtest_cache_key": cache_key,
                        }
            finally:
                status_line.caption(final_status_message)

    errors = st.session_state.get("backtest_errors", [])
    if errors:
        with st.expander("Incidents du backtest", expanded=True):
            for item in errors:
                st.write(item)

    results = st.session_state.get("backtest_results")
    meta = st.session_state.get("backtest_meta")
    if not results or not meta:
        st.info("Lancez la comparaison pour voir les resultats portefeuille et benchmarks.")
        return

    run_quality = _backtest_run_quality(results, meta)

    st.caption(
        f"Dernier backtest ({meta['timestamp']}) | "
        f"Modele: {meta['model_type']} | Fenetre: {meta['window_size']} | "
        f"Horizon: {meta['horizon_label']} ({meta['horizon_steps']} pas) | "
        f"Periodes: {meta['num_periods']} | "
        f"Frais: {float(meta.get('transaction_cost_bps', 0.0)):.0f} bps | "
        f"Benchmark: {meta['benchmark_label']} | "
        f"Source: {meta.get('result_source', 'Train')}"
    )
    st.caption(
        "Cadre de risque portefeuille: "
        f"{BACKTEST_RISK_PROFILE_LABELS.get(meta.get('risk_profile', ''), meta.get('risk_profile', 'n/a'))} | "
        f"{BACKTEST_PORTFOLIO_HORIZON_LABELS.get(meta.get('portfolio_horizon', ''), meta.get('portfolio_horizon', 'n/a'))}"
    )
    if meta.get("cutoff_dates"):
        st.caption("Dates de rebalance: " + ", ".join(meta["cutoff_dates"]))
    if "Source training" in results["asset_details"].columns:
        source_counts = (
            results["asset_details"]["Source training"]
            .value_counts(dropna=False)
            .to_dict()
        )
        source_parts = [
            f"{source.lower()}: {count}"
            for source, count in source_counts.items()
            if pd.notna(source)
        ]
        if source_parts:
            st.caption(
                "Sources d'entrainement reutilisees sur ce backtest: "
                + " | ".join(source_parts)
            )
    quality_cols = st.columns(5)
    quality_cols[0].metric(
        "Periodes backtest",
        run_quality["period_count"],
        help="Nombre de periodes utilisees pour le backtest courant.",
    )
    quality_cols[1].metric(
        "Poids max modele",
        f"{run_quality['max_weight']:.1%}" if pd.notna(run_quality["max_weight"]) else "n/a",
        help=HELP_TEXT["max_weight_metric"],
    )
    quality_cols[2].metric(
        "Positions effectives",
        f"{run_quality['effective_positions']:.2f}"
        if pd.notna(run_quality["effective_positions"])
        else "n/a",
        help=HELP_TEXT["effective_positions"],
    )
    quality_cols[3].metric(
        "Top 3 poids",
        f"{run_quality['top3_share']:.1%}" if pd.notna(run_quality["top3_share"]) else "n/a",
        help=HELP_TEXT["top3_weight_share"],
    )
    quality_cols[4].metric(
        "Turnover modele",
        f"{run_quality['turnover']:.2%}" if pd.notna(run_quality["turnover"]) else "n/a",
        help=HELP_TEXT["turnover"],
    )
    st.caption(
        "Concentration du portefeuille modele: "
        f"`{run_quality['concentration_label']}`"
        + (
            f" | HHI {run_quality['concentration_hhi']:.2f}"
            if pd.notna(run_quality["concentration_hhi"])
            else ""
        )
    )
    for warning in run_quality["warnings"]:
        st.warning(warning)

    strategy_metrics = _prepare_table_display(
        results["strategy_metrics"].sort_values("Rendement cumule", ascending=False),
        order=[
            "Strategie",
            "Rendement cumule",
            "Rendement annualise",
            "Volatilite annualisee",
            "Sharpe",
            "Max drawdown",
            "VaR 95%",
            "CVaR 95%",
            "Calmar",
            "Turnover moyen",
            "Periodes",
        ],
        percent_columns=[
            "Rendement annualise",
            "Rendement cumule",
            "Volatilite annualisee",
            "Max drawdown",
            "VaR 95%",
            "CVaR 95%",
            "Turnover moyen",
        ],
        number_columns=[(["Sharpe", "Calmar"], 2, "")],
    )
    st.markdown("### Scoreboard des strategies")
    st.dataframe(strategy_metrics, width="stretch", hide_index=True)
    _render_table_help("strategy_metrics")

    headline_metrics = results["strategy_metrics"].copy()
    headline_metrics = _prepare_table_display(
        headline_metrics.sort_values("Rendement cumule", ascending=False),
        order=[
            "Strategie",
            "Rendement cumule",
            "Max drawdown",
            "VaR 95%",
            "Turnover moyen",
            "Periodes",
        ],
        percent_columns=[
            "Rendement cumule",
            "Max drawdown",
            "VaR 95%",
            "Turnover moyen",
        ],
    )
    st.markdown("### Lecture portefeuille prioritaire")
    st.caption(
        "Lecture la plus defendable pour la demo: rendement cumule, drawdown, VaR et turnover."
    )
    st.dataframe(headline_metrics, width="stretch", hide_index=True)

    nav_chart = results["nav"].set_index("Date")
    st.markdown("### Evolution de 1 euro investi")
    _render_precise_nav_chart(nav_chart.reset_index())

    st.markdown("### Rendement par periode")
    st.caption(
        "Lecture complementaire plus differenciante que le drawdown: compare la performance de chaque strategie, periode par periode."
    )
    _render_period_return_chart(
        results["period_returns"],
        results.get("benchmark_label", meta["benchmark_label"]),
    )

    st.markdown("### Qualite du signal par actif")
    st.caption(
        "Largeur IC 95% et couverture IC 95% resumant l'incertitude des previsions et la frequence a laquelle le prix reel tombe dans cet intervalle."
    )
    forecast_summary = _format_percent_columns(
        results["forecast_summary"],
        [
            "ic95_couverture",
            "beat_naive",
            "rendement_predit",
            "rendement_naive",
            "rendement_reel",
        ],
    )
    forecast_summary = _prepare_table_display(
        forecast_summary,
        rename_map={
            "mae_prix": "MAE prix",
            "erreur_modele": "Erreur modele",
            "erreur_naive": "Erreur naive",
            "ic95_largeur": "Largeur IC 95%",
            "ic95_couverture": "Couverture IC 95%",
            "beat_naive": "Beat vs naive",
            "rendement_predit": "Rendement modele",
            "rendement_naive": "Rendement naive",
            "rendement_reel": "Rendement reel",
        },
        order=[
            "Actif",
            "MAE prix",
            "Erreur modele",
            "Erreur naive",
            "Largeur IC 95%",
            "Couverture IC 95%",
            "Beat vs naive",
            "Rendement modele",
            "Rendement naive",
            "Rendement reel",
        ],
        number_columns=[
            (
                ["MAE prix", "Erreur modele", "Erreur naive", "Largeur IC 95%"],
                2,
                "",
            )
        ],
    )
    st.dataframe(forecast_summary, width="stretch", hide_index=True)
    _render_table_help("forecast_summary")

    with st.expander("Allocations recentes des strategies"):
        allocation_history = results["allocation_history"].copy()
        allocation_history = _prepare_table_display(
            allocation_history.tail(9),
            percent_columns=["Turnover", "Cout"],
        )
        st.dataframe(allocation_history, width="stretch", hide_index=True)

    with st.expander("Details par periode"):
        period_returns = _format_percent_columns(
            results["period_returns"],
            [
                "Modele",
                "Naive",
                "Equal Weight",
                results.get("benchmark_label", meta["benchmark_label"]),
                "Turnover modele",
                "Turnover naive",
                "Turnover equal",
                "Cout modele",
                "Cout naive",
                "Cout equal",
            ],
        )
        period_returns = _prepare_table_display(
            period_returns.sort_values("Date realisee", ascending=False),
            order=[
                "Date de rebalance",
                "Date realisee",
                "Modele",
                "Naive",
                "Equal Weight",
                results.get("benchmark_label", meta["benchmark_label"]),
                "Turnover modele",
                "Turnover naive",
                "Turnover equal",
                "Cout modele",
                "Cout naive",
                "Cout equal",
            ],
        )
        st.dataframe(period_returns, width="stretch", hide_index=True)
        _render_table_help("period_details")

    st.markdown("### Export des resultats")
    export_cols = st.columns(3)
    export_meta = {
        "meta": meta,
        "cutoff_dates": results.get("cutoff_dates", []),
    }
    export_cols[0].download_button(
        "Telecharger metriques CSV",
        data=_df_to_csv_bytes(results["strategy_metrics"]),
        file_name="backtest_strategy_metrics.csv",
        mime="text/csv",
        help=HELP_TEXT["download_metrics"],
        width="stretch",
    )
    export_cols[1].download_button(
        "Telecharger periodes CSV",
        data=_df_to_csv_bytes(results["period_returns"]),
        file_name="backtest_period_returns.csv",
        mime="text/csv",
        help=HELP_TEXT["download_periods"],
        width="stretch",
    )
    export_cols[2].download_button(
        "Telecharger allocations CSV",
        data=_df_to_csv_bytes(results["allocation_history"]),
        file_name="backtest_allocation_history.csv",
        mime="text/csv",
        help=HELP_TEXT["download_allocations"],
        width="stretch",
    )

    export_cols_2 = st.columns(3)
    export_cols_2[0].download_button(
        "Telecharger forecast CSV",
        data=_df_to_csv_bytes(results["forecast_summary"]),
        file_name="backtest_forecast_summary.csv",
        mime="text/csv",
        help=HELP_TEXT["download_forecast"],
        width="stretch",
    )
    export_cols_2[1].download_button(
        "Telecharger details actifs CSV",
        data=_df_to_csv_bytes(results["asset_details"]),
        file_name="backtest_asset_details.csv",
        mime="text/csv",
        help=HELP_TEXT["download_asset_details"],
        width="stretch",
    )
    export_cols_2[2].download_button(
        "Telecharger resume JSON",
        data=_json_to_bytes(export_meta),
        file_name="backtest_summary.json",
        mime="application/json",
        help=HELP_TEXT["download_summary"],
        width="stretch",
    )


def render_executive_summary_tab(state):
    st.subheader(
        "Synthese Decisionnelle",
        help=HELP_TEXT["executive_summary"],
    )
    st.caption(
        "Lecture rapide du dernier backtest: verdict, positionnement du modele, "
        "classement des strategies et allocation actuelle."
    )
    results = st.session_state.get("backtest_results")
    meta = st.session_state.get("backtest_meta")
    if not results or not meta:
        st.info("Lancez d'abord un backtest pour generer la synthese decisionnelle.")
        return

    summary = _build_executive_summary_payload(
        results=results,
        meta=meta,
        asset_count=len(state.get("assets", [])),
    )
    with st.container(border=True):
        if summary["verdict_level"] == "success":
            st.success(f"Verdict: {summary['verdict_label']}")
        elif summary["verdict_level"] == "info":
            st.info(f"Verdict: {summary['verdict_label']}")
        else:
            st.warning(f"Verdict: {summary['verdict_label']}")
        st.write(summary["headline"])
        st.caption(summary["comparison_text"])
        verdict_cols = st.columns(4)
        verdict_cols[0].metric(
            "References battues",
            f"{summary['model_beats_count']}/{summary['comparison_count']}"
            if summary["comparison_count"] > 0
            else "n/a",
            help="Nombre de references que le modele depasse en rendement annualise.",
        )
        verdict_cols[1].metric(
            "Periodes observees",
            summary["period_count"],
            help="Nombre de decisions historiques simulees dans le backtest courant.",
        )
        verdict_cols[2].metric(
            "Actifs testes",
            summary["asset_count"],
            help="Nombre d'actifs inclus dans le dernier backtest.",
        )
        verdict_cols[3].metric(
            "Rendement cumule modele",
            f"{summary['model_cumulative_return']:.2%}"
            if pd.notna(summary["model_cumulative_return"])
            else "n/a",
            help="Performance totale du modele sur la fenetre de backtest.",
        )
        risk_cols = st.columns(4)
        risk_cols[0].metric(
            "Max drawdown modele",
            f"{summary['model_max_drawdown']:.2%}"
            if pd.notna(summary["model_max_drawdown"])
            else "n/a",
            help=HELP_TEXT["max_drawdown"],
        )
        risk_cols[1].metric(
            "Turnover modele",
            f"{summary['model_turnover']:.2%}"
            if pd.notna(summary["model_turnover"])
            else "n/a",
            help=HELP_TEXT["turnover"],
        )
        risk_cols[2].metric(
            "VaR 95% modele",
            f"{summary['model_var_95']:.2%}"
            if pd.notna(summary["model_var_95"])
            else "n/a",
            help=HELP_TEXT["var_95"],
        )
        risk_cols[3].metric(
            "CVaR 95% modele",
            f"{summary['model_cvar_95']:.2%}"
            if pd.notna(summary["model_cvar_95"])
            else "n/a",
            help=HELP_TEXT["cvar_95"],
        )

    secondary_cols = st.columns(4)
    secondary_cols[0].metric(
        "Beat rate vs Naive",
        f"{summary['beat_naive_rate']:.1%}"
        if pd.notna(summary["beat_naive_rate"])
        else "n/a",
        help=HELP_TEXT["beat_rate"],
    )
    secondary_cols[1].metric(
        "Meilleure strategie",
        summary["best_strategy"],
        help=HELP_TEXT["best_strategy"],
    )
    secondary_cols[2].metric(
        "Rendement annualise modele",
        f"{summary['model_annualized_return']:.2%}"
        if pd.notna(summary["model_annualized_return"])
        else "n/a",
        help=HELP_TEXT["annualized_return"],
    )
    secondary_cols[3].metric(
        "Sharpe modele",
        f"{summary['model_sharpe']:.2f}"
        if pd.notna(summary["model_sharpe"])
        else "n/a",
        help=HELP_TEXT["sharpe"],
    )

    st.markdown("### Conclusion")
    st.write(
        f"Configuration: {meta['model_type']} | fenetre {meta['window_size']} | "
        f"horizon {meta['horizon_label']} ({meta['horizon_steps']} pas) | "
        f"epochs {meta['epochs']} | benchmark {meta['benchmark_label']}"
    )
    st.write(f"Allocation actuelle: {summary['top_weights']}")

    ranking_frame = summary["ranking"].copy()
    if not ranking_frame.empty and len(ranking_frame) >= 2:
        available_strategies = ranking_frame["Strategie"].dropna().tolist()
        st.markdown("### Comparaison libre de deux strategies")
        compare_cols = st.columns(2)
        default_a_index = (
            available_strategies.index("Modele")
            if "Modele" in available_strategies
            else 0
        )
        strategy_a = compare_cols[0].selectbox(
            "Strategie A",
            options=available_strategies,
            index=default_a_index,
            key="executive_strategy_compare_a",
        )
        strategy_b_options = [
            strategy for strategy in available_strategies if strategy != strategy_a
        ]
        default_b_name = (
            "Equal Weight"
            if "Equal Weight" in strategy_b_options
            else strategy_b_options[0]
        )
        strategy_b = compare_cols[1].selectbox(
            "Strategie B",
            options=strategy_b_options,
            index=strategy_b_options.index(default_b_name),
            key="executive_strategy_compare_b",
        )

        period_returns = results.get("period_returns", pd.DataFrame())
        comparison_stats_cols = st.columns(3)
        comparison_stats_cols[0].metric("Strategie A", strategy_a)
        comparison_stats_cols[1].metric("Strategie B", strategy_b)
        if (
            period_returns is not None
            and not period_returns.empty
            and strategy_a in period_returns.columns
            and strategy_b in period_returns.columns
        ):
            pair_returns = period_returns[[strategy_a, strategy_b]].dropna()
            if not pair_returns.empty:
                pair_wins = int((pair_returns[strategy_a] > pair_returns[strategy_b]).sum())
                comparison_stats_cols[2].metric(
                    f"{strategy_a} > {strategy_b}",
                    f"{pair_wins}/{len(pair_returns)} periodes",
                    delta=f"{pair_wins / len(pair_returns):.1%}",
                    help="Nombre de periodes ou la strategie A depasse la strategie B.",
                )
            else:
                comparison_stats_cols[2].metric(
                    f"{strategy_a} > {strategy_b}",
                    "n/a",
                    help="Comparaison par periode indisponible.",
                )
        else:
            comparison_stats_cols[2].metric(
                f"{strategy_a} > {strategy_b}",
                "n/a",
                help="Comparaison par periode indisponible.",
            )

        comparison_metrics = _build_strategy_comparison_frame(
            _strategy_metric_lookup(results.get("strategy_metrics", pd.DataFrame())),
            strategy_a,
            strategy_b,
        )
        st.dataframe(comparison_metrics, width="stretch", hide_index=True)

    comparison_frame = summary["comparison_results"].copy()
    if not comparison_frame.empty:
        comparison_frame["Modele bat"] = comparison_frame["Modele bat"].map(
            lambda value: "Oui" if value is True else ("Non" if value is False else "n/a")
        )
        comparison_frame = _format_percent_columns(
            comparison_frame,
            ["Rendement annualise"],
        )
        st.markdown("### Position du modele face aux references")
        comparison_frame = _prepare_table_display(
            comparison_frame,
            rename_map={"Strategie": "Reference"},
            order=["Reference", "Modele bat", "Rendement annualise"],
        )
        st.dataframe(comparison_frame, width="stretch", hide_index=True)

    if not ranking_frame.empty:
        ranking_frame = _format_percent_columns(
            ranking_frame,
            [
                "Rendement cumule",
                "Rendement annualise",
                "Volatilite annualisee",
                "Max drawdown",
                "VaR 95%",
                "CVaR 95%",
                "Turnover moyen",
            ],
        )
        st.markdown("### Classement des approches")
        ranking_frame = _prepare_table_display(
            ranking_frame,
            order=[
                "Strategie",
                "Rendement cumule",
                "Rendement annualise",
                "Volatilite annualisee",
                "Sharpe",
                "Max drawdown",
                "VaR 95%",
                "CVaR 95%",
                "Calmar",
                "Turnover moyen",
                "Periodes",
            ],
            number_columns=[(["Sharpe", "Calmar"], 2, "")],
        )
        st.dataframe(ranking_frame, width="stretch", hide_index=True)



def render_allocation_tab(state):
    st.subheader(
        "Allocation Recommandee",
        help=HELP_TEXT["allocation_view"],
    )
    st.caption(
        "Derniere allocation issue du backtest: repartition des poids, "
        "niveau de concentration, signaux et contribution au risque."
    )
    results = st.session_state.get("backtest_results")
    meta = st.session_state.get("backtest_meta")
    if not results or not meta:
        st.info("Lancez d'abord un backtest pour alimenter la recommandation d'allocation.")
        return

    allocation_history = results.get("allocation_history")
    if allocation_history is None or allocation_history.empty:
        st.info("Aucune allocation disponible.")
        return

    latest_model_rows = _latest_allocation_rows(allocation_history, "Modele")
    latest_naive_rows = _latest_allocation_rows(allocation_history, "Naive")
    if latest_model_rows.empty:
        st.info("Aucune allocation modele disponible.")
        return

    latest_model_row = latest_model_rows.iloc[-1]
    latest_cutoff_date = pd.to_datetime(latest_model_row["Date de rebalance"])
    latest_future_date = pd.to_datetime(latest_model_row["Date realisee"])

    model_weights = _parse_weights_text(latest_model_row["Poids"])
    if model_weights.empty:
        st.info("Poids modele indisponibles.")
        return
    naive_weights = (
        _parse_weights_text(latest_naive_rows.iloc[-1]["Poids"])
        if not latest_naive_rows.empty
        else pd.Series(dtype=float)
    )

    equal_weight = 1.0 / len(model_weights)
    active_positions = int((model_weights > 1e-8).sum())
    effective_positions = _effective_position_count(model_weights)
    max_weight = float(model_weights.max())
    top3_share = _top_weight_share(model_weights, top_n=3)
    concentration_hhi = _concentration_hhi(model_weights)
    concentration_label = _concentration_label(model_weights)

    metric_lookup = (
        results["strategy_metrics"]
        .set_index("Strategie")
        .to_dict(orient="index")
    )
    model_metrics = metric_lookup.get("Modele", {})

    st.markdown("### Lecture rapide")
    quick_cols = st.columns(4)
    quick_cols[0].metric(
        "Date de rebalance",
        latest_cutoff_date.strftime("%Y-%m-%d"),
        help="Date a laquelle les poids du portefeuille ont ete recalcules.",
    )
    quick_cols[1].metric(
        "Positions actives",
        active_positions,
        help=HELP_TEXT["active_positions"],
    )
    quick_cols[2].metric(
        "Top 3 poids",
        f"{top3_share:.1%}",
        help=HELP_TEXT["top3_weight_share"],
    )
    quick_cols[3].metric(
        "Niveau concentration",
        concentration_label,
        help=HELP_TEXT["concentration_label"],
    )

    st.markdown("### Risque et performance")
    perf_cols = st.columns(4)
    perf_cols[0].metric(
        "Rendement cumule",
        f"{float(model_metrics.get('Rendement cumule', np.nan)):.2%}",
        help="Performance totale observee sur la fenetre de backtest.",
    )
    perf_cols[1].metric(
        "Max drawdown",
        f"{float(model_metrics.get('Max drawdown', np.nan)):.2%}",
        help=HELP_TEXT["max_drawdown"],
    )
    perf_cols[2].metric(
        "VaR 95%",
        f"{float(model_metrics.get('VaR 95%', np.nan)):.2%}",
        help=HELP_TEXT["var_95"],
    )
    perf_cols[3].metric(
        "Turnover moyen",
        f"{float(model_metrics.get('Turnover moyen', np.nan)):.2%}",
        help=HELP_TEXT["turnover"],
    )

    advanced_cols = st.columns(4)
    advanced_cols[0].metric(
        "Rendement annualise",
        f"{float(model_metrics.get('Rendement annualise', np.nan)):.2%}",
        help=HELP_TEXT["annualized_return"],
    )
    advanced_cols[1].metric(
        "Sharpe",
        f"{float(model_metrics.get('Sharpe', np.nan)):.2f}",
        help=HELP_TEXT["sharpe"],
    )
    advanced_cols[2].metric(
        "CVaR 95%",
        f"{float(model_metrics.get('CVaR 95%', np.nan)):.2%}",
        help=HELP_TEXT["cvar_95"],
    )
    advanced_cols[3].metric(
        "Calmar",
        f"{float(model_metrics.get('Calmar', np.nan)):.2f}",
        help=HELP_TEXT["calmar"],
    )

    structure_cols = st.columns(3)
    structure_cols[0].metric(
        "Poids max observe",
        f"{max_weight:.1%}",
        help=HELP_TEXT["max_weight_metric"],
    )
    structure_cols[1].metric(
        "Positions effectives",
        f"{effective_positions:.2f}",
        help=HELP_TEXT["effective_positions"],
    )
    structure_cols[2].metric(
        "HHI concentration",
        f"{concentration_hhi:.2f}",
        help=HELP_TEXT["concentration_hhi"],
    )

    if concentration_label == "Elevee" or max_weight >= 0.45:
        st.warning(
            "Portefeuille concentre: une petite partie des positions porte une part importante du portefeuille."
        )
    elif concentration_label == "Moyenne" or effective_positions < max(2.0, len(model_weights) / 2):
        st.info(
            "Diversification perfectible: la concentration reste moderee et merite d'etre surveillee."
        )

    weight_frame = pd.DataFrame(
        {
            "Actif": model_weights.index,
            "Poids modele": model_weights.values,
        }
    )
    weight_frame["Poids equal"] = equal_weight
    weight_frame["Ecart vs equal"] = (
        weight_frame["Poids modele"] - weight_frame["Poids equal"]
    )
    if not naive_weights.empty:
        weight_frame["Poids naive"] = (
            naive_weights.reindex(weight_frame["Actif"]).fillna(0.0).values
        )
        weight_frame["Ecart vs naive"] = (
            weight_frame["Poids modele"] - weight_frame["Poids naive"]
        )

    latest_asset_details = results.get("asset_details", pd.DataFrame()).copy()
    if not latest_asset_details.empty:
        latest_asset_details = latest_asset_details[
            latest_asset_details["Date de rebalance"]
            == latest_cutoff_date.strftime("%Y-%m-%d")
        ][
            [
                "Actif",
                "Rendement predit",
                "Rendement naive",
                "Rendement reel",
                "Erreur modele",
            ]
        ]
        weight_frame = weight_frame.merge(
            latest_asset_details,
            on="Actif",
            how="left",
        )

    try:
        trailing_returns = _load_trailing_returns_for_assets(
            state=state,
            tickers=list(model_weights.index),
            cutoff_date=latest_cutoff_date,
            history_lookback=int(meta.get("history_lookback", 252)),
        )
        risk_contribution = _compute_risk_contribution(
            model_weights, trailing_returns
        )
    except Exception:
        risk_contribution = pd.Series(dtype=float)
    if not risk_contribution.empty:
        weight_frame["Contribution risque"] = (
            risk_contribution.reindex(weight_frame["Actif"]).fillna(0.0).values
        )

    chart_data = weight_frame.set_index("Actif")[["Poids modele"]]
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("### Repartition cible du portefeuille")
        st.bar_chart(chart_data, width="stretch")
    if "Contribution risque" in weight_frame.columns:
        risk_chart = (
            weight_frame[["Actif", "Contribution risque"]]
            .sort_values("Contribution risque", ascending=False)
            .set_index("Actif")
        )
        with chart_cols[1]:
            st.markdown("### Contribution au risque")
            st.caption(
                "Compare le poids economique d'un actif a sa part dans le risque total."
            )
            st.bar_chart(risk_chart, width="stretch")

    display_frame = _format_percent_columns(
        weight_frame,
        [
            "Poids modele",
            "Poids equal",
            "Ecart vs equal",
            "Poids naive",
            "Ecart vs naive",
            "Rendement predit",
            "Rendement naive",
            "Rendement reel",
            "Contribution risque",
        ],
    )
    sorted_display_frame = display_frame.loc[
        weight_frame.sort_values("Poids modele", ascending=False).index
    ]
    sorted_display_frame = _prepare_table_display(
        sorted_display_frame,
        rename_map={
            "Poids modele": "Poids modele",
            "Poids equal": "Poids equal-weight",
            "Ecart vs equal": "Ecart vs equal-weight",
            "Poids naive": "Poids naive",
            "Ecart vs naive": "Ecart vs naive",
            "Rendement predit": "Signal modele",
            "Rendement naive": "Signal naive",
            "Rendement reel": "Rendement realise",
        },
        order=[
            "Actif",
            "Poids modele",
            "Poids equal-weight",
            "Ecart vs equal-weight",
            "Poids naive",
            "Ecart vs naive",
            "Signal modele",
            "Signal naive",
            "Rendement realise",
            "Contribution risque",
            "Erreur modele",
        ],
        number_columns=[(["Erreur modele"], 2, "")],
    )
    st.dataframe(
        sorted_display_frame,
        width="stretch",
        hide_index=True,
    )
    _render_table_help("allocation_main")

    top_overweights = display_frame.loc[
        weight_frame.sort_values("Ecart vs equal", ascending=False).head(3).index
    ]
    top_underweights = display_frame.loc[
        weight_frame.sort_values("Ecart vs equal", ascending=True).head(3).index
    ]
    col_over, col_under = st.columns(2)
    with col_over:
        st.markdown("### Principales surponderations")
        st.dataframe(
            _prepare_table_display(
                top_overweights[["Actif", "Poids modele", "Ecart vs equal"]],
                rename_map={"Ecart vs equal": "Ecart vs equal-weight"},
                order=["Actif", "Poids modele", "Ecart vs equal-weight"],
            ),
            width="stretch",
            hide_index=True,
        )
    with col_under:
        st.markdown("### Principales sous-ponderations")
        st.dataframe(
            _prepare_table_display(
                top_underweights[["Actif", "Poids modele", "Ecart vs equal"]],
                rename_map={"Ecart vs equal": "Ecart vs equal-weight"},
                order=["Actif", "Poids modele", "Ecart vs equal-weight"],
            ),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        f"Derniere projection: rebalance {latest_cutoff_date.strftime('%Y-%m-%d')} "
        f"-> realise {latest_future_date.strftime('%Y-%m-%d')}"
    )


def main():
    st.set_page_config(
        page_title="Cockpit Quantitatif Multi-Actifs",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state = render_sidebar()

    tab_predictions, tab_backtest, tab_summary, tab_allocation = st.tabs(
        [
            "Marche et Signaux",
            "Backtests et Benchmarks",
            "Synthese Decisionnelle",
            "Allocation Recommandee",
        ]
    )

    with tab_predictions:
        render_predictions_tab(state)

    with tab_backtest:
        render_backtest_tab(state)

    with tab_summary:
        render_executive_summary_tab(state)

    with tab_allocation:
        render_allocation_tab(state)


if __name__ == "__main__":
    main()
