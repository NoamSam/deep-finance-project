from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


RiskProfile = Literal["conservateur", "equilibre", "dynamique"]
Horizon = Literal["court", "moyen", "long"]
CovarianceMethod = Literal["empirical", "ridge", "ledoit_wolf"]


def lambda_from_profile(risk_profile: str, horizon: str) -> float:
    """Map (profil, horizon) -> lambda de penalite du risque."""
    risk_profile = risk_profile.lower().strip()
    horizon = horizon.lower().strip()

    base = {
        "conservateur": 10.0,
        "equilibre": 3.0,
        "dynamique": 1.0,
    }
    horizon_mult = {
        "court": 1.5,
        "moyen": 1.0,
        "long": 0.7,
    }

    if risk_profile not in base:
        raise ValueError(
            "risk_profile doit etre 'conservateur', 'equilibre' ou 'dynamique'."
        )
    if horizon not in horizon_mult:
        raise ValueError("horizon doit etre 'court', 'moyen' ou 'long'.")
    return float(base[risk_profile] * horizon_mult[horizon])


def filter_by_correlation(
    returns_df: pd.DataFrame, max_corr: float = 0.95
) -> list[str]:
    """Pre-filtre diversification: retire des actifs trop correles."""
    if returns_df.empty:
        return []
    corr = returns_df.corr().abs()
    keep: list[str] = []
    for col in corr.columns:
        if all(corr.loc[col, k] <= max_corr for k in keep):
            keep.append(col)
    return keep


def estimate_covariance(
    returns_df: pd.DataFrame,
    method: CovarianceMethod = "ledoit_wolf",
    ridge_alpha: float = 1e-4,
) -> np.ndarray:
    """Estimateur de covariance robuste."""
    if returns_df.shape[1] < 2:
        raise ValueError("Il faut au moins 2 actifs pour estimer une covariance.")

    if method == "ledoit_wolf":
        cov = LedoitWolf().fit(returns_df.values).covariance_
    elif method == "ridge":
        cov = returns_df.cov().values
        avg_var = float(np.trace(cov) / cov.shape[0])
        cov += np.eye(cov.shape[0]) * (ridge_alpha * max(avg_var, 1e-12))
    elif method == "empirical":
        cov = returns_df.cov().values
    else:
        raise ValueError("covariance_method inconnu.")

    # Stabilite numerique.
    cov = (cov + cov.T) / 2.0
    return cov


def _check_feasibility(
    n: int,
    allow_short: bool,
    max_weight: float | None,
    leverage_limit: float | None,
) -> None:
    if n < 2:
        raise ValueError("Il faut au moins 2 actifs.")

    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("max_weight doit etre > 0.")
        if n * max_weight < 1.0 - 1e-12:
            raise ValueError(
                "Contraintes infeasibles: n * max_weight < 1. "
                "Augmentez max_weight ou le nombre d'actifs."
            )

    if allow_short and leverage_limit is not None and leverage_limit < 1.0:
        raise ValueError("leverage_limit doit etre >= 1.0.")


def _initial_weights(n: int, max_weight: float | None) -> np.ndarray:
    if max_weight is None or 1.0 / n <= max_weight:
        return np.ones(n, dtype=float) / n

    # Construction d'un point de depart faisable sous long-only + cap.
    w = np.zeros(n, dtype=float)
    remaining = 1.0
    for i in range(n):
        add = min(max_weight, remaining)
        w[i] = add
        remaining -= add
        if remaining <= 1e-12:
            break
    return w


def _cap_long_only_weights(weights: pd.Series, max_weight: float | None) -> pd.Series:
    if max_weight is None:
        total = float(weights.sum())
        return weights / total if total > 0 else weights

    capped = weights.astype(float).copy()
    remaining = capped.index.tolist()
    fixed = pd.Series(0.0, index=capped.index, dtype=float)
    cap = float(max_weight)

    while remaining:
        residual = 1.0 - float(fixed.sum())
        if residual <= 1e-12:
            break
        proposed = capped.loc[remaining]
        proposed_sum = float(proposed.sum())
        if proposed_sum <= 1e-12:
            equal_weight = residual / len(remaining)
            fixed.loc[remaining] = equal_weight
            break
        proposed = proposed / proposed_sum * residual
        over = proposed[proposed > cap + 1e-12]
        if over.empty:
            fixed.loc[remaining] = proposed
            break
        fixed.loc[over.index] = cap
        remaining = [name for name in remaining if name not in over.index]

    total = float(fixed.sum())
    if total <= 1e-12:
        fixed[:] = 1.0 / len(fixed)
        total = 1.0
    return fixed / total


def _fallback_long_only_weights(
    mu: pd.Series,
    max_weight: float | None,
) -> pd.Series:
    scores = mu.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = scores - float(scores.min())
    scores = scores + 1e-12
    if float(scores.sum()) <= 1e-12:
        scores[:] = 1.0
    return _cap_long_only_weights(scores, max_weight=max_weight)


def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_profile: RiskProfile = "equilibre",
    horizon: Horizon = "moyen",
    allow_short: bool = False,
    max_weight: float | None = None,
    leverage_limit: float | None = None,
    l2_reg: float = 0.0,
) -> np.ndarray:
    """
    Optimisation mean-variance:
      max w^T mu - lambda * w^T cov w - l2_reg * ||w||^2
      s.c. sum(w)=1, bornes, (optionnel) contrainte de levier.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)

    if cov.shape != (n, n):
        raise ValueError("cov doit etre de shape (n,n) avec n=len(mu).")
    _check_feasibility(
        n=n,
        allow_short=allow_short,
        max_weight=max_weight,
        leverage_limit=leverage_limit,
    )

    lam = lambda_from_profile(risk_profile, horizon)

    def objective(w: np.ndarray) -> float:
        ret = float(w @ mu)
        var = float(w.T @ cov @ w)
        reg = float(l2_reg * (w @ w))
        return -(ret - lam * var - reg)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    if allow_short and leverage_limit is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: leverage_limit - np.sum(np.abs(w))}
        )

    if allow_short:
        bound_high = 1.0 if max_weight is None else float(max_weight)
        bounds = [(-bound_high, bound_high) for _ in range(n)]
    else:
        bound_high = 1.0 if max_weight is None else float(max_weight)
        bounds = [(0.0, bound_high) for _ in range(n)]

    w0 = _initial_weights(n, max_weight)
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(f"Optimisation echouee: {res.message}")

    w = np.asarray(res.x, dtype=float)
    w[np.abs(w) < 1e-12] = 0.0

    # Correction numerique legere.
    total = float(np.sum(w))
    if abs(total - 1.0) > 1e-8 and abs(total) > 1e-12:
        w = w / total
    return w


def run_portfolio_optimization(
    returns_hist: pd.DataFrame,
    mu_pred: pd.Series,
    risk_profile: RiskProfile = "equilibre",
    horizon: Horizon = "moyen",
    corr_filter: bool = False,
    max_corr: float = 0.95,
    allow_short: bool = False,
    max_weight: float | None = None,
    leverage_limit: float | None = None,
    covariance_method: CovarianceMethod = "ledoit_wolf",
    ridge_alpha: float = 1e-4,
    l2_reg: float = 0.0,
) -> pd.Series:
    """Pipeline complet: alignement, covariance, optimisation, retour poids."""
    if not isinstance(returns_hist, pd.DataFrame) or returns_hist.empty:
        raise ValueError("returns_hist doit etre un DataFrame non vide.")
    if not isinstance(mu_pred, pd.Series) or mu_pred.empty:
        raise ValueError("mu_pred doit etre une Series non vide.")

    common_assets = [a for a in mu_pred.index if a in returns_hist.columns]
    if len(common_assets) < 2:
        raise ValueError(
            "Pas assez d'actifs communs entre mu_pred et returns_hist."
        )

    r = returns_hist[common_assets].dropna()
    if r.shape[0] < 20:
        raise ValueError("Historique insuffisant: au moins 20 lignes recommandees.")
    mu = mu_pred.loc[common_assets].astype(float)

    if corr_filter:
        kept = filter_by_correlation(r, max_corr=max_corr)
        if len(kept) < 2:
            raise ValueError(
                "Filtre correlation trop strict: moins de 2 actifs conserves."
            )
        r = r[kept]
        mu = mu.loc[kept]

    cov = estimate_covariance(
        returns_df=r,
        method=covariance_method,
        ridge_alpha=ridge_alpha,
    )
    try:
        w = optimize_portfolio(
            mu=mu.values,
            cov=cov,
            risk_profile=risk_profile,
            horizon=horizon,
            allow_short=allow_short,
            max_weight=max_weight,
            leverage_limit=leverage_limit,
            l2_reg=l2_reg,
        )
        weights = pd.Series(w, index=mu.index, name="weight")
    except RuntimeError:
        if allow_short:
            raise
        weights = _fallback_long_only_weights(
            mu=mu,
            max_weight=max_weight,
        ).rename("weight")
    return weights.sort_values(ascending=False)
