"""Baseline Results Protocol v1.0 metric functions.

Migrated from the existing MASTER paper-baseline implementation in
``qlib-MASTER``; metric formulas are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ANNUALIZATION = 252


def prediction_metrics(prediction: pd.Series, label: pd.Series) -> dict[str, float]:
    data = pd.concat({"prediction": prediction, "label": label}, axis=1, join="inner").dropna()
    if not isinstance(data.index, pd.MultiIndex) or "datetime" not in data.index.names:
        raise ValueError("prediction and label must use a MultiIndex containing datetime")
    grouped = data.groupby(level="datetime", sort=True)
    ic = grouped.apply(lambda frame: frame["prediction"].corr(frame["label"], method="pearson"))
    rank_ic = grouped.apply(lambda frame: frame["prediction"].corr(frame["label"], method="spearman"))
    return {
        "IC": float(ic.mean()),
        "ICIR": _mean_over_sample_std(ic),
        "RankIC": float(rank_ic.mean()),
        "RankICIR": _mean_over_sample_std(rank_ic),
    }


def portfolio_metrics(daily_return_net: pd.Series | np.ndarray) -> dict[str, float | int]:
    returns = np.asarray(daily_return_net, dtype=float)
    returns = returns[np.isfinite(returns)]
    if np.any(returns <= -1):
        raise ValueError("daily net return must be greater than -1")
    logs = np.log1p(returns)
    count = int(logs.size)
    if count == 0:
        return {name: math.nan for name in ("AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")} | {
            "num_test_days": 0
        }
    mean_log = float(logs.mean())
    sample_std = float(logs.std(ddof=1)) if count >= 2 else math.nan
    annual_return = float(np.exp(mean_log * ANNUALIZATION) - 1)
    annual_std = sample_std * math.sqrt(ANNUALIZATION) if np.isfinite(sample_std) else math.nan
    nav = np.concatenate(([1.0], np.exp(np.cumsum(logs))))
    drawdown = nav / np.maximum.accumulate(nav) - 1
    max_drawdown = float(drawdown.min())
    sharpe = _safe_divide(math.sqrt(ANNUALIZATION) * mean_log, sample_std)
    downside = float(np.sqrt(np.mean(np.minimum(logs, 0.0) ** 2)))
    sortino = _safe_divide(math.sqrt(ANNUALIZATION) * mean_log, downside)
    calmar = _safe_divide(annual_return, abs(max_drawdown))
    return {
        "AR": annual_return,
        "STD": annual_std,
        "MDD": max_drawdown,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "num_test_days": count,
    }


def _mean_over_sample_std(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return math.nan
    return _safe_divide(float(values.mean()), float(values.std(ddof=1)))


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)
