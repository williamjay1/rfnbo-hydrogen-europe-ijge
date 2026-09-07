"""Parameter uncertainty utilities for Project 01.

This module samples assumptions only.  It does not silently encode a claim
about the true distributions; the production study must replace the default
illustrative priors with provenance-tagged distributions.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable

import numpy as np
import pandas as pd

from project01_joint_model import JointModelCosts


def _lognormal_from_median_cv(rng: np.random.Generator, median: float, cv: float, size: int) -> np.ndarray:
    if median <= 0 or cv < 0 or size <= 0:
        raise ValueError("median must be positive, cv nonnegative, and size positive")
    sigma2 = np.log1p(cv**2)
    return rng.lognormal(mean=np.log(median) - 0.5 * sigma2, sigma=np.sqrt(sigma2), size=size)


def sample_cost_parameters(
    n_draws: int,
    *,
    seed: int = 20260829,
    wind_capex_cv: float = 0.20,
    solar_capex_cv: float = 0.20,
    battery_energy_capex_cv: float = 0.25,
    electrolyser_capex_cv: float = 0.20,
    efficiency_sd: float = 0.02,
) -> pd.DataFrame:
    """Generate reproducible illustrative draws for a Monte Carlo registry."""

    if n_draws <= 0 or efficiency_sd < 0:
        raise ValueError("n_draws must be positive and efficiency_sd nonnegative")
    rng = np.random.default_rng(seed)
    draw = pd.DataFrame(
        {
            "wind_capex_eur_per_mw_year": _lognormal_from_median_cv(rng, 120_000.0, wind_capex_cv, n_draws),
            "solar_capex_eur_per_mw_year": _lognormal_from_median_cv(rng, 85_000.0, solar_capex_cv, n_draws),
            "battery_energy_capex_eur_per_mwh_year": _lognormal_from_median_cv(rng, 12_000.0, battery_energy_capex_cv, n_draws),
            "electrolyser_capex_eur_per_mw_year": _lognormal_from_median_cv(rng, 100_000.0, electrolyser_capex_cv, n_draws),
            "eta_battery_charge": np.clip(rng.normal(0.90, efficiency_sd, n_draws), 0.70, 0.99),
            "eta_battery_discharge": np.clip(rng.normal(0.90, efficiency_sd, n_draws), 0.70, 0.99),
            "eta_hydrogen_charge": np.clip(rng.normal(0.995, efficiency_sd / 2.0, n_draws), 0.90, 1.0),
            "eta_hydrogen_discharge": np.clip(rng.normal(0.995, efficiency_sd / 2.0, n_draws), 0.90, 1.0),
        }
    )
    draw.insert(0, "draw_id", np.arange(n_draws, dtype=int))
    draw.insert(1, "seed", seed)
    return draw


def costs_from_draw(base: JointModelCosts, draw: pd.Series) -> JointModelCosts:
    """Apply only fields present in a validated parameter draw."""

    allowed = set(asdict(base))
    updates = {key: float(draw[key]) for key in draw.index if key in allowed and key != "horizon_years"}
    return replace(base, **updates)


def quantile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a nonempty finite vector")
    return {
        "n": float(len(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def bounded_lhs_draws(
    n_draws: int = 16,
    *,
    seed: int = 20260830,
    include_base: bool = True,
) -> pd.DataFrame:
    """Create a reproducible bounded stress design for the selected-EU model.

    The output is a space-filling parameter box, not a probability model.  The
    capital-cost, yield and grid-fee limits reproduce the deterministic
    one-way envelope already used in the manuscript.  Efficiency and EUA
    multipliers are deliberately bounded engineering/price-proxy stresses.
    They must not be interpreted as calibrated probabilities or confidence
    intervals.
    """

    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    bounds = {
        "capex_factor": (0.80, 1.20),
        "yield_factor": (0.90, 1.10),
        "grid_fee_eur_per_mwh": (0.0, 6.0),
        "eta_battery_charge": (0.85, 0.95),
        "eta_battery_discharge": (0.85, 0.95),
        "eta_hydrogen_charge": (0.98, 1.00),
        "eta_hydrogen_discharge": (0.98, 1.00),
        "eua_price_factor": (0.80, 1.20),
    }
    rng = np.random.default_rng(seed)
    strata = (np.arange(n_draws, dtype=float) + rng.random(n_draws)) / n_draws
    values: dict[str, np.ndarray] = {}
    for name, (lower, upper) in bounds.items():
        shuffled = strata[rng.permutation(n_draws)]
        values[name] = lower + (upper - lower) * shuffled
    draws = pd.DataFrame(values)
    draws.insert(0, "draw_id", np.arange(1, n_draws + 1, dtype=int))
    draws.insert(1, "seed", seed)
    draws.insert(2, "design", "bounded_latin_hypercube")
    if include_base:
        base = {
            "draw_id": 0,
            "seed": seed,
            "design": "base",
            "capex_factor": 1.0,
            "yield_factor": 1.0,
            "grid_fee_eur_per_mwh": 3.0,
            # The reference draw must reproduce build_costs(draw=None), not
            # turn the storage system into an unphysical lossless benchmark.
            "eta_battery_charge": 0.90,
            "eta_battery_discharge": 0.90,
            "eta_hydrogen_charge": 0.995,
            "eta_hydrogen_discharge": 0.995,
            "eua_price_factor": 1.0,
        }
        draws = pd.concat([pd.DataFrame([base]), draws], ignore_index=True)
    return draws[["draw_id", "seed", "design", *bounds.keys()]]
