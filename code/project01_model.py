"""Small, auditable building blocks for Project 01.

This module is an MVP implementation, not a calibrated EU study.  It contains
legal-rule unit tests, origin-tagged battery accounting, and a small certified
dispatch LP. Real-data estimation must add the full Article 4 pathway ledger,
matching-period constraints, a public-data source-sink representation, and
validated zonal/interconnector network data. It cannot infer project-level
connection feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog


LOW_PRICE_THRESHOLD_EUR_PER_MWH = 20.0
EUA_MULTIPLIER = 0.36


def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _same_length(*arrays: np.ndarray) -> None:
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError("all time series must have the same length")


def article6_low_price_trigger(
    da_price_eur_per_mwh: Sequence[float] | np.ndarray,
    eua_price_eur_per_tco2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the exact Article 6 low-price/carbon-price trigger.

    The trigger is a temporal-correlation condition only.  This function does
    not certify additionality, geography, PPA provenance, redispatch, or
    storage qualification.
    """

    da = _as_float_array(da_price_eur_per_mwh, "da_price_eur_per_mwh")
    eua = _as_float_array(eua_price_eur_per_tco2, "eua_price_eur_per_tco2")
    _same_length(da, eua)
    finite = np.isfinite(da) & np.isfinite(eua)
    return finite & (
        (da <= LOW_PRICE_THRESHOLD_EUR_PER_MWH)
        | (da < EUA_MULTIPLIER * eua)
    )


def negative_price_trigger(
    da_price_eur_per_mwh: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the deliberately narrower negative-price-only comparator."""

    da = _as_float_array(da_price_eur_per_mwh, "da_price_eur_per_mwh")
    return np.isfinite(da) & (da <= 0.0)


def article6_temporal_flags(
    da_price_eur_per_mwh: Sequence[float] | np.ndarray,
    eua_price_eur_per_tco2: Sequence[float] | np.ndarray,
    source_temporally_matched: Sequence[bool] | np.ndarray,
    additionality: bool | Sequence[bool] = True,
    geography: bool | Sequence[bool] = True,
) -> Mapping[str, np.ndarray]:
    """Evaluate the temporal part of an Article 6 pathway.

    ``source_temporally_matched`` must be calculated by the caller from the
    applicable monthly or hourly ledger.  A low-price trigger relaxes only
    that temporal flag.  Additionality and geography remain separate flags.
    """

    da = _as_float_array(da_price_eur_per_mwh, "da_price_eur_per_mwh")
    eua = _as_float_array(eua_price_eur_per_tco2, "eua_price_eur_per_tco2")
    matched = np.asarray(source_temporally_matched, dtype=bool)
    if matched.ndim != 1:
        raise ValueError("source_temporally_matched must be one-dimensional")
    _same_length(da, eua, matched)

    def flag_array(value: bool | Sequence[bool], name: str) -> np.ndarray:
        if np.isscalar(value):
            return np.full(da.shape[0], bool(value), dtype=bool)
        array = np.asarray(value, dtype=bool)
        if array.ndim != 1 or array.shape[0] != da.shape[0]:
            raise ValueError(f"{name} must be scalar or match the time series")
        return array

    additionality_flag = flag_array(additionality, "additionality")
    geography_flag = flag_array(geography, "geography")
    low_price = article6_low_price_trigger(da, eua)
    temporal_ok = matched | low_price
    all_pathway_flags = additionality_flag & geography_flag & temporal_ok
    return {
        "low_price_trigger": low_price,
        "source_temporally_matched": matched,
        "temporal_ok": temporal_ok,
        "additionality_ok": additionality_flag,
        "geography_ok": geography_flag,
        "article6_pathway_ok": all_pathway_flags,
    }


def matching_period_key(timestamp: datetime, rule_year: int) -> tuple[Any, ...]:
    """Return a deterministic monthly or hourly matching-period key.

    The legal transition is represented explicitly: monthly matching through
    2029 and hourly matching from 2030.  Timezone conversion must happen before
    this function is called; the production pipeline will use UTC internally.
    """

    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    if rule_year <= 2029:
        return ("month", timestamp.year, timestamp.month)
    return (
        "hour",
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
    )


def battery_qualification_ledger(
    ppa_charge_mw: Sequence[float] | np.ndarray,
    grid_charge_mw: Sequence[float] | np.ndarray,
    discharge_mw: Sequence[float] | np.ndarray,
    *,
    eta_charge: float = 0.9,
    eta_discharge: float = 0.9,
    self_discharge: float = 0.0,
    initial_qualified_soc_mwh: float = 0.0,
    initial_nonqualified_soc_mwh: float = 0.0,
) -> Mapping[str, np.ndarray]:
    """Track qualified and nonqualified battery energy separately.

    PPA charging is treated as the qualified origin supplied by the caller;
    grid charging is always nonqualified in this ledger.  This function is an
    origin-accounting primitive.  It does not decide whether a PPA satisfies
    all Article 5 or Article 7 conditions, nor does it replace the monthly or
    hourly legal matching ledger.
    """

    ppa = _as_float_array(ppa_charge_mw, "ppa_charge_mw")
    grid = _as_float_array(grid_charge_mw, "grid_charge_mw")
    discharge = _as_float_array(discharge_mw, "discharge_mw")
    _same_length(ppa, grid, discharge)
    if np.any(ppa < 0) or np.any(grid < 0) or np.any(discharge < 0):
        raise ValueError("battery flows cannot be negative")
    if not 0 < eta_charge <= 1 or not 0 < eta_discharge <= 1:
        raise ValueError("battery efficiencies must be in (0, 1]")
    if not 0 <= self_discharge < 1:
        raise ValueError("self_discharge must be in [0, 1)")

    n = ppa.shape[0]
    q_soc = float(initial_qualified_soc_mwh)
    n_soc = float(initial_nonqualified_soc_mwh)
    q_soc_path = np.zeros(n)
    n_soc_path = np.zeros(n)
    q_discharge = np.zeros(n)
    n_discharge = np.zeros(n)
    unserved = np.zeros(n)

    for t in range(n):
        q_soc *= 1.0 - self_discharge
        n_soc *= 1.0 - self_discharge
        q_soc += eta_charge * ppa[t]
        n_soc += eta_charge * grid[t]

        q_available_output = eta_discharge * q_soc
        q_discharge[t] = min(discharge[t], q_available_output)
        q_soc -= q_discharge[t] / eta_discharge

        remaining = discharge[t] - q_discharge[t]
        n_available_output = eta_discharge * n_soc
        n_discharge[t] = min(remaining, n_available_output)
        n_soc -= n_discharge[t] / eta_discharge
        unserved[t] = remaining - n_discharge[t]

        q_soc_path[t] = q_soc
        n_soc_path[t] = n_soc

    return {
        "qualified_soc_mwh": q_soc_path,
        "nonqualified_soc_mwh": n_soc_path,
        "qualified_discharge_mw": q_discharge,
        "nonqualified_discharge_mw": n_discharge,
        "unserved_discharge_mw": unserved,
    }


@dataclass(frozen=True)
class DispatchResult:
    """Output of the certified-only dispatch smoke model."""

    status: int
    message: str
    objective_eur: float
    local_renewable_use_mw: np.ndarray
    source_ppa_to_electrolyser_mw: np.ndarray
    qualified_grid_use_mw: np.ndarray
    qualified_battery_charge_mw: np.ndarray
    qualified_battery_discharge_mw: np.ndarray
    electrolyser_load_mw: np.ndarray
    battery_soc_mwh: np.ndarray
    certified_hydrogen_kg: np.ndarray
    source_line_flow_mw: np.ndarray


def solve_certified_dispatch_mvp(
    local_res_mw: Sequence[float] | np.ndarray,
    source_ppa_availability_mw: Sequence[float] | np.ndarray,
    grid_eligible_availability_mw: Sequence[float] | np.ndarray,
    da_price_eur_per_mwh: Sequence[float] | np.ndarray,
    *,
    line_limit_mw: float,
    electrolyser_capacity_mw: float,
    battery_power_mw: float = 0.0,
    battery_energy_mwh: float = 0.0,
    eta_charge: float = 0.9,
    eta_discharge: float = 0.9,
    battery_degradation_eur_per_mwh: float = 0.0,
    grid_fee_eur_per_mwh: float = 0.0,
    ppa_price_eur_per_mwh: float = 0.0,
    hydrogen_yield_kg_per_mwh: float = 20.0,
    hydrogen_value_eur_per_kg: float = 50.0,
    initial_soc_mwh: float = 0.0,
) -> DispatchResult:
    """Solve a small certified-electricity dispatch LP.

    This is deliberately narrow.  It maximizes the value of certified
    hydrogen subject to a source line, local renewable supply, eligible grid
    supply, and one origin-tagged battery.  The caller must supply
    ``grid_eligible_availability_mw`` only after applying the legally reviewed
    pathway logic.  The production model must add nonqualified hydrogen,
    Article 4 pathways, monthly storage matching, project investment choices,
    outages, and the full network model.
    """

    local = _as_float_array(local_res_mw, "local_res_mw")
    source = _as_float_array(source_ppa_availability_mw, "source_ppa_availability_mw")
    grid = _as_float_array(
        grid_eligible_availability_mw, "grid_eligible_availability_mw"
    )
    da = _as_float_array(da_price_eur_per_mwh, "da_price_eur_per_mwh")
    _same_length(local, source, grid, da)
    if np.any(local < 0) or np.any(source < 0) or np.any(grid < 0):
        raise ValueError("availability cannot be negative")
    if min(line_limit_mw, electrolyser_capacity_mw, battery_power_mw, battery_energy_mwh) < 0:
        raise ValueError("capacities cannot be negative")
    if not 0 < eta_charge <= 1 or not 0 < eta_discharge <= 1:
        raise ValueError("battery efficiencies must be in (0, 1]")
    if hydrogen_yield_kg_per_mwh < 0 or hydrogen_value_eur_per_kg < 0:
        raise ValueError("hydrogen parameters cannot be negative")

    n = local.shape[0]
    # Variable order per hour: local, source, eligible grid, battery charge,
    # battery discharge, electrolyser load, followed by SOC for every hour.
    per_hour = 6
    soc_offset = per_hour * n

    def idx(t: int, component: int) -> int:
        return per_hour * t + component

    local_i, source_i, grid_i, bch_i, bdis_i, electro_i = range(6)
    n_vars = soc_offset + n
    soc_i = lambda t: soc_offset + t

    c = np.zeros(n_vars)
    c[[idx(t, local_i) for t in range(n)]] = 0.01
    c[[idx(t, source_i) for t in range(n)]] = ppa_price_eur_per_mwh
    c[[idx(t, grid_i) for t in range(n)]] = da + grid_fee_eur_per_mwh
    c[[idx(t, bch_i) for t in range(n)]] = battery_degradation_eur_per_mwh
    c[[idx(t, bdis_i) for t in range(n)]] = battery_degradation_eur_per_mwh
    c[[idx(t, electro_i) for t in range(n)]] = (
        -hydrogen_value_eur_per_kg * hydrogen_yield_kg_per_mwh
    )

    bounds: list[tuple[float, float]] = []
    for t in range(n):
        bounds.extend(
            [
                (0.0, float(local[t])),
                (0.0, float(source[t])),
                (0.0, float(grid[t])),
                (0.0, float(battery_power_mw)),
                (0.0, float(battery_power_mw)),
                (0.0, float(electrolyser_capacity_mw)),
            ]
        )
    bounds.extend((0.0, float(battery_energy_mwh)) for _ in range(n))

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for t in range(n):
        row = np.zeros(n_vars)
        row[idx(t, local_i)] = 1.0
        row[idx(t, source_i)] = 1.0
        row[idx(t, grid_i)] = 1.0
        row[idx(t, bdis_i)] = 1.0
        row[idx(t, bch_i)] = -1.0
        row[idx(t, electro_i)] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

        soc_row = np.zeros(n_vars)
        soc_row[soc_i(t)] = 1.0
        if t > 0:
            soc_row[soc_i(t - 1)] = -1.0
            initial = 0.0
        else:
            initial = float(initial_soc_mwh)
        soc_row[idx(t, bch_i)] = -eta_charge
        soc_row[idx(t, bdis_i)] = 1.0 / eta_discharge
        a_eq.append(soc_row)
        b_eq.append(initial)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for t in range(n):
        # Source renewable availability and the line limit both apply to the
        # PPA electricity delivered directly or sent to the qualified battery.
        source_row = np.zeros(n_vars)
        source_row[idx(t, source_i)] = 1.0
        source_row[idx(t, bch_i)] = 1.0
        a_ub.append(source_row)
        b_ub.append(float(source[t]))

        line_row = np.zeros(n_vars)
        line_row[idx(t, source_i)] = 1.0
        line_row[idx(t, bch_i)] = 1.0
        a_ub.append(line_row)
        b_ub.append(float(line_limit_mw))

    solution = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        return DispatchResult(
            status=int(solution.status),
            message=str(solution.message),
            objective_eur=float("nan"),
            local_renewable_use_mw=np.full(n, np.nan),
            source_ppa_to_electrolyser_mw=np.full(n, np.nan),
            qualified_grid_use_mw=np.full(n, np.nan),
            qualified_battery_charge_mw=np.full(n, np.nan),
            qualified_battery_discharge_mw=np.full(n, np.nan),
            electrolyser_load_mw=np.full(n, np.nan),
            battery_soc_mwh=np.full(n, np.nan),
            certified_hydrogen_kg=np.full(n, np.nan),
            source_line_flow_mw=np.full(n, np.nan),
        )

    x = solution.x
    local_use = np.array([x[idx(t, local_i)] for t in range(n)])
    source_to_electrolyser = np.array([x[idx(t, source_i)] for t in range(n)])
    grid_use = np.array([x[idx(t, grid_i)] for t in range(n)])
    battery_charge = np.array([x[idx(t, bch_i)] for t in range(n)])
    battery_discharge = np.array([x[idx(t, bdis_i)] for t in range(n)])
    electro_load = np.array([x[idx(t, electro_i)] for t in range(n)])
    soc = np.array([x[soc_i(t)] for t in range(n)])
    h2 = hydrogen_yield_kg_per_mwh * electro_load

    return DispatchResult(
        status=int(solution.status),
        message=str(solution.message),
        objective_eur=float(solution.fun),
        local_renewable_use_mw=local_use,
        source_ppa_to_electrolyser_mw=source_to_electrolyser,
        qualified_grid_use_mw=grid_use,
        qualified_battery_charge_mw=battery_charge,
        qualified_battery_discharge_mw=battery_discharge,
        electrolyser_load_mw=electro_load,
        battery_soc_mwh=soc,
        certified_hydrogen_kg=h2,
        source_line_flow_mw=source_to_electrolyser + battery_charge,
    )
