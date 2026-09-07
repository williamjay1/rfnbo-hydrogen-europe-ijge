"""Project 01 joint sizing and dispatch model.

This is an auditable linear-programming implementation of the project-level
core.  It is deliberately explicit about what is and is not certified:

* the Article 6 low-price/carbon-price trigger is evaluated separately from
  additionality, geography, and the validity of a grid-electricity pathway;
* renewable-origin and grid-origin battery state of charge are separate;
* network import and source-line caps are time-varying constraints;
* temporal matching is applied by monthly or hourly matching periods;
* hydrogen storage keeps certified and non-certified product separate.

The model is a development implementation. After the data-feasibility hard
gate, the permitted publication scope is an EU-27 bidding-zone/interconnector
representation. A publication run must provide validated public data,
explicitly labelled zonal network proxies, pathway evidence, and independently
reviewed cost and emissions parameters. It must not be presented as a
project-connection or private-network model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from project01_model import article6_low_price_trigger, matching_period_key, negative_price_trigger


@dataclass(frozen=True)
class JointModelInputs:
    """Hourly inputs and scenario flags for the project LP."""

    timestamps_utc: Sequence[datetime]
    wind_capacity_factor: Sequence[float]
    solar_capacity_factor: Sequence[float]
    day_ahead_price_eur_per_mwh: Sequence[float]
    eua_price_eur_per_tco2: Sequence[float]
    eligible_grid_capacity_mw: Sequence[float]
    network_import_capacity_mw: Sequence[float]
    source_line_capacity_mw: Sequence[float]
    network_base_flow_mw: Sequence[Sequence[float]] | None = None
    network_branch_capacity_mw: Sequence[float] | None = None
    network_ptdf_project: Sequence[float] | None = None
    additionality_ok: bool = True
    # When enabled, a grid-qualified dispatch must be backed by at least the
    # same amount of renewable generation over the modeled horizon.  This is a
    # conservative engineering representation of the Article 5 quantity
    # requirement; it is separate from the caller's evidence flag above.
    enforce_additionality_energy_balance: bool = False
    # Deliberately permissive counterfactual used only for provenance
    # ablation. The valid model keeps renewable- and grid-origin battery
    # inventories separate; this flag quantifies the bias caused by pooling.
    pool_battery_origin: bool = False
    geography_ok: bool = True
    grid_pathway_valid: bool = False
    price_trigger_mode: str = "exact"
    rule_year: int = 2029
    dt_hours: float = 1.0


@dataclass(frozen=True)
class JointModelCosts:
    """Annualized capacity and variable cost/value assumptions."""

    wind_capex_eur_per_mw_year: float = 100_000.0
    solar_capex_eur_per_mw_year: float = 70_000.0
    battery_power_capex_eur_per_mw_year: float = 50_000.0
    battery_energy_capex_eur_per_mwh_year: float = 20_000.0
    electrolyser_capex_eur_per_mw_year: float = 100_000.0
    hydrogen_storage_capex_eur_per_kg_year: float = 5.0
    renewable_variable_cost_eur_per_mwh: float = 0.0
    grid_fee_eur_per_mwh: float = 0.0
    battery_degradation_eur_per_mwh: float = 2.0
    hydrogen_yield_kg_per_mwh: float = 20.0
    certified_hydrogen_value_eur_per_kg: float = 6.0
    noncertified_hydrogen_value_eur_per_kg: float = 0.0
    eta_battery_charge: float = 0.90
    eta_battery_discharge: float = 0.90
    eta_hydrogen_charge: float = 0.995
    eta_hydrogen_discharge: float = 0.995
    horizon_years: float | None = None


@dataclass(frozen=True)
class JointDispatchResult:
    """LP solution and auditable hourly flows."""

    status: int
    message: str
    objective_eur: float
    capacities: Mapping[str, float]
    metrics: Mapping[str, float]
    flags: Mapping[str, np.ndarray]
    flows: Mapping[str, np.ndarray]


FLOW_NAMES = (
    "renewable_direct_mw",
    "renewable_to_battery_mw",
    "grid_certified_mw",
    "grid_noncertified_mw",
    "grid_to_battery_mw",
    "qualified_battery_discharge_mw",
    "nonqualified_battery_discharge_mw",
    "certified_electrolyser_load_mw",
    "noncertified_electrolyser_load_mw",
    "qualified_battery_soc_mwh",
    "nonqualified_battery_soc_mwh",
    "certified_hydrogen_delivery_kg",
    "noncertified_hydrogen_delivery_kg",
    "certified_hydrogen_soc_kg",
    "noncertified_hydrogen_soc_kg",
)


def _array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _validate_inputs(inputs: JointModelInputs) -> dict[str, np.ndarray]:
    timestamps = np.asarray(list(inputs.timestamps_utc), dtype=object)
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("timestamps_utc must be a nonempty one-dimensional sequence")
    arrays = {
        "wind_cf": _array(inputs.wind_capacity_factor, "wind_capacity_factor"),
        "solar_cf": _array(inputs.solar_capacity_factor, "solar_capacity_factor"),
        "da": _array(inputs.day_ahead_price_eur_per_mwh, "day_ahead_price_eur_per_mwh"),
        "eua": _array(inputs.eua_price_eur_per_tco2, "eua_price_eur_per_tco2"),
        "eligible_grid": _array(inputs.eligible_grid_capacity_mw, "eligible_grid_capacity_mw"),
        "network_grid": _array(inputs.network_import_capacity_mw, "network_import_capacity_mw"),
        "source_line": _array(inputs.source_line_capacity_mw, "source_line_capacity_mw"),
    }
    n = len(timestamps)
    for name, value in arrays.items():
        if len(value) != n:
            raise ValueError(f"{name} must have the same length as timestamps_utc")
        if name not in {"da", "eua"} and (np.any(~np.isfinite(value)) or np.any(value < 0)):
            raise ValueError(f"{name} must be finite and nonnegative")
    if np.any(~np.isfinite(arrays["da"])) or np.any(~np.isfinite(arrays["eua"])):
        raise ValueError("price series must be finite before model construction")
    if np.any(arrays["wind_cf"] > 1.0) or np.any(arrays["solar_cf"] > 1.0):
        raise ValueError("capacity factors must not exceed one")
    if inputs.rule_year < 2023:
        raise ValueError("rule_year should be a current or future RFNBO rule year")
    if inputs.dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    if inputs.network_base_flow_mw is None or inputs.network_branch_capacity_mw is None or inputs.network_ptdf_project is None:
        network_base = np.empty((n, 0), dtype=float)
        network_capacity = np.empty(0, dtype=float)
        network_ptdf = np.empty(0, dtype=float)
    else:
        network_base = np.asarray(inputs.network_base_flow_mw, dtype=float)
        network_capacity = _array(inputs.network_branch_capacity_mw, "network_branch_capacity_mw")
        network_ptdf = _array(inputs.network_ptdf_project, "network_ptdf_project")
        if network_base.ndim != 2 or network_base.shape[0] != n:
            raise ValueError("network_base_flow_mw must have shape (hours, branches)")
        if network_base.shape[1] != len(network_capacity) or len(network_capacity) != len(network_ptdf):
            raise ValueError("network branch arrays must have the same branch dimension")
        if np.any(~np.isfinite(network_base)) or np.any(~np.isfinite(network_capacity)) or np.any(network_capacity < 0) or np.any(~np.isfinite(network_ptdf)):
            raise ValueError("network branch inputs must be finite and capacities nonnegative")
    arrays["network_base"] = network_base
    arrays["network_capacity"] = network_capacity
    arrays["network_ptdf"] = network_ptdf
    return {"timestamps": timestamps, **arrays}


def build_article6_grid_eligibility(
    inputs: JointModelInputs,
) -> Mapping[str, np.ndarray]:
    """Construct the legally separated grid pathway inputs.

    ``grid_pathway_valid`` is intentionally an explicit scenario input.  The
    Article 6 trigger alone never certifies grid electricity, and this helper
    returns zero eligible grid when the non-price pathway evidence is absent.
    """

    arrays = _validate_inputs(inputs)
    if inputs.price_trigger_mode == "exact":
        trigger = article6_low_price_trigger(arrays["da"], arrays["eua"])
    elif inputs.price_trigger_mode == "negative_only":
        trigger = negative_price_trigger(arrays["da"])
    elif inputs.price_trigger_mode == "price20_only":
        # Article 6 price branch isolated for mechanism decomposition.  This
        # is a computational ablation, not a legal certification pathway.
        trigger = arrays["da"] <= 20.0
    elif inputs.price_trigger_mode == "eua_only":
        # Isolate the EUA-linked branch used by the Article 6 abstraction.
        # This is a computational ablation, not a legal certification pathway.
        trigger = arrays["da"] < 0.36 * arrays["eua"]
    elif inputs.price_trigger_mode == "none":
        trigger = np.zeros(len(arrays["da"]), dtype=bool)
    else:
        raise ValueError("price_trigger_mode must be exact, negative_only, price20_only, eua_only, or none")
    pathway = np.full(trigger.shape, bool(inputs.grid_pathway_valid), dtype=bool)
    additionality = np.full(trigger.shape, bool(inputs.additionality_ok), dtype=bool)
    geography = np.full(trigger.shape, bool(inputs.geography_ok), dtype=bool)
    eligible = arrays["eligible_grid"] * (trigger & pathway & additionality & geography)
    return {
        "low_price_trigger": trigger,
        "additionality_ok": additionality,
        "geography_ok": geography,
        "grid_pathway_valid": pathway,
        "eligible_grid_mw": eligible,
    }


def _period_groups(timestamps: np.ndarray, rule_year: int) -> list[np.ndarray]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, datetime):
            # pandas.Timestamp is datetime-compatible for the attributes used
            # by matching_period_key, while numpy datetime64 is not.
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()
            else:
                raise TypeError("timestamps_utc must contain datetime-like objects")
        key = matching_period_key(timestamp, rule_year)
        groups.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(groups.items(), key=lambda item: item[0])]


def _bounds_with_upper(value: float) -> tuple[float, float]:
    if value < 0 or not np.isfinite(value):
        raise ValueError("capacity upper bounds must be finite and nonnegative")
    return (0.0, float(value))


def _empty_result(n: int, status: int, message: str, flags: Mapping[str, np.ndarray]) -> JointDispatchResult:
    return JointDispatchResult(
        status=status,
        message=message,
        objective_eur=float("nan"),
        capacities={name: float("nan") for name in ["wind_mw", "solar_mw", "battery_power_mw", "battery_energy_mwh", "electrolyser_mw", "hydrogen_storage_kg"]},
        metrics={"certified_hydrogen_kg": float("nan"), "noncertified_hydrogen_kg": float("nan")},
        flags=flags,
        flows={name: np.full(n, np.nan) for name in FLOW_NAMES},
    )


def solve_joint_sizing_dispatch(
    inputs: JointModelInputs,
    costs: JointModelCosts,
    *,
    capacity_upper_bounds: Mapping[str, float] | None = None,
    certified_hydrogen_target_kg: float | None = None,
    noncertified_hydrogen_target_kg: float | None = None,
) -> JointDispatchResult:
    """Optimize renewable, battery, electrolyser and H2 storage capacities.

    The LP is a representative-system core rather than a full European
    power-flow model. Network inputs are therefore explicit hourly transfer or
    interconnector caps; the final EU-27 run must generate them from public
    zonal data or a documented flow-based market representation. A cap may not
    be described as a real project's connection limit.
    """

    arrays = _validate_inputs(inputs)
    n = len(arrays["timestamps"])
    flags = build_article6_grid_eligibility(inputs)
    upper = {
        "wind_mw": 2_000.0,
        "solar_mw": 2_000.0,
        "battery_power_mw": 2_000.0,
        "battery_energy_mwh": 8_000.0,
        "electrolyser_mw": 2_000.0,
        "hydrogen_storage_kg": 1_000_000.0,
    }
    if capacity_upper_bounds is not None:
        upper.update(capacity_upper_bounds)
    for value in upper.values():
        _bounds_with_upper(value)

    for name, value in vars(costs).items():
        if name.startswith("eta_"):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        elif name not in {"horizon_years"} and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if costs.hydrogen_yield_kg_per_mwh <= 0:
        raise ValueError("hydrogen_yield_kg_per_mwh must be positive")
    horizon_years = costs.horizon_years
    if horizon_years is None:
        horizon_years = n * inputs.dt_hours / 8760.0
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")

    # Hourly flow variables followed by six capacity variables.
    per_hour = len(FLOW_NAMES)
    cap_names = ["wind_mw", "solar_mw", "battery_power_mw", "battery_energy_mwh", "electrolyser_mw", "hydrogen_storage_kg"]
    cap_offset = per_hour * n
    cap_index = {name: cap_offset + i for i, name in enumerate(cap_names)}
    n_vars = cap_offset + len(cap_names)

    def vi(t: int, name: str) -> int:
        return per_hour * t + FLOW_NAMES.index(name)

    c = np.zeros(n_vars)
    c[cap_index["wind_mw"]] = costs.wind_capex_eur_per_mw_year * horizon_years
    c[cap_index["solar_mw"]] = costs.solar_capex_eur_per_mw_year * horizon_years
    c[cap_index["battery_power_mw"]] = costs.battery_power_capex_eur_per_mw_year * horizon_years
    c[cap_index["battery_energy_mwh"]] = costs.battery_energy_capex_eur_per_mwh_year * horizon_years
    c[cap_index["electrolyser_mw"]] = costs.electrolyser_capex_eur_per_mw_year * horizon_years
    c[cap_index["hydrogen_storage_kg"]] = costs.hydrogen_storage_capex_eur_per_kg_year * horizon_years
    for t in range(n):
        dt = inputs.dt_hours
        grid_cost = arrays["da"][t] + costs.grid_fee_eur_per_mwh
        for name in ["grid_certified_mw", "grid_noncertified_mw", "grid_to_battery_mw"]:
            c[vi(t, name)] = grid_cost * dt
        for name in ["renewable_direct_mw", "renewable_to_battery_mw"]:
            c[vi(t, name)] = costs.renewable_variable_cost_eur_per_mwh * dt
        for name in ["renewable_to_battery_mw", "grid_to_battery_mw", "qualified_battery_discharge_mw", "nonqualified_battery_discharge_mw"]:
            c[vi(t, name)] += costs.battery_degradation_eur_per_mwh * dt
        # If a delivery target is supplied, the LP is in cost-minimization
        # mode and hydrogen value must not be counted as a second objective.
        # Without a target, the same formulation supports value-maximization.
        # Delivery variables are average hourly rates (kg/h), so their value
        # contribution is integrated over the time step.
        c[vi(t, "certified_hydrogen_delivery_kg")] = 0.0 if certified_hydrogen_target_kg is not None else -costs.certified_hydrogen_value_eur_per_kg * dt
        c[vi(t, "noncertified_hydrogen_delivery_kg")] = 0.0 if noncertified_hydrogen_target_kg is not None else -costs.noncertified_hydrogen_value_eur_per_kg * dt

    for target_name, target in {
        "certified_hydrogen_target_kg": certified_hydrogen_target_kg,
        "noncertified_hydrogen_target_kg": noncertified_hydrogen_target_kg,
    }.items():
        if target is not None and (target < 0 or not np.isfinite(target)):
            raise ValueError(f"{target_name} must be finite and nonnegative when supplied")

    # Six dynamic equalities per hour: electricity-origin ledgers, two battery
    # SOC equations and two certified/non-certified H2 SOC equations.
    target_equalities = int(certified_hydrogen_target_kg is not None) + int(noncertified_hydrogen_target_kg is not None)
    n_eq = 6 * n + 4 + target_equalities
    a_eq = lil_matrix((n_eq, n_vars), dtype=float)
    b_eq = np.zeros(n_eq)
    row = 0
    for t in range(n):
        a_eq[row, vi(t, "certified_electrolyser_load_mw")] = 1.0
        a_eq[row, vi(t, "renewable_direct_mw")] = -1.0
        a_eq[row, vi(t, "qualified_battery_discharge_mw")] = -1.0
        a_eq[row, vi(t, "grid_certified_mw")] = -1.0
        row += 1
        a_eq[row, vi(t, "noncertified_electrolyser_load_mw")] = 1.0
        a_eq[row, vi(t, "nonqualified_battery_discharge_mw")] = -1.0
        a_eq[row, vi(t, "grid_noncertified_mw")] = -1.0
        row += 1

        a_eq[row, vi(t, "qualified_battery_soc_mwh")] = 1.0
        a_eq[row, vi(t, "renewable_to_battery_mw")] = -costs.eta_battery_charge * inputs.dt_hours
        a_eq[row, vi(t, "qualified_battery_discharge_mw")] = inputs.dt_hours / costs.eta_battery_discharge
        if inputs.pool_battery_origin:
            a_eq[row, vi(t, "grid_to_battery_mw")] = -costs.eta_battery_charge * inputs.dt_hours
            a_eq[row, vi(t, "nonqualified_battery_discharge_mw")] = inputs.dt_hours / costs.eta_battery_discharge
        if t > 0:
            a_eq[row, vi(t - 1, "qualified_battery_soc_mwh")] = -1.0
        row += 1
        a_eq[row, vi(t, "nonqualified_battery_soc_mwh")] = 1.0
        if not inputs.pool_battery_origin:
            a_eq[row, vi(t, "grid_to_battery_mw")] = -costs.eta_battery_charge * inputs.dt_hours
            a_eq[row, vi(t, "nonqualified_battery_discharge_mw")] = inputs.dt_hours / costs.eta_battery_discharge
        if t > 0:
            a_eq[row, vi(t - 1, "nonqualified_battery_soc_mwh")] = -1.0
        row += 1

        a_eq[row, vi(t, "certified_hydrogen_soc_kg")] = 1.0
        a_eq[row, vi(t, "certified_electrolyser_load_mw")] = -costs.eta_hydrogen_charge * costs.hydrogen_yield_kg_per_mwh * inputs.dt_hours
        a_eq[row, vi(t, "certified_hydrogen_delivery_kg")] = inputs.dt_hours / costs.eta_hydrogen_discharge
        if t > 0:
            a_eq[row, vi(t - 1, "certified_hydrogen_soc_kg")] = -1.0
        row += 1
        a_eq[row, vi(t, "noncertified_hydrogen_soc_kg")] = 1.0
        a_eq[row, vi(t, "noncertified_electrolyser_load_mw")] = -costs.eta_hydrogen_charge * costs.hydrogen_yield_kg_per_mwh * inputs.dt_hours
        a_eq[row, vi(t, "noncertified_hydrogen_delivery_kg")] = inputs.dt_hours / costs.eta_hydrogen_discharge
        if t > 0:
            a_eq[row, vi(t - 1, "noncertified_hydrogen_soc_kg")] = -1.0
        row += 1

    # Close both storage ledgers over the modeled horizon.  This prevents the
    # objective from valuing energy left in storage at the artificial endpoint.
    for name in ["qualified_battery_soc_mwh", "nonqualified_battery_soc_mwh", "certified_hydrogen_soc_kg", "noncertified_hydrogen_soc_kg"]:
        a_eq[row, vi(n - 1, name)] = 1.0
        row += 1

    if certified_hydrogen_target_kg is not None:
        for t in range(n):
            # The delivery variable is kg/h; the target is an integrated kg
            # quantity and therefore needs the time-step factor.
            a_eq[row, vi(t, "certified_hydrogen_delivery_kg")] = inputs.dt_hours
        b_eq[row] = float(certified_hydrogen_target_kg)
        row += 1
    if noncertified_hydrogen_target_kg is not None:
        for t in range(n):
            a_eq[row, vi(t, "noncertified_hydrogen_delivery_kg")] = inputs.dt_hours
        b_eq[row] = float(noncertified_hydrogen_target_kg)
        row += 1

    branch_count = arrays["network_capacity"].shape[0]
    matching_groups = _period_groups(arrays["timestamps"], inputs.rule_year)
    additionality_balance = bool(
        inputs.enforce_additionality_energy_balance
        and inputs.grid_pathway_valid
        and inputs.additionality_ok
        and inputs.geography_ok
    )
    ub_rows = 10 * n + len(matching_groups) + 2 * n * branch_count + int(additionality_balance)
    a_ub = lil_matrix((ub_rows, n_vars), dtype=float)
    b_ub = np.zeros(ub_rows)
    row = 0
    for t in range(n):
        # Renewable availability and source-line capacity.
        a_ub[row, vi(t, "renewable_direct_mw")] = 1.0
        a_ub[row, vi(t, "renewable_to_battery_mw")] = 1.0
        a_ub[row, cap_index["wind_mw"]] = -arrays["wind_cf"][t]
        a_ub[row, cap_index["solar_mw"]] = -arrays["solar_cf"][t]
        row += 1
        a_ub[row, vi(t, "renewable_direct_mw")] = 1.0
        a_ub[row, vi(t, "renewable_to_battery_mw")] = 1.0
        b_ub[row] = arrays["source_line"][t]
        row += 1
        # Physical grid import cap, independent from the RFNBO eligibility cap.
        for name in ["grid_certified_mw", "grid_noncertified_mw", "grid_to_battery_mw"]:
            a_ub[row, vi(t, name)] = 1.0
        b_ub[row] = arrays["network_grid"][t]
        row += 1
        a_ub[row, vi(t, "grid_certified_mw")] = 1.0
        b_ub[row] = flags["eligible_grid_mw"][t]
        row += 1
        # Battery charge/discharge power, combined across origins.
        a_ub[row, vi(t, "renewable_to_battery_mw")] = 1.0
        a_ub[row, vi(t, "grid_to_battery_mw")] = 1.0
        a_ub[row, cap_index["battery_power_mw"]] = -1.0
        row += 1
        a_ub[row, vi(t, "qualified_battery_discharge_mw")] = 1.0
        a_ub[row, vi(t, "nonqualified_battery_discharge_mw")] = 1.0
        a_ub[row, cap_index["battery_power_mw"]] = -1.0
        row += 1
        a_ub[row, vi(t, "qualified_battery_soc_mwh")] = 1.0
        a_ub[row, vi(t, "nonqualified_battery_soc_mwh")] = 1.0
        a_ub[row, cap_index["battery_energy_mwh"]] = -1.0
        row += 1
        a_ub[row, vi(t, "certified_electrolyser_load_mw")] = 1.0
        a_ub[row, vi(t, "noncertified_electrolyser_load_mw")] = 1.0
        a_ub[row, cap_index["electrolyser_mw"]] = -1.0
        row += 1
        a_ub[row, vi(t, "certified_hydrogen_soc_kg")] = 1.0
        a_ub[row, vi(t, "noncertified_hydrogen_soc_kg")] = 1.0
        a_ub[row, cap_index["hydrogen_storage_kg"]] = -1.0
        row += 1
        # H2 delivery is limited by the modeled storage tank capacity as a
        # conservative fixed-output-power proxy; a production run can replace
        # this with an independently sized compressor/output constraint.
        a_ub[row, vi(t, "certified_hydrogen_delivery_kg")] = 1.0
        a_ub[row, vi(t, "noncertified_hydrogen_delivery_kg")] = 1.0
        a_ub[row, cap_index["hydrogen_storage_kg"]] = -1.0
        row += 1

        # Optional PTDF/DC branch constraints.  The project withdrawal is the
        # total physical grid import.  Background flows are supplied by the
        # external network model and remain separate from the RFNBO ledger.
        for branch, ptdf in enumerate(arrays["network_ptdf"]):
            for name in ["grid_certified_mw", "grid_noncertified_mw", "grid_to_battery_mw"]:
                a_ub[row, vi(t, name)] = ptdf
            b_ub[row] = arrays["network_capacity"][branch] - arrays["network_base"][t, branch]
            row += 1
            for name in ["grid_certified_mw", "grid_noncertified_mw", "grid_to_battery_mw"]:
                a_ub[row, vi(t, name)] = -ptdf
            b_ub[row] = arrays["network_capacity"][branch] + arrays["network_base"][t, branch]
            row += 1

    # Temporal matching: certified H2 electricity in each legal matching
    # period cannot exceed contemporaneous renewable electricity plus the
    # explicitly enabled Article 6 low-price pathway.  The low-price pathway
    # is not available unless grid_pathway_valid and the other flags are true.
    for indices in matching_groups:
        for t in indices:
            a_ub[row, vi(int(t), "certified_electrolyser_load_mw")] += inputs.dt_hours
            a_ub[row, vi(int(t), "renewable_direct_mw")] -= inputs.dt_hours
            a_ub[row, vi(int(t), "renewable_to_battery_mw")] -= inputs.dt_hours
            if inputs.pool_battery_origin and flags["eligible_grid_mw"][int(t)] > 0:
                a_ub[row, vi(int(t), "grid_to_battery_mw")] -= inputs.dt_hours
            a_ub[row, vi(int(t), "grid_certified_mw")] -= inputs.dt_hours
        row += 1

    if additionality_balance:
        # A price-triggered grid pathway relaxes temporal correlation, not the
        # Article 5 quantity condition. Use one cumulative energy-coverage
        # inequality over the modeled period: model-qualified grid energy must
        # not exceed the energy produced by the optimized wind/PV assets. This
        # is a conservative quantity surrogate for an equivalent PPA/own-supply
        # amount. It deliberately does not assert the Article 5 age, support,
        # production, or contract tests, and it does not reimpose hourly
        # matching that Article 6 can relax.
        for t in range(n):
            a_ub[row, vi(t, "grid_certified_mw")] = inputs.dt_hours
            a_ub[row, cap_index["wind_mw"]] -= arrays["wind_cf"][t] * inputs.dt_hours
            a_ub[row, cap_index["solar_mw"]] -= arrays["solar_cf"][t] * inputs.dt_hours
        row += 1

    assert row == ub_rows

    bounds: list[tuple[float, float]] = [(0.0, None)] * n_vars
    for t in range(n):
        # All dynamic variables are nonnegative; capacity variables receive
        # scenario-specific finite upper bounds below.
        for name in FLOW_NAMES:
            bounds[vi(t, name)] = (0.0, None)
        if inputs.pool_battery_origin:
            # The permissive counterfactual assigns all battery discharge to
            # the certified ledger and removes the unrelated non-certified
            # product route from the comparison.
            for name in [
                "nonqualified_battery_discharge_mw",
                "grid_noncertified_mw",
                "noncertified_electrolyser_load_mw",
                "noncertified_hydrogen_delivery_kg",
                "nonqualified_battery_soc_mwh",
                "noncertified_hydrogen_soc_kg",
            ]:
                bounds[vi(t, name)] = (0.0, 0.0)
    for name in cap_names:
        bounds[cap_index[name]] = _bounds_with_upper(upper[name])

    solution = linprog(
        c,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        return _empty_result(n, int(solution.status), str(solution.message), flags)

    x = solution.x
    flows = {name: np.asarray([x[vi(t, name)] for t in range(n)], dtype=float) for name in FLOW_NAMES}
    capacities = {name: float(x[index]) for name, index in cap_index.items()}
    renewable_available = arrays["wind_cf"] * capacities["wind_mw"] + arrays["solar_cf"] * capacities["solar_mw"]
    grid_total = flows["grid_certified_mw"] + flows["grid_noncertified_mw"] + flows["grid_to_battery_mw"]
    source_line_flow = flows["renewable_direct_mw"] + flows["renewable_to_battery_mw"]
    metrics = {
        "certified_hydrogen_kg": float((flows["certified_hydrogen_delivery_kg"] * inputs.dt_hours).sum()),
        "noncertified_hydrogen_kg": float((flows["noncertified_hydrogen_delivery_kg"] * inputs.dt_hours).sum()),
        "renewable_available_mwh": float((renewable_available * inputs.dt_hours).sum()),
        "renewable_used_mwh": float(((flows["renewable_direct_mw"] + flows["renewable_to_battery_mw"]) * inputs.dt_hours).sum()),
        "renewable_curtailment_mwh": float(((renewable_available - flows["renewable_direct_mw"] - flows["renewable_to_battery_mw"]) * inputs.dt_hours).sum()),
        "grid_import_mwh": float((grid_total * inputs.dt_hours).sum()),
        "certified_grid_mwh": float((flows["grid_certified_mw"] * inputs.dt_hours).sum()),
        "low_price_trigger_hours": float(flags["low_price_trigger"].sum()),
        "grid_certified_hours": float((flows["grid_certified_mw"] > 1e-8).sum()),
        "source_line_binding_hours": float((source_line_flow >= arrays["source_line"] - 1e-7).sum()),
        "network_import_binding_hours": float((grid_total >= arrays["network_grid"] - 1e-7).sum()),
        "battery_throughput_mwh": float(((flows["renewable_to_battery_mw"] + flows["grid_to_battery_mw"] + flows["qualified_battery_discharge_mw"] + flows["nonqualified_battery_discharge_mw"]) * inputs.dt_hours).sum()),
        "certified_share_of_hydrogen": float(
            (flows["certified_hydrogen_delivery_kg"] * inputs.dt_hours).sum()
            / max(
                (flows["certified_hydrogen_delivery_kg"] * inputs.dt_hours).sum()
                + (flows["noncertified_hydrogen_delivery_kg"] * inputs.dt_hours).sum(),
                1e-12,
            )
        ),
    }
    metrics["capacity_upper_bound_count"] = float(sum(
        abs(capacities[name] - upper[name]) <= max(1e-7, 1e-7 * upper[name])
        for name in cap_names
    ))
    for name in cap_names:
        metrics[f"{name}_at_upper_bound"] = float(
            abs(capacities[name] - upper[name]) <= max(1e-7, 1e-7 * upper[name])
        )
    if branch_count:
        branch_flows = arrays["network_base"] + grid_total[:, None] * arrays["network_ptdf"][None, :]
        metrics["ptdf_branch_binding_hours"] = float(
            np.any(np.abs(branch_flows) >= arrays["network_capacity"][None, :] - 1e-7, axis=1).sum()
        )
    if certified_hydrogen_target_kg is not None:
        metrics["lcoh_eur_per_kg"] = float(solution.fun / max(certified_hydrogen_target_kg, 1e-12))
    flows["source_line_flow_mw"] = source_line_flow
    flows["network_import_mw"] = grid_total
    flows["renewable_available_mw"] = renewable_available
    return JointDispatchResult(
        status=int(solution.status),
        message=str(solution.message),
        objective_eur=float(solution.fun),
        capacities=capacities,
        metrics=metrics,
        flags=flags,
        flows=flows,
    )
