"""Shared, auditable utilities for the IJGE engineering screen.

The module keeps the experimental runners deliberately small.  Every LP is
solved in a fresh Python process through ``run_esb_upgrade_case.py``; this
avoids carrying HiGHS/SciPy memory from one annual solve into another.  It is
not a legal RFNBO compliance engine or a physical network model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from project01_joint_model import JointModelCosts, JointModelInputs, solve_joint_sizing_dispatch


PROJECT = Path(os.environ.get("PROJECT01_ROOT", Path(__file__).resolve().parents[1]))
SRC = Path(os.environ.get("PROJECT01_CODE_DIR", Path(__file__).resolve().parent))
UPGRADE_ROOT = Path(
    os.environ.get(
        "PROJECT01_UPGRADE_ROOT",
        str(PROJECT / "results" / "esb_upgrade_20260904"),
    )
)

CASE_META: dict[str, dict[str, object]] = {
    "no_trigger": {
        "price_trigger_mode": "none",
        "grid_pathway_valid": False,
        "label": "renewable only baseline",
    },
    "negative_only_proxy": {
        "price_trigger_mode": "negative_only",
        "grid_pathway_valid": True,
        "label": "negative price pathway proxy",
    },
    "price20_only_proxy": {
        "price_trigger_mode": "price20_only",
        "grid_pathway_valid": True,
        "label": "EUR 20 per MWh branch proxy",
    },
    "eua_only_proxy": {
        "price_trigger_mode": "eua_only",
        "grid_pathway_valid": True,
        "label": "EUA linked branch proxy",
    },
    "article6_price_proxy": {
        "price_trigger_mode": "exact",
        "grid_pathway_valid": True,
        "label": "joint RFNBO price proxy",
    },
}

CAPACITY_UPPER_BOUNDS = {
    "wind_mw": 500.0,
    "solar_mw": 500.0,
    "battery_power_mw": 500.0,
    "battery_energy_mwh": 2_000.0,
    "electrolyser_mw": 500.0,
    "hydrogen_storage_kg": 500_000.0,
}

REQUIRED_CACHE_COLUMNS = (
    "timestamp_utc",
    "wind_cf",
    "solar_cf",
    "day_ahead_price_eur_per_mwh",
    "eua_price_eur_tco2",
)


def json_default(value: object) -> object:
    """Serialize pathlib and NumPy values without silently stringifying arrays."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    """Hash the implementation files that determine a solve result."""

    files = [
        SRC / "project01_joint_model.py",
        SRC / "project01_model.py",
        SRC / "esb_upgrade_core.py",
        SRC / "run_esb_upgrade_case.py",
        SRC / "uncertainty.py",
    ]
    return {path.name: sha256_file(path) for path in files if path.exists()}


def load_hourly_cache(path: Path) -> pd.DataFrame:
    """Read an already-derived public cache and enforce its UTC data contract."""

    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, parse_dates=["timestamp_utc"])
    missing = [column for column in REQUIRED_CACHE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if data.empty:
        raise ValueError(f"{path.name} is empty")
    timestamps = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    data = data.copy()
    data["timestamp_utc"] = timestamps
    data = data.sort_values("timestamp_utc").reset_index(drop=True)
    if data["timestamp_utc"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate UTC timestamps")
    for column in REQUIRED_CACHE_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains non-finite {column}")
    for column in ("wind_cf", "solar_cf"):
        values = data[column].to_numpy(dtype=float)
        if (values < 0).any() or (values > 1).any():
            raise ValueError(f"{path.name} has {column} outside [0, 1]")
    return data


def profile_shift(values: np.ndarray, shift_positions: int) -> np.ndarray:
    """Apply a reproducible circular shift on the retained UTC sequence.

    Some public cells have a small number of missing paired UTC hours.  We do
    not impute them.  The shift therefore acts on the retained, ordered UTC
    sequence and records both the requested nominal hour offset and the actual
    index offset in every result record.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("profile_shift requires a nonempty vector")
    return np.roll(values, int(shift_positions))


def _draw_value(draw: Mapping[str, object] | None, name: str, default: float) -> float:
    if draw is None:
        return default
    return float(draw.get(name, default))


def build_costs(n_hours: int, draw: Mapping[str, object] | None = None) -> JointModelCosts:
    """Build the fixed primary cost base or one declared bounded stress draw."""

    if n_hours <= 0:
        raise ValueError("n_hours must be positive")
    capex_factor = _draw_value(draw, "capex_factor", 1.0)
    yield_factor = _draw_value(draw, "yield_factor", 1.0)
    grid_fee = _draw_value(draw, "grid_fee_eur_per_mwh", 3.0)
    return JointModelCosts(
        wind_capex_eur_per_mw_year=120_000.0 * capex_factor,
        solar_capex_eur_per_mw_year=85_000.0 * capex_factor,
        battery_power_capex_eur_per_mw_year=45_000.0 * capex_factor,
        battery_energy_capex_eur_per_mwh_year=12_000.0 * capex_factor,
        electrolyser_capex_eur_per_mw_year=100_000.0 * capex_factor,
        hydrogen_storage_capex_eur_per_kg_year=2.0 * capex_factor,
        renewable_variable_cost_eur_per_mwh=1.0,
        grid_fee_eur_per_mwh=grid_fee,
        battery_degradation_eur_per_mwh=2.0,
        hydrogen_yield_kg_per_mwh=20.0 * yield_factor,
        certified_hydrogen_value_eur_per_kg=7.0,
        noncertified_hydrogen_value_eur_per_kg=0.0,
        eta_battery_charge=_draw_value(draw, "eta_battery_charge", 0.90),
        eta_battery_discharge=_draw_value(draw, "eta_battery_discharge", 0.90),
        eta_hydrogen_charge=_draw_value(draw, "eta_hydrogen_charge", 0.995),
        eta_hydrogen_discharge=_draw_value(draw, "eta_hydrogen_discharge", 0.995),
        horizon_years=n_hours / 8760.0,
    )


def _finite_max_excess(values: np.ndarray, limit: float | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    limit_array = np.asarray(limit, dtype=float)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.maximum(values - limit_array, 0.0).max(initial=0.0))


def solve_public_cache_case(spec: Mapping[str, object]) -> dict[str, object]:
    """Solve one explicitly specified case and emit all auditable diagnostics."""

    cache = Path(str(spec["cache"]))
    case = str(spec["case"])
    if case not in CASE_META:
        raise ValueError(f"unknown case {case}")
    data = load_hourly_cache(cache)
    n = len(data)
    draw_raw = spec.get("draw")
    draw: dict[str, object] | None = None
    if draw_raw is not None:
        if not isinstance(draw_raw, Mapping):
            raise TypeError("draw must be an object when supplied")
        draw = dict(draw_raw)
    shift_positions = int(spec.get("profile_shift_positions", 0))
    requested_shift_hours = int(spec.get("profile_shift_requested_hours", shift_positions))
    wind = profile_shift(data["wind_cf"].to_numpy(dtype=float), shift_positions)
    solar = profile_shift(data["solar_cf"].to_numpy(dtype=float), shift_positions)
    prices = data["day_ahead_price_eur_per_mwh"].to_numpy(dtype=float)
    eua_factor = _draw_value(draw, "eua_price_factor", 1.0)
    eua = data["eua_price_eur_tco2"].to_numpy(dtype=float) * eua_factor
    meta = CASE_META[case]
    costs = build_costs(n, draw)
    target = float(spec.get("target_certified_hydrogen_kg", 10_000_000.0 * costs.horizon_years))
    grid_cap = float(spec.get("grid_cap_mw", 1_000.0))
    source_cap = float(spec.get("source_cap_mw", 250.0))
    if not (math.isfinite(grid_cap) and grid_cap >= 0 and math.isfinite(source_cap) and source_cap >= 0):
        raise ValueError("engineering caps must be finite and nonnegative")
    inputs = JointModelInputs(
        timestamps_utc=data["timestamp_utc"].to_list(),
        wind_capacity_factor=wind,
        solar_capacity_factor=solar,
        day_ahead_price_eur_per_mwh=prices,
        eua_price_eur_per_tco2=eua,
        eligible_grid_capacity_mw=np.full(n, grid_cap),
        network_import_capacity_mw=np.full(n, grid_cap),
        source_line_capacity_mw=np.full(n, source_cap),
        additionality_ok=True,
        enforce_additionality_energy_balance=True,
        geography_ok=True,
        grid_pathway_valid=bool(meta["grid_pathway_valid"]),
        price_trigger_mode=str(meta["price_trigger_mode"]),
        rule_year=int(spec.get("rule_year", 2030)),
        dt_hours=1.0,
    )
    result = solve_joint_sizing_dispatch(
        inputs,
        costs,
        capacity_upper_bounds=CAPACITY_UPPER_BOUNDS,
        certified_hydrogen_target_kg=target,
        noncertified_hydrogen_target_kg=0.0,
    )
    exact_trigger = (prices <= 20.0) | (prices < 0.36 * eua)
    profile_proxy = wind + solar
    trigger_share = float(exact_trigger.mean())
    profile_trigger_share = float(profile_proxy[exact_trigger].sum() / max(profile_proxy.sum(), 1e-12))
    overlap_ratio = float(profile_trigger_share / trigger_share) if trigger_share > 0 else float("nan")
    record: dict[str, object] = {
        "scenario_id": str(spec["scenario_id"]),
        "experiment": str(spec["experiment"]),
        "zone": str(spec["zone"]),
        "data_year": int(spec["data_year"]),
        "profile_variant": str(spec["profile_variant"]),
        "cache": str(cache),
        "cache_sha256": sha256_file(cache),
        "case": case,
        "case_label": str(meta["label"]),
        "status": int(result.status),
        "message": result.message,
        "rule_year": int(spec.get("rule_year", 2030)),
        "hours": n,
        "represented_hours": n,
        "horizon_years": float(costs.horizon_years or n / 8760.0),
        "timestamp_min_utc": data["timestamp_utc"].iloc[0].isoformat(),
        "timestamp_max_utc": data["timestamp_utc"].iloc[-1].isoformat(),
        "timestamps_strictly_utc": bool(str(data["timestamp_utc"].dt.tz) == "UTC"),
        "profile_shift_positions": shift_positions,
        "profile_shift_requested_hours": requested_shift_hours,
        "grid_cap_mw": grid_cap,
        "source_cap_mw": source_cap,
        "target_certified_hydrogen_kg": target,
        "objective_eur": float(result.objective_eur),
        "eua_price_factor": eua_factor,
        "capex_factor": _draw_value(draw, "capex_factor", 1.0),
        "yield_factor": _draw_value(draw, "yield_factor", 1.0),
        "grid_fee_eur_per_mwh": _draw_value(draw, "grid_fee_eur_per_mwh", 3.0),
        "eta_battery_charge": _draw_value(draw, "eta_battery_charge", 0.90),
        "eta_battery_discharge": _draw_value(draw, "eta_battery_discharge", 0.90),
        "eta_hydrogen_charge": _draw_value(draw, "eta_hydrogen_charge", 0.995),
        "eta_hydrogen_discharge": _draw_value(draw, "eta_hydrogen_discharge", 0.995),
        "draw_id": int(draw["draw_id"]) if draw is not None and "draw_id" in draw else -1,
        "draw_design": str(draw.get("design", "base")) if draw is not None else "base",
        "exact_trigger_hour_share": trigger_share,
        "renewable_profile_in_trigger_share": profile_trigger_share,
        "renewable_price_overlap_ratio": overlap_ratio,
        "grid_pathway_valid_assumption": bool(meta["grid_pathway_valid"]),
        "additionality_energy_balance_assumption": True,
        "network_evidence": "none; constant synthetic engineering import cap, not observed congestion",
        "boundary": "selected European public-data engineering screen; not legal RFNBO certification, project-level congestion, real-time operation, or EU-27 prevalence",
    }
    for field in (
        "source_record_id",
        "source_coverage_ratio",
        "source_expected_hours",
        "source_dropped_incomplete_hours",
        "forecast_provenance",
    ):
        if field in spec:
            record[field] = spec[field]
    record.update({f"capacity_{name}": float(value) for name, value in result.capacities.items()})
    record.update({name: float(value) for name, value in result.metrics.items()})
    for name, upper in CAPACITY_UPPER_BOUNDS.items():
        value = float(result.capacities.get(name, float("nan")))
        record[f"capacity_upper_hit_{name}"] = bool(np.isfinite(value) and abs(value - upper) <= 1e-7)

    if result.status == 0:
        flows = result.flows
        record.update(
            {
                "target_error_kg": abs(float(result.metrics["certified_hydrogen_kg"]) - target),
                "max_grid_import_excess_mw": _finite_max_excess(flows["network_import_mw"], grid_cap),
                "max_source_line_excess_mw": _finite_max_excess(flows["source_line_flow_mw"], source_cap),
                "max_eligible_grid_excess_mw": _finite_max_excess(flows["grid_certified_mw"], result.flags["eligible_grid_mw"]),
                "terminal_battery_soc_abs_mwh": float(
                    max(abs(flows["qualified_battery_soc_mwh"][-1]), abs(flows["nonqualified_battery_soc_mwh"][-1]))
                ),
                "terminal_hydrogen_soc_abs_kg": float(
                    max(abs(flows["certified_hydrogen_soc_kg"][-1]), abs(flows["noncertified_hydrogen_soc_kg"][-1]))
                ),
                "renewable_quantity_guard_margin_mwh": float(
                    result.metrics["renewable_available_mwh"] - result.metrics["certified_grid_mwh"]
                ),
            }
        )
    else:
        for name in (
            "target_error_kg",
            "max_grid_import_excess_mw",
            "max_source_line_excess_mw",
            "max_eligible_grid_excess_mw",
            "terminal_battery_soc_abs_mwh",
            "terminal_hydrogen_soc_abs_kg",
            "renewable_quantity_guard_margin_mwh",
        ):
            record[name] = float("nan")
    return record


def write_job_specs(
    jobs: Iterable[Mapping[str, object]],
    spec_dir: Path,
    result_dir: Path,
) -> list[tuple[Path, Path, str]]:
    """Persist immutable per-solve specifications before dispatching workers."""

    spec_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, Path, str]] = []
    for raw in jobs:
        spec = dict(raw)
        scenario_id = str(spec["scenario_id"])
        spec_path = spec_dir / f"{scenario_id}.json"
        result_path = result_dir / f"{scenario_id}.json"
        write_json(spec_path, spec)
        entries.append((spec_path, result_path, scenario_id))
    return entries


def run_worker_specs(entries: list[tuple[Path, Path, str]], workers: int) -> list[dict[str, object]]:
    """Run a fixed set of per-solve workers with a bounded process count."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    worker = SRC / "run_esb_upgrade_case.py"
    if not worker.exists():
        raise FileNotFoundError(worker)
    pending = list(entries)
    running: dict[subprocess.Popen[str], tuple[Path, Path, str]] = {}
    failures: list[dict[str, object]] = []
    completed = 0
    total = len(entries)
    started = time.time()
    while pending or running:
        while pending and len(running) < workers:
            spec_path, result_path, scenario_id = pending.pop(0)
            command = [sys.executable, str(worker), "--spec", str(spec_path), "--output", str(result_path)]
            process = subprocess.Popen(
                command,
                cwd=str(SRC),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            running[process] = (spec_path, result_path, scenario_id)
        if not running:
            continue
        time.sleep(0.2)
        for process, (_, result_path, scenario_id) in list(running.items()):
            if process.poll() is None:
                continue
            stdout, stderr = process.communicate()
            completed += 1
            running.pop(process)
            if process.returncode != 0 or not result_path.exists():
                failures.append(
                    {
                        "scenario_id": scenario_id,
                        "returncode": process.returncode,
                        "stdout_tail": stdout[-2000:],
                        "stderr_tail": stderr[-4000:],
                    }
                )
                status = "FAILED"
            else:
                status = "OK"
            elapsed = time.time() - started
            print(json.dumps({"progress": f"{completed}/{total}", "scenario_id": scenario_id, "status": status, "elapsed_s": round(elapsed, 1)}), flush=True)
    if failures:
        raise RuntimeError(json.dumps({"worker_failures": failures}, indent=2))
    records = [json.loads(result_path.read_text(encoding="utf-8")) for _, result_path, _ in entries]
    return records


def require_technical_validity(frame: pd.DataFrame, *, target_tolerance_kg: float = 1e-4) -> None:
    """Fail loudly on solver or engineering-ledger violations; never drop them."""

    if frame.empty:
        raise RuntimeError("no records produced")
    if not frame["status"].eq(0).all():
        bad = frame.loc[~frame["status"].eq(0), ["scenario_id", "status", "message"]]
        raise RuntimeError(f"nonzero solver status:\n{bad.to_string(index=False)}")
    checks = {
        "target_error_kg": target_tolerance_kg,
        "max_grid_import_excess_mw": 1e-5,
        "max_source_line_excess_mw": 1e-5,
        "max_eligible_grid_excess_mw": 1e-5,
        "terminal_battery_soc_abs_mwh": 1e-5,
        "terminal_hydrogen_soc_abs_kg": 1e-5,
    }
    for column, limit in checks.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values > limit).any():
            bad = frame.loc[values.isna() | (values > limit), ["scenario_id", column]]
            raise RuntimeError(f"technical check failed for {column} > {limit}:\n{bad.to_string(index=False)}")
    if not frame["timestamps_strictly_utc"].astype(bool).all():
        raise RuntimeError("at least one record failed the UTC timestamp check")
    if frame["cache_sha256"].astype(str).str.len().ne(64).any():
        raise RuntimeError("at least one result lacks a valid cache SHA256")


def pair_cases(frame: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    """Pivot all named cases into a comparison table with explicit savings."""

    lcoh = frame.pivot_table(index=index, columns="case", values="lcoh_eur_per_kg", aggfunc="first")
    required = {"no_trigger", "article6_price_proxy"}
    missing = required - set(lcoh.columns)
    if missing:
        raise RuntimeError(f"comparison lacks required cases: {sorted(missing)}")
    result = lcoh.reset_index()
    result["joint_saving_pct"] = 100.0 * (result["no_trigger"] - result["article6_price_proxy"]) / result["no_trigger"]
    for capacity in ("wind_mw", "solar_mw", "battery_power_mw", "battery_energy_mwh", "electrolyser_mw", "hydrogen_storage_kg"):
        pivot = frame.pivot_table(index=index, columns="case", values=f"capacity_{capacity}", aggfunc="first")
        if {"no_trigger", "article6_price_proxy"}.issubset(pivot.columns):
            result[f"joint_delta_{capacity}"] = (pivot["article6_price_proxy"] - pivot["no_trigger"]).to_numpy(dtype=float)
    return result
