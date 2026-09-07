"""E3: bounded Latin-hypercube engineering stress on the hourly 2019 screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from esb_upgrade_core import (
    PROJECT,
    UPGRADE_ROOT,
    code_hashes,
    pair_cases,
    require_technical_validity,
    run_worker_specs,
    write_job_specs,
    write_json,
)
from uncertainty import bounded_lhs_draws


CACHE_DIR = Path(os.environ.get("PROJECT01_PRIMARY_CACHE", str(PROJECT / "datasets" / "public_eu_2019")))
CASES = ("no_trigger", "article6_price_proxy")


def cell_label(cache: Path) -> str:
    return cache.name.removesuffix("_2019.csv.gz")


def make_jobs(draws: pd.DataFrame, preflight: bool) -> list[dict[str, object]]:
    caches = sorted(CACHE_DIR.glob("*_2019.csv.gz"))
    if len(caches) != 18:
        raise RuntimeError(f"expected 18 primary caches, found {len(caches)}")
    jobs: list[dict[str, object]] = []
    for cache in caches:
        zone = cell_label(cache)
        if preflight and zone != "SK":
            continue
        for _, draw_row in draws.iterrows():
            if preflight and int(draw_row["draw_id"]) not in {0, 1}:
                continue
            draw = {key: value.item() if isinstance(value, np.generic) else value for key, value in draw_row.to_dict().items()}
            for case in CASES:
                scenario_id = f"e3_{zone}_draw{int(draw['draw_id']):02d}_{case}"
                jobs.append(
                    {
                        "scenario_id": scenario_id,
                        "experiment": "E3_bounded_parameter_latin_hypercube",
                        "cache": str(cache),
                        "zone": zone,
                        "data_year": 2019,
                        "profile_variant": "energy_charts_country_generation_q995",
                        "case": case,
                        "rule_year": 2030,
                        "grid_cap_mw": 1_000.0,
                        "source_cap_mw": 250.0,
                        "draw": draw,
                    }
                )
    expected = 4 if preflight else 18 * len(draws) * len(CASES)
    if len(jobs) != expected:
        raise RuntimeError(f"expected {expected} E3 jobs, built {len(jobs)}")
    return jobs


def make_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zone, group in comparison.groupby("zone", sort=True):
        base = group.loc[group["draw_id"].eq(0)]
        if len(base) != 1:
            raise RuntimeError(f"missing unique base draw for {zone}")
        record: dict[str, object] = {
            "zone": zone,
            "draws_including_base": int(len(group)),
            "base_joint_saving_pct": float(base["joint_saving_pct"].iloc[0]),
            "saving_min_pct": float(group["joint_saving_pct"].min()),
            "saving_max_pct": float(group["joint_saving_pct"].max()),
            "saving_range_pp": float(group["joint_saving_pct"].max() - group["joint_saving_pct"].min()),
            "zero_response_draws": int((group["joint_saving_pct"].abs() <= 1e-7).sum()),
            "positive_response_draws": int((group["joint_saving_pct"] > 1e-7).sum()),
        }
        for capacity in ("wind_mw", "solar_mw", "battery_power_mw", "battery_energy_mwh", "electrolyser_mw", "hydrogen_storage_kg"):
            column = f"joint_delta_{capacity}"
            if column in group:
                record[f"{capacity}_delta_min"] = float(group[column].min())
                record[f"{capacity}_delta_max"] = float(group[column].max())
        rows.append(record)
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=UPGRADE_ROOT / "parameter_lhs")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    draws = bounded_lhs_draws(args.draws, seed=args.seed, include_base=True)
    root = args.out / ("preflight" if args.preflight else "full")
    root.mkdir(parents=True, exist_ok=True)
    draws.to_csv(root / "parameter_draws.csv", index=False, float_format="%.10g")
    jobs = make_jobs(draws, args.preflight)
    write_json(root / "experiment_design.json", {
        "experiment": "E3 bounded Latin-hypercube parameter stress",
        "preflight": args.preflight,
        "draws_excluding_base": args.draws,
        "draws_including_base": int(len(draws)),
        "seed": args.seed,
        "parameter_draws": draws.to_dict(orient="records"),
        "jobs": jobs,
        "interpretation": "This is a finite engineering stress box, not a probability model, sampling distribution, confidence interval, or direction-stability claim.",
        "code_hashes": code_hashes(),
    })
    entries = write_job_specs(jobs, root / "specs", root / "cases")
    records = run_worker_specs(entries, args.workers)
    frame = pd.DataFrame(records).sort_values(["zone", "draw_id", "case"]).reset_index(drop=True)
    frame.to_csv(root / "parameter_lhs_results.csv", index=False, float_format="%.10g")
    require_technical_validity(frame)
    comparison = pair_cases(frame, ["zone", "draw_id"])
    draw_columns = ["draw_id", "draw_design", "capex_factor", "yield_factor", "grid_fee_eur_per_mwh", "eua_price_factor", "eta_battery_charge", "eta_battery_discharge", "eta_hydrogen_charge", "eta_hydrogen_discharge"]
    draw_values = frame.loc[frame["case"].eq("article6_price_proxy"), draw_columns].drop_duplicates()
    comparison = comparison.merge(draw_values, on="draw_id", how="left", validate="many_to_one")
    comparison.to_csv(root / "parameter_lhs_comparison.csv", index=False, float_format="%.10g")
    summary = make_summary(comparison)
    summary.to_csv(root / "parameter_lhs_interval_summary.csv", index=False, float_format="%.10g")
    cap_columns = [column for column in frame.columns if column.startswith("capacity_upper_hit_")]
    capacity_hits = {column: int(frame[column].astype(bool).sum()) for column in cap_columns}
    manifest = {
        "status": "PREFLIGHT_COMPLETE" if args.preflight else "E3_COMPLETE",
        "experiment": "E3 bounded Latin-hypercube parameter stress",
        "records": int(len(frame)),
        "expected_records": len(jobs),
        "cells": int(frame["zone"].nunique()),
        "draws_including_base": int(frame["draw_id"].nunique()),
        "seed": args.seed,
        "all_solver_status_zero": bool(frame["status"].eq(0).all()),
        "minimum_saving_pct": float(comparison["joint_saving_pct"].min()),
        "maximum_saving_pct": float(comparison["joint_saving_pct"].max()),
        "capacity_upper_bound_hits": capacity_hits,
        "interpretation": "Reported ranges are finite bounded-stress outcomes only. They are not probability intervals or confidence intervals.",
        "outputs": {
            "draws": str(root / "parameter_draws.csv"),
            "records": str(root / "parameter_lhs_results.csv"),
            "comparison": str(root / "parameter_lhs_comparison.csv"),
            "interval_summary": str(root / "parameter_lhs_interval_summary.csv"),
        },
    }
    write_json(root / "parameter_lhs_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
