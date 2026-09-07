"""E2: one-hour cross-period and published forecast-profile sensitivity runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


MANIFEST_PATH = Path(
    os.environ.get(
        "PROJECT01_CROSS_PERIOD_MANIFEST",
        str(PROJECT / "datasets" / "public_cross_period" / "public_cross_period_manifest.json"),
    )
)
CASES = ("no_trigger", "negative_only_proxy", "price20_only_proxy", "eua_only_proxy", "article6_price_proxy")
MIN_COVERAGE = 0.99


def selected_records() -> list[dict[str, object]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = list(manifest["records"])
    selected: list[dict[str, object]] = []
    for record in records:
        hours = int(record["hours"])
        expected = int(record["expected_hours"])
        coverage = hours / expected
        if coverage < MIN_COVERAGE:
            continue
        item = dict(record)
        item["coverage_ratio"] = coverage
        selected.append(item)
    if len(selected) != 14:
        raise RuntimeError(f"expected all 14 retained records to meet >=99% coverage, found {len(selected)}")
    return selected


def record_id(record: dict[str, object]) -> str:
    zone = str(record["country_or_zone"]).replace("-", "_")
    year = int(record["year"])
    variant = str(record["profile_variant"])
    return f"{zone}_{year}_{variant}"


def make_jobs(preflight: bool) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for record in selected_records():
        source_id = record_id(record)
        if preflight and source_id != "DE_LU_2019_observed_generation":
            continue
        forecast_provenance = (
            "published day-ahead generation profile sensitivity only; issue-time provenance is not available and no forecast-vintage compliance claim is made"
            if str(record["profile_variant"]) == "dayahead_forecast"
            else "observed generation profile"
        )
        for case in CASES:
            jobs.append(
                {
                    "scenario_id": f"e2_{source_id}_{case}",
                    "experiment": "E2_cross_period_and_forecast_profile",
                    "cache": str(record["cache"]),
                    "zone": str(record["country_or_zone"]),
                    "data_year": int(record["year"]),
                    "profile_variant": str(record["profile_variant"]),
                    "case": case,
                    "rule_year": 2030,
                    "grid_cap_mw": 1_000.0,
                    "source_cap_mw": 250.0,
                    "source_record_id": source_id,
                    "source_coverage_ratio": float(record["coverage_ratio"]),
                    "source_expected_hours": int(record["expected_hours"]),
                    "source_dropped_incomplete_hours": int(record["dropped_incomplete_hours"]),
                    "forecast_provenance": forecast_provenance,
                }
            )
    expected = 5 if preflight else 14 * len(CASES)
    if len(jobs) != expected:
        raise RuntimeError(f"expected {expected} E2 jobs, built {len(jobs)}")
    return jobs


def make_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    index = ["source_record_id", "zone", "data_year", "profile_variant"]
    comparison = pair_cases(frame, index)
    return comparison.sort_values(["zone", "data_year", "profile_variant"]).reset_index(drop=True)


def forecast_observed_pairs(comparison: pd.DataFrame) -> pd.DataFrame:
    observed = comparison.loc[comparison["profile_variant"].eq("observed_generation")].copy()
    forecast = comparison.loc[comparison["profile_variant"].eq("dayahead_forecast")].copy()
    key = ["zone", "data_year"]
    keep = key + ["joint_saving_pct", "joint_delta_wind_mw", "joint_delta_solar_mw", "joint_delta_electrolyser_mw"]
    observed = observed[keep].rename(columns={column: f"observed_{column}" for column in keep if column not in key})
    forecast = forecast[keep].rename(columns={column: f"forecast_{column}" for column in keep if column not in key})
    pairs = observed.merge(forecast, on=key, how="inner", validate="one_to_one")
    if not pairs.empty:
        pairs["forecast_minus_observed_saving_pp"] = pairs["forecast_joint_saving_pct"] - pairs["observed_joint_saving_pct"]
        pairs["forecast_minus_observed_solar_delta_mw"] = pairs["forecast_joint_delta_solar_mw"] - pairs["observed_joint_delta_solar_mw"]
    return pairs.sort_values(key).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=UPGRADE_ROOT / "cross_period")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.out / ("preflight" if args.preflight else "full")
    source_records = selected_records()
    jobs = make_jobs(args.preflight)
    write_json(root / "experiment_design.json", {
        "experiment": "E2 cross-period and published forecast-profile sensitivity",
        "preflight": args.preflight,
        "minimum_complete_utc_coverage": MIN_COVERAGE,
        "selected_source_records": source_records,
        "jobs": jobs,
        "fixed_components": ["hourly model", "cost base", "2030 matching-rule scenario", "engineering caps"],
        "varied_components": ["year", "observed generation profile", "published Belgian day-ahead profile"],
        "forecast_boundary": "The Belgian forecast profile has no authenticated issue-time field in the retained public source and is not forecast-vintage validation.",
        "code_hashes": code_hashes(),
    })
    entries = write_job_specs(jobs, root / "specs", root / "cases")
    records = run_worker_specs(entries, args.workers)
    frame = pd.DataFrame(records).sort_values(["source_record_id", "case"]).reset_index(drop=True)
    frame.to_csv(root / "cross_period_results.csv", index=False, float_format="%.10g")
    require_technical_validity(frame)
    comparison = make_comparison(frame)
    comparison.to_csv(root / "cross_period_comparison.csv", index=False, float_format="%.10g")
    pairs = forecast_observed_pairs(comparison)
    pairs.to_csv(root / "forecast_observed_comparison.csv", index=False, float_format="%.10g")
    branch = frame.pivot_table(
        index=["source_record_id", "zone", "data_year", "profile_variant"],
        columns="case",
        values="lcoh_eur_per_kg",
        aggfunc="first",
    ).reset_index()
    if {"article6_price_proxy", "price20_only_proxy"}.issubset(branch.columns):
        branch["joint_minus_price20_eur_per_kg"] = branch["article6_price_proxy"] - branch["price20_only_proxy"]
    branch.to_csv(root / "cross_period_branch_comparison.csv", index=False, float_format="%.10g")
    manifest = {
        "status": "PREFLIGHT_COMPLETE" if args.preflight else "E2_COMPLETE",
        "experiment": "E2 cross-period and published forecast-profile sensitivity",
        "records": int(len(frame)),
        "expected_records": len(jobs),
        "source_cells": int(frame["source_record_id"].nunique()),
        "all_solver_status_zero": bool(frame["status"].eq(0).all()),
        "minimum_source_coverage_ratio": float(frame["source_coverage_ratio"].min()),
        "forecast_profile_cells": int(frame.loc[frame["profile_variant"].eq("dayahead_forecast"), "source_record_id"].nunique()),
        "observed_profile_cells": int(frame.loc[frame["profile_variant"].eq("observed_generation"), "source_record_id"].nunique()),
        "forecast_observed_pairs": int(len(pairs)),
        "joint_saving_min_pct": float(comparison["joint_saving_pct"].min()),
        "joint_saving_max_pct": float(comparison["joint_saving_pct"].max()),
        "joint_saving_median_pct": float(comparison["joint_saving_pct"].median()),
        "forecast_boundary": "Published Belgian day-ahead profile sensitivity only; not authenticated forecast-vintage validation, real-time control evidence, or a forecast-error estimate.",
        "outputs": {
            "records": str(root / "cross_period_results.csv"),
            "comparison": str(root / "cross_period_comparison.csv"),
            "branch_comparison": str(root / "cross_period_branch_comparison.csv"),
            "forecast_observed": str(root / "forecast_observed_comparison.csv"),
        },
    }
    write_json(root / "cross_period_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
