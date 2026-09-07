"""E1: hold prices fixed and circularly shift renewable profiles in each cell.

This is a mechanism falsification experiment.  It preserves each cache's
price/EUA distributions and trigger count while changing their temporal
coincidence with the retained wind and solar profiles.  It is a synthetic
engineering counterfactual, never an observed policy intervention.
"""

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


CACHE_DIR = Path(os.environ.get("PROJECT01_PRIMARY_CACHE", str(PROJECT / "datasets" / "public_eu_2019")))
PRIMARY_COMPARISON = Path(
    os.environ.get(
        "PROJECT01_PRIMARY_COMPARISON",
        str(PROJECT / "results" / "public_eu_2019_hourly_2030" / "hourly_2030_baseline_exact.csv"),
    )
)
SHIFTS = (0, 2_190, 4_380, 6_570)
CASES = ("no_trigger", "article6_price_proxy")


def cell_label(cache: Path) -> str:
    return cache.name.removesuffix("_2019.csv.gz")


def make_jobs(preflight: bool) -> list[dict[str, object]]:
    caches = sorted(CACHE_DIR.glob("*_2019.csv.gz"))
    if len(caches) != 18:
        raise RuntimeError(f"expected 18 primary caches, found {len(caches)}")
    jobs: list[dict[str, object]] = []
    for cache in caches:
        zone = cell_label(cache)
        for shift in SHIFTS:
            for case in CASES:
                if preflight and not (zone == "SK" and shift == 0):
                    continue
                scenario_id = f"e1_{zone}_shift{shift:04d}_{case}"
                jobs.append(
                    {
                        "scenario_id": scenario_id,
                        "experiment": "E1_temporal_phase_shift",
                        "cache": str(cache),
                        "zone": zone,
                        "data_year": 2019,
                        "profile_variant": "energy_charts_country_generation_q995",
                        "case": case,
                        "rule_year": 2030,
                        "profile_shift_positions": shift,
                        "profile_shift_requested_hours": shift,
                        "grid_cap_mw": 1_000.0,
                        "source_cap_mw": 250.0,
                    }
                )
    expected = 2 if preflight else 18 * len(SHIFTS) * len(CASES)
    if len(jobs) != expected:
        raise RuntimeError(f"expected {expected} E1 jobs, built {len(jobs)}")
    return jobs


def descriptive_within_cell_slope(comparison: pd.DataFrame) -> float:
    """A descriptive, no-p-value within-cell timing-response slope."""

    numerator = 0.0
    denominator = 0.0
    for _, group in comparison.groupby("zone", sort=False):
        x = group["renewable_price_overlap_ratio"].to_numpy(dtype=float)
        y = group["joint_saving_pct"].to_numpy(dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        numerator += float(np.dot(x, y))
        denominator += float(np.dot(x, x))
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    comparison = pair_cases(frame, ["zone", "profile_shift_requested_hours"])
    context = (
        frame.loc[frame["case"].eq("article6_price_proxy"), [
            "zone", "profile_shift_requested_hours", "renewable_price_overlap_ratio",
            "renewable_profile_in_trigger_share", "exact_trigger_hour_share",
        ]]
        .drop_duplicates()
    )
    comparison = comparison.merge(
        context,
        on=["zone", "profile_shift_requested_hours"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for zone, group in comparison.groupby("zone", sort=True):
        rows.append(
            {
                "zone": zone,
                "shift_cases": int(len(group)),
                "saving_min_pct": float(group["joint_saving_pct"].min()),
                "saving_max_pct": float(group["joint_saving_pct"].max()),
                "saving_range_pp": float(group["joint_saving_pct"].max() - group["joint_saving_pct"].min()),
                "overlap_ratio_min": float(group["renewable_price_overlap_ratio"].min()),
                "overlap_ratio_max": float(group["renewable_price_overlap_ratio"].max()),
                "overlap_ratio_range": float(group["renewable_price_overlap_ratio"].max() - group["renewable_price_overlap_ratio"].min()),
                "solar_delta_range_mw": float(group["joint_delta_solar_mw"].max() - group["joint_delta_solar_mw"].min()),
                "wind_delta_range_mw": float(group["joint_delta_wind_mw"].max() - group["joint_delta_wind_mw"].min()),
                "electrolyser_delta_range_mw": float(group["joint_delta_electrolyser_mw"].max() - group["joint_delta_electrolyser_mw"].min()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)
    indicators = {
        "cells": int(summary.shape[0]),
        "shift_positions": list(SHIFTS),
        "median_saving_range_pp": float(summary["saving_range_pp"].median()),
        "maximum_saving_range_pp": float(summary["saving_range_pp"].max()),
        "cells_saving_range_at_least_0_25pp": int((summary["saving_range_pp"] >= 0.25).sum()),
        "median_overlap_ratio_range": float(summary["overlap_ratio_range"].median()),
        "descriptive_within_cell_slope_pp_per_overlap_ratio": descriptive_within_cell_slope(comparison),
        "interpretation_rule": (
            "No inferential test is used. The manuscript may call temporal coincidence a supported "
            "mechanism only if the executed shift contrasts show material engineering variation and the "
            "predefined overlap indicator co-varies transparently with that variation; otherwise it is a boundary diagnostic."
        ),
    }
    return comparison, summary, indicators


def zero_shift_reproduction(comparison: pd.DataFrame, expected_cells: int) -> dict[str, object]:
    old = pd.read_csv(PRIMARY_COMPARISON).rename(columns={"baseline_lcoh": "old_baseline_lcoh", "exact_lcoh": "old_joint_lcoh"})
    current = comparison.loc[comparison["profile_shift_requested_hours"].eq(0), ["zone", "no_trigger", "article6_price_proxy"]]
    joined = old.merge(current, on="zone", how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("zero-shift reproduction could not join the prior primary results")
    baseline_error = (joined["old_baseline_lcoh"] - joined["no_trigger"]).abs()
    joint_error = (joined["old_joint_lcoh"] - joined["article6_price_proxy"]).abs()
    report = {
        "joined_cells": int(len(joined)),
        "maximum_absolute_baseline_lcoh_error_eur_per_kg": float(baseline_error.max()),
        "maximum_absolute_joint_lcoh_error_eur_per_kg": float(joint_error.max()),
        "tolerance_eur_per_kg": 1e-7,
    }
    if report["joined_cells"] != expected_cells or max(baseline_error.max(), joint_error.max()) > 1e-7:
        raise RuntimeError(f"zero-shift reproduction failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=UPGRADE_ROOT / "phase_shift")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.out / ("preflight" if args.preflight else "full")
    jobs = make_jobs(args.preflight)
    write_json(root / "experiment_design.json", {
        "experiment": "E1 temporal phase shift",
        "preflight": args.preflight,
        "jobs": jobs,
        "fixed_components": ["day-ahead price", "EUA proxy", "targets", "engineering caps", "cost base"],
        "varied_component": "joint wind and solar profile timing on the retained UTC sequence",
        "shifts_requested_hours": list(SHIFTS),
        "code_hashes": code_hashes(),
    })
    entries = write_job_specs(jobs, root / "specs", root / "cases")
    records = run_worker_specs(entries, args.workers)
    frame = pd.DataFrame(records).sort_values(["zone", "profile_shift_requested_hours", "case"]).reset_index(drop=True)
    frame.to_csv(root / "phase_shift_results.csv", index=False, float_format="%.10g")
    require_technical_validity(frame)
    comparison, summary, indicators = summarize(frame)
    comparison.to_csv(root / "phase_shift_comparison.csv", index=False, float_format="%.10g")
    summary.to_csv(root / "phase_shift_summary.csv", index=False, float_format="%.10g")
    reproduction = zero_shift_reproduction(comparison, expected_cells=1 if args.preflight else 18)
    manifest = {
        "status": "PREFLIGHT_COMPLETE" if args.preflight else "E1_COMPLETE",
        "experiment": "E1 temporal phase-shift mechanism falsification",
        "records": int(len(frame)),
        "expected_records": len(jobs),
        "all_solver_status_zero": bool(frame["status"].eq(0).all()),
        "zero_shift_reproduction": reproduction,
        "mechanism_indicators": indicators,
        "data_boundary": "Circular shifts are synthetic and operate on retained paired UTC observations; missing observations were not imputed.",
        "outputs": {
            "records": str(root / "phase_shift_results.csv"),
            "comparison": str(root / "phase_shift_comparison.csv"),
            "summary": str(root / "phase_shift_summary.csv"),
        },
    }
    write_json(root / "phase_shift_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
