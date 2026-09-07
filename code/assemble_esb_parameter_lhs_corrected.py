"""Repair E3's reference point without discarding valid bounded-LHS solves.

The original E3 preparation assigned unit storage efficiencies to draw 0.  That
is outside the declared finite stress box and does not reproduce the primary
engineering configuration.  This utility preserves the already-computed
draws 1--12 only after checking every stored draw value, recomputes the 36
draw-0 solves with the primary 0.90/0.995 efficiencies, and emits a separate,
non-overwriting corrected result directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from esb_upgrade_core import (
    UPGRADE_ROOT,
    code_hashes,
    pair_cases,
    require_technical_validity,
    run_worker_specs,
    write_job_specs,
    write_json,
)
from run_esb_parameter_lhs import make_jobs, make_summary
from uncertainty import bounded_lhs_draws


DRAW_COLUMNS = [
    "draw_id",
    "capex_factor",
    "yield_factor",
    "grid_fee_eur_per_mwh",
    "eua_price_factor",
    "eta_battery_charge",
    "eta_battery_discharge",
    "eta_hydrogen_charge",
    "eta_hydrogen_discharge",
]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_matching_lhs_record(record: dict[str, object], expected: dict[str, object]) -> None:
    if int(record.get("status", -1)) != 0:
        raise RuntimeError(f"Predecessor solve is not optimal: {record.get('scenario_id')}")
    if int(record.get("draw_id", -1)) != int(expected["draw_id"]):
        raise RuntimeError(f"Draw ID mismatch in {record.get('scenario_id')}")
    for column in DRAW_COLUMNS[1:]:
        if not np.isclose(float(record[column]), float(expected[column]), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Draw mismatch for {record.get('scenario_id')}: {column}")
    require_technical_validity(pd.DataFrame([record]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=UPGRADE_ROOT / "parameter_lhs" / "full")
    parser.add_argument("--out", type=Path, default=UPGRADE_ROOT / "parameter_lhs_corrected" / "full")
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    source = args.source
    out = args.out
    predecessor_manifest = source / "parameter_lhs_manifest.json"
    predecessor_results = source / "parameter_lhs_results.csv"
    if not predecessor_manifest.exists() or not predecessor_results.exists():
        raise FileNotFoundError("The predecessor E3 run must be complete before base repair")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite corrected E3 output: {out}")

    predecessor = read_json(predecessor_manifest)
    if predecessor.get("status") != "E3_COMPLETE" or int(predecessor.get("records", -1)) != 468:
        raise RuntimeError("The predecessor E3 run is incomplete")
    old_draws = pd.read_csv(source / "parameter_draws.csv")
    draws = bounded_lhs_draws(args.draws, seed=args.seed, include_base=True)
    if len(draws) != 13 or not np.isclose(float(draws.loc[draws["draw_id"].eq(0), "eta_battery_charge"].iloc[0]), 0.90):
        raise RuntimeError("Corrected base draw was not constructed")
    if len(old_draws) != len(draws):
        raise RuntimeError("Predecessor draw count does not match the declared design")
    # The aggregate CSV was written with ``%.10g`` formatting, so comparing it
    # at machine precision would reject harmless decimal serialization noise.
    # The persisted per-solve specs retain the full precision and are the
    # authoritative design record; validate those exactly, while also checking
    # that the aggregate draw table is a rounded representation of the same
    # design.
    for draw_id in range(1, args.draws + 1):
        old = old_draws.loc[old_draws["draw_id"].eq(draw_id)].iloc[0]
        new = draws.loc[draws["draw_id"].eq(draw_id)].iloc[0]
        spec_candidates = sorted((source / "specs").glob(f"e3_*_draw{draw_id:02d}_article6_price_proxy.json"))
        if not spec_candidates:
            raise FileNotFoundError(f"no persisted spec found for draw {draw_id}")
        spec_draw = read_json(spec_candidates[0])["draw"]
        for column in DRAW_COLUMNS[1:]:
            if not np.isclose(float(spec_draw[column]), float(new[column]), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Persisted LHS spec {draw_id} differs from corrected declaration: {column}")
            if not np.isclose(float(old[column]), float(new[column]), rtol=0.0, atol=1e-8):
                raise RuntimeError(f"Predecessor aggregate LHS draw {draw_id} differs from corrected declaration: {column}")

    jobs = make_jobs(draws, preflight=False)
    entries = write_job_specs(jobs, out / "specs", out / "cases")
    design = {
        "experiment": "E3 bounded Latin-hypercube parameter stress, corrected reference draw",
        "predecessor": str(source),
        "predecessor_status": predecessor.get("status"),
        "reason": "The predecessor's draw 0 had idealized 1.0 storage efficiencies. Draws 1--12 are unchanged and fall inside the predeclared stress bounds; draw 0 is recomputed with the primary 0.90/0.995 efficiencies.",
        "draws_excluding_base": args.draws,
        "draws_including_base": int(len(draws)),
        "seed": args.seed,
        "parameter_draws": draws.to_dict(orient="records"),
        "jobs": jobs,
        "interpretation": "This is a finite engineering stress box, not a probability model, sampling distribution, confidence interval, or direction-stability claim.",
        "code_hashes": code_hashes(),
    }
    write_json(out / "experiment_design.json", design)

    base_entries: list[tuple[Path, Path, str]] = []
    reused = 0
    for spec_path, result_path, scenario_id in entries:
        spec = read_json(spec_path)
        draw = dict(spec["draw"])
        draw_id = int(draw["draw_id"])
        if draw_id == 0:
            base_entries.append((spec_path, result_path, scenario_id))
            continue
        source_result = source / "cases" / f"{scenario_id}.json"
        if not source_result.exists():
            raise FileNotFoundError(source_result)
        record = read_json(source_result)
        require_matching_lhs_record(record, draw)
        shutil.copy2(source_result, result_path)
        reused += 1

    base_records = run_worker_specs(base_entries, args.workers)
    if len(base_records) != 36:
        raise RuntimeError(f"Expected 36 recomputed reference solves, got {len(base_records)}")
    records = [read_json(path) for path in sorted((out / "cases").glob("*.json"))]
    frame = pd.DataFrame(records).sort_values(["zone", "draw_id", "case"]).reset_index(drop=True)
    if len(frame) != 468:
        raise RuntimeError(f"Corrected E3 has {len(frame)} records, not 468")
    require_technical_validity(frame)
    frame.to_csv(out / "parameter_lhs_results.csv", index=False, float_format="%.10g")
    comparison = pair_cases(frame, ["zone", "draw_id"])
    draw_values = frame.loc[frame["case"].eq("article6_price_proxy"), DRAW_COLUMNS].drop_duplicates()
    comparison = comparison.merge(draw_values, on="draw_id", how="left", validate="many_to_one")
    comparison.to_csv(out / "parameter_lhs_comparison.csv", index=False, float_format="%.10g")
    summary = make_summary(comparison)
    summary.to_csv(out / "parameter_lhs_interval_summary.csv", index=False, float_format="%.10g")
    draws.to_csv(out / "parameter_draws.csv", index=False, float_format="%.10g")
    cap_columns = [column for column in frame.columns if column.startswith("capacity_upper_hit_")]
    capacity_hits = {column: int(frame[column].astype(bool).sum()) for column in cap_columns}
    manifest = {
        "status": "E3_CORRECTED_BASE_COMPLETE",
        "experiment": "E3 bounded Latin-hypercube parameter stress, corrected reference draw",
        "records": int(len(frame)),
        "expected_records": 468,
        "cells": int(frame["zone"].nunique()),
        "draws_including_base": int(frame["draw_id"].nunique()),
        "seed": args.seed,
        "all_solver_status_zero": bool(frame["status"].eq(0).all()),
        "reused_predecessor_lhs_results": reused,
        "recomputed_primary_base_results": len(base_records),
        "predecessor": str(source),
        "minimum_saving_pct": float(comparison["joint_saving_pct"].min()),
        "maximum_saving_pct": float(comparison["joint_saving_pct"].max()),
        "capacity_upper_bound_hits": capacity_hits,
        "base_reference": {
            "capex_factor": 1.0,
            "yield_factor": 1.0,
            "grid_fee_eur_per_mwh": 3.0,
            "eta_battery_charge": 0.90,
            "eta_battery_discharge": 0.90,
            "eta_hydrogen_charge": 0.995,
            "eta_hydrogen_discharge": 0.995,
            "eua_price_factor": 1.0,
        },
        "interpretation": "Reported ranges are finite bounded-stress outcomes only. They are not probability intervals or confidence intervals.",
        "outputs": {
            "draws": str(out / "parameter_draws.csv"),
            "records": str(out / "parameter_lhs_results.csv"),
            "comparison": str(out / "parameter_lhs_comparison.csv"),
            "interval_summary": str(out / "parameter_lhs_interval_summary.csv"),
        },
    }
    write_json(out / "parameter_lhs_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
