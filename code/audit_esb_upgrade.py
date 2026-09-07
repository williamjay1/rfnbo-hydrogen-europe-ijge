"""Technical and numerical audit for the three public-data experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esb_upgrade_core import UPGRADE_ROOT, code_hashes, require_technical_validity, write_json


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def within_cell_slope(comparison: pd.DataFrame) -> float:
    numerator = 0.0
    denominator = 0.0
    for _, group in comparison.groupby("zone", sort=False):
        x = group["renewable_price_overlap_ratio"].to_numpy(dtype=float, copy=True)
        y = group["joint_saving_pct"].to_numpy(dtype=float, copy=True)
        x -= x.mean()
        y -= y.mean()
        numerator += float(np.dot(x, y))
        denominator += float(np.dot(x, x))
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=UPGRADE_ROOT)
    args = parser.parse_args()
    root = args.root
    e1_root = root / "phase_shift" / "full"
    e2_root = root / "cross_period" / "full"
    # The preliminary E3 directory is retained for provenance but is not
    # canonical: its central reference used lossless storage efficiencies.
    # The corrected directory retains the validated bounded-LHS draws and
    # recomputes the reference draw with the primary engineering settings.
    e3_root = root / "parameter_lhs_corrected" / "full"
    e1 = read_csv(e1_root / "phase_shift_results.csv")
    e1_comp = read_csv(e1_root / "phase_shift_comparison.csv")
    e1_summary = read_csv(e1_root / "phase_shift_summary.csv")
    e1_manifest = read_json(e1_root / "phase_shift_manifest.json")
    e2 = read_csv(e2_root / "cross_period_results.csv")
    e2_comp = read_csv(e2_root / "cross_period_comparison.csv")
    e2_forecast = read_csv(e2_root / "forecast_observed_comparison.csv")
    e2_manifest = read_json(e2_root / "cross_period_manifest.json")
    e3 = read_csv(e3_root / "parameter_lhs_results.csv")
    e3_comp = read_csv(e3_root / "parameter_lhs_comparison.csv")
    e3_summary = read_csv(e3_root / "parameter_lhs_interval_summary.csv")
    e3_manifest = read_json(e3_root / "parameter_lhs_manifest.json")

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for label, frame in (("E1", e1), ("E2", e2), ("E3", e3)):
        try:
            require_technical_validity(frame)
            check(f"{label} solver and ledger validity", True, {"records": int(len(frame))})
        except Exception as error:  # retain the audit record rather than hiding failures
            check(f"{label} solver and ledger validity", False, str(error))
        check(
            f"{label} UTC and delivery-target checks",
            bool(frame["timestamps_strictly_utc"].astype(bool).all())
            and bool(pd.to_numeric(frame["target_error_kg"], errors="coerce").le(1e-4).all()),
            {
                "all_timestamps_strictly_utc": bool(frame["timestamps_strictly_utc"].astype(bool).all()),
                "maximum_target_error_kg": float(pd.to_numeric(frame["target_error_kg"], errors="coerce").max()),
            },
        )
        physical_columns = [
            "max_grid_import_excess_mw",
            "max_source_line_excess_mw",
            "max_eligible_grid_excess_mw",
            "terminal_battery_soc_abs_mwh",
            "terminal_hydrogen_soc_abs_kg",
        ]
        physical_maxima = {
            column: float(pd.to_numeric(frame[column], errors="coerce").max())
            for column in physical_columns
        }
        check(
            f"{label} energy-balance and capacity-limit checks",
            all(value <= 1e-5 for value in physical_maxima.values()),
            physical_maxima,
        )

    check("E1 solve count", len(e1) == 144, {"actual": int(len(e1)), "expected": 144})
    check("E1 unique scenario IDs", e1["scenario_id"].is_unique, int(e1["scenario_id"].nunique()))
    check("E1 cell coverage", e1["zone"].nunique() == 18 and set(e1["profile_shift_requested_hours"].unique()) == {0, 2190, 4380, 6570}, {
        "cells": int(e1["zone"].nunique()), "shifts": sorted(e1["profile_shift_requested_hours"].unique().tolist()),
    })
    # ``low_price_trigger_hours`` is an optimization outcome (the number of
    # hours actually used by the grid pathway), so it can change after a
    # profile shift.  The exogenous trigger share is the correct invariant.
    trigger_ranges = e1.groupby("zone")["exact_trigger_hour_share"].agg(lambda values: float(values.max() - values.min()))
    trigger_stable = trigger_ranges.le(1e-12).all()
    check("E1 price triggers held fixed within each cell", bool(trigger_stable), trigger_ranges.to_dict())
    check("E1 zero-shift reproduction", bool(e1_manifest["zero_shift_reproduction"]["maximum_absolute_baseline_lcoh_error_eur_per_kg"] <= 1e-7 and e1_manifest["zero_shift_reproduction"]["maximum_absolute_joint_lcoh_error_eur_per_kg"] <= 1e-7), e1_manifest["zero_shift_reproduction"])
    check("E1 comparison rows", len(e1_comp) == 72, {"actual": int(len(e1_comp)), "expected": 72})

    check("E2 solve count", len(e2) == 70, {"actual": int(len(e2)), "expected": 70})
    check("E2 source coverage", e2["source_record_id"].nunique() == 14 and float(e2["source_coverage_ratio"].min()) >= 0.99, {
        "source_cells": int(e2["source_record_id"].nunique()), "minimum_coverage": float(e2["source_coverage_ratio"].min()),
    })
    check("E2 branch completeness", e2.groupby("source_record_id")["case"].nunique().eq(5).all(), e2.groupby("source_record_id")["case"].nunique().to_dict())
    check("E2 forecast labels", e2.loc[e2["profile_variant"].eq("dayahead_forecast"), "forecast_provenance"].astype(str).str.contains("no forecast-vintage", regex=False).all(), "forecast profiles retain the non-vintage boundary")
    check("E2 forecast-observed pair coverage", len(e2_forecast) == 5, {"actual": int(len(e2_forecast)), "expected": 5})

    check("E3 solve count", len(e3) == 468, {"actual": int(len(e3)), "expected": 468})
    check("E3 corrected-reference manifest", e3_manifest.get("status") == "E3_CORRECTED_BASE_COMPLETE", e3_manifest.get("status"))
    check("E3 cells and deterministic draws", e3["zone"].nunique() == 18 and e3["draw_id"].nunique() == 13, {
        "cells": int(e3["zone"].nunique()), "draws": int(e3["draw_id"].nunique()),
    })
    check("E3 paired cases", e3.groupby(["zone", "draw_id"])["case"].nunique().eq(2).all(), int(e3.groupby(["zone", "draw_id"])["case"].nunique().min()))
    check("E3 uses finite ranges not probability labels", not any("p05" in col.lower() or "p95" in col.lower() for col in e3_summary.columns), list(e3_summary.columns))
    base_e3 = e3.loc[e3["draw_id"].eq(0), ["zone", "case", "lcoh_eur_per_kg"]].rename(columns={"lcoh_eur_per_kg": "e3_base_lcoh"})
    base_e1 = e1.loc[e1["profile_shift_requested_hours"].eq(0), ["zone", "case", "lcoh_eur_per_kg"]].rename(columns={"lcoh_eur_per_kg": "e1_primary_lcoh"})
    base_reproduction = base_e3.merge(base_e1, on=["zone", "case"], how="inner", validate="one_to_one")
    base_error = float((base_reproduction["e3_base_lcoh"] - base_reproduction["e1_primary_lcoh"]).abs().max())
    check("E3 reference reproduces primary configuration", len(base_reproduction) == 36 and base_error <= 1e-7, {"rows": int(len(base_reproduction)), "maximum_absolute_lcoh_error_eur_per_kg": base_error})
    corrected_reference = e3_manifest.get("base_reference", {})
    expected_reference = {
        "capex_factor": 1.0,
        "yield_factor": 1.0,
        "grid_fee_eur_per_mwh": 3.0,
        "eta_battery_charge": 0.90,
        "eta_battery_discharge": 0.90,
        "eta_hydrogen_charge": 0.995,
        "eta_hydrogen_discharge": 0.995,
        "eua_price_factor": 1.0,
    }
    check("E3 reference parameters match primary settings", all(np.isclose(float(corrected_reference.get(key, np.nan)), value, rtol=0.0, atol=1e-12) for key, value in expected_reference.items()), corrected_reference)

    all_frames = pd.concat([e1, e2, e3], ignore_index=True)
    capacity_columns = [column for column in all_frames.columns if column.startswith("capacity_upper_hit_")]
    capacity_hits = {column: int(all_frames[column].fillna(False).astype(bool).sum()) for column in capacity_columns}
    check("all cache hashes valid", all_frames["cache_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all(), int(all_frames["cache_sha256"].nunique()))

    technical_pass = all(bool(item["passed"]) for item in checks)
    e1_mechanism = {
        "median_within_cell_saving_range_pp": float(e1_summary["saving_range_pp"].median()),
        "maximum_within_cell_saving_range_pp": float(e1_summary["saving_range_pp"].max()),
        "cells_with_saving_range_at_least_0_25pp": int((e1_summary["saving_range_pp"] >= 0.25).sum()),
        "median_overlap_ratio_range": float(e1_summary["overlap_ratio_range"].median()),
        "descriptive_within_cell_slope_pp_per_overlap_ratio": within_cell_slope(e1_comp),
        "decision_rule": "No automatic scientific conclusion is assigned. The manuscript must call temporal coincidence a mechanism only to the extent supported by the full displayed shift contrasts and overlap diagnostic, without inferential p-values.",
    }
    e2_summary = {
        "source_cells": int(e2_comp["source_record_id"].nunique()),
        "minimum_source_coverage_ratio": float(e2["source_coverage_ratio"].min()),
        "joint_saving_min_pct": float(e2_comp["joint_saving_pct"].min()),
        "joint_saving_median_pct": float(e2_comp["joint_saving_pct"].median()),
        "joint_saving_max_pct": float(e2_comp["joint_saving_pct"].max()),
        "forecast_observed_pairs": int(len(e2_forecast)),
        "forecast_minus_observed_saving_min_pp": float(e2_forecast["forecast_minus_observed_saving_pp"].min()),
        "forecast_minus_observed_saving_max_pp": float(e2_forecast["forecast_minus_observed_saving_pp"].max()),
    }
    e3_summary_values = {
        "cells": int(len(e3_summary)),
        "saving_min_pct": float(e3_summary["saving_min_pct"].min()),
        "saving_max_pct": float(e3_summary["saving_max_pct"].max()),
        "maximum_within_cell_saving_range_pp": float(e3_summary["saving_range_pp"].max()),
        "total_zero_response_draws": int(e3_summary["zero_response_draws"].sum()),
        "total_positive_response_draws": int(e3_summary["positive_response_draws"].sum()),
    }
    ledger_rows = [
        {"claim_id": "E1-COUNT", "claim": "E1 executed LP solves", "value": len(e1), "unit": "solves", "source": str(e1_root / "phase_shift_results.csv")},
        {"claim_id": "E1-REPRO", "claim": "Maximum zero-shift LCOH reproduction error", "value": max(e1_manifest["zero_shift_reproduction"]["maximum_absolute_baseline_lcoh_error_eur_per_kg"], e1_manifest["zero_shift_reproduction"]["maximum_absolute_joint_lcoh_error_eur_per_kg"]), "unit": "EUR per kg", "source": str(e1_root / "phase_shift_manifest.json")},
        {"claim_id": "E1-RANGE", "claim": "Median within-cell phase-shift saving range", "value": e1_mechanism["median_within_cell_saving_range_pp"], "unit": "percentage points", "source": str(e1_root / "phase_shift_summary.csv")},
        {"claim_id": "E2-COUNT", "claim": "E2 executed LP solves", "value": len(e2), "unit": "solves", "source": str(e2_root / "cross_period_results.csv")},
        {"claim_id": "E2-CELLS", "claim": "E2 eligible source profile-year cells", "value": e2_summary["source_cells"], "unit": "cells", "source": str(e2_root / "cross_period_comparison.csv")},
        {"claim_id": "E3-COUNT", "claim": "E3 executed LP solves", "value": len(e3), "unit": "solves", "source": str(e3_root / "parameter_lhs_results.csv")},
        {"claim_id": "E3-RANGE", "claim": "E3 finite stress saving range", "value": f"{e3_summary_values['saving_min_pct']:.10g} to {e3_summary_values['saving_max_pct']:.10g}", "unit": "percent", "source": str(e3_root / "parameter_lhs_interval_summary.csv")},
    ]
    pd.DataFrame(ledger_rows).to_csv(root / "numerical_claim_ledger.csv", index=False)
    input_manifest = all_frames[["experiment", "cache", "cache_sha256", "zone", "data_year", "profile_variant", "hours", "timestamp_min_utc", "timestamp_max_utc"]].drop_duplicates().sort_values(["experiment", "cache"])
    input_manifest.to_csv(root / "input_cache_manifest.csv", index=False)
    report = {
        "status": "PASS" if technical_pass else "FAIL",
        "technical_pass": technical_pass,
        "checks": checks,
        "code_hashes": code_hashes(),
        "capacity_upper_bound_hits": capacity_hits,
        "E1_mechanism_diagnostics": e1_mechanism,
        "E2_cross_period_diagnostics": e2_summary,
        "E3_bounded_stress_diagnostics": e3_summary_values,
        "scientific_boundary": "Selected European public-data engineering screen only. Results do not establish EU-27 prevalence, project-level congestion, legal RFNBO certification, authenticated forecast-vintage performance, real-time operation, or bankability.",
        "outputs": {
            "numerical_claim_ledger": str(root / "numerical_claim_ledger.csv"),
            "input_cache_manifest": str(root / "input_cache_manifest.csv"),
        },
    }
    write_json(root / "esb_upgrade_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if technical_pass else 1)


if __name__ == "__main__":
    main()
