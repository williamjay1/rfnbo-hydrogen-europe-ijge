"""Build a country-level public-data extension for the Project 01 model.

This extension deliberately has a different estimand from the selected
bidding-zone screen.  It pairs public country-level wind and solar generation
shapes from Energy-Charts with a matched public bidding-zone price snapshot
from Figshare/ENTSO-E.  The output is a set of independent country/market-area
stress cells, not a pooled EU-27 estimate and not a physical network model.
Missing observations are dropped only after a complete source interval has
been required; no profile or price imputation is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("PROJECT01_ROOT", Path(__file__).resolve().parents[1]))
RAW_ROOT = Path(os.environ.get("PROJECT01_RAW_ROOT", PROJECT / "raw"))
RAW_GENERATION = Path(
    os.environ.get(
        "PROJECT01_GENERATION_2019_DIR",
        str(RAW_ROOT / "energy_charts_public_power_2019"),
    )
)
RAW_PRICE = Path(
    os.environ.get(
        "PROJECT01_PRICE_SNAPSHOT",
        str(RAW_ROOT / "public_entsoe_figshare" / "entsoe-hourly-prices.csv"),
    )
)
EUA_PATH = Path(
    os.environ.get(
        "PROJECT01_EUA_CACHE",
        str(PROJECT / "datasets" / "dev_cache" / "eua_primary_auction_daily_dev.csv.gz"),
    )
)
OUT = Path(os.environ.get("PROJECT01_PRIMARY_CACHE", str(PROJECT / "datasets" / "public_eu_2019")))


# One public market area is selected per available EU member-state profile.
# DE-LU is retained as the coupled German-Luxembourg market area and is not
# described as a Germany-only observation.  IE_SEM is deliberately absent
# because it mixes the Republic of Ireland with Northern Ireland (UK).
AREA_MAP = {
    "AT": {"country": "AT", "price_area": "AT", "label": "AT", "member_state_scope": "Austria"},
    "BE": {"country": "BE", "price_area": "BE", "label": "BE", "member_state_scope": "Belgium"},
    "BG": {"country": "BG", "price_area": "BG", "label": "BG", "member_state_scope": "Bulgaria"},
    "CZ": {"country": "CZ", "price_area": "CZ", "label": "CZ", "member_state_scope": "Czechia"},
    "DE": {"country": "DE", "price_area": "DE_LU", "label": "DE_LU", "member_state_scope": "Germany and Luxembourg"},
    "DK": {"country": "DK", "price_area": "DK1", "label": "DK1_country_profile", "member_state_scope": "Denmark; country profile paired with DK1 price"},
    "EE": {"country": "EE", "price_area": "EE", "label": "EE", "member_state_scope": "Estonia"},
    "ES": {"country": "ES", "price_area": "ES", "label": "ES", "member_state_scope": "Spain"},
    "FR": {"country": "FR", "price_area": "FR", "label": "FR", "member_state_scope": "France"},
    "GR": {"country": "GR", "price_area": "GR", "label": "GR", "member_state_scope": "Greece"},
    "HR": {"country": "HR", "price_area": "HR", "label": "HR", "member_state_scope": "Croatia"},
    "IT": {"country": "IT", "price_area": "IT_North", "label": "IT_North_country_profile", "member_state_scope": "Italy; country profile paired with IT-North price"},
    "LT": {"country": "LT", "price_area": "LT", "label": "LT", "member_state_scope": "Lithuania"},
    "NL": {"country": "NL", "price_area": "NL", "label": "NL", "member_state_scope": "Netherlands"},
    "PT": {"country": "PT", "price_area": "PT", "label": "PT", "member_state_scope": "Portugal"},
    "RO": {"country": "RO", "price_area": "RO", "label": "RO", "member_state_scope": "Romania"},
    "SI": {"country": "SI", "price_area": "SI", "label": "SI", "member_state_scope": "Slovenia"},
    "SK": {"country": "SK", "price_area": "SK", "label": "SK", "member_state_scope": "Slovakia"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_generation(country: str) -> tuple[pd.DataFrame, dict[str, object]]:
    path = RAW_GENERATION / f"public_power_{country}_2019.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for item in payload.get("data", []):
        stamp = str(item.get("timestamp", ""))
        if not stamp.startswith("2019-"):
            continue
        values = item.get("values", {}) or {}
        records.append({
            "timestamp_raw": stamp,
            "wind_mw": values.get("wind_onshore"),
            "solar_mw": values.get("solar"),
        })
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ValueError(f"no 2019 records in {path}")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_raw"], utc=True)
    raw["wind_mw"] = pd.to_numeric(raw["wind_mw"], errors="coerce")
    raw["solar_mw"] = pd.to_numeric(raw["solar_mw"], errors="coerce")
    interval = int(payload.get("interval_minutes") or 60)
    expected_per_hour = max(1, 60 // interval)
    raw["hour_utc"] = raw["timestamp_utc"].dt.floor("h")
    grouped = raw.groupby("hour_utc", sort=True)
    hourly = grouped[["wind_mw", "solar_mw"]].mean()
    counts = grouped.size().rename("source_rows")
    nonmissing = grouped[["wind_mw", "solar_mw"]].count().min(axis=1).rename("nonmissing_series_rows")
    hourly = hourly.join(counts).join(nonmissing).reset_index().rename(columns={"hour_utc": "timestamp_utc"})
    complete = hourly["source_rows"].eq(expected_per_hour) & hourly["nonmissing_series_rows"].eq(expected_per_hour)
    hourly = hourly.loc[complete].copy()
    hourly["wind_mw"] = hourly["wind_mw"].clip(lower=0.0)
    hourly["solar_mw"] = hourly["solar_mw"].clip(lower=0.0)
    return hourly[["timestamp_utc", "wind_mw", "solar_mw"]], {
        "raw_path": str(path),
        "raw_sha256": sha256(path),
        "raw_rows_local_2019": int(len(raw)),
        "source_interval_minutes": interval,
        "expected_rows_per_utc_hour": expected_per_hour,
        "complete_hourly_rows": int(len(hourly)),
        "dropped_incomplete_hourly_rows": int((~complete).sum()),
        "timezone": payload.get("timezone"),
        "available_from": payload.get("available_from"),
        "available_until": payload.get("available_until"),
        "license": payload.get("license"),
    }


def normalize(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    clean = pd.to_numeric(series, errors="coerce").clip(lower=0.0)
    scale = float(clean.quantile(0.995))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalize a non-positive generation series")
    cf = (clean / scale).clip(lower=0.0, upper=1.0)
    return cf, {"q995_mw": scale, "clipped_above_q995_fraction": float((clean > scale).mean())}


def load_prices() -> pd.DataFrame:
    frame = pd.read_csv(RAW_PRICE, usecols=["Datetime", "Area", "Value"])
    frame["timestamp_utc"] = pd.to_datetime(frame["Datetime"], utc=True)
    frame["price_eur_per_mwh"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame[frame["timestamp_utc"].dt.year.eq(2019)].copy()
    return frame[["timestamp_utc", "Area", "price_eur_per_mwh"]]


def load_eua() -> pd.DataFrame:
    frame = pd.read_csv(EUA_PATH, parse_dates=["timestamp_utc"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame[frame["timestamp_utc"].dt.year.eq(2019)].copy()
    frame["date_utc"] = frame["timestamp_utc"].dt.normalize()
    return frame[["date_utc", "eua_price_eur_tco2"]].drop_duplicates("date_utc")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    eua = load_eua()
    records: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for source_country, meta in AREA_MAP.items():
        generation, generation_report = load_generation(source_country)
        area_prices = prices[prices["Area"].eq(meta["price_area"])].copy()
        duplicate_price_timestamps = int(area_prices["timestamp_utc"].duplicated().sum())
        area_prices = area_prices.groupby("timestamp_utc", as_index=False)["price_eur_per_mwh"].mean()
        merged = generation.merge(area_prices, on="timestamp_utc", how="inner")
        merged["date_utc"] = merged["timestamp_utc"].dt.normalize()
        merged = merged.merge(eua, on="date_utc", how="left").drop(columns=["date_utc"])
        required = ["wind_mw", "solar_mw", "price_eur_per_mwh", "eua_price_eur_tco2"]
        before = len(merged)
        merged = merged.dropna(subset=required).sort_values("timestamp_utc").reset_index(drop=True)
        wind_cf, wind_report = normalize(merged["wind_mw"])
        solar_cf, solar_report = normalize(merged["solar_mw"])
        output = pd.DataFrame({
            "timestamp_utc": merged["timestamp_utc"],
            "wind_cf": wind_cf,
            "solar_cf": solar_cf,
            "day_ahead_price_eur_per_mwh": merged["price_eur_per_mwh"],
            "eua_price_eur_tco2": merged["eua_price_eur_tco2"],
        })
        path = OUT / f"{meta['label']}_2019.csv.gz"
        output.to_csv(path, index=False, compression="gzip", float_format="%.8g")
        trigger = (output["day_ahead_price_eur_per_mwh"] <= 20.0) | (output["day_ahead_price_eur_per_mwh"] < 0.36 * output["eua_price_eur_tco2"])
        coverage = {
            "source_country": source_country,
            "label": meta["label"],
            "price_area": meta["price_area"],
            "member_state_scope": meta["member_state_scope"],
            "raw_generation_rows": generation_report["raw_rows_local_2019"],
            "complete_generation_hours": generation_report["complete_hourly_rows"],
            "price_rows_2019": int(len(area_prices)),
            "duplicate_price_timestamps_before_aggregation": duplicate_price_timestamps,
            "joined_rows_before_missing_filter": int(before),
            "final_complete_rows": int(len(output)),
            "dropped_after_join": int(before - len(output)),
            "timestamp_min": str(output["timestamp_utc"].min()),
            "timestamp_max": str(output["timestamp_utc"].max()),
            "negative_price_hours": int((output["day_ahead_price_eur_per_mwh"] < 0).sum()),
            "price20_hours": int((output["day_ahead_price_eur_per_mwh"] <= 20).sum()),
            "exact_trigger_hours": int(trigger.sum()),
            "wind_cf_q995_mw": wind_report["q995_mw"],
            "solar_cf_q995_mw": solar_report["q995_mw"],
            "source_license": generation_report["license"],
            "cache_path": str(path),
            "cache_sha256": sha256(path),
        }
        coverage.update({f"generation_{key}": value for key, value in generation_report.items() if key not in {"raw_path", "raw_sha256", "license"}})
        coverage_rows.append(coverage)
        records.append({"source_country": source_country, "cache": str(path), "rows": int(len(output)), "coverage": coverage})
        print(json.dumps({"country": source_country, "label": meta["label"], "rows": len(output), "trigger_hours": int(trigger.sum())}), flush=True)

    coverage_frame = pd.DataFrame(coverage_rows).sort_values("label").reset_index(drop=True)
    coverage_frame.to_csv(OUT / "coverage_audit.csv", index=False, float_format="%.8g")
    manifest = {
        "status": "PUBLIC_EU_MEMBER_MARKET_AREA_2019_CACHE_COMPLETE",
        "estimand": "independent country-profile/market-area engineering stress cells; not a pooled EU-27 estimator",
        "period": {"requested_local_year": 2019, "price_time_standard": "UTC", "profile_time_standard": "source local timestamps converted to UTC with offsets, then complete UTC hours"},
        "sources": {
            "generation_endpoint": "https://api.energy-charts.info/v2/public_power",
            "generation_license": "retained per raw response and manifest; CC BY 4.0 attribution text returned by the API",
            "price_snapshot": str(RAW_PRICE),
            "price_source_doi": "10.6084/m9.figshare.12178929.v1",
            "price_source_url": "https://figshare.com/articles/dataset/entsoe-hourly-prices_csv/12178929",
            "price_license": "CC BY 4.0",
            "eua_proxy": str(EUA_PATH),
        },
        "area_map": AREA_MAP,
        "included_cells": records,
        "excluded_from_this_extension": {
            "CY": "public Energy-Charts 2019 solar series returned no numeric values",
            "FI": "public Energy-Charts 2019 solar series returned no numeric values",
            "LV": "public Energy-Charts 2019 solar series returned no numeric values",
            "PL": "public Energy-Charts 2019 solar series returned no numeric values",
            "SE": "public Energy-Charts 2019 solar series returned no numeric values; Swedish PVGIS point profiles remain in the separate selected-zone sample",
            "MT": "not present in the public Energy-Charts country catalog tested",
            "IE": "IE_SEM price area mixes the Republic of Ireland and Northern Ireland and is excluded from the EU-only extension",
        },
        "processing": {
            "generation_aggregation": "source interval rows are retained only when every expected sub-hourly row has both wind_onshore and solar; arithmetic hourly mean is used",
            "generation_normalization": "each country profile divided by its 2019 99.5th percentile and clipped to [0,1]; no capacity factor claim is made",
            "missingness": "inner price/profile join and complete-case filtering; no imputation",
            "eua": "daily EEX primary-auction development proxy joined by UTC date",
        },
        "limitations": [
            "Country generation shapes are not project availability or contractual renewable production.",
            "A country profile paired with DK1 or IT-North price is a market-area stress cell, not a zonal generation observation.",
            "DE-LU is a coupled bidding area and is not separated into Germany and Luxembourg.",
            "No physical network flow, PTDF, N-1, redispatch or project connection data are inferred.",
            "The extension does not establish EU-27 prevalence, legal RFNBO eligibility or forecast-vintage compliance.",
        ],
        "coverage_audit": str(OUT / "coverage_audit.csv"),
    }
    (OUT / "public_eu_2019_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "cells": len(records), "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
