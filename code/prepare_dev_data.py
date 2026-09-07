"""Prepare auditable development caches for Project 01.

The files produced by this module are development/QC artifacts only.  They
are intentionally not the final EU-27 evidence set because the OPSD package is
a legacy extract and the EEX file contains auction-date EUA observations, not
an hourly secondary-market settlement series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ZONES = [
    "DE_LU",
    "FR",
    "NL",
    "BE",
    "ES",
    "IT_NORD",
    "DK_1",
    "PL",
    "AT",
    "CZ",
    "SE_4",
    "NO_2",
]
DEFAULT_PV_COUNTRIES = ["AT", "BE", "CZ", "DE", "DK", "ES", "FR", "IT", "NL", "PL"]
DEFAULT_WIND_BZ = ["NO1", "NO2", "NO3", "NO4", "NO5", "SW1", "SW2", "SW3", "SW4", "CNOR", "NORD", "SARD", "SUD", "CSUD", "SICI", "DK1", "DK2"]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, source: str, role: str) -> dict[str, object]:
    return {
        "source": source,
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "md5": md5_file(path),
        "readonly": not bool(path.stat().st_mode & 0o200),
    }


def _available_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def load_opsd_legacy(path: Path, zones: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load only usable legacy OPSD columns and report missingness.

    The 2020 OPSD package is retained for pipeline development and historical
    calibration checks.  It must not be silently presented as a 2021–2026
    current ENTSO-E dataset.
    """

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["utc_timestamp", "cet_cest_timestamp"]
    mapping: dict[str, dict[str, str | None]] = {}
    for zone in zones:
        mapping[zone] = {
            "price": _available_column(
                columns,
                [f"{zone}_price_day_ahead", f"{zone}_price_day_ahead_entsoe_transparency"],
            ),
            "load": _available_column(
                columns, [f"{zone}_load_actual_entsoe_transparency"]
            ),
            "solar": _available_column(
                columns,
                [f"{zone}_solar_generation_actual", f"{zone}_solar_generation_actual_entsoe_transparency"],
            ),
            "wind": _available_column(
                columns,
                [
                    f"{zone}_wind_onshore_generation_actual",
                    f"{zone}_wind_generation_actual",
                ],
            ),
        }
        usecols.extend(v for v in mapping[zone].values() if v is not None)
    usecols = list(dict.fromkeys(usecols))

    frame = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=["utc_timestamp"],
        low_memory=False,
    )
    frame = frame.sort_values("utc_timestamp").reset_index(drop=True)
    frame = frame.rename(columns={"utc_timestamp": "timestamp_utc"})
    for zone, fields in mapping.items():
        for field, original in fields.items():
            target = f"{zone}_{field}"
            if original is None:
                frame[target] = np.nan
            else:
                frame[target] = pd.to_numeric(frame[original], errors="coerce")
                if original != target:
                    frame = frame.drop(columns=[original])

    numeric_columns = [c for c in frame.columns if c != "timestamp_utc" and c != "cet_cest_timestamp"]
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    coverage = {
        "rows": int(len(frame)),
        "timestamp_min_utc": str(frame["timestamp_utc"].min()),
        "timestamp_max_utc": str(frame["timestamp_utc"].max()),
        "duplicate_timestamps": int(frame["timestamp_utc"].duplicated().sum()),
        "monotonic_timestamp": bool(frame["timestamp_utc"].is_monotonic_increasing),
        "zones": {},
    }
    for zone in zones:
        coverage["zones"][zone] = {
            "columns": mapping[zone],
            "missing_fraction": {
                field: float(frame[f"{zone}_{field}"].isna().mean()) for field in mapping[zone]
            },
        }
    return frame, coverage


def _find_header_row(raw: pd.DataFrame) -> int:
    for index, row in raw.iterrows():
        values = [str(value).strip() for value in row.tolist() if pd.notna(value)]
        if any(value == "Date" for value in values) and any("Auction Price" in value for value in values):
            return int(index)
    raise ValueError("Could not find the EEX auction table header")


def _column_name(columns: Iterable[object], *, exact: str | None = None, contains: str | None = None) -> object:
    for column in columns:
        text = str(column).strip()
        if exact is not None and text == exact:
            return column
        if contains is not None and contains.lower() in text.lower():
            return column
    raise ValueError(f"Could not find EEX column exact={exact!r}, contains={contains!r}")


def load_eex_eua_reports(extract_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse EU primary auction rows from the EEX annual workbooks."""

    files = sorted(extract_root.rglob("emission-spot-primary-market-auction-report-*-data.*"))
    if not files:
        raise FileNotFoundError("No extracted EEX annual reports found")
    records: list[pd.DataFrame] = []
    years: list[int] = []
    for file in files:
        match = re.search(r"report-(\d{4})-data", file.name)
        if not match:
            continue
        year = int(match.group(1))
        raw = pd.read_excel(file, sheet_name=0, header=None)
        header_row = _find_header_row(raw)
        table = raw.iloc[header_row + 1 :].copy()
        table.columns = raw.iloc[header_row].tolist()
        date_col = _column_name(table.columns, exact="Date")
        price_col = _column_name(table.columns, contains="Auction Price")
        auction_col = _column_name(table.columns, exact="Auction Name")
        country_col = next((c for c in table.columns if str(c).strip() == "Country"), None)
        volume_col = _column_name(table.columns, contains="Auction Volume")
        status_col = next((c for c in table.columns if str(c).strip() == "Status"), None)
        base_columns = [date_col, price_col, auction_col, volume_col] + ([country_col] if country_col is not None else [])
        selected = table[base_columns + ([status_col] if status_col is not None else [])].copy()
        selected.columns = ["date", "eua_price_eur_tco2", "auction_name", "auction_volume_tco2"] + (["country"] if country_col is not None else []) + (["status"] if status_col is not None else [])
        if country_col is None:
            selected["country"] = ""
        selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
        selected["eua_price_eur_tco2"] = pd.to_numeric(selected["eua_price_eur_tco2"], errors="coerce")
        selected["auction_volume_tco2"] = pd.to_numeric(selected["auction_volume_tco2"], errors="coerce")
        selected["country_text"] = selected["country"].astype(str).str.strip().str.upper()
        selected["auction_text"] = selected["auction_name"].astype(str).str.strip().str.upper()
        selected = selected[
            selected["country_text"].eq("EU")
            | selected["auction_text"].str.contains(r"CAP\d.*EU|EU PRIMARY|(?:^|[- ])EU(?:$|[- ])", regex=True, na=False)
        ]
        if status_col is not None:
            selected = selected[
                selected["status"].isna()
                | selected["status"].astype(str).str.contains("successful", case=False, na=False)
            ]
        records.append(selected)
        years.append(year)

    auctions = pd.concat(records, ignore_index=True)
    auctions = auctions.dropna(subset=["date", "eua_price_eur_tco2"])
    auctions["weight"] = auctions["auction_volume_tco2"].fillna(1.0).clip(lower=1.0)
    auctions["weighted_price"] = auctions["eua_price_eur_tco2"] * auctions["weight"]
    daily = (
        auctions.groupby("date", as_index=False)
        .agg(weighted_price=("weighted_price", "sum"), weight=("weight", "sum"), auction_count=("date", "size"))
    )
    daily["eua_price_eur_tco2"] = daily["weighted_price"] / daily["weight"]
    daily = daily.drop(columns=["weighted_price", "weight"]).sort_values("date")
    report = {
        "annual_files": len(files),
        "years": years,
        "eu_auction_rows": int(len(auctions)),
        "unique_auction_dates": int(len(daily)),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "aggregation": "auction-volume-weighted mean when multiple EU auctions occur on one date",
        "proxy_warning": "EEX primary auction observations are forward-filled to daily/hourly development timestamps; use as a proxy only.",
    }
    return daily, report


def load_emhires_pv(extract_root: Path, countries: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    path = next(extract_root.rglob("EMHIRESPV_TSh_CF_Country_19862015.xlsx"))
    usecols = ["Date", "Year"] + countries
    frame = pd.read_excel(path, sheet_name=0, usecols=usecols)
    frame["timestamp_utc"] = pd.to_datetime(frame["Date"], errors="coerce").dt.tz_localize("UTC")
    frame = frame.drop(columns=["Date"]).sort_values("timestamp_utc").reset_index(drop=True)
    for country in countries:
        frame[country] = pd.to_numeric(frame[country], errors="coerce")
    year_2015 = frame[frame["Year"].eq(2015)].copy()
    report = {
        "source_file": str(path),
        "rows_all_years": int(len(frame)),
        "rows_2015": int(len(year_2015)),
        "timestamp_min_2015_utc_assumed": str(year_2015["timestamp_utc"].min()),
        "timestamp_max_2015_utc_assumed": str(year_2015["timestamp_utc"].max()),
        "countries": countries,
        "timestamp_assumption": "source workbook has naive hourly Date; development cache localizes it to UTC pending source-time-zone confirmation",
        "missing_fraction_2015": {country: float(year_2015[country].isna().mean()) for country in countries},
        "value_range_2015": {country: [float(year_2015[country].min()), float(year_2015[country].max())] for country in countries},
    }
    return year_2015[["timestamp_utc", "Year"] + countries], report


def load_emhires_wind(extract_root: Path, zones: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    path = next(extract_root.rglob("TS.CF.BZN.30yr.txt"))
    frame = pd.read_csv(path, sep=r"\s+", index_col=0)
    frame.columns = [str(c).strip().strip('"') for c in frame.columns]
    wanted = [zone for zone in zones if zone in frame.columns]
    frame = frame[wanted].apply(pd.to_numeric, errors="coerce")
    frame["timestamp_utc"] = pd.date_range("1986-01-01", periods=len(frame), freq="h", tz="UTC")
    frame["Year"] = frame["timestamp_utc"].dt.year
    year_2015 = frame[frame["Year"].eq(2015)].copy()
    report = {
        "source_file": str(path),
        "rows_all_years": int(len(frame)),
        "rows_2015": int(len(year_2015)),
        "bidding_zones_available": wanted,
        "bidding_zones_not_available": [zone for zone in zones if zone not in wanted],
        "timestamp_assumption": "30-year sequence is mapped to 1986-01-01 through 2015-12-31 UTC for development alignment",
        "missing_fraction_2015": {zone: float(year_2015[zone].isna().mean()) for zone in wanted},
        "value_range_2015": {zone: [float(year_2015[zone].min()), float(year_2015[zone].max())] for zone in wanted},
    }
    return year_2015[["timestamp_utc", "Year"] + wanted], report


def build_daily_eua_alignment(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    source = daily.copy()
    source["timestamp_utc"] = pd.to_datetime(source.pop("date"), utc=True)
    source = source.set_index("timestamp_utc").sort_index()
    aligned = source.reindex(index)
    aligned["eua_observation_available"] = aligned["eua_price_eur_tco2"].notna()
    aligned["eua_price_eur_tco2"] = aligned["eua_price_eur_tco2"].ffill()
    aligned["eua_forward_filled"] = ~aligned["eua_observation_available"] & aligned["eua_price_eur_tco2"].notna()
    aligned = aligned.reset_index(names="timestamp_utc")
    return aligned


def prepare(raw_root: Path, extract_root: Path, out_dir: Path, zones: list[str]) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    opsd_path = raw_root / "opsd_time_series_60min_singleindex_2020-10-06.csv"
    wind_path = raw_root / "EMHIRES_WIND_ONSHORE_BZN.zip"
    pv_path = raw_root / "EMHIRES_PV_COUNTRY.zip"
    eex_path = raw_root / "eex_emission_spot_primary_market_auction_report_2012_2025_data.zip"

    opsd, opsd_report = load_opsd_legacy(opsd_path, zones)
    eua, eua_report = load_eex_eua_reports(extract_root / "eex_eua")
    pv, pv_report = load_emhires_pv(extract_root / "emhires_pv", DEFAULT_PV_COUNTRIES)
    wind, wind_report = load_emhires_wind(extract_root / "emhires_wind", DEFAULT_WIND_BZ)

    opsd.to_csv(out_dir / "opsd_legacy_hourly_dev.csv.gz", index=False, compression="gzip", float_format="%.8g")
    pv.to_csv(out_dir / "emhires_pv_2015_dev.csv.gz", index=False, compression="gzip", float_format="%.8g")
    wind.to_csv(out_dir / "emhires_wind_2015_dev.csv.gz", index=False, compression="gzip", float_format="%.8g")

    start = pd.Timestamp(opsd["timestamp_utc"].min())
    end = pd.Timestamp(opsd["timestamp_utc"].max())
    eua_aligned = build_daily_eua_alignment(eua, start, end)
    eua_aligned.to_csv(out_dir / "eua_primary_auction_daily_dev.csv.gz", index=False, compression="gzip", float_format="%.8g")

    raw_records = [
        file_record(wind_path, source="JRC EMHIRES / Zenodo mirror", role="wind profiles"),
        file_record(pv_path, source="JRC EMHIRES / Zenodo mirror", role="solar profiles"),
        file_record(opsd_path, source="OPSD time series package", role="legacy prices and generation"),
        file_record(eex_path, source="EEX primary auction report archive", role="EUA development proxy"),
    ]
    report: dict[str, object] = {
        "project": "Project01",
        "scope": "EU-27 target; development cache is not yet EU-27 complete",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_files": raw_records,
        "opsd": opsd_report,
        "eua": eua_report,
        "eua_alignment": {
            "rows": int(len(eua_aligned)),
            "start": str(eua_aligned["timestamp_utc"].min()),
            "end": str(eua_aligned["timestamp_utc"].max()),
            "exact_auction_date_fraction": float(eua_aligned["eua_observation_available"].mean()),
            "forward_filled_fraction": float(eua_aligned["eua_forward_filled"].mean()),
        },
        "emhires_pv": pv_report,
        "emhires_wind": wind_report,
        "quality_gate": {
            "status": "DEVELOPMENT_ONLY",
            "final_primary_price_source_required": "ENTSO-E Transparency Platform current price and generation data after security-token access is granted",
            "final_network_source_required": "validated bidding-zone/interconnector capacities and flow/redispatch series",
            "final_eua_source_required": "documented allowance-price alignment choice; EEX auction proxy must be sensitivity-tested against licensed or official secondary-market series",
        },
    }
    (out_dir / "data_feasibility_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("PROJECT01_RAW_ROOT", repo_root / "raw")),
    )
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=Path(os.environ.get("PROJECT01_EXTRACT_ROOT", repo_root / "raw_extract")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("PROJECT01_DEV_CACHE", repo_root / "datasets" / "dev_cache")),
    )
    args = parser.parse_args()
    report = prepare(args.raw_root, args.extract_root, args.out_dir, DEFAULT_ZONES)
    print(json.dumps({"status": report["quality_gate"]["status"], "out_dir": str(args.out_dir), "opsd_rows": report["opsd"]["rows"], "eua_dates": report["eua"]["unique_auction_dates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
