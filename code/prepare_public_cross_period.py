"""Prepare a no-credential cross-period public-data extension for Project 01.

The extension is deliberately separate from the selected-zone 2015 primary
screen.  It uses public measured renewable generation and public day-ahead
prices to test whether the qualitative rule comparison survives different
years and source datasets.  It does not create a network-flow observation or
legal RFNBO certification record.

Raw files are read from the immutable E: repository.  Every derived cache and
manifest is written to D:.  The Belgian Elia field ``dayaheadforecast`` is
retained as a published forecast input, but forecast issue-time provenance is
not inferred from the field name.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("PROJECT01_ROOT", Path(__file__).resolve().parents[1]))
RAW = Path(os.environ.get("PROJECT01_RAW_ROOT", PROJECT / "raw"))
EEX_EXTRACT = Path(
    os.environ.get(
        "PROJECT01_EEX_EXTRACT",
        str(PROJECT / "raw_extract" / "eex_eua"),
    )
)
OUT = Path(
    os.environ.get(
        "PROJECT01_CROSS_PERIOD_OUT",
        str(PROJECT / "datasets" / "public_cross_period"),
    )
)

DE_YEARS = [2019, 2020, 2021, 2022]
BE_YEARS = [2021, 2022, 2023, 2024, 2025]

BE_SOLAR = RAW / "elia/solar_belgium.csv"
BE_WIND = [
    RAW / "elia/wind_federal_offshore.csv",
    RAW / "elia/wind_flanders_onshore_dso.csv",
    RAW / "elia/wind_flanders_onshore_elia.csv",
    RAW / "elia/wind_wallonia_onshore_dso.csv",
    RAW / "elia/wind_wallonia_onshore_elia.csv",
]


def _energy_chart_production(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    data: dict[str, object] = {"timestamp_utc": timestamps}
    source_names: list[str] = []
    for item in payload["production_types"]:
        name = str(item["name"])
        source_names.append(name)
        if name in {"Wind offshore", "Wind onshore", "Solar"}:
            data[name] = pd.to_numeric(item["data"], errors="coerce")
    frame = pd.DataFrame(data).sort_values("timestamp_utc").reset_index(drop=True)
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"duplicate Energy-Charts timestamps: {path}")
    metadata = {
        "path": str(path),
        "license_info": payload.get("license_info"),
        "deprecated": payload.get("deprecated"),
        "production_type_names": source_names,
        "rows": int(len(frame)),
        "timestamp_min_utc": str(frame["timestamp_utc"].min()),
        "timestamp_max_utc": str(frame["timestamp_utc"].max()),
    }
    return frame, metadata


def _energy_chart_price(path: Path, local_tz: str, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
            "day_ahead_price_eur_per_mwh": pd.to_numeric(payload["price"], errors="coerce"),
        }
    ).sort_values("timestamp_utc")
    local_year = frame["timestamp_utc"].dt.tz_convert(local_tz).dt.year.eq(year)
    frame = frame.loc[local_year].copy()
    frame["hour_utc"] = frame["timestamp_utc"].dt.floor("h")
    grouped = (
        frame.groupby("hour_utc", as_index=False)
        .agg(day_ahead_price_eur_per_mwh=("day_ahead_price_eur_per_mwh", "mean"), source_rows=("timestamp_utc", "size"))
        .rename(columns={"hour_utc": "timestamp_utc"})
    )
    metadata = {
        "path": str(path),
        "license_info": payload.get("license_info"),
        "deprecated": payload.get("deprecated"),
        "source_rows_local_year": int(len(frame)),
        "hourly_rows": int(len(grouped)),
        "price_missing_hourly": int(grouped["day_ahead_price_eur_per_mwh"].isna().sum()),
        "timestamp_min_utc": str(grouped["timestamp_utc"].min()),
        "timestamp_max_utc": str(grouped["timestamp_utc"].max()),
        "source_frequency": "hourly in 2021-2024 files; 15-minute and hourly observations aggregated by arithmetic hourly mean in the 2025 Belgium file",
    }
    return grouped, metadata


def _installed_power(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    years = [int(value) for value in payload["time"]]
    values: dict[int, dict[str, float]] = {year: {} for year in years}
    for item in payload["production_types"]:
        name = str(item["name"])
        if name in {"Wind offshore", "Wind onshore", "Solar AC"}:
            for year, value in zip(years, item["data"]):
                if value is not None and np.isfinite(float(value)):
                    values[year][name] = float(value) * 1000.0
    return values


def _load_eua_daily() -> tuple[pd.DataFrame, dict[str, object]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_dev_data import load_eex_eua_reports

    daily, report = load_eex_eua_reports(EEX_EXTRACT)
    daily = daily.rename(columns={"date": "eua_date"})
    daily["eua_date"] = pd.to_datetime(daily["eua_date"]).dt.normalize()
    return daily[["eua_date", "eua_price_eur_tco2"]], report


def _attach_eua(frame: pd.DataFrame, eua: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("timestamp_utc").copy()
    result["eua_date"] = result["timestamp_utc"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    right = eua.sort_values("eua_date").copy()
    right["eua_date"] = pd.to_datetime(right["eua_date"]).astype("datetime64[ns]")
    result = pd.merge_asof(result, right, on="eua_date", direction="backward")
    result["eua_observation_available"] = result["eua_date"].isin(set(right["eua_date"]))
    result["eua_forward_filled"] = result["eua_price_eur_tco2"].notna() & ~result["eua_observation_available"]
    return result.drop(columns=["eua_date"])


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "timestamp_utc",
        "wind_cf",
        "solar_cf",
        "day_ahead_price_eur_per_mwh",
        "eua_price_eur_tco2",
        "eua_observation_available",
        "eua_forward_filled",
    ]
    required = [column for column in columns if column not in frame.columns]
    if required:
        raise ValueError(f"missing cache columns: {required}")
    frame[columns].sort_values("timestamp_utc").to_csv(path, index=False, compression="gzip", float_format="%.10g")


def _make_de_year(year: int, eua: pd.DataFrame) -> dict[str, object]:
    power_path = RAW / "energy_charts" / f"energy_charts_public_power_de_{year}.json"
    price_path = RAW / "energy_charts" / f"energy_charts_price_de_lu_{year}.json"
    power, power_meta = _energy_chart_production(power_path)
    price, price_meta = _energy_chart_price(price_path, "Europe/Berlin", year)
    local_year = power["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.year.eq(year)
    power = power.loc[local_year].copy()
    power["hour_utc"] = power["timestamp_utc"].dt.floor("h")
    hourly = (
        power.groupby("hour_utc", as_index=False)
        .agg(
            wind_offshore_mw=("Wind offshore", "mean"),
            wind_onshore_mw=("Wind onshore", "mean"),
            solar_mw=("Solar", "mean"),
            source_rows=("timestamp_utc", "size"),
        )
        .rename(columns={"hour_utc": "timestamp_utc"})
    )
    installed = _installed_power(RAW / "energy_charts/installed_power_de.json")[year]
    hourly["wind_mw"] = hourly["wind_offshore_mw"] + hourly["wind_onshore_mw"]
    hourly["wind_cf"] = hourly["wind_mw"] / (installed["Wind offshore"] + installed["Wind onshore"])
    hourly["solar_cf"] = hourly["solar_mw"] / installed["Solar AC"]
    merged = hourly.merge(price, on="timestamp_utc", how="inner")
    merged = _attach_eua(merged, eua)
    clip_counts = {
        "wind_cf_below_zero": int((merged["wind_cf"] < 0).sum()),
        "wind_cf_above_one": int((merged["wind_cf"] > 1).sum()),
        "solar_cf_below_zero": int((merged["solar_cf"] < 0).sum()),
        "solar_cf_above_one": int((merged["solar_cf"] > 1).sum()),
    }
    valid = (
        merged[["wind_cf", "solar_cf", "day_ahead_price_eur_per_mwh", "eua_price_eur_tco2"]]
        .notna()
        .all(axis=1)
    )
    merged = merged.loc[valid].copy()
    for column in ["wind_cf", "solar_cf"]:
        merged[column] = merged[column].clip(0.0, 1.0)
    output = OUT / f"DE_LU_{year}_observed.csv.gz"
    _write_cache(merged, output)
    return {
        "cache": str(output),
        "country_or_zone": "DE-LU",
        "year": year,
        "profile_variant": "observed_generation",
        "hours": int(len(merged)),
        "expected_hours": int(8784 if year == 2020 else 8760),
        "dropped_incomplete_hours": int(len(hourly.merge(price, on="timestamp_utc", how="inner")) - len(merged)),
        "installed_capacity_mw": installed,
        "raw_power": power_meta,
        "price": price_meta,
        "clip_counts_before_bounds": clip_counts,
        "source_rows_per_hour": sorted(pd.to_numeric(merged.get("source_rows", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()),
    }


def _elia_hourly(path: Path, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    frame = frame.rename(columns={"datetime": "timestamp_utc", "dayaheadforecast": "forecast_mw"})
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.loc[frame["timestamp_utc"].dt.tz_convert("Europe/Brussels").dt.year.eq(year)].copy()
    for column in ["measured", "forecast_mw", "monitoredcapacity"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    negative_measured = int((frame["measured"] < 0).sum())
    frame["measured_mw"] = frame["measured"].clip(lower=0.0)
    frame["hour_utc"] = frame["timestamp_utc"].dt.floor("h")
    hourly = (
        frame.groupby("hour_utc", as_index=False)
        .agg(
            measured_mw=("measured_mw", "mean"),
            forecast_mw=("forecast_mw", "mean"),
            monitoredcapacity_mw=("monitoredcapacity", "mean"),
            source_rows=("timestamp_utc", "size"),
        )
        .rename(columns={"hour_utc": "timestamp_utc"})
    )
    return hourly, {
        "path": str(path),
        "rows_local_year": int(len(frame)),
        "hours": int(len(hourly)),
        "measured_missing_hourly": int(hourly["measured_mw"].isna().sum()),
        "forecast_missing_hourly": int(hourly["forecast_mw"].isna().sum()),
        "capacity_missing_hourly": int(hourly["monitoredcapacity_mw"].isna().sum()),
        "negative_measured_15min_clipped": negative_measured,
    }


def _make_be_year(year: int, eua: pd.DataFrame, variant: str) -> dict[str, object]:
    price_path = RAW / "energy_charts_be" / f"price_BE_{year}.json"
    price, price_meta = _energy_chart_price(price_path, "Europe/Brussels", year)
    solar, solar_meta = _elia_hourly(BE_SOLAR, year)
    wind_frames: list[pd.DataFrame] = []
    wind_meta: list[dict[str, object]] = []
    for path in BE_WIND:
        frame, meta = _elia_hourly(path, year)
        wind_frames.append(frame.rename(columns={
            "measured_mw": f"measured_{len(wind_frames)}",
            "forecast_mw": f"forecast_{len(wind_frames)}",
            "monitoredcapacity_mw": f"capacity_{len(wind_frames)}",
            "source_rows": f"source_rows_{len(wind_frames)}",
        }))
        wind_meta.append(meta)
    wind = wind_frames[0]
    for other in wind_frames[1:]:
        wind = wind.merge(other, on="timestamp_utc", how="outer")
    wind_measured = [f"measured_{i}" for i in range(len(wind_frames))]
    wind_forecast = [f"forecast_{i}" for i in range(len(wind_frames))]
    wind_capacity = [f"capacity_{i}" for i in range(len(wind_frames))]
    wind["wind_mw"] = wind[wind_measured].sum(axis=1, min_count=len(wind_measured))
    wind["wind_forecast_mw"] = wind[wind_forecast].sum(axis=1, min_count=len(wind_forecast))
    wind["wind_capacity_mw"] = wind[wind_capacity].sum(axis=1, min_count=len(wind_capacity))
    solar = solar.rename(columns={
        "measured_mw": "solar_mw",
        "forecast_mw": "solar_forecast_mw",
        "monitoredcapacity_mw": "solar_capacity_mw",
    })
    merged = wind[["timestamp_utc", "wind_mw", "wind_forecast_mw", "wind_capacity_mw"]].merge(
        solar[["timestamp_utc", "solar_mw", "solar_forecast_mw", "solar_capacity_mw"]],
        on="timestamp_utc",
        how="outer",
    ).merge(price, on="timestamp_utc", how="outer")
    merged = _attach_eua(merged.sort_values("timestamp_utc"), eua)
    if variant == "observed_generation":
        wind_column, solar_column = "wind_mw", "solar_mw"
    elif variant == "dayahead_forecast":
        wind_column, solar_column = "wind_forecast_mw", "solar_forecast_mw"
    else:
        raise ValueError(f"unknown Belgium variant: {variant}")
    merged["wind_cf"] = merged[wind_column] / merged["wind_capacity_mw"]
    merged["solar_cf"] = merged[solar_column] / merged["solar_capacity_mw"]
    source = merged.copy()
    valid = source[["wind_cf", "solar_cf", "day_ahead_price_eur_per_mwh", "eua_price_eur_tco2"]].notna().all(axis=1)
    merged = source.loc[valid].copy()
    clip_counts = {
        "wind_cf_below_zero": int((merged["wind_cf"] < 0).sum()),
        "wind_cf_above_one": int((merged["wind_cf"] > 1).sum()),
        "solar_cf_below_zero": int((merged["solar_cf"] < 0).sum()),
        "solar_cf_above_one": int((merged["solar_cf"] > 1).sum()),
    }
    merged["wind_cf"] = merged["wind_cf"].clip(0.0, 1.0)
    merged["solar_cf"] = merged["solar_cf"].clip(0.0, 1.0)
    output = OUT / f"BE_{year}_{variant}.csv.gz"
    _write_cache(merged, output)
    return {
        "cache": str(output),
        "country_or_zone": "BE",
        "year": year,
        "profile_variant": variant,
        "hours": int(len(merged)),
        "expected_hours": int(8784 if year == 2024 else 8760),
        "dropped_incomplete_hours": int(len(source) - len(merged)),
        "raw_solar": solar_meta,
        "raw_wind": wind_meta,
        "price": price_meta,
        "clip_counts_before_bounds": clip_counts,
        "negative_generation_policy": "negative measured Elia values clipped to zero before hourly averaging; forecast values not clipped unless final CF bound requires it",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eua, eua_report = _load_eua_daily()
    records: list[dict[str, object]] = []
    for year in DE_YEARS:
        records.append(_make_de_year(year, eua))
    for year in BE_YEARS:
        records.append(_make_be_year(year, eua, "observed_generation"))
        records.append(_make_be_year(year, eua, "dayahead_forecast"))
    manifest = {
        "status": "PUBLIC_CROSS_PERIOD_CACHE_COMPLETE",
        "scope": "public historical generation and price stress test; not EU-27 network estimation or legal certification",
        "eua_report": eua_report,
        "sources": {
            "energy_charts": "https://www.energy-charts.info/",
            "smard_bundesnetzagentur": "https://www.smard.de/en",
            "elia_generation_data": "https://www.elia.be/en/grid-data/power-generation",
            "eex_primary_auction_reports": "https://www.eex.com/en/markets/emissions/eua-primary-auction-report",
        },
        "records": records,
        "time_standard": "UTC hour bins; local Europe/Berlin or Europe/Brussels calendar year used only to select source-year observations; DST is therefore represented by the UTC hour count",
        "forecast_boundary": "Belgian dayaheadforecast is used as a published forecast profile for robustness only; no issue-time field is inferred or used to claim forecast-vintage compliance",
        "raw_data_policy": "raw E: files are read-only; all derived caches and this manifest are on D:",
    }
    manifest_path = OUT / "public_cross_period_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "records": len(records), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
