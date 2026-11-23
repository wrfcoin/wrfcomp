"""Generate a spatially balanced 500-station CONUS reference list from NOAA ISD.

Downloads NOAA NCEI's official `isd-history.csv`, filters for active US stations
spanning 2020 through the end of the 2024 validation year inside the contiguous
United States, and applies greedy farthest-point sampling so the 500 selected
stations cover both dense corridors and rural/mountain gaps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

app = typer.Typer(help="Build canonical CONUS reference station list for WRFComp.")


def _select_spatially_diverse(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Pick stations by greedy farthest-point sampling so coverage is even across CONUS.

    Stations with a real WBAN (primary ASOS/airport sites) are preferred; others are used only if there
    are not enough. The first pick is the longest record, ties broken by station_id, so the result is
    deterministic for a given input file.
    """
    df = df.copy()
    df["has_wban"] = df["WBAN"] != "99999"
    df["span"] = df["END"].astype(int) - df["BEGIN"].astype(int)
    pool = df[df["has_wban"]]
    if len(pool) < target_count:
        pool = df
    pool = pool.sort_values(["span", "station_id"], ascending=[False, True]).reset_index(drop=True)
    if len(pool) <= target_count:
        return pool.drop(columns=["has_wban", "span"])

    lat = np.radians(pool["LAT"].to_numpy())
    lon = np.radians(pool["LON"].to_numpy())
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    chosen = [0]
    min_dist = np.linalg.norm(xyz - xyz[0], axis=1)
    while len(chosen) < target_count:
        nxt = int(np.argmax(min_dist))  # argmax returns the first maximum: deterministic
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(xyz - xyz[nxt], axis=1))

    out = pool.iloc[chosen].sort_values("station_id")
    return out.drop(columns=["has_wban", "span"])


@app.command()
def main(
    out: Path = typer.Option(Path("configs/stations_conus.csv"), help="Output CSV path."),
    count: int = typer.Option(500, help="Number of CONUS stations to select."),
    min_begin: int = typer.Option(20200101, help="Earliest required station start YYYYMMDD."),
    # isd-history.csv stops at 2025-08 (checked 2026-11-23), so requiring a later end date selects nothing.
    min_end: int = typer.Option(20241231, help="Minimum active end date YYYYMMDD."),
) -> None:
    typer.echo(f"Fetching NOAA ISD station history from {ISD_HISTORY_URL}...")
    df = pd.read_csv(ISD_HISTORY_URL, dtype={"USAF": str, "WBAN": str}, low_memory=False)

    # Filter to contiguous United States (CONUS) with valid coordinates & active span
    df = df.dropna(subset=["LAT", "LON", "ELEV(M)", "BEGIN", "END"])
    conus_mask = (
        (df["CTRY"] == "US")
        & (~df["STATE"].isin(["AK", "HI", "PR", "VI", "GU", "AS", "MP"]))
        & (df["LAT"].between(24.5, 49.5))
        & (df["LON"].between(-125.0, -66.5))
        & (df["BEGIN"].astype(int) <= min_begin)
        & (df["END"].astype(int) >= min_end)
        & (df["USAF"] != "999999")
    )
    filtered = df.loc[conus_mask].copy()
    filtered["station_id"] = filtered["USAF"].str.zfill(6) + filtered["WBAN"].str.zfill(5)
    filtered = filtered.drop_duplicates(subset=["station_id"])

    typer.echo(f"Found {len(filtered)} active CONUS stations; selecting {count} spatially stratified sites.")
    selected = _select_spatially_diverse(filtered, target_count=count)

    export_df = selected.rename(
        columns={
            "STATION NAME": "station_name",
            "STATE": "state",
            "LAT": "lat",
            "LON": "lon",
            "ELEV(M)": "elevation_m",
            "BEGIN": "begin_yyyymmdd",
            "END": "end_yyyymmdd",
        }
    )[["station_id", "station_name", "state", "lat", "lon", "elevation_m", "begin_yyyymmdd", "end_yyyymmdd"]]

    out.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out, index=False)
    typer.echo(f"Saved {len(export_df)} stations to {out}")


if __name__ == "__main__":
    app()