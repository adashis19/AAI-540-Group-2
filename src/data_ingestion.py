"""
Reproducible BRFSS data ingestion and harmonisation.

Reads the raw SAS-XPT files for 2021, 2022 and 2023, selects the harmonised
feature set, derives the binary heart-disease target from _MICHD, converts
"don't know / refused" survey codes to missing values, and writes a tidy
parquet file per year plus a combined parquet.

Run:  python -m src.data_ingestion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyreadstat

# Allow "python src/data_ingestion.py" as well as "-m src.data_ingestion".
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """BRFSS XPT column names can carry trailing spaces; strip them."""
    df.columns = [c.strip() for c in df.columns]
    return df


def load_year(year: int) -> pd.DataFrame:
    """Load one BRFSS year, keep only needed raw columns, clean and label it."""
    path = C.XPT_FILES[year]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file for {year}: {path}. Run the download step first."
        )

    needed_raw = list(C.FEATURE_RENAME.keys()) + C.LEAKAGE_COLS
    # Read full file (usecols keeps memory reasonable: ~15 cols vs 300+).
    df, _ = pyreadstat.read_xport(
        str(path), encoding=C.XPT_ENCODING, usecols=needed_raw
    )
    df = _strip_columns(df)

    # --- Target -----------------------------------------------------------
    # _MICHD: 1 -> has CHD/MI, 2 -> no. Anything else -> drop (unknown label).
    michd = df[C.TARGET_RAW]
    target = michd.map({1.0: 1, 2.0: 0})
    df[C.TARGET] = target

    # --- Rename features ---------------------------------------------------
    df = df.rename(columns=C.FEATURE_RENAME)

    keep = C.ALL_FEATURES + [C.TARGET]
    df = df[keep].copy()
    df["survey_year"] = year

    # --- Convert don't-know / refused codes to NaN -------------------------
    for col, codes in C.DK_REFUSED_CODES.items():
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(codes))

    # --- Map "none" day codes (88) to 0 ------------------------------------
    for col, codes in C.ZERO_CODES.items():
        if col in df.columns:
            df[col] = df[col].replace({c: 0 for c in codes})

    # --- BMI is stored *100 in BRFSS; convert to real BMI -----------------
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"] / 100.0
        # Guard against implausible values (BRFSS caps ~12-98).
        df.loc[(df["bmi"] < 10) | (df["bmi"] > 100), "bmi"] = pd.NA
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # --- Drop rows without a valid label -----------------------------------
    df = df.dropna(subset=[C.TARGET]).reset_index(drop=True)
    df[C.TARGET] = df[C.TARGET].astype(int)

    return df


def run(save: bool = True) -> pd.DataFrame:
    frames = []
    for year in (2021, 2022, 2023):
        print(f"[ingest] loading {year} ...", flush=True)
        dfy = load_year(year)
        pos = int(dfy[C.TARGET].sum())
        print(
            f"[ingest] {year}: rows={len(dfy):,} "
            f"positives={pos:,} ({pos / len(dfy):.2%})"
        )
        if save:
            out = C.PROCESSED_DIR / f"brfss_{year}.parquet"
            dfy.to_parquet(out, index=False)
            print(f"[ingest] wrote {out}")
        frames.append(dfy)

    combined = pd.concat(frames, ignore_index=True)
    if save:
        out = C.PROCESSED_DIR / "brfss_combined.parquet"
        combined.to_parquet(out, index=False)
        print(f"[ingest] wrote {out}  (rows={len(combined):,})")
    return combined


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest and harmonise BRFSS data.")
    ap.add_argument("--no-save", action="store_true", help="do not write parquet")
    args = ap.parse_args()
    run(save=not args.no_save)
