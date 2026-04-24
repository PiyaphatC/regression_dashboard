"""
01_combine_sources.py
---------------------
Combines station-level accessibility summary files from three survey sources
(A21, AEC, CUTI) into a single master dataset.

Inputs  : clean_station_summary_A21.csv
          clean_station_summary_AEC.csv
          clean_station_summary_CUTI.csv
Output  : Output/combined_station_summary.csv

Usage   : python 01_combine_sources.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "A21":  os.path.join(BASE_DIR, "clean_station_summary_A21.csv"),
    "AEC":  os.path.join(BASE_DIR, "clean_station_summary_AEC.csv"),
    "CUTI": os.path.join(BASE_DIR, "clean_station_summary_CUTI.csv"),
}

OUTPUT_DIR  = os.path.join(BASE_DIR, "Output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_station_summary.csv")

# Prefix attached to A21 station names by the survey form template
A21_STATION_PREFIX = "_PTSE_ฟอร์มกรอกข้อมูล_"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source(path: str, source_label: str) -> pd.DataFrame:
    """Read one source CSV, strip BOM from column names, and tag rows."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Remove any residual BOM characters from column names
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    df.insert(0, "source", source_label)
    return df


def clean_station_name(name: str, source: str) -> str:
    """Normalise station name: strip survey-form prefix (A21 only)."""
    name = str(name).strip()
    if source == "A21" and name.startswith(A21_STATION_PREFIX):
        name = name[len(A21_STATION_PREFIX):]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for label, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue

        df = load_source(path, label)
        df["station"] = df.apply(
            lambda row: clean_station_name(row["station"], row["source"]), axis=1
        )

        print(f"  Loaded {label:4s}: {len(df):>3} stations  |  {path}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No source files were loaded. Check file paths.")

    combined = pd.concat(frames, ignore_index=True)

    # Sanity checks
    duplicates = combined.duplicated(subset=["station"], keep=False)
    if duplicates.any():
        print(
            f"\n  [WARNING] {duplicates.sum()} rows share a station name "
            f"across sources. Review before modelling:\n"
            f"{combined.loc[duplicates, ['source', 'station']].to_string(index=False)}"
        )

    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(
        f"\n  Combined: {len(combined)} stations total  →  {OUTPUT_FILE}"
    )
    print(f"  Columns : {list(combined.columns[:6])} ... ({len(combined.columns)} total)")


if __name__ == "__main__":
    main()
