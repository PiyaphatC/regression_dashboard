# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project builds a **station-level ridership elasticity model** for public transport stations in Bangkok. The pipeline:

1. Parse raw field survey XLSX files → per-source station summary CSVs
2. Combine the 3 source CSVs into one master file
3. Merge external POI data (in progress)
4. Run regression / elasticity analysis per station

## Environment

Uses **conda** (managed via VS Code ms-python extension). Run scripts with:

```bash
python "main for A21.py"
python "Accessible-Public-Transportation-Station/combine.py"
```

No requirements.txt exists. Core dependencies: `pandas`, `openpyxl`, `tqdm`.

## Data Architecture

### Raw Inputs (3 survey sources)
| Folder | Source | Stations |
|---|---|---|
| `Raw files (A21)/` | BTS Skytrain (N-line) | ~52 |
| `Raw files (AEC)/` | AEC dataset | ~25 |
| `Raw files (CUTI)/` | CUTI dataset | ~97 |

Each XLSX file = one station. Columns include Thai-language category names, distance, lat/lon.

### Processing Scripts
- **`main for A21.py`** — processes `Raw files (A21)/` → `clean_station_summary_A21.csv`. More robust than `main.py` (handles variant spellings, uses `set(category_map.values())` for dedup).
- **`main.py`** — older template for AEC/CUTI sources; same logic with minor column-name differences.
- **`01_combine_sources.py`** — combines all 3 CSVs → `Output/combined_station_summary.csv` (174 stations). Strips A21 survey-form prefix from station names, adds `source` column, warns on cross-source duplicates.
- **`02_elasticity_model.py`** — log-log OLS elasticity model. Configurable feature set, optional external ridership merge, HC3-robust SEs, VIF diagnostics. Outputs `Output/elasticity_results.csv` and `Output/elasticity_results.txt`.
- **`Accessible-Public-Transportation-Station/combine.py`** — legacy combiner (references outdated filename; replaced by `01_combine_sources.py`).
- **`Accessible-Public-Transportation-Station/main_part_2.py`** — filters to 100m radius and renames columns to `{mode}_count_100m` / `{mode}_mean_100m`.

### Output Schema
`clean_station_summary_*.csv` and `All_poi_part1.csv` share the same wide format:

```
station | {mode}_count | {mode}_min_dist | {mode}_mean_dist
```

Transport modes: `bus_stop`, `win`, `songtaew`, `minibus`, `taxi`, `tuktuk`, `kiss_ride`, `park_ride_car`, `park_ride_moto`, `bike_parking`, `bike_share`, `scooter_share`

After part 2 processing, columns become `{mode}_count_100m` and `{mode}_mean_100m`.

### POI Deduplication Logic
Each raw row is a POI observation from a surveyor. POIs are deduplicated per station by rounding lat/lon to 5 decimal places → `poi_id`. Only minimum distance is kept per unique POI.

## Pending Work

- **Ridership data**: `02_elasticity_model.py` is ready but requires a ridership column or file. Set `RIDERSHIP_FILE` and `RIDERSHIP_COL` in the config block.
- **POI integration**: External POI data is being retrieved; add new columns to `FEATURE_CONFIG` in `02_elasticity_model.py` once merged.
- **`Output/combined_station_summary.csv`** is the current master file (174 stations). `Output/elasticity_results.*` will appear after the model runs.

## Key Design Conventions

- Station name is the join key across all datasets; keep it clean and consistent.
- Column naming pattern: `{mode}_{metric}` (e.g., `bus_stop_count`, `win_min_dist`).
- Encoding: all CSVs written with `encoding="utf-8-sig"` (BOM) to support Thai characters in Excel.
- A21 raw filenames include a prefix `_PTSE_ฟอร์มกรอกข้อมูล_` that must be stripped to get the clean station name.
