# Bangkok Ridership Elasticity — Project Notes

## Project Overview
Log-log OLS regression of Bangkok rail station ridership (daily entries) against sidewalk quality and last-mile feeder mode variables. Outputs are elasticity coefficients. Dashboard is built with Streamlit (`app.py`).

## Data
- **Main dataset:** `Output/combined_station_summary_expanded_rev16.csv`
  - 193 stations, 87 columns
  - Target: `entry` (average daily entries)
  - `station_typology`: Local Station, Interchange Station, Major Hub, Destination Gateway, Intermodal Transportation Hub
- **Ridership source:** `Output/ridership_raw.xlsx`

## Current Best Model
**R² = 0.4328 | R²_adj = 0.4114 | n = 193 | OLS with HC3 robust SE**

| Variable | Role | Expected sign | Coef | p |
|---|---|---|---|---|
| `surface_pct_1` | Sidewalk surface quality (%) | + | +0.1771 | 0.058 . |
| `shade_pct_1` | Sidewalk shade (%) | + | +0.0501 | 0.302 |
| `obs_pct_1` | Obstacle-free sidewalk (%) | + | +0.0529 | 0.489 |
| `win_count` | Motorcycle taxi stops | + | +0.5710 | <0.001 *** |
| `bike_share_count` | Bike share stations | + | +0.6215 | <0.001 *** |
| `taxi_count` | Taxi stands | + | +0.5963 | <0.001 *** |
| `bus_stop_mean_dist` | Mean distance to bus stops (m) | − | −0.0557 | 0.289 |

This variable set is coded as `DEFAULT_FEATURES` in `app.py`.

## Critical Data Finding
`surface_pct_1` is **mathematically identical** to `sidewalk_length_surface_1 / sw_total_length × 100` (r = 1.000). They represent the same measurement — the percentage of total sidewalk with surface quality rating = 1 (good). Never include both in the same model.

The three genuinely independent sidewalk quality dimensions are:
1. **Surface** → `surface_pct_1` (or equivalently `sidewalk_length_surface_1`)
2. **Shade** → `shade_pct_1` (or `sidewalk_length_shade_1`)
3. **Obstacle-free** → `obs_pct_1` (or `sidewalk_length_obstacle_1`)

## Variable Encoding
Quality ratings follow a −1 / 0 / 1 scale:
- `_neg1` = poor quality
- `_0` = neutral/medium
- `_1` = good quality

Percentage columns (`barrier_pct_*`, `shade_pct_*`, `obs_pct_*`) sum to ~100% per station.
Length columns (`sidewalk_length_surface_*`, `sidewalk_length_shade_*`, `sidewalk_length_obstacle_*`) are in metres.

## Adjustable Buffer Radius
- `Output/features_by_radius.csv` contains pre-computed features at 15 radii (100m–1500m, step 100m)
- Sidebar slider lets users pick a buffer radius; all model results update accordingly
- Tab 4 ("Buffer Sensitivity") shows how R² and coefficients change across radii
- Pre-computed via `buffer_radius_explorer/precompute_radii.py` in the main project repo

## Model Notes
- All models use log-log specification: `log(y) ~ log(x + offset)` with offset = 1.0
- `bus_stop_count` has a wrong-sign (negative) coefficient — use `bus_stop_mean_dist` instead if bus connectivity is needed
- `win_count`, `bike_share_count`, `taxi_count` are the strongest predictors (last-mile feeders)
- Shade and obstacle-free sidewalk are directionally correct but not statistically significant after controlling for feeder modes
