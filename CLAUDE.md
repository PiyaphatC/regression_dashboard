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
**R² = 0.5002 | R²_adj = 0.4485 | n = 193 | 18 variables | OLS with HC3 robust SE**

| Variable | Role | Expected sign | Coef | p |
|---|---|---|---|---|
| `bus_stop_count` | Bus stops | + | −0.0582 | 0.765 |
| `win_count` | Motorcycle taxi stops | + | +0.4362 | 0.001 ** |
| `taxi_count` | Taxi stands | + | +0.6628 | 0.007 ** |
| `park_ride_car_count` | Park & ride lots | + | −0.1383 | 0.450 |
| `bike_parking_count` | Bike parking spots | + | −0.2283 | 0.098 . |
| `bike_share_count` | Bike share stations | + | +0.5607 | 0.005 ** |
| `bike_parking_mean_dist` | Mean dist to bike parking (m) | − | +0.1639 | 0.005 ** |
| `bike_share_mean_dist` | Mean dist to bike share (m) | − | +0.0149 | 0.823 |
| `bus_stop_mean_dist` | Mean dist to bus stops (m) | − | −0.0265 | 0.736 |
| `taxi_mean_dist` | Mean dist to taxi stands (m) | − | −0.0156 | 0.818 |
| `win_mean_dist` | Mean dist to motorcycle taxis (m) | − | +0.0446 | 0.456 |
| `sw_total_length` | Total sidewalk length (m) | + | +0.3671 | 0.014 * |
| `sidewalk_length_surface_neg1` | Poor-surface sidewalk (m) | − | −0.0756 | 0.023 * |
| `sidewalk_length_shade_neg1` | Unshaded sidewalk (m) | − | −0.0634 | 0.013 * |
| `sidewalk_length_obstacle_neg1` | Obstructed sidewalk (m) | − | +0.0031 | 0.909 |
| `POP25` | Population (25km buffer) | + | +0.0724 | 0.678 |
| `PRIM25` | Primary schools | + | +0.0364 | 0.738 |
| `STU25` | Students | + | +0.0188 | 0.773 |

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
- `win_count`, `bike_share_count`, `taxi_count` are the strongest predictors (last-mile feeders)
- `sw_total_length`, `sidewalk_length_surface_neg1`, `sidewalk_length_shade_neg1` are significant sidewalk variables
- `bus_stop_count`, `park_ride_car_count`, `bike_parking_count` have wrong-sign (negative) coefficients
- `bike_parking_mean_dist` has a wrong-sign (positive) coefficient
- Socio-economic variables (`POP25`, `PRIM25`, `STU25`) are not significant but included for theoretical completeness
- Dashboard has two Policy Recommendation tabs comparing exact prediction vs elasticity approximation for walkability upgrade scenarios
