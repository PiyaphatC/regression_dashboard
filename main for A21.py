import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

folder_path = "/mnt/d/Living labs/PTSE/Raw files (A21)"

# --- CATEGORY MAP (FINAL) ---
category_map = {
    "ป้ายรถเมล์": "bus_stop",
    "วินมอเตอร์ไซต์": "win",
    "คิวรถสองแถว": "songtaew",
    "คิวรถสี่ล้อเล็ก": "minibus",
    "คิวรถแท็กซี่": "taxi",
    "คิวรถสามล้อ": "tuktuk",
    "จุดจอดรับส่งสำหรับรถยนต์": "kiss_ride",
    "จุดจอดรับส่งสำหรับรถยนต์ (Kiss & Ride)": "kiss_ride",
    "จุดจอดแล้วจร (รถยนต์)": "park_ride_car",
    "จุดจอดแล้วจร(รถยนต์)": "park_ride_car",
    "จุดจอดแล้วจร (มอเตอร์ไซต์)": "park_ride_moto",
    "จุดจอดแล้วจร(มอเตอร์ไซต์)": "park_ride_moto",
    "จุดจอดจักรยาน": "bike_parking",
    "Bike-Sharing": "bike_share",
    "Scooter-sharing": "scooter_share",
}

results = []

for file in os.listdir(folder_path):
    if not file.endswith(".xlsx"):
        continue

    file_path = os.path.join(folder_path, file)
    print(f"Processing: {file}")

    df = pd.read_excel(file_path, sheet_name=0)

    # --- CLEAN COLUMNS ---
    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    # --- DETECT KEY COLUMNS ---
    distance_col = [c for c in df.columns if "ระยะทาง" in c][0]
    lat_col = [c for c in df.columns if "ละติจูด (" in c][-1]
    lon_col = [c for c in df.columns if "ลองติจูด (" in c][-1]

    # --- CLEAN CATEGORY TEXT ---
    df["ประเภท_clean"] = (
        df["ประเภท"]
        .astype(str)
        .str.strip()
        .str.replace(" (Kiss & Ride)", "", regex=False)
        .str.replace("จุดจอดแล้วจร (รถยนต์)", "จุดจอดแล้วจร(รถยนต์)", regex=False)
        .str.replace("จุดจอดแล้วจร (มอเตอร์ไซต์)", "จุดจอดแล้วจร(มอเตอร์ไซต์)", regex=False)
        .map(category_map)
    )

    # --- CLEAN NUMERIC ---
    df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # --- CREATE POI ID ---
    df["poi_id"] = (
        df[lat_col].round(5).astype(str) + "_" +
        df[lon_col].round(5).astype(str)
    )

    station_name = os.path.splitext(file)[0]
    station_data = {"station": station_name}

    # --- LOOP EACH CATEGORY ---
    for clean in set(category_map.values()):

        df_cat = df[
            (df["ประเภท_clean"] == clean) &
            (df["รายละเอียด"].notna())
        ].copy()

        # remove rows without lat/lon
        df_cat = df_cat.dropna(subset=[lat_col, lon_col])

        if df_cat.empty:
            station_data[f"{clean}_count"] = 0
            station_data[f"{clean}_min_dist"] = 0
            station_data[f"{clean}_mean_dist"] = 0
            continue

        # --- DEDUPLICATE POI ---
        df_unique = (
            df_cat.groupby("poi_id", as_index=False)
            .agg({distance_col: "min"})
        )

        # --- METRICS ---
        station_data[f"{clean}_count"] = len(df_unique)
        station_data[f"{clean}_min_dist"] = df_unique[distance_col].min()
        station_data[f"{clean}_mean_dist"] = df_unique[distance_col].mean()

    results.append(station_data)

# --- SAVE OUTPUT ---
summary_df = pd.DataFrame(results)

summary_df.to_csv(
    "/mnt/d/Living labs/PTSE/clean_station_summary_A21.csv",
    index=False,
    encoding="utf-8-sig"
)

print(" Done")