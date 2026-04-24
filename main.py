import pandas as pd
import os
from tqdm import tqdm

folder_path = "/mnt/d/Living labs/PTSE/Raw files ()"

category_map = {
    "ป้ายรถเมล์": "bus_stop",
    "วินมอเตอร์ไซต์": "win",
    "คิวรถสองแถว": "songtaew",
    "คิวรถสี่ล้อเล็ก": "minibus",
    "คิวรถแท็กซี่": "taxi",
    "คิวรถสามล้อ": "tuktuk",
    "จุดจอดรับส่งสำหรับรถยนต์": "kiss_ride",
    "จุดจอดแล้วจร(รถยนต์)": "park_ride_car",
    "จุดจอดแล้วจร(มอเตอร์ไซต์)": "park_ride_moto",
    "จุดจอดจักรยาน": "bike_parking",
    "Bike-Sharing": "bike_share",
    "Scooter-sharing": "scooter_share",
}

results = []

for file in tqdm(os.listdir(folder_path)):
    if file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        # print(file_path)
        df = pd.read_excel(file_path, sheet_name= 0)

        # Clean columns
        df.columns = (
            df.columns
            .str.replace("\n", " ")
            .str.strip()
        )

        # Detect distance column
        distance_col = [c for c in df.columns if "ระยะทาง" in c][0]

        # Clean category
        df["ประเภท_clean"] = (
            df["ประเภท"]
            .astype(str)
            .str.strip()
            .map(category_map)
        )

        station_name = os.path.splitext(file)[0]
        station_data = {"station": station_name}

        # --- ADD THIS BEFORE the loop (only once per file) ---
        lat_col = [c for c in df.columns if "ละติจูด" in c][0]
        lon_col = [c for c in df.columns if "ลองติจูด" in c][0]

        df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        df["poi_id"] = (
            df[lat_col].round(5).astype(str) + "_" +
            df[lon_col].round(5).astype(str)
        )

# --- REPLACE YOUR LOOP WITH THIS ---
        for raw, clean in category_map.items():

            df_cat = df[
                (df["ประเภท_clean"] == clean) &
                (df["รายละเอียด"].notna())
            ].copy()

            # remove rows without lat/lon (important)
            df_cat = df_cat.dropna(subset=[lat_col, lon_col])

            if df_cat.empty:
                station_data[f"{clean}_count"] = 0
                station_data[f"{clean}_min_dist"] = 0
                station_data[f"{clean}_mean_dist"] = 0
                continue

            # 🔥 KEY FIX: deduplicate POI using lat/lon
            df_unique = (
                df_cat.groupby("poi_id", as_index=False)
                .agg({distance_col: "min"})
            )

            count = len(df_unique)

            min_dist = df_unique[distance_col].min()
            mean_dist = df_unique[distance_col].mean()

            station_data[f"{clean}_count"] = count
            station_data[f"{clean}_min_dist"] = min_dist if pd.notna(min_dist) else 0
            station_data[f"{clean}_mean_dist"] = mean_dist if pd.notna(mean_dist) else 0

        results.append(station_data)

summary_df = pd.DataFrame(results)
summary_df.to_csv("/mnt/d/Living labs/PTSE/clean_station_summary_A21.csv", index=False, encoding="utf-8-sig")

print(" Done")