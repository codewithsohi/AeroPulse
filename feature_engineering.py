import os
import numpy as np
import pandas as pd
from parser import parse_metar_to_numeric

def build_tabular_dataset(raw_csv_path: str, processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Parses raw METAR strings using parser.py, regularizes to 1-hour timesteps,
    and constructs sequential 6-hour lag features. Caches to data/processed for instant reloads.
    """
    os.makedirs(processed_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(raw_csv_path))[0]
    cache_path = os.path.join(processed_dir, f"{base_name}_features.csv")
    if os.path.exists(cache_path):
        print(f"[+] Loading cached feature matrix from {cache_path}...")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("[+] Reading raw METAR file...")
    df_raw = pd.read_csv(raw_csv_path)
    # IEM provides columns: station, valid (YYYY-MM-DD HH:MM), metar
    df_raw["valid"] = pd.to_datetime(df_raw["valid"], errors="coerce")
    df_raw = df_raw.dropna(subset=["valid", "metar"]).sort_values("valid")

    parsed_records = []
    print(f"[+] Parsing {len(df_raw):,} raw METAR strings with parser.py...")
    for _, row in df_raw.iterrows():
        obs = parse_metar_to_numeric(str(row["metar"]))
        if obs:
            parsed_records.append({
                "timestamp": row["valid"],
                "visibility_m": obs.visibility_m,
                "ceiling_ft": obs.ceiling_ft,
                "temp_c": obs.temp_c,
                "dewpoint_c": obs.dewpoint_c,
                "dewpoint_depression": obs.dewpoint_depression,
                "altimeter_hpa": obs.altimeter_hpa,
                "wind_speed": obs.wind_speed,
                "wind_u": obs.wind_u,
                "wind_v": obs.wind_v,
            })

    df = pd.DataFrame(parsed_records)
    
    # 1. Round observation times to nearest full hour
    df["timestamp"] = df["timestamp"].dt.round("1h")
    # Keep the last observation if multiple exist within the same hour
    df = df.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

    # 2. Resample to strict 1-hour intervals to ensure consistent time steps
    df = df.resample("1h").asfreq()

    # Linear interpolation for short 1-2 hour gaps; drop remaining long missing blocks
    df = df.interpolate(method="time", limit=2)

    # 3. Cyclical time-of-day features (crucial for diurnal fog & temp cycles)
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # 4. Generate 6-hour sequential lags (t-5 to t-0)
    feature_cols = [
        "visibility_m", "ceiling_ft", "temp_c", "dewpoint_c",
        "dewpoint_depression", "altimeter_hpa", "wind_speed", "wind_u", "wind_v"
    ]
    
    lagged_dfs = [df[["hour_sin", "hour_cos"]]]
    for col in feature_cols:
        for lag in range(6):  # lag 0 is t_0, lag 5 is t_-5
            lagged_dfs.append(df[col].shift(lag).rename(f"{col}_t_{lag}"))

    # 5. Domain Atmospheric Trend Features (Deltas & Rolling Stats)
    df_features = pd.concat(lagged_dfs, axis=1)

    # Pressure tendency (Altimeter change over 3 hours: classic storm/front indicator)
    df_features["altimeter_tendency_3h"] = (
        df_features["altimeter_hpa_t_0"] - df_features["altimeter_hpa_t_2"]
    )
    # Dewpoint depression tendency (nearing 0 indicates fog condensation)
    df_features["dewpoint_depr_tendency_3h"] = (
        df_features["dewpoint_depression_t_0"] - df_features["dewpoint_depression_t_2"]
    )
    # Visibility 6-hour moving average & min
    df_features["vis_min_6h"] = df["visibility_m"].rolling(6).min()
    df_features["vis_mean_6h"] = df["visibility_m"].rolling(6).mean()

    # 6. Targets for t+1 (Next Hour)
    df_features["target_visibility_m"] = df["visibility_m"].shift(-1)
    df_features["target_ceiling_ft"] = df["ceiling_ft"].shift(-1)
    
    # Drop rows with NaN caused by shifting
    clean_df = df_features.dropna()
    print(f"[SUCCESS] Feature matrix ready: {clean_df.shape[0]} samples with {clean_df.shape[1]} features.")
    clean_df.to_csv(cache_path)
    print(f"[+] Cached feature matrix saved to {cache_path}")
    return clean_df