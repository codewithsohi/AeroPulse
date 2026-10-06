import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from parser import parse_metar_to_numeric, determine_flight_category

class MetarNowcaster:
    def __init__(self, model_dir: str = "models"):
        self.feature_names = joblib.load(f"{model_dir}/feature_names.joblib")
        self.vis_model = xgb.XGBRegressor()
        self.vis_model.load_model(f"{model_dir}/xgboost_visibility.json")
        self.ceil_model = xgb.XGBRegressor()
        self.ceil_model.load_model(f"{model_dir}/xgboost_ceiling.json")

    def predict_next_hour(self, past_6_metars: list[str]) -> dict:
        """
        Accepts a list of 6 raw METAR strings ordered chronologically [t-5, t-4, t-3, t-2, t-1, t0].
        Returns predicted visibility, ceiling, and flight category for t+1.
        """
        assert len(past_6_metars) == 6, "Must provide exactly 6 hourly observations."
        parsed = [parse_metar_to_numeric(m) for m in past_6_metars]

        # Extract features for each lag (reverse order so index 0 = t_0, index 5 = t_-5)
        row_dict = {}
        for lag, obs in enumerate(reversed(parsed)):
            row_dict[f"visibility_m_t_{lag}"] = obs.visibility_m
            row_dict[f"ceiling_ft_t_{lag}"] = obs.ceiling_ft
            row_dict[f"temp_c_t_{lag}"] = obs.temp_c or 15.0
            row_dict[f"dewpoint_c_t_{lag}"] = obs.dewpoint_c or 10.0
            row_dict[f"dewpoint_depression_t_{lag}"] = obs.dewpoint_depression or 5.0
            row_dict[f"altimeter_hpa_t_{lag}"] = obs.altimeter_hpa or 1013.25
            row_dict[f"wind_speed_t_{lag}"] = obs.wind_speed or 0.0
            row_dict[f"wind_u_t_{lag}"] = obs.wind_u or 0.0
            row_dict[f"wind_v_t_{lag}"] = obs.wind_v or 0.0

        # Derived tendencies
        row_dict["altimeter_tendency_3h"] = row_dict["altimeter_hpa_t_0"] - row_dict["altimeter_hpa_t_2"]
        row_dict["dewpoint_depr_tendency_3h"] = row_dict["dewpoint_depression_t_0"] - row_dict["dewpoint_depression_t_2"]
        vis_series = [obs.visibility_m for obs in parsed]
        row_dict["vis_min_6h"] = min(vis_series)
        row_dict["vis_mean_6h"] = sum(vis_series) / len(vis_series)
        row_dict["hour_sin"] = 0.0  # Optional default or calculate from current UTC hour
        row_dict["hour_cos"] = 1.0

        # Create 1-row DataFrame aligned to feature order
        df_input = pd.DataFrame([row_dict])[self.feature_names]

        # Predict continuous values
        pred_vis = float(np.clip(self.vis_model.predict(df_input)[0], 0, 10000))
        pred_ceil = float(np.clip(self.ceil_model.predict(df_input)[0], 0, 20000))
        pred_category = determine_flight_category(pred_vis, pred_ceil)

        return {
            "predicted_visibility_m": round(pred_vis, 1),
            "predicted_ceiling_ft": round(pred_ceil, 1),
            "predicted_flight_category": pred_category,
        }

if __name__ == "__main__":
    print("=" * 65)
    print("      AEROPULSE: TIME-SERIES NOWCASTING (t+1 PREDICTION)")
    print("=" * 65)
    nowcaster = MetarNowcaster()

    # 6 consecutive hourly METAR observations demonstrating incoming fog:
    sample_metar_sequence = [
        "KJFK 010000Z 24010KT 10SM CLR 20/15 A2992",       # t-5: Clear VFR
        "KJFK 010100Z 24009KT 10SM CLR 19/15 A2991",       # t-4: Cooling
        "KJFK 010200Z 24008KT 8SM SCT040 18/15 A2990",     # t-3: Clouds forming
        "KJFK 010300Z 24006KT 6SM BKN030 17/15 A2989",     # t-2: Ceiling lowering
        "KJFK 010400Z 20004KT 3SM BR BKN015 16/15 A2988",  # t-1: Mist, IFR approaching
        "KJFK 010500Z 00000KT 1SM FG OVC005 15/15 A2987",  # t_0: Fog formed, LIFR
    ]

    print("\n[+] Input: Past 6 Hours of Hourly METARs (t-5 to t0):")
    for i, m in enumerate(sample_metar_sequence):
        t_label = f"t-{5-i}" if (5-i) > 0 else "t_0 (latest)"
        print(f"  [{t_label}]: {m}")

    result = nowcaster.predict_next_hour(sample_metar_sequence)
    print("\n[+] AeroPulse Model Forecast for Next Hour (t+1):")
    print(f"  * Predicted Visibility:   {result['predicted_visibility_m']:,} meters")
    print(f"  * Predicted Cloud Ceiling: {result['predicted_ceiling_ft']:,} feet AGL")
    print(f"  * Predicted Flight Rules:  {result['predicted_flight_category']}")
    print("=" * 65)