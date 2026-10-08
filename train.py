import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from feature_engineering import build_tabular_dataset

def train_nowcasting_models(dataset_paths: list[str] | str | None = None, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    # 1. Resolve raw CSV paths
    if dataset_paths is None:
        raw_files = sorted(glob.glob("data/raw/*.csv"))
    elif isinstance(dataset_paths, str):
        raw_files = [dataset_paths]
    else:
        raw_files = sorted(dataset_paths)

    if not raw_files:
        raise FileNotFoundError("No raw METAR CSV files found in data/raw/ to train on.")

    print("=" * 65)
    print("      AEROPULSE: MULTI-AIRPORT MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"[+] Discovered {len(raw_files)} airport dataset(s):")
    for f in raw_files:
        print(f"    - {os.path.basename(f)}")

    # 2. Process each airport independently & split chronologically
    train_dfs, val_dfs, test_dfs = [], [], []

    for file_path in raw_files:
        station_name = os.path.basename(file_path).split("_")[0]
        print(f"\n[+] Processing and caching feature matrix for {station_name}...")
        df_station = build_tabular_dataset(file_path)

        n = len(df_station)
        train_idx = int(n * 0.70)
        val_idx = int(n * 0.85)

        train_dfs.append(df_station.iloc[:train_idx])
        val_dfs.append(df_station.iloc[train_idx:val_idx])
        test_dfs.append(df_station.iloc[val_idx:])
        print(f"    -> {station_name}: {n:,} total hours | Train: {train_idx:,} | Val: {val_idx - train_idx:,} | Test: {n - val_idx:,}")

    # Combine per-station splits into universal multi-airport matrices
    df_train = pd.concat(train_dfs, axis=0, ignore_index=True)
    df_val = pd.concat(val_dfs, axis=0, ignore_index=True)
    df_test = pd.concat(test_dfs, axis=0, ignore_index=True)

    print("\n[+] Universal Multi-Airport Split Summary:")
    print(f"    * Train Samples:      {len(df_train):,}")
    print(f"    * Validation Samples: {len(df_val):,}")
    print(f"    * Test Samples:       {len(df_test):,}")

    # Separate features and targets
    target_cols = ["target_visibility_m", "target_ceiling_ft"]
    feature_cols = [c for c in df_train.columns if c not in target_cols]

    X_train, y_vis_train, y_ceil_train = df_train[feature_cols], df_train["target_visibility_m"], df_train["target_ceiling_ft"]
    X_val, y_vis_val, y_ceil_val = df_val[feature_cols], df_val["target_visibility_m"], df_val["target_ceiling_ft"]
    X_test, y_vis_test, y_ceil_test = df_test[feature_cols], df_test["target_visibility_m"], df_test["target_ceiling_ft"]

    # 3. Persistence Baseline: t+1 = t_0 (Aviation gold standard rule)
    persist_vis_pred = X_test["visibility_m_t_0"]
    persist_vis_mae = mean_absolute_error(y_vis_test, persist_vis_pred)
    persist_vis_rmse = np.sqrt(mean_squared_error(y_vis_test, persist_vis_pred))

    persist_ceil_pred = X_test["ceiling_ft_t_0"]
    persist_ceil_mae = mean_absolute_error(y_ceil_test, persist_ceil_pred)
    persist_ceil_rmse = np.sqrt(mean_squared_error(y_ceil_test, persist_ceil_pred))

    print("\n" + "=" * 65)
    print("      BASELINE BENCHMARK: PERSISTENCE MODEL (t+1 = t_0)")
    print("=" * 65)
    print(f"  * Visibility Persistence -> MAE: {persist_vis_mae:8.1f} m  | RMSE: {persist_vis_rmse:8.1f} m")
    print(f"  * Ceiling Persistence    -> MAE: {persist_ceil_mae:8.1f} ft | RMSE: {persist_ceil_rmse:8.1f} ft")

    # 4. Train XGBoost Visibility Regressor
    print("\n" + "=" * 65)
    print("      STAGE 1/2: TRAINING XGBOOST VISIBILITY REGRESSOR")
    print("=" * 65)
    vis_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=25
    )
    vis_model.fit(
        X_train, y_vis_train,
        eval_set=[(X_val, y_vis_val)],
        verbose=25
    )

    vis_preds = vis_model.predict(X_test)
    xgb_vis_mae = mean_absolute_error(y_vis_test, vis_preds)
    xgb_vis_rmse = np.sqrt(mean_squared_error(y_vis_test, vis_preds))
    vis_improvement = ((persist_vis_mae - xgb_vis_mae) / persist_vis_mae) * 100

    print(f"\n[+] Visibility Test Evaluation:")
    print(f"    * XGBoost MAE:            {xgb_vis_mae:8.1f} m  | RMSE: {xgb_vis_rmse:8.1f} m")
    print(f"    * Persistence MAE:        {persist_vis_mae:8.1f} m  | RMSE: {persist_vis_rmse:8.1f} m")
    print(f"    * Improvement over Baseline: {vis_improvement:+.2f}%")

    # 5. Train XGBoost Cloud Ceiling Regressor
    print("\n" + "=" * 65)
    print("      STAGE 2/2: TRAINING XGBOOST CEILING REGRESSOR")
    print("=" * 65)
    ceil_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=25
    )
    ceil_model.fit(
        X_train, y_ceil_train,
        eval_set=[(X_val, y_ceil_val)],
        verbose=25
    )

    ceil_preds = ceil_model.predict(X_test)
    xgb_ceil_mae = mean_absolute_error(y_ceil_test, ceil_preds)
    xgb_ceil_rmse = np.sqrt(mean_squared_error(y_ceil_test, ceil_preds))
    ceil_improvement = ((persist_ceil_mae - xgb_ceil_mae) / persist_ceil_mae) * 100

    print(f"\n[+] Ceiling Test Evaluation:")
    print(f"    * XGBoost MAE:            {xgb_ceil_mae:8.1f} ft | RMSE: {xgb_ceil_rmse:8.1f} ft")
    print(f"    * Persistence MAE:        {persist_ceil_mae:8.1f} ft | RMSE: {persist_ceil_rmse:8.1f} ft")
    print(f"    * Improvement over Baseline: {ceil_improvement:+.2f}%")

    # 6. Save Model Artifacts
    vis_model.save_model(os.path.join(model_dir, "xgboost_visibility.json"))
    ceil_model.save_model(os.path.join(model_dir, "xgboost_ceiling.json"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_names.joblib"))
    print("\n" + "=" * 65)
    print(f"[SUCCESS] Trained models and feature schema saved to '{model_dir}/'")
    print("=" * 65)

if __name__ == "__main__":
    import glob, sys
    custom_target = sys.argv[1] if len(sys.argv) > 1 else None
    train_nowcasting_models(custom_target)