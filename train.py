import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from feature_engineering import build_tabular_dataset

def train_nowcasting_models(dataset_path: str, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    df = build_tabular_dataset(dataset_path)

    # Separate features and targets
    target_cols = ["target_visibility_m", "target_ceiling_ft"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    X = df[feature_cols]
    y_vis = df["target_visibility_m"]
    y_ceil = df["target_ceiling_ft"]

    # 1. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    X_train, y_vis_train, y_ceil_train = X.iloc[:train_idx], y_vis.iloc[:train_idx], y_ceil.iloc[:train_idx]
    X_val, y_vis_val, y_ceil_val = X.iloc[train_idx:val_idx], y_vis.iloc[train_idx:val_idx], y_ceil.iloc[train_idx:val_idx]
    X_test, y_vis_test, y_ceil_test = X.iloc[val_idx:], y_vis.iloc[val_idx:], y_ceil.iloc[val_idx:]

    print(f"Dataset split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 2. Persistence Baseline: t+1 = t_0 (rule of thumb in aviation nowcasting)
    persist_vis_pred = X_test["visibility_m_t_0"]
    persist_vis_mae = mean_absolute_error(y_vis_test, persist_vis_pred)
    persist_vis_rmse = np.sqrt(mean_squared_error(y_vis_test, persist_vis_pred))
    print("\n--- BASELINE: PERSISTENCE MODEL ---")
    print(f"Visibility Persistence MAE:  {persist_vis_mae:.1f} m | RMSE: {persist_vis_rmse:.1f} m")

    # 3. Train XGBoost Visibility Regressor
    print("\n--- TRAINING: XGBOOST VISIBILITY REGRESSOR ---")
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
        verbose=False
    )
    
    vis_preds = vis_model.predict(X_test)
    xgb_vis_mae = mean_absolute_error(y_vis_test, vis_preds)
    xgb_vis_rmse = np.sqrt(mean_squared_error(y_vis_test, vis_preds))
    print(f"XGBoost Visibility MAE:      {xgb_vis_mae:.1f} m | RMSE: {xgb_vis_rmse:.1f} m")
    print(f"Improvement over Baseline:   {((persist_vis_mae - xgb_vis_mae)/persist_vis_mae)*100:.2f}%")

    # 4. Train XGBoost Cloud Ceiling Regressor
    print("\n--- TRAINING: XGBOOST CEILING REGRESSOR ---")
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
        verbose=False
    )

    # 5. Save Model Artifacts
    vis_model.save_model(os.path.join(model_dir, "xgboost_visibility.json"))
    ceil_model.save_model(os.path.join(model_dir, "xgboost_ceiling.json"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_names.joblib"))
    print(f"\n[SUCCESS] Models successfully saved to {model_dir}/")

if __name__ == "__main__":
    import glob, sys
    if len(sys.argv) > 1:
        target_csv = sys.argv[1]
    else:
        raw_files = glob.glob("data/raw/*.csv")
        target_csv = raw_files[0] if raw_files else "data/raw/KJFK_2023_2023.csv"
    train_nowcasting_models(target_csv)