import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from feature_engineering import build_tabular_dataset

def plot_evaluation(dataset_path: str, model_dir: str = "models"):
    sns.set_theme(style="whitegrid")
    df = build_tabular_dataset(dataset_path)
    
    feature_cols = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    X = df[feature_cols]
    y_vis = df["target_visibility_m"]

    # Load test split (last 15%)
    val_idx = int(len(df) * 0.85)
    X_test = X.iloc[val_idx:]
    y_test = y_vis.iloc[val_idx:]

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "xgboost_visibility.json"))
    preds = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: 72-hour Event Tracking
    sample_window = 72
    axes[0].plot(range(sample_window), y_test.iloc[:sample_window], label="Actual Visibility", color="#1f77b4", lw=2)
    axes[0].plot(range(sample_window), preds[:sample_window], label="XGBoost Predicted (t+1)", color="#ff7f0e", linestyle="--", lw=2)
    axes[0].plot(range(sample_window), X_test["visibility_m_t_0"].iloc[:sample_window], label="Persistence (t_0)", color="gray", alpha=0.5)
    axes[0].set_title(f"AeroPulse Nowcasting: 72-Hour Visibility Tracking (Meters)", fontsize=13)
    axes[0].set_xlabel("Hours", fontsize=11)
    axes[0].set_ylabel("Visibility (Meters)", fontsize=11)
    axes[0].legend()

    # Plot 2: Top 10 Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, ax=axes[1], palette="viridis", legend=False)
    axes[1].set_title("Top 10 Most Predictive Atmospheric Features", fontsize=13)
    axes[1].set_xlabel("Gain / Importance", fontsize=11)

    plt.tight_layout()
    plt.savefig("nowcasting_evaluation.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved evaluation figure: nowcasting_evaluation.png")

if __name__ == "__main__":
    import glob, sys
    if len(sys.argv) > 1:
        target_csv = sys.argv[1]
    else:
        raw_files = glob.glob("data/raw/*.csv")
        target_csv = raw_files[0] if raw_files else "data/raw/KJFK_2023_2023.csv"
    plot_evaluation(target_csv)