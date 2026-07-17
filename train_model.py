"""
train_model.py — Sales Prediction System: standalone training pipeline.

Run this script *once* (or whenever the data changes) to regenerate
``model.pkl``, ``scaler.pkl``, and ``features.pkl``.  After that,
``streamlit run app.py`` picks them up without re-training.

Usage:
    python train_model.py
    python train_model.py --csv path/to/Advertising.csv

Convention: exceptions are propagated with explicit ``raise`` — no silent
``except: pass`` blocks.  This keeps Pylance type-narrowing happy and
surfaces real failures loudly.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils import (
    FEATURE_COLS,
    TARGET_COL,
    clean_data,
    evaluate_model,
    load_data,
    save_artifacts,
)

# ─── Optional XGBoost ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor  # type: ignore[import]
    _XGB_AVAILABLE = True
except ImportError as _xgb_err:
    _XGB_AVAILABLE = False
    print(f"[warn] XGBoost not available ({_xgb_err}).  Skipping XGBRegressor.")

RANDOM_STATE = 42


def _build_model_registry() -> dict[str, object]:
    """Return ordered dict of model name → unfitted estimator."""
    reg: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.01, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }
    if _XGB_AVAILABLE:
        reg["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    return reg


def _tune_best_tree(
    best_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Run GridSearchCV for the best tree-based model and return the refit estimator.

    Args:
        best_name: Name of the top-performing tree model from the comparison table.
        X_train: Scaled training features.
        y_train: Training target.

    Returns:
        Best estimator (already refit on full training set by GridSearchCV).
    """
    if best_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        }
        base = RandomForestRegressor(random_state=RANDOM_STATE)

    elif best_name == "Gradient Boosting":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)

    elif best_name == "XGBoost" and _XGB_AVAILABLE:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
        base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)  # type: ignore[misc]

    else:  # Decision Tree fallback
        param_grid = {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        }
        base = DecisionTreeRegressor(random_state=RANDOM_STATE)

    total_fits = 5
    for v in param_grid.values():
        total_fits *= len(v)
    print(f"  GridSearchCV: {len(param_grid)} params, {total_fits} total fits …")

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV R²  : {float(gs.best_score_):.4f}")
    return gs.best_estimator_


def main(csv_path: str = "Advertising.csv") -> None:
    """End-to-end pipeline: load → clean → split → train → tune → save."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("=" * 60)
    print("Sales Prediction System — Training Pipeline")
    print(f"scikit-learn : {sklearn.__version__}")
    print("=" * 60)

    # ── Step 1: Load & inspect ────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    df = load_data(csv_path)
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Nulls   : {int(df.isnull().sum().sum())}")
    print(f"  Dupes   : {int(df.duplicated().sum())}")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("\n[2/5] Cleaning data …")
    df = clean_data(df)
    print(f"  Post-clean shape: {df.shape}")

    # ── Step 3: Split & scale ─────────────────────────────────────────────────
    print("\n[3/5] Splitting and scaling …")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
        # No stratify= — continuous regression target
    )
    print(f"  Train rows : {X_train.shape[0]}")
    print(f"  Test rows  : {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    # ── Step 4: Train & compare ───────────────────────────────────────────────
    print("\n[4/5] Training all models …")
    registry = _build_model_registry()
    results: list[dict] = []
    fitted: dict[str, object] = {}

    for name, mdl in registry.items():
        mdl.fit(X_train_sc, y_train)  # type: ignore[union-attr]
        fitted[name] = mdl
        metrics = evaluate_model(mdl, X_test_sc, y_test, model_name=name)
        cv_scores = cross_val_score(mdl, X_train_sc, y_train, cv=5, scoring="r2")
        metrics["CV_R2_mean"] = round(float(cv_scores.mean()), 4)
        results.append(metrics)
        print(f"  {name:<25}: R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}")

    comparison = (
        pd.DataFrame(results)
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    print("\nModel Comparison Table (sorted by Test R²):")
    print(comparison.to_string(index=False))

    # Pick best tree-based model for tuning
    tree_candidates = ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]
    best_name: str = comparison["Model"].iloc[0]  # overall best
    for cand in comparison["Model"].tolist():
        if cand in tree_candidates:
            best_name = cand
            break
    print(f"\nSelected for tuning: {best_name}")

    # ── Step 5: Tune & save ───────────────────────────────────────────────────
    print("\n[5/5] Tuning best model & saving artifacts …")
    best_model = _tune_best_tree(best_name, X_train_sc, y_train)

    tuned = evaluate_model(best_model, X_test_sc, y_test, model_name=f"{best_name} (Tuned)")
    print(f"  Tuned Test R²  : {tuned['R2']:.4f}")
    print(f"  Tuned Test RMSE: {tuned['RMSE']:.4f}")
    print(f"  Tuned Test MAE : {tuned['MAE']:.4f}")

    save_artifacts(best_model, scaler, FEATURE_COLS)
    print("\nArtifacts saved:")
    for fname in ["model.pkl", "scaler.pkl", "features.pkl"]:
        size_kb = Path(fname).stat().st_size / 1024
        print(f"  {fname:<15}: {size_kb:.1f} KB")

    print("\n[DONE] Training complete.  Run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Sales Prediction model.")
    parser.add_argument(
        "--csv",
        default="Advertising.csv",
        help="Path to the Advertising CSV file (default: Advertising.csv)",
    )
    args = parser.parse_args()
    main(args.csv)
