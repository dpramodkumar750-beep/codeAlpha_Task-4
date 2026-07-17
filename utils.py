"""
utils.py — Sales Prediction System shared helpers.

Used by:
  - CodeAlpha_SalesPrediction.ipynb  (exploration / modelling)
  - train_model.py                   (standalone pipeline script)
  - app.py                           (Streamlit dashboard)

Convention: exceptions are propagated with explicit ``raise`` so that
Pylance can narrow types correctly and callers are never silently surprised.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = ["TV", "Radio", "Newspaper"]
TARGET_COL: str = "Sales"

MODEL_PATH: Path = Path("model.pkl")
SCALER_PATH: Path = Path("scaler.pkl")
FEATURES_PATH: Path = Path("features.pkl")


# ─── RMSE helper (version-safe) ───────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSE, compatible with scikit-learn < 1.4 and >= 1.4.

    ``mean_squared_error(..., squared=False)`` was deprecated in sklearn 1.4
    and removed in 1.5.  We branch on the version so neither a deprecation
    warning nor an AttributeError occurs.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Root mean squared error as a Python float.
    """
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 4):
        from sklearn.metrics import root_mean_squared_error  # type: ignore[import]
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: str | Path = "Advertising.csv") -> pd.DataFrame:
    """Load the Advertising dataset, dropping the stray pandas-exported index.

    The raw CSV has an unnamed first column (``""``) that is simply the
    row-index pandas writes when you call ``df.to_csv()`` without
    ``index=False``.  Loading with ``index_col=0`` treats it as the
    DataFrame index so it never enters the feature matrix.

    Args:
        csv_path: Path to the CSV file.  Defaults to ``"Advertising.csv"``
            (relative to the current working directory).

    Returns:
        A :class:`~pandas.DataFrame` with columns: TV, Radio, Newspaper, Sales.

    Raises:
        FileNotFoundError: If the CSV file is not found at *csv_path*.
        ValueError: If expected columns are missing after loading.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path.resolve()}'.\n"
            "Place Advertising.csv in the project root, then re-run."
        )

    df = pd.read_csv(path, index_col=0)

    expected = {"TV", "Radio", "Newspaper", "Sales"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}.  "
            f"Actual columns: {list(df.columns)}"
        )

    return df


# ─── Data cleaning ────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the Advertising DataFrame in-place.

    Steps
    -----
    1. Drop duplicate rows (only if any exist — reported either way).
    2. Assert all spend/sales values are non-negative.

    Outlier policy: IQR-based outliers in Newspaper are **kept** — they
    represent genuine large campaigns, not recording errors.

    Args:
        df: Raw Advertising DataFrame as returned by :func:`load_data`.

    Returns:
        Cleaned DataFrame (may be the same object if no changes were needed).

    Raises:
        ValueError: If negative values are detected in any column.
    """
    # 1. Duplicates
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        warnings.warn(f"Dropped {n_dup} duplicate row(s).", stacklevel=2)

    # 2. Non-negativity assertion
    for col in df.columns:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            raise ValueError(
                f"Column '{col}' contains {n_neg} negative value(s), "
                "which is physically impossible for advertising spend / sales.  "
                "Inspect the data before proceeding."
            )

    return df


# ─── Model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str = "Model",
) -> dict[str, float | str]:
    """Compute R², MAE, MSE, and RMSE for a fitted model on a test set.

    Args:
        model: A fitted scikit-learn–compatible estimator.
        X_test: Test features (already scaled if required).
        y_test: True target values.
        model_name: Display name for the model, included in the returned dict.

    Returns:
        Dictionary with keys: ``"Model"``, ``"R2"``, ``"MAE"``,
        ``"MSE"``, ``"RMSE"``.
    """
    y_pred = model.predict(X_test)
    return {
        "Model": model_name,
        "R2": round(float(r2_score(y_test, y_pred)), 4),
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "MSE": round(float(mean_squared_error(y_test, y_pred)), 4),
        "RMSE": round(_rmse(y_test.values, y_pred), 4),
    }


# ─── Single-point prediction ──────────────────────────────────────────────────

def predict_sales(
    tv: float,
    radio: float,
    newspaper: float,
    model: Any,
    scaler: StandardScaler,
    feature_names: list[str],
) -> float:
    """Predict sales for a single set of advertising budgets.

    The input is built as a **named-column DataFrame** — not a raw list or
    numpy array — so that column-order bugs raise a loud ``KeyError`` instead
    of silently corrupting the prediction if the scaler / feature order ever
    changes.

    Args:
        tv: TV advertising budget (same units as the training data, i.e. $000).
        radio: Radio advertising budget ($000).
        newspaper: Newspaper advertising budget ($000).
        model: Fitted scikit-learn–compatible regressor.
        scaler: Fitted :class:`~sklearn.preprocessing.StandardScaler`.
        feature_names: Ordered list of feature column names as used during
            training (typically ``FEATURE_COLS``).

    Returns:
        Predicted sales as a Python float.
    """
    input_df = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=feature_names,
    )
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=feature_names,
    )
    return float(model.predict(input_scaled)[0])


# ─── Batch forecasting ────────────────────────────────────────────────────────

def forecast_sales(
    df_new: pd.DataFrame,
    model: Any,
    scaler: StandardScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    """Predict sales for a batch of advertising budget rows.

    Args:
        df_new: DataFrame with columns matching *feature_names*
            (TV, Radio, Newspaper).
        model: Fitted scikit-learn–compatible regressor.
        scaler: Fitted :class:`~sklearn.preprocessing.StandardScaler`.
        feature_names: Ordered list of feature column names.

    Returns:
        Copy of *df_new* with an additional ``"Predicted_Sales"`` column.

    Raises:
        ValueError: If *df_new* is missing any required feature columns.
    """
    missing_cols = set(feature_names) - set(df_new.columns)
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing columns: {missing_cols}.  "
            f"Expected: {feature_names}"
        )

    df_out = df_new[feature_names].copy()
    df_out_scaled = pd.DataFrame(
        scaler.transform(df_out),
        columns=feature_names,
        index=df_out.index,
    )
    df_result = df_new.copy()
    df_result["Predicted_Sales"] = model.predict(df_out_scaled)
    return df_result


# ─── Artifact persistence ─────────────────────────────────────────────────────

def save_artifacts(
    model: Any,
    scaler: StandardScaler,
    feature_names: list[str],
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    features_path: Path = FEATURES_PATH,
) -> None:
    """Persist the trained model, scaler, and feature list with joblib.

    Args:
        model: Fitted estimator to save.
        scaler: Fitted StandardScaler to save.
        feature_names: Ordered list of feature column names.
        model_path: Destination path for the model pickle.
        scaler_path: Destination path for the scaler pickle.
        features_path: Destination path for the feature names pickle.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    features_path: Path = FEATURES_PATH,
) -> tuple[Any, StandardScaler, list[str]]:
    """Load persisted model, scaler, and feature list from disk.

    Args:
        model_path: Path to the model pickle.
        scaler_path: Path to the scaler pickle.
        features_path: Path to the feature names pickle.

    Returns:
        Tuple of ``(model, scaler, feature_names)``.

    Raises:
        FileNotFoundError: If any of the three pickle files is missing.
    """
    for p in (model_path, scaler_path, features_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Artifact not found: '{p}'.  "
                "Run train_model.py first to generate all .pkl files."
            )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names
