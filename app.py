"""
app.py — Sales Prediction System: Streamlit dashboard.

Loads model.pkl, scaler.pkl, features.pkl produced by train_model.py.
Does NOT retrain anything at runtime.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ─── Page config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Import helpers ───────────────────────────────────────────────────────────
try:
    from utils import (
        FEATURE_COLS,
        TARGET_COL,
        clean_data,
        evaluate_model,
        forecast_sales,
        load_artifacts,
        load_data,
        predict_sales,
    )
except ImportError as exc:
    st.error(
        f"**ImportError:** {exc}\n\n"
        "Make sure `utils.py` is in the same directory as `app.py`."
    )
    st.stop()

# ─── Global style ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ── Metric cards ─────────────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Prediction result ────────────────────────── */
    .prediction-box {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    }
    .prediction-box h1 { color: #d1fae5; font-size: 3rem; margin: 0; }
    .prediction-box p  { color: #a7f3d0; font-size: 1.1rem; margin-top: 0.5rem; }

    /* ── Tab bar ──────────────────────────────────── */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #94a3b8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a78bfa;
        border-bottom: 2px solid #a78bfa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":   "#334155",
})


# ─── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_model_artifacts():
    """Load model, scaler, and feature list once and cache them."""
    try:
        return load_artifacts()
    except FileNotFoundError as exc:
        st.error(
            f"**Artifacts not found:** {exc}\n\n"
            "Run `python train_model.py` first to generate `model.pkl`, "
            "`scaler.pkl`, and `features.pkl`."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def _load_dataset(csv_path: str = "Advertising.csv") -> pd.DataFrame:
    """Load and clean the Advertising dataset once."""
    try:
        df = load_data(csv_path)
        return clean_data(df)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()


# ─── Load artifacts ───────────────────────────────────────────────────────────
model, scaler, feature_names = _load_model_artifacts()
df = _load_dataset()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Sales Prediction")
    st.markdown("---")
    st.markdown("### 🎯 Set Advertising Budget")

    # Use dataset stats for sensible slider bounds
    tv_min,    tv_max    = float(df["TV"].min()),        float(df["TV"].max())
    radio_min, radio_max = float(df["Radio"].min()),     float(df["Radio"].max())
    news_min,  news_max  = float(df["Newspaper"].min()), float(df["Newspaper"].max())

    tv_val    = st.slider("📺 TV Budget ($000)",        tv_min,    tv_max,    float(df["TV"].median()),        step=0.5, key="tv_budget")
    radio_val = st.slider("📻 Radio Budget ($000)",     radio_min, radio_max, float(df["Radio"].median()),     step=0.5, key="radio_budget")
    news_val  = st.slider("📰 Newspaper Budget ($000)", news_min,  news_max,  float(df["Newspaper"].median()), step=0.5, key="news_budget")

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Sales", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Model loaded:**")
    st.code(type(model).__name__, language=None)
    st.markdown(f"**Features:** {', '.join(feature_names)}")
    st.markdown(f"**Dataset:** {df.shape[0]} rows")

# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown("# 📈 Sales Prediction System")
st.markdown(
    "**CodeAlpha Data Science Internship — Task 4** | "
    "Predict sales from TV, Radio & Newspaper advertising spend."
)

tab_predict, tab_data, tab_eda, tab_models, tab_insights = st.tabs([
    "🎯 Predict Sales",
    "📊 Dataset",
    "🔍 EDA",
    "🤖 Compare Models",
    "💡 Business Insights",
])

# ── Tab 1: Predict Sales ──────────────────────────────────────────────────────
with tab_predict:
    col1, col2, col3 = st.columns(3)
    col1.metric("📺 TV Budget",        f"${tv_val:.1f}K")
    col2.metric("📻 Radio Budget",     f"${radio_val:.1f}K")
    col3.metric("📰 Newspaper Budget", f"${news_val:.1f}K")

    if predict_btn:
        pred = predict_sales(tv_val, radio_val, news_val, model, scaler, feature_names)
        st.markdown(
            f"""
            <div class="prediction-box">
                <h1>🛒 {pred:.2f}</h1>
                <p>Predicted Sales (units)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(f"✅ Predicted Sales: **{pred:.2f} units** for a "
                   f"${tv_val:.0f}K TV / ${radio_val:.0f}K Radio / ${news_val:.0f}K Newspaper budget.")
    else:
        st.info("👈 Adjust the sliders in the sidebar and click **Predict Sales**.")

    # Batch forecasting section
    st.markdown("---")
    st.markdown("### 📋 Batch Forecasting")
    st.markdown("Enter multiple campaign budgets to forecast sales in bulk.")

    default_batch = pd.DataFrame({
        "TV": [230.0, 150.0, 80.0, 50.0],
        "Radio": [40.0, 25.0, 10.0, 5.0],
        "Newspaper": [15.0, 30.0, 50.0, 10.0],
    })
    batch_input = st.data_editor(
        default_batch,
        num_rows="dynamic",
        use_container_width=True,
        key="batch_editor",
    )

    if st.button("📊 Forecast Batch", use_container_width=True):
        try:
            batch_result = forecast_sales(batch_input, model, scaler, feature_names)
            batch_result["Predicted_Sales"] = batch_result["Predicted_Sales"].round(2)
            st.dataframe(batch_result, use_container_width=True)
            csv_dl = batch_result.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Forecast CSV",
                csv_dl,
                "sales_forecast.csv",
                "text/csv",
            )
        except ValueError as exc:
            st.error(str(exc))

# ── Tab 2: Dataset ────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### 📊 Advertising Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Null values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.dataframe(df, use_container_width=True)

    st.markdown("### 📈 Descriptive Statistics")
    st.dataframe(df.describe().T.round(3), use_container_width=True)

# ── Tab 3: EDA ────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### 🔍 Exploratory Data Analysis")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Correlation Heatmap")
        fig_hm, ax_hm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",    # named arg 'cmap', NOT a variable named 'cm'
            ax=ax_hm,
            linewidths=0.5,
            linecolor="#334155",
            annot_kws={"size": 11, "weight": "bold"},
            vmin=-1,
            vmax=1,
        )
        ax_hm.set_title("Correlation Heatmap", color="#e2e8f0", pad=10)
        st.pyplot(fig_hm)
        plt.close(fig_hm)

    with col_r:
        st.markdown("#### Distribution of Sales")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
        sns.histplot(df["Sales"], kde=True, ax=ax_hist, color="#a78bfa", bins=25, alpha=0.7)
        ax_hist.axvline(float(df["Sales"].mean()),   color="white",   linestyle="--", label=f"Mean={float(df['Sales'].mean()):.1f}")
        ax_hist.axvline(float(df["Sales"].median()), color="#f59e0b", linestyle=":",  label=f"Median={float(df['Sales'].median()):.1f}")
        ax_hist.legend()
        ax_hist.set_title("Sales Distribution", color="#e2e8f0")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    st.markdown("#### Feature vs Sales (Scatter + Regression Line)")
    feat_colors = ["#a78bfa", "#60a5fa", "#34d399"]
    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(16, 5))
    for ax, feat, clr in zip(axes_sc, FEATURE_COLS, feat_colors):
        sns.regplot(
            x=feat, y="Sales", data=df, ax=ax,
            scatter_kws={"alpha": 0.5, "color": clr, "s": 30},
            line_kws={"color": "#f59e0b", "linewidth": 2},
            ci=95,
        )
        ax.set_title(f"{feat} vs Sales", color="#e2e8f0")
    fig_sc.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)

    st.markdown("#### Boxplots — Outlier Detection")
    fig_bp, axes_bp = plt.subplots(1, 4, figsize=(16, 5))
    box_colors = ["#a78bfa", "#60a5fa", "#34d399", "#f59e0b"]
    for ax, col, clr in zip(axes_bp, df.columns, box_colors):
        ax.boxplot(
            df[col].values,
            patch_artist=True,
            boxprops=dict(facecolor=clr, alpha=0.6),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#94a3b8"),
            capprops=dict(color="#94a3b8"),
            flierprops=dict(marker="o", color=clr, alpha=0.5),
        )
        ax.set_title(col, color="#e2e8f0")
        ax.set_xticks([])
    fig_bp.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    st.pyplot(fig_bp)
    plt.close(fig_bp)

# ── Tab 4: Compare Models ─────────────────────────────────────────────────────
with tab_models:
    st.markdown("### 🤖 Model Comparison")
    st.info(
        "Run `python train_model.py` to regenerate the comparison table.  "
        "The table below re-trains all models with a fixed random seed for display — "
        "it does **not** overwrite `model.pkl`."
    )

    if st.button("🔄 Run Model Comparison", use_container_width=True):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeRegressor

        with st.spinner("Training 7 models … this takes ~15 seconds."):
            X_c = df[FEATURE_COLS]
            y_c = df[TARGET_COL]
            X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
            sc_c = StandardScaler()
            X_tr_sc = pd.DataFrame(sc_c.fit_transform(X_tr), columns=FEATURE_COLS, index=X_tr.index)
            X_te_sc = pd.DataFrame(sc_c.transform(X_te),      columns=FEATURE_COLS, index=X_te.index)

            mdl_reg: dict[str, object] = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression":  Ridge(alpha=1.0, random_state=42),
                "Lasso Regression":  Lasso(alpha=0.01, random_state=42),
                "Decision Tree":     DecisionTreeRegressor(random_state=42),
                "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            try:
                from xgboost import XGBRegressor  # type: ignore[import]
                mdl_reg["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
            except ImportError:
                pass

            rows = []
            for nm, mdl_ in mdl_reg.items():
                mdl_.fit(X_tr_sc, y_tr)  # type: ignore[union-attr]
                m = evaluate_model(mdl_, X_te_sc, y_te, model_name=nm)
                cv = cross_val_score(mdl_, X_tr_sc, y_tr, cv=5, scoring="r2")
                m["CV R² (mean)"] = round(float(cv.mean()), 4)
                m["CV R² (std)"]  = round(float(cv.std()),  4)
                rows.append(m)

            comp = (
                pd.DataFrame(rows)
                .rename(columns={"R2": "Test R²"})
                .sort_values("Test R²", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(comp.round(4), use_container_width=True)

            # Bar chart
            fig_bar, axes_bar = plt.subplots(1, 2, figsize=(14, 5))
            bar_clrs = plt.cm.viridis(np.linspace(0.2, 0.9, len(comp)))
            axes_bar[0].barh(comp["Model"].tolist()[::-1], comp["Test R²"].tolist()[::-1], color=bar_clrs)
            axes_bar[0].set_title("Test R² (higher = better)", color="#e2e8f0")
            axes_bar[1].barh(comp["Model"].tolist()[::-1], comp["RMSE"].tolist()[::-1], color=bar_clrs)
            axes_bar[1].set_title("RMSE (lower = better)", color="#e2e8f0")
            for ax_ in axes_bar:
                ax_.set_facecolor("#1e293b")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)
    else:
        st.caption("Click the button above to run the comparison (takes ~15s).")

# ── Tab 5: Business Insights ──────────────────────────────────────────────────
with tab_insights:
    st.markdown("### 💡 Business Insights")

    from sklearn.linear_model import LinearRegression as _LR
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _SS

    X_i = df[FEATURE_COLS]
    y_i = df[TARGET_COL]
    X_tr_i, _, y_tr_i, _ = _tts(X_i, y_i, test_size=0.2, random_state=42)
    sc_i = _SS()
    X_tr_i_sc = sc_i.fit_transform(X_tr_i)
    lr_i = _LR().fit(X_tr_i_sc, y_tr_i)

    # ROI proxy: unscale coefficients back to original units
    raw_coefs = lr_i.coef_ / sc_i.scale_
    roi_df = (
        pd.DataFrame({"Channel": FEATURE_COLS, "ROI_proxy (Δ Sales per $000 spend)": raw_coefs})
        .sort_values("ROI_proxy (Δ Sales per $000 spend)", ascending=False)
        .reset_index(drop=True)
    )

    corr_s = df[FEATURE_COLS].corrwith(df[TARGET_COL]).sort_values(ascending=False)
    corr_df = pd.DataFrame({
        "Channel": corr_s.index,
        "Pearson r with Sales": corr_s.values.round(4),
    })

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Channel Correlation with Sales")
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        bars_c = ax_corr.bar(
            corr_df["Channel"], corr_df["Pearson r with Sales"],
            color=["#a78bfa", "#60a5fa", "#34d399"]
        )
        for b, v in zip(bars_c, corr_df["Pearson r with Sales"].tolist()):
            ax_corr.text(b.get_x() + b.get_width() / 2, float(v) + 0.01,
                         f"{float(v):.3f}", ha="center", va="bottom", color="#e2e8f0")
        ax_corr.set_ylim(0, 1)
        ax_corr.set_title("Pearson Correlation with Sales", color="#e2e8f0")
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    with col_b:
        st.markdown("#### ROI Proxy (Linear Model Coefficients)")
        st.dataframe(roi_df.round(4), use_container_width=True, hide_index=True)

        fig_roi, ax_roi = plt.subplots(figsize=(6, 4))
        bars_r = ax_roi.bar(
            roi_df["Channel"], roi_df["ROI_proxy (Δ Sales per $000 spend)"],
            color=["#a78bfa", "#60a5fa", "#34d399"]
        )
        for b, v in zip(bars_r, roi_df["ROI_proxy (Δ Sales per $000 spend)"].tolist()):
            ax_roi.text(b.get_x() + b.get_width() / 2, float(v) + 0.001,
                         f"{float(v):.4f}", ha="center", va="bottom", color="#e2e8f0")
        ax_roi.set_title("ROI Proxy: Δ Sales per $000 Spend", color="#e2e8f0")
        ax_roi.set_ylabel("Δ Sales per $000", color="#94a3b8")
        st.pyplot(fig_roi)
        plt.close(fig_roi)

    weakest  = roi_df.iloc[-1]["Channel"]
    strongest = roi_df.iloc[0]["Channel"]
    st.markdown("---")
    st.markdown(
        f"**📌 Directional Insight:** `{weakest}` has the lowest ROI proxy "
        f"({roi_df.iloc[-1]['ROI_proxy (Δ Sales per $000 spend)']:.4f} Δ Sales per $000).  "
        f"The model suggests reallocating `{weakest}` budget toward `{strongest}`."
    )
    st.warning(
        "⚠️ **Caveat:** This analysis is based on 200 rows and 3 features.  "
        "Treat directional findings as hypotheses to validate with more data, "
        "not as prescriptive budget decisions."
    )

    if hasattr(model, "feature_importances_"):
        st.markdown("#### Tree Model Feature Importances")
        imp_df = pd.DataFrame({
            "Channel": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        st.dataframe(imp_df.round(4), use_container_width=True, hide_index=True)

        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        imp_bars = ax_imp.bar(imp_df["Channel"], imp_df["Importance"], color=["#a78bfa", "#60a5fa", "#34d399"])
        for b, v in zip(imp_bars, imp_df["Importance"].tolist()):
            ax_imp.text(b.get_x() + b.get_width() / 2, float(v) + 0.005,
                        f"{float(v):.3f}", ha="center", va="bottom", color="#e2e8f0")
        ax_imp.set_title(f"{type(model).__name__} Feature Importances", color="#e2e8f0")
        ax_imp.set_ylabel("Importance (unitless)", color="#94a3b8")
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        st.info(
            "**Correlation vs. Tree importance:** Correlation captures linear relationships only.  "
            "Tree importances capture non-linear effects and interactions — so rankings can differ.  "
            "For example, Radio may rank higher in tree importance than correlation suggests if it "
            "interacts with TV spend in a non-additive way."
        )
