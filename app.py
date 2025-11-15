# app.py

import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

# Optional: uncomment if you want real AI summaries with OpenAI
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="InsightHub - One-Click Data Analytics",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Minimal Custom CSS (Modern UI)
# -----------------------------
st.markdown(
    """
<style>
/* Global page styling */
main, .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* Background */
body {
    background: #f3f4f6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Title area */
.insight-hero {
    padding: 1.25rem 1.5rem;
    border-radius: 1.25rem;
    background: radial-gradient(circle at top left, #4f46e5 0, #0ea5e9 45%, #0f172a 100%);
    color: #f9fafb;
    box-shadow: 0 18px 45px rgba(15,23,42,0.35);
}

/* Cards */
.card {
    background: #ffffff;
    border-radius: 1.25rem;
    padding: 1.25rem 1.5rem;
    box-shadow:
        0 18px 40px rgba(15,23,42,0.08),
        0 1px 2px rgba(15,23,42,0.05);
    border: 1px solid rgba(148,163,184,0.25);
}

/* Section titles */
.section-title {
    font-weight: 600;
    font-size: 1.1rem;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.section-caption {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 0.75rem;
}

/* KPI metrics tweak */
div[data-testid="metric-container"] {
    background: #f9fafb;
    border-radius: 0.75rem;
    padding: 0.75rem 0.75rem;
    box-shadow: 0 2px 4px rgba(15,23,42,0.04);
    border: 1px solid rgba(148,163,184,0.35);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 0.75rem;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.45);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #020617;
    color: #e5e7eb;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb;
}
section[data-testid="stSidebar"] label {
    color: #e5e7eb;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    border: none;
    background: linear-gradient(135deg, #4f46e5, #0ea5e9);
    color: white;
    font-weight: 500;
}
.stButton > button:hover {
    filter: brightness(1.05);
}

/* Download button */
[data-testid="baseButton-secondary"] {
    border-radius: 999px !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #0ea5e9) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
<div class="insight-hero">
  <h1 style="margin-bottom:0.3rem;font-size:1.9rem;font-weight:650;">
    📊 InsightHub – One-Click Data Analytics for Non-Tech Users
  </h1>
  <p style="margin:0;font-size:0.98rem;color:#e5e7eb;">
    Upload your CSV or Excel file and instantly turn raw data into clean insights, KPIs, 
    visual dashboards, and simple forecasts — without writing a single line of code.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")  # small spacing

# -----------------------------
# Utility Functions
# -----------------------------
def load_data(uploaded_file: BytesIO) -> pd.DataFrame:
    """Load CSV or Excel into a pandas DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip column names, try to parse dates, coerce numeric columns."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse date-like columns
    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "day"]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    # Coerce numeric-looking object columns to numbers
    for col in df.select_dtypes(include=["object"]).columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.4:
            df[col] = numeric

    return df


def detect_date_column(df: pd.DataFrame):
    """Try to detect a date/time column."""
    dt_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
    if dt_cols:
        return dt_cols[0]

    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "day", "week", "month", "year"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed
                    return col
            except Exception:
                continue
    return None


def detect_target_column(df: pd.DataFrame):
    """Try to detect a 'target' metric like sales/revenue/amount."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None

    priority_keywords = [
        "revenue",
        "sales",
        "amount",
        "gmv",
        "turnover",
        "value",
        "profit",
        "net",
        "total",
    ]
    for col in numeric_cols:
        if any(k in col.lower() for k in priority_keywords):
            return col

    return numeric_cols[-1]


def compute_kpis(df: pd.DataFrame, target_col: str, date_col: str = None):
    """Return a dict of simple KPIs."""
    series = df[target_col].dropna()
    kpis = {}
    if series.empty:
        return kpis

    kpis["Total"] = float(series.sum())
    kpis["Average"] = float(series.mean())
    kpis["Max"] = float(series.max())
    kpis["Min"] = float(series.min())

    if date_col is not None:
        tmp = df[[date_col, target_col]].dropna()
        tmp = tmp.sort_values(by=date_col)
        if len(tmp) >= 2:
            first_val = tmp[target_col].iloc[0]
            last_val = tmp[target_col].iloc[-1]
            if first_val != 0:
                growth = (last_val - first_val) / abs(first_val) * 100
                kpis["Growth (%)"] = float(growth)
            else:
                kpis["Growth (%)"] = np.nan

    return kpis


def make_forecast(df: pd.DataFrame, target_col: str, date_col: str, periods: int = 5):
    """Simple linear-regression-based forecast over time."""
    ts = df[[date_col, target_col]].dropna().copy()
    ts = ts.sort_values(by=date_col)
    if len(ts) < 5:
        return None, None

    ts["t"] = ts[date_col].map(datetime.toordinal)
    X = ts[["t"]]
    y = ts[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = ts[date_col].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
    future_t = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_pred = model.predict(future_t)

    forecast_df = pd.DataFrame({date_col: future_dates, target_col: future_pred})
    combined = pd.concat([ts[[date_col, target_col]], forecast_df], ignore_index=True)
    return combined, forecast_df


def rule_based_summary(df: pd.DataFrame, target_col: str = None, date_col: str = None, kpis: dict = None) -> str:
    """Simple heuristic summary to mimic AI insight."""
    lines = []

    n_rows, n_cols = df.shape
    lines.append(f"- The dataset contains **{n_rows} rows** and **{n_cols} columns**.")

    if target_col:
        lines.append(f"- The main numeric metric seems to be **`{target_col}`**.")
        if kpis:
            total = kpis.get("Total")
            avg = kpis.get("Average")
            mx = kpis.get("Max")
            mn = kpis.get("Min")
            if total is not None:
                lines.append(f"- Total `{target_col}` across the dataset is **{total:,.2f}**.")
            if avg is not None:
                lines.append(f"- On average, `{target_col}` is **{avg:,.2f}**.")
            if mx is not None and mn is not None:
                lines.append(
                    f"- The range of `{target_col}` goes from **{mn:,.2f}** to **{mx:,.2f}**."
                )

    if len(df.select_dtypes(include=[np.number]).columns) >= 2:
        corr = df.select_dtypes(include=[np.number]).corr()
        corr_pairs = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
            .stack()
            .sort_values(ascending=False)
        )
        if not corr_pairs.empty:
            top_pair = corr_pairs.index[0]
            top_val = corr_pairs.iloc[0]
            if abs(top_val) > 0.6:
                lines.append(
                    f"- There is a **strong correlation ({top_val:.2f})** between `{top_pair[0]}` and `{top_pair[1]}`."
                )

    if date_col and target_col:
        k_growth = (kpis or {}).get("Growth (%)", None)
        if k_growth is not None and not np.isnan(k_growth):
            if k_growth > 0:
                lines.append(
                    f"- Over time, `{target_col}` shows an **overall upward trend** of about **{k_growth:.1f}%**."
                )
            else:
                lines.append(
                    f"- Over time, `{target_col}` shows an **overall downward trend** of about **{k_growth:.1f}%**."
                )

    lines.append(
        "- You can use these insights to identify high-performing segments, monitor trends, and take data-driven decisions."
    )

    return "\n".join(lines)


def llm_summary_stub(df: pd.DataFrame, target_col: str, date_col: str, kpis: dict) -> str:
    """Stub function showing how you'd call an LLM."""
    text_preview = df.head(10).to_markdown()
    prompt = f"""
You are a business analyst. Summarize key insights from this dataset.

Target column: {target_col}
Date column: {date_col}
KPIs: {kpis}

Sample data:
{text_preview}

Return a concise business summary in bullet points.
"""
    # Example for later OpenAI integration
    # completion = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return completion["choices"][0]["message"]["content"].strip()
    return "LLM summary placeholder. Plug in your OpenAI key and use `llm_summary_stub`."


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

show_raw = st.sidebar.checkbox("Show raw data", value=False)
sidebar_analysis = st.sidebar.multiselect(
    "Choose analysis sections",
    options=["Overview", "EDA", "Visualizations", "Forecast", "AI Summary"],
    default=["Overview", "EDA", "Visualizations", "AI Summary"],
)

# -----------------------------
# Main App Logic
# -----------------------------
if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to get started.")
else:
    df_raw = load_data(uploaded_file)
    if df_raw is None:
        st.stop()

    df = clean_dataframe(df_raw)
    st.success("✅ File uploaded and cleaned successfully!")

    # --- Data Preview Card ---
    with st.container():
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧹 Cleaned Data Preview</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-caption">First 100 rows after automatic cleaning and type detection.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(100), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📄 Raw Data Snapshot</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-caption">Compare with the original data to see how InsightHub cleaned it.</div>',
                unsafe_allow_html=True,
            )
            if show_raw:
                st.dataframe(df_raw.head(50), use_container_width=True)
            else:
                st.caption("Enable **Show raw data** in the sidebar to view the original dataset.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Detect key columns
    date_col = detect_date_column(df)
    target_col = detect_target_column(df)

    # --- Column Detection + Overrides Card ---
    st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔎 Smart Column Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">InsightHub auto-detects your time and value columns, but you can override them anytime.</div>',
        unsafe_allow_html=True,
    )

    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown(
            f"**Detected Date Column:** `{date_col}`" if date_col else "**Detected Date Column:** _None_"
        )
    with info_cols[1]:
        st.markdown(
            f"**Detected Target Metric:** `{target_col}`" if target_col else "**Detected Target Metric:** _None_"
        )

    with st.expander("⚙️ Override detected columns (optional)", expanded=False):
        all_cols = df.columns.tolist()
        date_override = st.selectbox("Select date column (optional)", options=["(Auto)"] + all_cols)
        if date_override != "(Auto)":
            try:
                df[date_override] = pd.to_datetime(df[date_override], errors="coerce")
                date_col = date_override
            except Exception:
                st.warning("Could not parse selected column as dates. Keeping previous detection.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_override = st.selectbox(
            "Select target metric column (optional)",
            options=["(Auto)"] + numeric_cols,
        )
        if target_override != "(Auto)":
            target_col = target_override

    st.markdown("</div>", unsafe_allow_html=True)

    # Compute KPIs
    kpis = compute_kpis(df, target_col, date_col) if target_col else {}

    # --------------
    # Overview
    # --------------
    if "Overview" in sidebar_analysis:
        st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📌 Overview & KPIs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">A quick health check of your key performance metric.</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        if kpis:
            c1.metric("Total", f"{kpis.get('Total', 0):,.2f}")
            c2.metric("Average", f"{kpis.get('Average', 0):,.2f}")
            c3.metric("Max", f"{kpis.get('Max', 0):,.2f}")
            growth = kpis.get("Growth (%)", None)
            if growth is not None and not np.isnan(growth):
                c4.metric("Growth (%)", f"{growth:.1f}%")
            else:
                c4.metric("Min", f"{kpis.get('Min', 0):,.2f}")
        else:
            st.write("Not enough numeric data to compute KPIs.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------
    # EDA
    # --------------
    if "EDA" in sidebar_analysis:
        st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Distribution, completeness, and relationships across your dataset.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Basic Summary Statistics**")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        st.markdown("**Missing Values per Column**")
        missing = df.isna().sum()
        missing_df = missing[missing > 0].reset_index()
        missing_df.columns = ["Column", "MissingCount"]
        if missing_df.empty:
            st.write("No missing values detected.")
        else:
            fig_missing = px.bar(
                missing_df,
                x="Column",
                y="MissingCount",
                title="Missing Values by Column",
            )
            fig_missing.update_layout(
                template="plotly_white",
                margin=dict(t=40, r=10, l=10, b=10),
            )
            st.plotly_chart(fig_missing, use_container_width=True)

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            st.markdown("**Correlation Heatmap (Numeric Columns)**")
            corr = numeric_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(
                template="plotly_white",
                margin=dict(t=60, r=40, l=40, b=40),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write("Not enough numeric columns to compute correlations.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------
    # Visualizations
    # --------------
    if "Visualizations" in sidebar_analysis:
        st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Visualizations</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Trends, distributions, and relationships across key metrics.</div>',
            unsafe_allow_html=True,
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Time series if possible
            if date_col and target_col:
                st.markdown("**Time Series: Target Metric Over Time**")
                ts = df[[date_col, target_col]].dropna().sort_values(by=date_col)
                fig_ts = px.line(
                    ts,
                    x=date_col,
                    y=target_col,
                    title=f"{target_col} over time",
                )
                fig_ts.update_layout(
                    template="plotly_white",
                    margin=dict(t=50, r=20, l=10, b=40),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("**Distribution of a Numeric Column**")
            num_col_choice = st.selectbox("Select numeric column", options=numeric_cols)
            fig_hist = px.histogram(
                df,
                x=num_col_choice,
                nbins=30,
                title=f"Distribution of {num_col_choice}",
            )
            fig_hist.update_layout(
                template="plotly_white",
                margin=dict(t=50, r=20, l=10, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if len(numeric_cols) >= 2:
                st.markdown("**Scatter Plot (Relationship between Two Metrics)**")
                x_col = st.selectbox("X-axis", options=numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", options=numeric_cols, key="scatter_y")
                # Safe trendline: fall back if statsmodels is missing
                try:
                    fig_scatter = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        trendline="ols",
                        title=f"{y_col} vs {x_col}",
                    )
                except ModuleNotFoundError:
                    fig_scatter = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs {x_col}",
                    )
                fig_scatter.update_layout(
                    template="plotly_white",
                    margin=dict(t=50, r=20, l=10, b=40),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.write("No numeric columns available for visualization.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------
    # Forecast
    # --------------
    if "Forecast" in sidebar_analysis:
        st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Simple Forecast</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">A quick linear forecast based on your historical data.</div>',
            unsafe_allow_html=True,
        )

        if date_col and target_col:
            periods = st.slider("Forecast periods (days)", min_value=3, max_value=30, value=7)
            combined, forecast_df = make_forecast(df, target_col, date_col, periods=periods)

            if combined is None:
                st.write("Not enough data points to build a forecast.")
            else:
                fig_forecast = go.Figure()
                hist_mask = combined[date_col] <= df[date_col].max()

                fig_forecast.add_trace(
                    go.Scatter(
                        x=combined.loc[hist_mask, date_col],
                        y=combined.loc[hist_mask, target_col],
                        mode="lines+markers",
                        name="Historical",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=combined.loc[~hist_mask, date_col],
                        y=combined.loc[~hist_mask, target_col],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(dash="dash"),
                    )
                )
                fig_forecast.update_layout(
                    title=f"Simple Forecast for {target_col}",
                    xaxis_title=str(date_col),
                    yaxis_title=str(target_col),
                    template="plotly_white",
                    margin=dict(t=60, r=20, l=10, b=40),
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                with st.expander("View forecast values"):
                    st.dataframe(forecast_df, use_container_width=True)
        else:
            st.info("Need a date column and numeric target metric to generate a forecast.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------
    # AI Summary
    # --------------
    if "AI Summary" in sidebar_analysis:
        st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI-Style Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">A quick narrative of what stands out in your dataset.</div>',
            unsafe_allow_html=True,
        )

        summary = rule_based_summary(df, target_col, date_col, kpis)
        st.markdown(summary)

        # Uncomment to use real LLM summary (after setting OPENAI_API_KEY)
        # if st.button("✨ Generate GPT Summary"):
        #     with st.spinner("Calling LLM..."):
        #         llm_text = llm_summary_stub(df, target_col, date_col, kpis)
        #     st.markdown("---")
        #     st.markdown(llm_text)

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------
    # Download cleaned data
    # --------------
    st.markdown('<div class="card" style="margin-top:1.25rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⬇️ Export Cleaned Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Download the transformed dataset as a CSV file for further analysis or sharing.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Download options", expanded=True):
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="Download cleaned data as CSV",
            data=buf,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
