# -------------------------------------------------------------
# InsightHub – Premium UI Version
# -------------------------------------------------------------

import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="InsightHub - One-Click Analytics",
    page_icon="📊",
    layout="wide",
)

# -------------------------------------------------------------
# CUSTOM CSS FOR PREMIUM UI
# -------------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 30px;
    padding-bottom: 8px;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

.metric-card {
    background: #1f2937;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: inset 0px 0px 10px rgba(255,255,255,0.05);
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #4ade80;
}

.metric-label {
    font-size: 14px;
    color: #d1d5db;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PLOTLY THEME FIX – CLEAN DARK WITHOUT WHITE BARS
# -------------------------------------------------------------
pio.templates.default = "plotly_dark"

CUSTOM_PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E1E1E1", size=14),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)

# -------------------------------------------------------------
# UTILITY FUNCTIONS (UNCHANGED LOGIC)
# -------------------------------------------------------------
def load_data(uploaded_file: BytesIO) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Upload CSV/Excel.")
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Convert date-like columns
    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "day"]):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # Convert numeric-like text
    for col in df.select_dtypes(include=["object"]).columns:
        cleaned = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.4:
            df[col] = numeric

    return df


def detect_date_column(df):
    dt_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
    if len(dt_cols) > 0:
        return dt_cols[0]

    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "day", "week", "month", "year"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    df[col] = parsed
                    return col
            except:
                continue

    return None


def detect_target_column(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None

    priority = ["revenue", "sales", "amount", "value", "gmv", "profit", "total", "net"]

    for col in num_cols:
        if any(p in col.lower() for p in priority):
            return col

    return num_cols[-1]


def compute_kpis(df, target_col, date_col=None):
    series = df[target_col].dropna()
    kpis = {}

    if series.empty:
        return kpis

    kpis["Total"] = float(series.sum())
    kpis["Average"] = float(series.mean())
    kpis["Max"] = float(series.max())
    kpis["Min"] = float(series.min())

    # Growth if time exists
    if date_col:
        tmp = df[[date_col, target_col]].dropna().sort_values(by=date_col)
        if len(tmp) >= 2:
            first, last = tmp[target_col].iloc[0], tmp[target_col].iloc[-1]
            if first != 0:
                kpis["Growth (%)"] = (last - first) / abs(first) * 100

    return kpis


def make_forecast(df, target_col, date_col, periods=7):
    ts = df[[date_col, target_col]].dropna().sort_values(by=date_col)
    if len(ts) < 5:
        return None, None

    ts["t"] = ts[date_col].map(datetime.toordinal)
    X = ts[["t"]]
    y = ts[target_col]

    model = LinearRegression()
    model.fit(X, y)

    last_date = ts[date_col].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
    future_t = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

    future_pred = model.predict(future_t)

    forecast_df = pd.DataFrame({date_col: future_dates, target_col: future_pred})
    combined = pd.concat([ts[[date_col, target_col]], forecast_df], ignore_index=True)

    return combined, forecast_df


def rule_based_summary(df, target_col=None, date_col=None, kpis=None):
    lines = []
    n_rows, n_cols = df.shape
    lines.append(f"- The dataset contains **{n_rows} rows** and **{n_cols} columns**.")

    if target_col:
        lines.append(f"- Target metric detected: **{target_col}**.")
        if kpis:
            lines.append(f"- Total: **{kpis['Total']:,.2f}**, Average: **{kpis['Average']:,.2f}**.")

    return "\n".join(lines)


# -------------------------------------------------------------
# SIDEBAR UI
# -------------------------------------------------------------
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

show_raw = st.sidebar.checkbox("Show raw data", False)

sections = st.sidebar.multiselect(
    "Select analysis sections",
    ["Overview", "EDA", "Visualizations", "Forecast", "AI Summary"],
    default=["Overview", "EDA", "Visualizations", "AI Summary"],
)

# -------------------------------------------------------------
# MAIN APP UI
# -------------------------------------------------------------
st.title("📊 InsightHub – One-Click Data Analytics")
st.caption("Premium UI • Smart Analytics • Automatic EDA • Clean Visuals")

if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file to begin.")
    st.stop()

df_raw = load_data(uploaded_file)
df = clean_dataframe(df_raw)

st.success("✅ File uploaded & cleaned successfully!")

# Cleaned Data Preview
with st.container():
    st.markdown("<div class='section-title'>🧹 Cleaned Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)

# Detect columns
date_col = detect_date_column(df)
target_col = detect_target_column(df)

# Smart detection display
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Detected Date Column:** `{date_col}`")
    c2.markdown(f"**Detected Target Metric:** `{target_col}`")
    st.markdown("</div>", unsafe_allow_html=True)

# Override
with st.expander("⚙ Override detected columns"):
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    date_choice = st.selectbox("Date column", ["(Auto)"] + all_cols)
    if date_choice != "(Auto)":
        df[date_choice] = pd.to_datetime(df[date_choice], errors="coerce")
        date_col = date_choice

    target_choice = st.selectbox("Target metric", ["(Auto)"] + num_cols)
    if target_choice != "(Auto)":
        target_col = target_choice

# KPIs
kpis = compute_kpis(df, target_col, date_col)

# -------------------------------------------------------------
# OVERVIEW
# -------------------------------------------------------------
if "Overview" in sections:
    st.markdown("<div class='section-title'>📌 Overview & KPIs</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    if kpis:
        c1.markdown("<div class='metric-card'><div class='metric-label'>Total</div>"
                    f"<div class='metric-value'>{kpis['Total']:,.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown("<div class='metric-card'><div class='metric-label'>Average</div>"
                    f"<div class='metric-value'>{kpis['Average']:,.2f}</div></div>", unsafe_allow_html=True)
        c3.markdown("<div class='metric-card'><div class='metric-label'>Max</div>"
                    f"<div class='metric-value'>{kpis['Max']:,.2f}</div></div>", unsafe_allow_html=True)
        growth = kpis.get("Growth (%)")
        if growth:
            c4.markdown("<div class='metric-card'><div class='metric-label'>Growth (%)</div>"
                        f"<div class='metric-value'>{growth:.1f}%</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# EDA
# -------------------------------------------------------------
if "EDA" in sections:
    st.markdown("<div class='section-title'>🔍 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    missing = df.isna().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "MissingCount"]

    if not missing_df.empty:
        fig_missing = px.bar(missing_df, x="Column", y="MissingCount")
        fig_missing.update_layout(**CUSTOM_PLOTLY_THEME)
        st.plotly_chart(fig_missing, use_container_width=True)

# -------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------
if "Visualizations" in sections:
    st.markdown("<div class='section-title'>📈 Visualizations</div>", unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histogram
    st.markdown("### Distribution of Numeric Column")
    numeric_col = st.selectbox("Select numeric column", num_cols)

    fig_hist = px.histogram(df, x=numeric_col, nbins=30,
                            color_discrete_sequence=["#60a5fa"], opacity=0.85)
    fig_hist.update_layout(**CUSTOM_PLOTLY_THEME)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter
    if len(num_cols) >= 2:
        st.markdown("### Scatter Plot")

        x_col = st.selectbox("X-axis", num_cols, key="x")
        y_col = st.selectbox("Y-axis", num_cols, key="y")

        fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                 trendline="ols",
                                 color_discrete_sequence=["#3b82f6"])
        fig_scatter.update_layout(**CUSTOM_PLOTLY_THEME)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Time series
    if date_col:
        st.markdown("### Time Series")
        fig_ts = px.line(df.sort_values(by=date_col), x=date_col, y=target_col,
                         markers=True, color_discrete_sequence=["#38bdf8"])
        fig_ts.update_layout(**CUSTOM_PLOTLY_THEME)
        st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------------------------------------------
# FORECAST
# -------------------------------------------------------------
if "Forecast" in sections:
    st.markdown("<div class='section-title'>🔮 Forecast</div>", unsafe_allow_html=True)

    if date_col and target_col:
        periods = st.slider("Forecast periods (days)", 3, 30, 7)

        combined, forecast_df = make_forecast(df, target_col, date_col, periods)
        if combined is not None:
            fig_f = go.Figure()

            hist_mask = combined[date_col] <= df[date_col].max()

            fig_f.add_trace(go.Scatter(
                x=combined.loc[hist_mask, date_col],
                y=combined.loc[hist_mask, target_col],
                mode="lines+markers",
                name="Historical",
            ))

            fig_f.add_trace(go.Scatter(
                x=combined.loc[~hist_mask, date_col],
                y=combined.loc[~hist_mask, target_col],
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash")
            ))

            fig_f.update_layout(**CUSTOM_PLOTLY_THEME)
            st.plotly_chart(fig_f, use_container_width=True)

# -------------------------------------------------------------
# AI SUMMARY
# -------------------------------------------------------------
if "AI Summary" in sections:
    st.markdown("<div class='section-title'>🧠 AI Summary</div>", unsafe_allow_html=True)

    summary = rule_based_summary(df, target_col, date_col, kpis)
    st.markdown(summary)

# -------------------------------------------------------------
# DOWNLOAD CLEANED DATA
# -------------------------------------------------------------
with st.expander("⬇️ Download Cleaned Data"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download cleaned data as CSV", buf, "cleaned_data.csv", "text/csv")
