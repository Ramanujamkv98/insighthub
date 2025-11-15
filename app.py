# InsightHub 4.0 – Universal Cleaner + Safe EDA + GPT Insights (OpenAI v1)

import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from openai import OpenAI

# ------------------ Page config & basic theme ------------------ #
st.set_page_config(
    page_title="InsightHub 4.0 – GPT Auto EDA",
    page_icon="📊",
    layout="wide",
)

pio.templates.default = "plotly_dark"

st.markdown(
    """
<style>
body { font-family: "Inter", system-ui, sans-serif; }
.block-container { padding-top: 1.1rem; }
.card {
    background: #0f172a;
    padding: 1rem 1.25rem;
    border-radius: 16px;
    border: 1px solid #1e293b;
    margin-bottom: 1rem;
}
.metric-card {
    background: #020617;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    border: 1px solid #1f2937;
}
.metric-label {
    font-size: 0.8rem;
    color: #9ca3af;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e5e7eb;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 InsightHub 4.0 – GPT Auto EDA")
st.caption("Upload a dataset → Universal cleaning → Safe EDA → GPT insights + Q&A.")

# ------------------ OpenAI client ------------------ #
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("Please add OPENAI_API_KEY in your Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


# ------------------ Universal Cleaning Engine ------------------ #
def auto_clean_df(df: pd.DataFrame):
    """
    Universal cleaning function for real-world messy data.

    Handles:
    - Unnamed columns
    - All-null rows
    - Currency symbols ($, ₹), commas, %
    - Accounting negatives (e.g., (500) → -500)
    - Object columns that are numeric-like or datetime-like
    - Inf / -inf
    Returns:
      cleaned_df, cleaning_info (dict of what changed)
    """
    df = df.copy()
    info = {
        "dropped_columns": [],
        "rows_dropped_all_null": 0,
        "numeric_converted": [],
        "datetime_converted": [],
    }

    # Strip column name whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Drop "Unnamed" columns from Excel
    mask_unnamed = df.columns.str.contains("^unnamed", case=False, regex=True)
    dropped = df.columns[mask_unnamed].tolist()
    if dropped:
        info["dropped_columns"].extend(dropped)
        df = df.loc[:, ~mask_unnamed]

    # Drop fully empty rows
    before_rows = len(df)
    df = df.dropna(how="all")
    info["rows_dropped_all_null"] = before_rows - len(df)

    # Replace obvious inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Handle object columns: try datetime first, then numeric-like
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        s = df[col].astype(str).str.strip()

        # Try datetime
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.7:  # 70% convertible → treat as date
            df[col] = dt
            info["datetime_converted"].append(col)
            continue

        # Try numeric-like: remove $, ₹, %, commas, parentheses for negatives
        cleaned = (
            s.str.replace(r"\((.*)\)", r"-\1", regex=True)  # (500) → -500
            .str.replace("[₹$,]", "", regex=True)
            .str.replace("%", "", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )
        cleaned = cleaned.replace("", np.nan)

        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().mean() >= 0.5:  # at least 50% appear numeric
            df[col] = num
            info["numeric_converted"].append(col)

    # Final inf cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, info


def detect_date_column(df: pd.DataFrame):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
        name = str(col).lower()
        if any(k in name for k in ["date", "day", "week", "month", "year"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.6:
                    return col
            except Exception:
                continue
    return None


def detect_target_column(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    priority = ["revenue", "sales", "amount", "gmv", "turnover", "profit", "value", "total"]
    for col in numeric_cols:
        low = str(col).lower()
        if any(p in low for p in priority):
            return col
    # fallback: numeric column with highest variance
    return df[numeric_cols].var().sort_values(ascending=False).index[0]


def build_profile_for_gpt(df: pd.DataFrame, info: dict, max_rows: int = 200) -> str:
    sample = df.head(max_rows)
    profile = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "cleaning_info": info,
        "missing_counts": df.isna().sum().to_dict(),
        "sample_csv": sample.to_csv(index=False),
    }
    return json.dumps(profile, indent=2)


# ------------------ GPT: Insights on cleaned data ------------------ #
def call_gpt_insights(df_clean: pd.DataFrame, cleaning_info: dict) -> str:
    """
    Ask GPT for English-language insights & recommendations only.
    No code execution from GPT.
    """
    profile = build_profile_for_gpt(df_clean, cleaning_info)

    system_msg = """
You are a senior business data analyst.

You will get a JSON summary of a cleaned pandas DataFrame.
Your job is to write a clear, business-friendly narrative.

Respond ONLY in Markdown with:
- 5–10 bullet points of key insights.
- Then a section "Recommended next analyses" with 3–5 suggestions.

Focus on:
- Trends over time if a date/week column exists.
- Which metrics move together (high correlation).
- Which categories (channels, segments, etc.) over- or under-perform.
- Outliers, anomalies, or suspicious data quality issues.
Do NOT output any Python code.
"""

    user_msg = f"""
Here is the dataset profile:

{profile}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()


# ------------------ GPT: Ask-the-data Q&A ------------------ #
def call_gpt_qa(df_clean: pd.DataFrame, insights_md: str, question: str) -> str:
    sample = df_clean.head(200).to_csv(index=False)
    system_msg = """
You are a helpful data analyst assistant.
You answer questions about the dataset using the sample and the prior insights.

If something cannot be seen clearly from the sample, say so honestly.
Always answer in clear, structured Markdown.
"""
    user_msg = f"""
CLEANED DATA SAMPLE (CSV):
{sample}

EARLIER INSIGHTS:
{insights_md}

QUESTION:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ------------------ EDA visualization helpers (no GPT) ------------------ #
def compute_kpis(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target:
        return {}
    series = df[target].dropna()
    if series.empty:
        return {}
    return {
        "target_col": target,
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def render_kpi_cards(kpis: dict):
    if not kpis:
        st.write("No numeric target detected for KPIs.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-label">Target Metric</div>'
        f'<div class="metric-value">{kpis["target_col"]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-label">Total</div>'
        f'<div class="metric-value">{kpis["sum"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="metric-label">Average</div>'
        f'<div class="metric-value">{kpis["mean"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><div class="metric-label">Max</div>'
        f'<div class="metric-value">{kpis["max"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )


def make_missing_bar(df: pd.DataFrame):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None
    mdf = missing.reset_index()
    mdf.columns = ["column", "missing"]
    fig = px.bar(
        mdf,
        x="column",
        y="missing",
        title="Missing Values per Column",
        text="missing",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def make_corr_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap (Numeric Columns)",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    return fig


def make_target_distribution(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target or target not in df.columns:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None
    fig = px.histogram(
        df,
        x=target,
        nbins=30,
        title=f"Distribution of {target}",
    )
    return fig


def make_time_series(df: pd.DataFrame):
    date_col = detect_date_column(df)
    target = detect_target_column(df)
    if not date_col or not target:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None
    tmp = df[[date_col, target]].dropna()
    if tmp.empty:
        return None
    tmp = tmp.sort_values(by=date_col)
    fig = px.line(
        tmp,
        x=date_col,
        y=target,
        title=f"{target} over Time ({date_col})",
    )
    return fig


def make_category_breakdown(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target or target not in df.columns:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None

    cat_cols = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 20:
                cat_cols.append((col, nunique))

    if not cat_cols:
        return None

    # choose the smallest cardinality dimension for clean bar chart
    cat_cols = sorted(cat_cols, key=lambda x: x[1])
    cat_col = cat_cols[0][0]

    tmp = df.groupby(cat_col)[target].mean().reset_index()
    tmp = tmp.sort_values(by=target, ascending=False)
    fig = px.bar(
        tmp,
        x=cat_col,
        y=target,
        title=f"Average {target} by {cat_col}",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def make_top_scatter(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target or target not in df.columns:
        return None

    num_df = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    if num_df.shape[1] < 1:
        return None

    # choose feature with highest abs correlation with target
    corr = num_df.corrwith(df[target]).abs().sort_values(ascending=False)
    feature = corr.index[0]
    tmp = df[[feature, target]].dropna()
    if tmp.empty:
        return None
    fig = px.scatter(
        tmp,
        x=feature,
        y=target,
        trendline="ols",
        title=f"{feature} vs {target}",
    )
    return fig


# ------------------ Sidebar: upload + controls ------------------ #
st.sidebar.header("📂 Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

# ------------------ Raw preview ------------------ #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Raw Data Preview")
st.dataframe(df_raw.head(200), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ------------------ Run universal cleaner + GPT (cached) ------------------ #
@st.cache_data(show_spinner=False)
def run_pipeline(data: pd.DataFrame):
    cleaned, info = auto_clean_df(data)
    insights_md = call_gpt_insights(cleaned, info)
    return cleaned, info, insights_md


with st.spinner("🧹 Cleaning data + asking GPT for insights..."):
    df_clean, clean_info, gpt_insights = run_pipeline(df_raw)

# ------------------ Tabs layout ------------------ #
tab_overview, tab_charts, tab_insights, tab_ask = st.tabs(
    ["📌 Overview", "📊 Charts", "🧠 GPT Insights", "💬 Ask the Data"]
)

# ---- Overview tab ---- #
with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧹 Cleaned Data (Head)")
    st.dataframe(df_clean.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Key Metrics")
    kpis = compute_kpis(df_clean)
    render_kpi_cards(kpis)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 Data Quality Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape:**", df_clean.shape)
        st.write("**Dropped columns:**", clean_info["dropped_columns"] or "None")
        st.write("**Rows dropped (all null):**", clean_info["rows_dropped_all_null"])

    with col2:
        st.write("**Converted to numeric:**", clean_info["numeric_converted"] or "None")
        st.write("**Converted to datetime:**", clean_info["datetime_converted"] or "None")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Charts tab ---- #
with tab_charts:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Charts (Safe & Deterministic)")

    figs = []

    ts_fig = make_time_series(df_clean)
    if ts_fig:
        figs.append(("Time Series", ts_fig))

    dist_fig = make_target_distribution(df_clean)
    if dist_fig:
        figs.append(("Target Distribution", dist_fig))

    miss_fig = make_missing_bar(df_clean)
    if miss_fig:
        figs.append(("Missing Values", miss_fig))

    corr_fig = make_corr_heatmap(df_clean)
    if corr_fig:
        figs.append(("Correlation Heatmap", corr_fig))

    cat_fig = make_category_breakdown(df_clean)
    if cat_fig:
        figs.append(("Category Breakdown", cat_fig))

    scatter_fig = make_top_scatter(df_clean)
    if scatter_fig:
        figs.append(("Top Scatter", scatter_fig))

    if not figs:
        st.write("Not enough structure in the data to build charts automatically.")
    else:
        for title, fig in figs:
            st.markdown(f"### {title}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- GPT Insights tab ---- #
with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 GPT Insights & Recommendations")
    st.markdown(gpt_insights)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Ask the Data tab ---- #
with tab_ask:
    st.subheader("💬 Ask Questions About This Data")

    question = st.text_area(
        "Ask anything (e.g., 'Which channels seem most efficient?', "
        "'Any anomalies in revenue?', 'What should I investigate next?')",
        height=120,
    )
    ask_btn = st.button("Ask AI")

    if ask_btn and question.strip():
        with st.spinner("Thinking..."):
            answer = call_gpt_qa(df_clean, gpt_insights, question)
        st.markdown("### 🔎 Answer")
        st.markdown(answer)
