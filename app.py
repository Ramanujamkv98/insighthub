# ======================================================================
# DataPilot – Pro Visual Version (No Heavy NLP)
# KPIs + Plotly Visuals + GPT Insights + Q&A
# Cloud Run + OpenAI v1.x Compatible
# ======================================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# ======================================================================
# STREAMLIT CONFIG
# ======================================================================
st.set_page_config(page_title="DataPilot", layout="wide")
st.title("📊 DataPilot – AI-Assisted Data Explorer")

# ======================================================================
# OPENAI CLIENT (Cloud Run: uses env var)
# ======================================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Set it in Cloud Run → Environment variables.")
    st.stop()

client = OpenAI(api_key=api_key)

# ======================================================================
# SEMANTIC COLUMN MAP
# ======================================================================
SEMANTIC_MAP = {
    "revenue": ["revenue", "sales", "gmv", "turnover", "amount"],
    "units_sold": ["units", "sold", "qty", "quantity"],
    "daily_demand": ["demand", "orders", "order_qty"],
    "inventory_on_hand": ["inventory", "stock", "onhand"],
    "stockout_flag": ["oos", "stockout", "out_of_stock"],
    "spend": ["spend", "cost", "budget", "marketing_spend"],
    "profit": ["profit", "net_profit"],
    "expenses": ["expense", "expenses"],
}

def semantic_match(col: str):
    col_l = col.lower()
    for canonical, synonyms in SEMANTIC_MAP.items():
        if any(word in col_l for word in synonyms):
            return canonical
    return None

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: semantic_match(col) or col for col in df.columns}
    return df.rename(columns=rename_map)

# ======================================================================
# KPI RULE ENGINE
# ======================================================================
KPI_RULES = {
    "retail": {
        "keywords": ["revenue", "units_sold"],
        "kpis": {
            "Total Revenue": lambda df: df.get("revenue", pd.Series(dtype=float)).sum(),
            "Avg Revenue per Sale": lambda df: df.get("revenue", pd.Series(dtype=float)).mean(),
            "Units Sold": lambda df: df.get("units_sold", pd.Series(dtype=float)).sum(),
        },
    },
    "marketing": {
        "keywords": ["spend"],
        "kpis": {
            "Total Spend": lambda df: df.filter(regex="spend").sum().sum(),
            "ROI": lambda df: (
                df.get("revenue", pd.Series(dtype=float)).sum()
                / df.filter(regex="spend").sum().sum()
            ) if "revenue" in df.columns and df.filter(regex="spend").sum().sum() > 0 else None,
        },
    },
    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Avg Daily Demand": lambda df: df.get("daily_demand", pd.Series(dtype=float)).mean(),
            "Avg Inventory On-Hand": lambda df: df.get("inventory_on_hand", pd.Series(dtype=float)).mean(),
        },
    },
}

def detect_kpi_group(df: pd.DataFrame):
    scores = {g: sum(k in df.columns for k in r["keywords"]) for g, r in KPI_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def compute_kpis(df: pd.DataFrame):
    group = detect_kpi_group(df)
    if not group:
        return {}

    results = {}
    for name, func in KPI_RULES[group]["kpis"].items():
        try:
            val = func(df)
            if val is not None and not pd.isna(val):
                results[name] = float(val)
        except Exception:
            pass
    return results

# ======================================================================
# AUTO CLEANING
# ======================================================================
def auto_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnamed index-like columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        # Try numeric cleanup for object columns
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        # Try date parsing based on column name
        if any(keyword in col.lower() for keyword in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df

# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df: pd.DataFrame):
    # Limit sample size to keep token usage manageable
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
You are a senior analytics consultant.
Return ONLY valid JSON in this structure:

{{
  "insights": "short paragraph of 4–6 sentences with key trends and risks",
  "charts": [
    "Chart idea 1",
    "Chart idea 2",
    "Chart idea 3"
  ]
}}

Base your analysis ONLY on the dataset sample below.
If something cannot be inferred, do not guess.

Dataset sample:
{sample}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
    )

    text = res.choices[0].message.content

    # Try to extract JSON from the response
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass

    return {"insights": "GPT failed to produce valid JSON.", "charts": []}

# ======================================================================
# SIMPLE HELPERS FOR VISUALS
# ======================================================================
def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["number"]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame, max_unique: int = 50):
    cats = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            if df[col].nunique(dropna=True) <= max_unique:
                cats.append(col)
    return cats

def get_date_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("⬅️ Upload a dataset to begin.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

df_clean = auto_clean_df(df_raw)
df_sem = harmonize_columns(df_clean)

# ======================================================================
# LAYOUT: TABS
# ======================================================================
tab_overview, tab_visuals, tab_diag, tab_ai = st.tabs(
    ["📌 Overview & KPIs", "📈 Visual Explorer", "🧪 Diagnostics", "🤖 AI Insights & Q&A"]
)

# ======================================================================
# TAB 1: OVERVIEW & KPIs
# ======================================================================
with tab_overview:
    st.subheader("📌 Executive Summary")

    kpis = compute_kpis(df_sem)

    if kpis:
        cols = st.columns(len(kpis))
        for (label, value), col in zip(kpis.items(), cols):
            col.metric(label, f"{value:,.2f}")
    else:
        st.write(
            "No domain-specific KPIs detected. "
            "Try including columns like revenue, units, spend, inventory, etc."
        )

    st.markdown("---")
    st.subheader("🧾 Raw Data Preview")
    st.dataframe(df_raw.head(20))

    st.subheader("🧹 Cleaned + Semantic-Aligned Data")
    st.dataframe(df_sem.head(20))

# ======================================================================
# TAB 2: VISUAL EXPLORER (PLOTLY)
# ======================================================================
with tab_visuals:
    st.subheader("📈 Data Visual Explorer (Plotly)")

    num_cols = get_numeric_columns(df_sem)
    cat_cols = get_categorical_columns(df_sem)
    date_cols = get_date_columns(df_sem)

    if not num_cols and not cat_cols and not date_cols:
        st.info("No suitable numeric, categorical, or date columns detected for visualization.")
    else:
        viz_type = st.selectbox(
            "Select visualization type",
            [
                "Histogram (Numeric)",
                "Bar (Categorical)",
                "Time Series (Date + Numeric)",
                "Scatter (Numeric vs Numeric)",
            ],
        )

        if viz_type == "Histogram (Numeric)":
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                col = st.selectbox("Select numeric column", num_cols)
                fig = px.histogram(df_sem, x=col, nbins=30, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Bar (Categorical)":
            if not cat_cols:
                st.warning("No suitable categorical columns available.")
            else:
                col = st.selectbox("Select categorical column", cat_cols)
                metric = st.selectbox("Metric", ["Count"] + num_cols)

                if metric == "Count":
                    counts = df_sem[col].value_counts().reset_index()
                    counts.columns = [col, "Count"]
                    fig = px.bar(counts, x=col, y="Count", title=f"Value counts for {col}")
                else:
                    grouped = df_sem.groupby(col)[metric].sum().reset_index()
                    fig = px.bar(grouped, x=col, y=metric, title=f"{metric} by {col}")

                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Time Series (Date + Numeric)":
            if not date_cols:
                st.warning("No date-like columns detected.")
            elif not num_cols:
                st.warning("No numeric columns available for aggregation.")
            else:
                date_col = st.selectbox("Select date column", date_cols)
                num_col = st.selectbox("Select numeric column", num_cols)
                agg_fn = st.selectbox("Aggregation", ["sum", "mean"])

                df_ts = df_sem[[date_col, num_col]].dropna()
                df_ts = df_ts.sort_values(by=date_col)

                if agg_fn == "sum":
                    grouped = df_ts.groupby(date_col)[num_col].sum().reset_index()
                else:
                    grouped = df_ts.groupby(date_col)[num_col].mean().reset_index()

                fig = px.line(
                    grouped,
                    x=date_col,
                    y=num_col,
                    title=f"{agg_fn.title()} of {num_col} over time ({date_col})",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Scatter (Numeric vs Numeric)":
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for a scatter plot.")
            else:
                x_col = st.selectbox("X-axis", num_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", num_cols, key="scatter_y")
                color_col = st.selectbox("Color (optional)", [None] + cat_cols, key="scatter_color")

                if color_col and color_col in df_sem.columns:
                    fig = px.scatter(df_sem, x=x_col, y=y_col, color=color_col,
                                     title=f"{y_col} vs {x_col} colored by {color_col}")
                else:
                    fig = px.scatter(df_sem, x=x_col, y=y_col,
                                     title=f"{y_col} vs {x_col}")

                st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 3: DIAGNOSTICS
# ======================================================================
with tab_diag:
    st.subheader("🧪 Data Diagnostics")

    st.markdown("### Numeric Summary")
    num_cols = get_numeric_columns(df_sem)
    if num_cols:
        st.dataframe(df_sem[num_cols].describe().T)
    else:
        st.write("No numeric columns to summarize.")

    st.markdown("---")
    st.markdown("### Correlation Matrix (Numeric Columns)")
    if len(num_cols) >= 2:
        corr = df_sem[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Need at least two numeric columns to show correlations.")

# ======================================================================
# TAB 4: AI INSIGHTS & Q&A
# ======================================================================
with tab_ai:
    st.subheader("🤖 AI Insights")

    if st.button("Generate AI Insights from Dataset"):
        with st.spinner("Analyzing data with GPT…"):
            gpt = ask_gpt(df_sem)

        st.subheader("📘 Key Insights")
        st.write(gpt.get("insights", ""))

        st.subheader("📊 Suggested Charts")
        charts = gpt.get("charts", [])
        if isinstance(charts, list) and charts:
            for i, c in enumerate(charts, 1):
                st.markdown(f"**{i}.** {c}")
        else:
            st.write("No chart suggestions returned.")

    st.markdown("---")
    st.subheader("💬 Ask a Question About Your Data")

    query = st.text_area("Your question")

    if st.button("Ask GPT About This Dataset"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            sample = df_sem.head(50).to_csv(index=False)

            prompt = f"""
Dataset sample:
{sample}

You are a business analyst. Answer the user's question using ONLY this data.
If the answer cannot be derived, say so clearly.

Question: {query}
"""

            with st.spinner("Thinking…"):
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

            st.write(res.choices[0].message.content)
