# ======================================================================
# DataPilot – Lean Pro Version (Cloud Run Optimized)
# ======================================================================

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from openai import OpenAI

# ======================================================================
# STREAMLIT CONFIG
# ======================================================================
st.set_page_config(page_title="DataPilot", layout="wide")
st.title("📊 DataPilot – AI-Assisted Data Explorer")

# ======================================================================
# OPENAI CLIENT
# ======================================================================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Add it in Cloud Run → Variables & Secrets.")
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
        if any(x in col_l for x in synonyms):
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
            )
            if "revenue" in df.columns and df.filter(regex="spend").sum().sum() > 0
            else None,
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
    for label, func in KPI_RULES[group]["kpis"].items():
        try:
            val = func(df)
            if val is not None and not pd.isna(val):
                results[label] = float(val)
        except:
            pass
    return results


# ======================================================================
# AUTO CLEANING
# ======================================================================
def auto_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        # Clean numbers
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        # Try date parsing
        if any(key in col.lower() for key in ["date", "time", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df: pd.DataFrame):
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
Return ONLY valid JSON with keys:
- "insights": a short paragraph
- "charts": a list of chart ideas

Dataset sample:
{sample}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    text = res.choices[0].message.content

    # Extract JSON
    try:
        start = text.find("{")
        end = text.rfind("}")
        parsed = json.loads(text[start:end+1])
        return parsed
    except:
        return {"insights": "GPT failed to generate valid JSON.", "charts": []}


# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("⬅️ Upload a dataset to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

df_clean = auto_clean_df(df_raw)
df_sem = harmonize_columns(df_clean)


# ======================================================================
# KPI DASHBOARD
# ======================================================================
st.subheader("📌 Executive Summary")
kpis = compute_kpis(df_sem)

if kpis:
    cols = st.columns(len(kpis))
    for (label, value), col in zip(kpis.items(), cols):
        col.metric(label, f"{value:,.2f}")
else:
    st.write("No KPIs detected.")


# ======================================================================
# DATA PREVIEW
# ======================================================================
st.subheader("🧾 Raw Data")
st.dataframe(df_raw.head(20))

st.subheader("🧹 Cleaned + Semantic-Aligned Data")
st.dataframe(df_sem.head(20))


# ======================================================================
# VISUALIZATION STUDIO
# ======================================================================
st.subheader("📊 Visualize Your Data")

numeric_cols = df_sem.select_dtypes(include=["int64", "float64"]).columns.tolist()
all_cols = df_sem.columns.tolist()

if len(all_cols) > 0:

    chart_type = st.selectbox(
        "Select chart type", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"]
    )

    x_axis = st.selectbox("X-axis", all_cols)
    y_axis = st.selectbox("Y-axis", numeric_cols)

    if st.button("Generate Chart"):
        if chart_type == "Line Chart":
            fig = px.line(df_sem, x=x_axis, y=y_axis)

        elif chart_type == "Bar Chart":
            fig = px.bar(df_sem, x=x_axis, y=y_axis)

        elif chart_type == "Scatter Plot":
            fig = px.scatter(df_sem, x=x_axis, y=y_axis)

        elif chart_type == "Histogram":
            fig = px.histogram(df_sem, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# GPT INSIGHTS
# ======================================================================
st.subheader("🤖 AI Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Thinking…"):
        gpt = ask_gpt(df_sem)

    st.write("### 📘 Key Insights")
    st.write(gpt.get("insights", ""))

    st.write("### 📊 Suggested Charts")
    for idea in gpt.get("charts", []):
        st.write("- " + idea)


# ======================================================================
# Q&A
# ======================================================================
st.subheader("💬 Ask a Question")

query = st.text_area("Ask anything about your dataset:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        sample = df_sem.head(40).to_csv(index=False)
        prompt = f"""
Dataset:
{sample}

Question: {query}
Answer clearly in business language. Use only the data provided.
"""
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        st.write(res.choices[0].message.content)
