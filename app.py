# ======================================================================
# DataPilot – Stable Streamlit Deployment Version
# OpenAI v1.x | Plotly | Pandas | Fast + Safe
# ======================================================================

import os
import json
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
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
    st.error(
        """
        ❌ OPENAI_API_KEY not found.

        Fix this:
        • Streamlit Cloud → Settings → Secrets:
          OPENAI_API_KEY = your_api_key_here
        """
    )
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
    "expenses": ["expense", "expenses"]
}

def semantic_match(col):
    col_l = col.lower()
    for canonical, synonyms in SEMANTIC_MAP.items():
        if any(x in col_l for x in synonyms):
            return canonical
    return None

def harmonize_columns(df):
    rename_map = {col: semantic_match(col) or col for col in df.columns}
    return df.rename(columns=rename_map)


# ======================================================================
# KPI RULE ENGINE
# ======================================================================
KPI_RULES = {
    "retail": {
        "keywords": ["revenue", "units_sold"],
        "kpis": {
            "Total Revenue": lambda df: df["revenue"].sum(),
            "Avg Revenue per Sale": lambda df: df["revenue"].mean(),
            "Units Sold": lambda df: df["units_sold"].sum(),
        },
    },
    "marketing": {
        "keywords": ["spend"],
        "kpis": {
            "Total Spend": lambda df: df[[c for c in df.columns if 'spend' in c]].sum().sum(),
            "ROI": lambda df: (
                df["revenue"].sum() /
                df[[c for c in df.columns if 'spend' in c]].sum().sum()
            ) if "revenue" in df else None,
        },
    },
    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Avg Daily Demand": lambda df: df["daily_demand"].mean(),
            "Avg Inventory On-Hand": lambda df: df["inventory_on_hand"].mean(),
        },
    },
}

def detect_kpi_group(df):
    scores = {g: sum(k in df.columns for k in r["keywords"])
              for g, r in KPI_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def compute_kpis(df):
    group = detect_kpi_group(df)
    if not group:
        return {}

    results = {}
    for name, func in KPI_RULES[group]["kpis"].items():
        try:
            results[name] = func(df)
        except:
            pass
    return results


# ======================================================================
# AUTO CLEANING
# ======================================================================
def auto_clean_df(df):
    df = df.copy()

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        if any(x in col.lower() for x in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df):
    sample = df.head(40).to_csv(index=False)

    prompt = f"""
Return ONLY JSON with keys:
- "insights"
- "charts"

Rules:
- "charts" must be a list (no code)
Sample:
{sample}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    text = res.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {"insights": "GPT failed to produce valid JSON.", "charts": []}


# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("⬅️ Upload a dataset to begin.")
    st.stop()

df_raw = (
    pd.read_csv(uploaded)
    if uploaded.name.endswith(".csv")
    else pd.read_excel(uploaded)
)

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
# DATA PREVIEWS
# ======================================================================
st.subheader("🧾 Raw Data Preview")
st.dataframe(df_raw.head(20))

st.subheader("🧹 Cleaned + Semantic-Aligned Data")
st.dataframe(df_sem.head(20))


# ======================================================================
# GPT AUTO INSIGHTS
# ======================================================================
st.subheader("🤖 GPT Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing with AI…"):
        gpt = ask_gpt(df_sem)

    st.subheader("📘 Key Insights")
    st.write(gpt.get("insights", ""))

    st.subheader("📊 Suggested Charts")
    st.write(gpt.get("charts", []))


# ======================================================================
# Q&A SECTION
# ======================================================================
st.subheader("💬 Ask a Question About Your Data")

query = st.text_area("Your question")

if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        sample = df_sem.head(50).to_csv(index=False)
        prompt = f"Dataset:\n{sample}\n\nQuestion: {query}\nAnswer clearly in business language."

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        st.write(res.choices[0].message.content)
