# ======================================================================
# DataPilot – Medium Version (GCP-Ready)
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
# OPENAI CLIENT (Cloud Run ENV VAR)
# ======================================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Set it in Cloud Run → Variables → Add Variable.")
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
    return col

def harmonize_columns(df):
    return df.rename(columns={col: semantic_match(col) for col in df.columns})

# ======================================================================
# KPI ENGINE
# ======================================================================
KPI_RULES = {
    "retail": {
        "keywords": ["revenue", "units_sold"],
        "kpis": {
            "Total Revenue": lambda df: df.get("revenue", pd.Series()).sum(),
            "Units Sold": lambda df: df.get("units_sold", pd.Series()).sum(),
            "Avg Revenue per Sale": lambda df: df.get("revenue", pd.Series()).mean(),
        },
    },
    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Avg Daily Demand": lambda df: df.get("daily_demand", pd.Series()).mean(),
            "Avg Inventory On-Hand": lambda df: df.get("inventory_on_hand", pd.Series()).mean(),
        },
    },
    "marketing": {
        "keywords": ["spend"],
        "kpis": {
            "Total Spend": lambda df: df.filter(regex="spend").sum().sum(),
            "ROI": lambda df:
                (df.get("revenue", pd.Series()).sum() /
                 df.filter(regex="spend").sum().sum())
                if df.filter(regex="spend").sum().sum() else None,
        },
    },
}

def detect_kpi_group(df):
    scores = {
        g: sum(k in df.columns for k in rule["keywords"])
        for g, rule in KPI_RULES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def compute_kpis(df):
    group = detect_kpi_group(df)
    if not group:
        return {}
    results = {}
    for name, func in KPI_RULES[group]["kpis"].items():
        try:
            val = func(df)
            if val is not None and not pd.isna(val):
                results[name] = float(val)
        except:
            pass
    return results

# ======================================================================
# AUTO CLEANING
# ======================================================================
def auto_clean_df(df: pd.DataFrame) -> pd.DataFrame:
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
    sample = df.head(30).astype(str).to_csv(index=False)

    prompt = f"""
Return only JSON.
Keys:
- "insights": a short paragraph
- "charts": a list of chart ideas

Data sample:
{sample}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        text = res.choices[0].message.content

        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end+1])
    except:
        return {"insights": "GPT error.", "charts": []}

# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("⬅️ Upload a dataset to begin analysis.")
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
# PREVIEWS
# ======================================================================
st.subheader("🧾 Raw Data Preview")
st.dataframe(df_raw.head(20))

st.subheader("🧹 Cleaned + Semantic-Aligned Data")
st.dataframe(df_sem.head(20))

# ======================================================================
# GPT INSIGHTS
# ======================================================================
st.subheader("🤖 AI Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing with AI..."):
        out = ask_gpt(df_sem)

    st.write("### 📘 Key Insights")
    st.write(out.get("insights", ""))

    st.write("### 📊 Suggested Charts")
    for i, c in enumerate(out.get("charts", []), 1):
        st.write(f"**{i}.** {c}")

# ======================================================================
# Q&A
# ======================================================================
st.subheader("💬 Ask a question about your data")

q = st.text_area("Your question")

if st.button("Ask"):
    if not q.strip():
        st.warning("Enter a question.")
    else:
        sample = df_sem.head(40).to_csv(index=False)
        prompt = f"""
Dataset:
{sample}

Question: {q}

Answer clearly. Use only available columns.
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        st.write(res.choices[0].message.content)
