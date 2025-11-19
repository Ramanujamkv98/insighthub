# ======================================================================
# DataPilot – MVP
# OpenAI v1.x | Plotly | Pandas | Fast + Safe
# ======================================================================

import os
import json
import re
import numpy as np
import pandas as pd
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
            ) if "revenue" in df.columns and df.filter(regex="spend").sum().sum() != 0 else None,
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
    scores = {
        g: sum(k in df.columns for k in r["keywords"])
        for g, r in KPI_RULES.items()
    }
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
        # Try numeric cleanup
        if df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        # Try date parsing
        if any(x in col.lower() for x in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df

# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df: pd.DataFrame):
    # Take a small sample to keep prompt small
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
Return ONLY valid JSON with keys:
- "insights": a short paragraph
- "charts": a list of suggested chart ideas (strings only)

Dataset sample:
{sample}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    text = res.choices[0].message.content

    # Try to parse JSON; if it fails, fall back gracefully
    try:
        # Simple heuristic: find first '{' and last '}' and try json.loads
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
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx"]
)

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
# KPI DASHBOARD
# ======================================================================
st.subheader("📌 Executive Summary")

kpis = compute_kpis(df_sem)

if kpis:
    cols = st.columns(len(kpis))
    for (label, value), col in zip(kpis.items(), cols):
        col.metric(label, f"{value:,.2f}")
else:
    st.write("No KPIs detected based on this dataset.")

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
    charts = gpt.get("charts", [])
    if isinstance(charts, list):
        for i, c in enumerate(charts, 1):
            st.markdown(f"**{i}.** {c}")
    else:
        st.write(charts)

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
        prompt = (
            f"Dataset:\n{sample}\n\n"
            f"Question: {query}\n\n"
            f"Answer clearly in business language. "
            f"Do not hallucinate columns that don't exist."
        )

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        st.write(res.choices[0].message.content)
