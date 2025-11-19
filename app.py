# ======================================================================
# DataPilot – FULL Semantic Version (Cloud Run optimized)
# Semantic Understanding + KPIs + Visualizations + AI Insights
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
st.title("🧠 DataPilot – AI-Assisted Semantic Data Explorer")

# ======================================================================
# OPENAI CLIENT (Cloud Run Compatible – NO st.secrets)
# ======================================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY missing. Add it in Cloud Run → Variables.")
    st.stop()

client = OpenAI(api_key=api_key)

# ======================================================================
# SEMANTIC COLUMN MAP
# ======================================================================
SEMANTIC_TAGS = {
    "revenue": ["revenue", "sales", "gmv", "turnover", "amount", "income"],
    "units_sold": ["units", "qty", "sold", "quantity", "volume"],
    "spend": ["spend", "ad_spend", "marketing_spend", "cost", "budget"],
    "profit": ["profit", "margin", "net"],
    "expenses": ["expense", "expenses", "operational_cost"],
    "inventory": ["inventory", "stock", "onhand", "inv"],
    "demand": ["demand", "orders", "order_qty", "demand_qty"],
    "date": ["date", "day", "time", "timestamp"],
    "customer": ["customer", "client", "user", "buyer"],
    "id": ["id", "code", "sku", "item"],
    "rating": ["rating", "score", "feedback"],
}


def detect_semantic_label(col: str):
    """Return semantic meaning of a column"""
    col_l = col.lower()
    for label, synonyms in SEMANTIC_TAGS.items():
        if any(word in col_l for word in synonyms):
            return label
    return None


# ======================================================================
# DATASET TYPE DETECTION
# ======================================================================
DATASET_SIGNATURES = {
    "Retail Sales": ["revenue", "units_sold", "profit"],
    "Marketing Performance": ["spend", "revenue", "customer"],
    "Inventory & Supply Chain": ["inventory", "demand"],
    "Finance": ["expenses", "revenue", "profit"],
    "E-Commerce": ["customer", "orders", "gmv"],
    "Survey / Feedback": ["rating", "feedback"],
    "Healthcare": ["patient", "treatment"],
    "HR / Employees": ["salary", "employee"],
}

def detect_dataset_type(semantic_cols):
    scores = {}
    for ds_type, keywords in DATASET_SIGNATURES.items():
        score = sum(1 for k in keywords if k in semantic_cols)
        scores[ds_type] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "General Dataset"

    return best


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

        if any(k in col.lower() for k in ["date", "time", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df, dataset_type):
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
You are a senior data analyst.

Dataset type: {dataset_type}

Return ONLY VALID JSON with keys:
- "insights"
- "recommended_kpis"
- "recommended_charts"

Dataset sample:
{sample}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = res.choices[0].message.content

    try:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end+1])
    except:
        return {
            "insights": ["GPT could not parse insights."],
            "recommended_kpis": [],
            "recommended_charts": [],
        }


# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("⬅ Upload a dataset to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
df_clean = auto_clean_df(df_raw)

# Detect semantic labels
semantic_map = {col: detect_semantic_label(col) for col in df_clean.columns}
semantic_cols = [v for v in semantic_map.values() if v]

dataset_type = detect_dataset_type(semantic_cols)


# ======================================================================
# SEMANTIC OVERVIEW
# ======================================================================
st.subheader("🧠 Semantic Understanding")

semantic_df = pd.DataFrame({
    "Column": df_clean.columns,
    "Meaning": [semantic_map[c] or "unknown" for c in df_clean.columns]
})

st.dataframe(semantic_df)
st.success(f"**Detected Dataset Type → {dataset_type}**")


# ======================================================================
# KPI SUGGESTIONS
# ===================
