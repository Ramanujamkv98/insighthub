# ======================================================================
# DataPilot – FULL Semantic Version (Cloud Run optimized)
# Semantic Understanding of Column Headers + Dataset Type Detection
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
# OPENAI CLIENT
# ======================================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Add it in Cloud Run → Variables")
    st.stop()

client = OpenAI(api_key=api_key)

# ======================================================================
# SEMANTIC COLUMN MAP
# (Synonym dictionary for understanding header meanings)
# ======================================================================
SEMANTIC_TAGS = {
    "revenue": ["revenue", "sales", "gmv", "turnover", "amount", "income"],
    "units_sold": ["units", "qty", "sold", "quantity", "volume"],
    "spend": ["spend", "ad_spend", "marketing_spend", "cost", "budget"],
    "profit": ["profit", "margin", "net"],
    "expenses": ["expense", "expenses", "operational_cost"],
    "inventory": ["inventory", "stock", "onhand", "inv"],
    "demand": ["demand", "orders", "order_qty"],
    "date": ["date", "day", "time", "timestamp"],
    "customer": ["customer", "client", "user", "buyer"],
    "id": ["id", "code", "sku", "item"],
}


def detect_semantic_label(col: str):
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
    "Inventory & Supply Chain": ["inventory", "demand", "stockout"],
    "Finance": ["expenses", "revenue", "profit"],
    "E-Commerce": ["order", "customer", "gmv", "units_sold"],
    "HR / Employees": ["employee", "salary", "department"],
    "Operations / Manufacturing": ["production", "inventory", "defect"],
    "Survey / Feedback": ["rating", "feedback", "comment"],
    "Healthcare": ["patient", "treatment", "diagnosis"],
}

def detect_dataset_type(semantic_cols: list[str]):
    scores = {}
    for ds_type, keywords in DATASET_SIGNATURES.items():
        matches = sum(1 for k in keywords if k in semantic_cols)
        scores[ds_type] = matches
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Dataset"


# ======================================================================
# AUTOMATIC CLEANING
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

        if any(k in col.lower() for k in ["date", "time", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df: pd.DataFrame, dataset_type: str):
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
You are a senior data analyst.

Dataset type: {dataset_type}

Analyze the data sample and return ONLY VALID JSON with:

- "insights": 4–6 key insights
- "recommended_metrics": 3–5 KPIs relevant to dataset type
- "recommended_charts": 4–6 visualization ideas

Data sample:
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
            "insights": ["GPT failed to parse JSON."],
            "recommended_metrics": [],
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

# Detect semantic column meanings
semantic_map = {col: detect_semantic_label(col) for col in df_clean.columns}
semantic_cols = [v for v in semantic_map.values() if v]

# Identify dataset type
dataset_type = detect_dataset_type(semantic_cols)


# ======================================================================
# SHOW SEMANTIC ANALYSIS
# ======================================================================
st.subheader("🧠 Semantic Understanding of Your Dataset")

semantic_df = pd.DataFrame({
    "Column": list(df_clean.columns),
    "Semantic Meaning": [semantic_map[col] or "unknown" for col in df_clean.columns]
})

st.dataframe(semantic_df)

st.success(f"**Detected Dataset Type → {dataset_type}**")


# ======================================================================
# KPI SUGGESTIONS
# ======================================================================
st.subheader("📌 KPI Dashboard (Auto-Detected)")

def compute_kpis(df, semantic_map):
    results = {}

    # Revenue
    rev_col = [c for c, s in semantic_map.items() if s == "revenue"]
    if rev_col:
        results["Total Revenue"] = df[rev_col[0]].sum()

    # Units sold
    u_col = [c for c, s in semantic_map.items() if s == "units_sold"]
    if u_col:
        results["Units Sold"] = df[u_col[0]].sum()

    # Spend
    s_col = [c for c, s in semantic_map.items() if s == "spend"]
    if s_col:
        results["Total Spend"] = df[s_col[0]].sum()

    # ROI if both exist
    if rev_col and s_col and df[s_col[0]].sum() > 0:
        results["ROI"] = df[rev_col[0]].sum() / df[s_col[0]].sum()

    return results

kpis = compute_kpis(df_clean, semantic_map)

if kpis:
    cols = st.columns(len(kpis))
    for (k, v), col in zip(kpis.items(), cols):
        col.metric(k, f"{v:,.2f}")
else:
    st.info("No KPIs detected.")



# ======================================================================
# RAW + CLEANED DATA
# ======================================================================
st.subheader("🧾 Raw Data Preview")
st.dataframe(df_raw.head(20))

st.subheader("🧹 Cleaned Data")
st.dataframe(df_clean.head(20))


# ======================================================================
# VISUALIZATION STUDIO
# ======================================================================
st.subheader("📊 Visualize Your Data")

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df_clean.columns.tolist()

if all_cols:

    chart_type = st.selectbox("Chart type", ["Line", "Bar", "Scatter", "Histogram"])
    x_axis = st.selectbox("X-axis", all_cols)
    y_axis = st.selectbox("Y-axis", numeric_cols)

    if st.button("Generate Chart"):
        if chart_type == "Line":
            fig = px.line(df_clean, x=x_axis, y=y_axis)
        elif chart_type == "Bar":
            fig = px.bar(df_clean, x=x_axis, y=y_axis)
        elif chart_type == "Scatter":
            fig = px.scatter(df_clean, x=x_axis, y=y_axis)
        else:
            fig = px.histogram(df_clean, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# GPT INSIGHTS
# ======================================================================
st.subheader("🤖 AI Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Thinking…"):
        g = ask_gpt(df_clean, dataset_type)

    st.write("### 🔍 Key Insights")
    for i in g["insights"]:
        st.write("•", i)

    st.write("### 📌 Recommended KPIs")
    for i in g["recommended_metrics"]:
        st.write("•", i)

    st.write("### 📊 Recommended Charts")
    for i in g["recommended_charts"]:
        st.write("•", i)


# ======================================================================
# Q&A SECTION
# ======================================================================
st.subheader("💬 Ask a Question")

question = st.text_area("Ask anything about your dataset")

if st.button("Ask"):
    sample = df_clean.head(40).to_csv(index=False)
    prompt = f"""
Dataset type: {dataset_type}

Dataset:
{sample}

Question: {question}

Answer using only the data and correct business language.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    st.write(res.choices[0].message.content)
