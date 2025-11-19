# ======================================================================
# DataPilot – Full Version (Cloud Run Ready)
# Smart Visualization + Semantic Column Meaning + KPI Engine + GPT Insights
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
st.title("🧠 DataPilot – AI-Assisted Data Explorer")

# ======================================================================
# OPENAI CLIENT
# ======================================================================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Set it in Cloud Run → Variables.")
    st.stop()

client = OpenAI(api_key=api_key)

# ======================================================================
# SEMANTIC COLUMN MAP (Temporary version — simple & stable)
# ======================================================================
SEMANTIC_TAGS = {
    "revenue": ["revenue", "sales", "gmv", "turnover", "amount", "income", "earning"],
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
    return "unknown"

# ======================================================================
# DATASET TYPE DETECTION
# ======================================================================
DATASET_SIGNATURES = {
    "Retail Sales": ["revenue", "units_sold", "profit", "product"],
    "Marketing Performance": ["spend", "revenue", "customer"],
    "Inventory & Supply Chain": ["inventory", "demand", "stock"],
    "Finance": ["expenses", "revenue", "profit"],
    "E-Commerce": ["order", "customer", "gmv"],
    "HR / Employees": ["employee", "salary", "department"],
    "Operations": ["production", "inventory", "defect"],
}

def detect_dataset_type(semantic_cols):
    scores = {}
    for ds_type, keys in DATASET_SIGNATURES.items():
        scores[ds_type] = sum(1 for k in keys if k in semantic_cols)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Dataset"

# ======================================================================
# AUTO CLEANING
# ======================================================================
def auto_clean_df(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        # Try numeric cleaning
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        # Date parsing
        if "date" in col.lower() or "day" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df

# ======================================================================
# GPT INSIGHTS
# ======================================================================
def ask_gpt(df: pd.DataFrame, dataset_type: str):
    sample = df.head(40).astype(str).to_csv(index=False)

    prompt = f"""
You are a senior data analyst.

Dataset Type: {dataset_type}

Return ONLY valid JSON:
- insights: 5 bullet points
- recommended_metrics: 3 KPIs
- recommended_charts: 4 charts

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
        return {"insights": [], "recommended_metrics": [], "recommended_charts": []}

# ======================================================================
# FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not uploaded:
    st.info("⬅ Upload a file to get started")
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
df_clean = auto_clean_df(df_raw)

# ======================================================================
# SEMANTIC COLUMN DETECTION
# ======================================================================
semantic_map = {col: detect_semantic_label(col) for col in df_clean.columns}
semantic_cols = list(semantic_map.values())
dataset_type = detect_dataset_type(semantic_cols)

st.subheader("🧠 Semantic Understanding of Your Dataset")
st.dataframe(
    pd.DataFrame({"Column": list(df_clean.columns),
                  "Semantic Meaning": list(semantic_map.values())})
)

st.success(f"📌 Detected Dataset Type: **{dataset_type}**")

# ======================================================================
# KPI ENGINE
# ======================================================================
def compute_kpis(df, semantic_map):
    k = {}

    # Revenue
    rev_cols = [c for c, m in semantic_map.items() if m == "revenue"]
    if rev_cols:
        k["Total Revenue"] = df[rev_cols[0]].sum()

    # Units sold
    u_cols = [c for c, m in semantic_map.items() if m == "units_sold"]
    if u_cols:
        k["Units Sold"] = df[u_cols[0]].sum()

    # Spend
    s_cols = [c for c, m in semantic_map.items() if m == "spend"]
    if s_cols:
        k["Total Spend"] = df[s_cols[0]].sum()

    # ROI
    if rev_cols and s_cols and df[s_cols[0]].sum() > 0:
        k["ROI"] = df[rev_cols[0]].sum() / df[s_cols[0]].sum()

    return k

kpis = compute_kpis(df_clean, semantic_map)

st.subheader("📈 KPI Dashboard")
if kpis:
    cols = st.columns(len(kpis))
    for (name, val), col in zip(kpis.items(), cols):
        col.metric(name, f"{val:,.2f}")
else:
    st.info("No KPIs detected.")

# ======================================================================
# RAW + CLEANED DATA
# ======================================================================
st.subheader("🧾 Raw Data Preview")
st.dataframe(df_raw.head(10))

st.subheader("🧹 Cleaned Data")
st.dataframe(df_clean.head(10))

# ======================================================================
# SMART VISUALIZATION ENGINE (No Spaghetti Charts)
# ======================================================================
st.subheader("📊 Visualize Your Data")

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df_clean.columns.tolist()

chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
x_axis = st.selectbox("X-axis", all_cols)
y_axis = st.selectbox("Y-axis", numeric_cols)

unique_x = df_clean[x_axis].nunique()
auto_agg = unique_x > 50

agg_choice = st.selectbox(
    "Aggregation",
    ["None", "Sum", "Average", "Median", "Count"],
    disabled=auto_agg
)

if auto_agg:
    st.warning(
        f"⚠ {unique_x} unique values detected for '{x_axis}'. "
        f"To avoid messy charts, aggregation will be automatically applied."
    )

if st.button("Generate Chart"):
    df_plot = df_clean.copy()

    if auto_agg or agg_choice != "None":
        if agg_choice == "Sum" or auto_agg:
            df_plot = df_clean.groupby(x_axis)[y_axis].sum().reset_index()
        elif agg_choice == "Average":
            df_plot = df_clean.groupby(x_axis)[y_axis].mean().reset_index()
        elif agg_choice == "Median":
            df_plot = df_clean.groupby(x_axis)[y_axis].median().reset_index()
        elif agg_choice == "Count":
            df_plot = df_clean.groupby(x_axis)[y_axis].count().reset_index()

    if chart_type == "Line":
        fig = px.line(df_plot, x=x_axis, y=y_axis)
    elif chart_type == "Bar":
        fig = px.bar(df_plot, x=x_axis, y=y_axis)
    elif chart_type == "Scatter":
        fig = px.scatter(df_plot, x=x_axis, y=y_axis)
    else:
        fig = px.histogram(df_plot, x=x_axis)

    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# GPT INSIGHTS
# ======================================================================
st.subheader("🤖 AI Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing…"):
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
st.subheader("💬 Ask a Question About Your Data")

question = st.text_area("Ask a question")
if st.button("Ask"):
    sample = df_clean.head(40).to_csv(index=False)
    prompt = f"""
Dataset Type: {dataset_type}

Dataset:
{sample}

Question: {question}

Answer based ONLY on the data.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    st.write(res.choices[0].message.content)
