# ======================================================================
# DataPilot 6.1.4 – Semantic KPI Edition (Clean Code, No Hidden Unicode)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import json

# ======================================================================
# Streamlit Configuration
# ======================================================================
st.set_page_config(
    page_title="DataPilot",
    layout="wide"
)

st.title("DataPilot")


# ======================================================================
# OpenAI Client
# ======================================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ======================================================================
# Semantic Mapping
# ======================================================================
SEMANTIC_MAP = {
    "revenue": ["revenue", "sales", "gmv", "turnover", "amount", "income", "amt", "rev"],
    "units_sold": ["units", "qty", "quantity", "sold", "units_sold"],
    "daily_demand": ["demand", "daily_demand", "orders", "order_qty"],
    "inventory_on_hand": ["inventory", "stock", "onhand", "available_qty"],
    "stockout_flag": ["stockout", "out_of_stock", "oos"],
    "lead_time_days": ["leadtime", "lead_time"],
    "patients_in": ["admissions", "patients_in", "inflow"],
    "patients_out": ["patients_out", "discharges", "outflow"],
    "surgery_count": ["surgeries", "surgery"],
    "spend": ["spend", "cost", "budget", "marketing_spend", "ad_spend"],
    "profit": ["profit", "net_profit"],
    "expenses": ["expense", "expenses", "costs"]
}


# ======================================================================
# Semantic Column Renaming
# ======================================================================
def semantic_match(col):
    col_l = col.lower()
    for key, synonyms in SEMANTIC_MAP.items():
        if any(term in col_l for term in synonyms):
            return key
    return None


def harmonize_columns(df):
    renamed = {}
    for col in df.columns:
        match = semantic_match(col)
        renamed[col] = match if match else col
    return df.rename(columns=renamed)


# ======================================================================
# KPI Rules
# ======================================================================
KPI_RULES = {
    "retail": {
        "keywords": ["revenue", "units_sold"],
        "kpis": {
            "Total Revenue": lambda df: df["revenue"].sum(),
            "Average Revenue per Sale": lambda df: df["revenue"].mean(),
            "Highest Revenue Sale": lambda df: df["revenue"].max(),
            "Total Units Sold": lambda df: df["units_sold"].sum()
        }
    },

    "marketing": {
        "keywords": ["spend"],
        "kpis": {
            "Total Marketing Spend": lambda df: df[[c for c in df.columns if "spend" in c]].sum().sum(),
            "Total Revenue": lambda df: df["revenue"].sum() if "revenue" in df else None,
            "ROI Revenue to Spend": lambda df: (
                df["revenue"].sum() / df[[c for c in df.columns if "spend" in c]].sum().sum()
                if "revenue" in df else None
            )
        }
    },

    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Average Daily Demand": lambda df: df["daily_demand"].mean(),
            "Total Stockouts": lambda df: df["stockout_flag"].sum() if "stockout_flag" in df else None,
            "Average Inventory On Hand": lambda df: df["inventory_on_hand"].mean()
        }
    },

    "finance": {
        "keywords": ["profit", "expenses"],
        "kpis": {
            "Total Expenses": lambda df: df["expenses"].sum(),
            "Total Profit": lambda df: df["profit"].sum() if "profit" in df else None,
            "Total Revenue": lambda df: df["revenue"].sum() if "revenue" in df else None
        }
    }
}


# ======================================================================
# Dataset Type Detection
# ======================================================================
def detect_kpi_group(df):
    cols = df.columns
    scores = {}

    for group, rule in KPI_RULES.items():
        score = sum(1 for kw in rule["keywords"] if kw in cols)
        scores[group] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ======================================================================
# Compute KPIs Based on Dataset Type
# ======================================================================
def compute_semantic_kpis(df):
    group = detect_kpi_group(df)

    if group is None:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {}

        c = numeric.columns[0]
        return {
            f"Total {c.title()}": numeric[c].sum(),
            f"Average {c.title()}": numeric[c].mean(),
            f"Max {c.title()}": numeric[c].max()
        }

    results = {}
    for label, fn in KPI_RULES[group]["kpis"].items():
        try:
            value = fn(df)
            if value is not None:
                results[label] = value
        except:
            pass

    return results


# ======================================================================
# Cleaning Function
# ======================================================================
def auto_clean_df(df):
    df = df.copy()

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("$", "")
                .str.replace("₹", "")
                .str.replace("Rs", "")
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        if any(term in col.lower() for term in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# GPT Auto EDA
# ======================================================================
def ask_gpt_for_analysis(df):
    sample = df.head(40).to_csv(index=False)

    prompt = f"""
You are a senior data analyst.
Use only this dataset sample:

{sample}

Return a JSON object with:
1. cleaning_code - defines clean_df(df). Do not read or write files.
2. eda_code - defines make_figures(df). Must use Plotly and return a dict.
3. insights - clear business insights.

Return JSON only.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = res.choices[0].message.content.replace("```json", "").replace("```", "")
    return json.loads(raw)


# ======================================================================
# Execute GPT Code Safely
# ======================================================================
def run_dynamic_code(df, code, func_name):
    namespace = {}
    exec(code, {"df": df, "px": px, "pd": pd, "np": np}, namespace)
    return namespace[func_name](df)


# ======================================================================
# File Upload
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)


# ======================================================================
# Clean and Align
# ======================================================================
df_clean = auto_clean_df(df_raw)
df_semantic = harmonize_columns(df_clean)


# ======================================================================
# KPI Display
# ======================================================================
st.subheader("Executive Summary")

kpis = compute_semantic_kpis(df_semantic)

if len(kpis) == 0:
    st.write("No meaningful KPIs detected for this dataset.")
else:
    cols = st.columns(len(kpis))
    for (label, value), col in zip(kpis.items(), cols):
        col.markdown(
            f"""
            <div style="padding:16px; border-radius:10px; background:#10141a; border:1px solid #2a2a2a;">
                <div style="font-size:14px; color:#aaaaaa;">{label}</div>
                <div style="font-size:22px; font-weight:600; margin-top:6px; color:white;">
                    {value:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ======================================================================
# Data Preview
# ======================================================================
st.subheader("Raw Data")
st.dataframe(df_raw.head(50), use_container_width=True)

st.subheader("Cleaned and Semantic-Aligned Data")
st.dataframe(df_semantic.head(50), use_container_width=True)


# ======================================================================
# GPT Auto EDA Button
# ======================================================================
st.subheader("GPT Auto Analysis")

if st.button("Run GPT Analysis"):
    with st.spinner("Generating insights and charts..."):
        gpt = ask_gpt_for_analysis(df_semantic)

    st.subheader("Insights")
    st.write(gpt["insights"])

    df2 = run_dynamic_code(df_semantic, gpt["cleaning_code"], "clean_df")
    figures = run_dynamic_code(df2, gpt["eda_code"], "make_figures")

    st.subheader("AI-Generated Charts")
    for fig in figures.values():
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Ask Data Questions
# ======================================================================
st.subheader("Ask Questions About This Dataset")

question = st.text_area("Enter your question")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            sample = df_semantic.head(50).to_csv(index=False)

            prompt = f"""
Dataset sample:
{sample}

Question: {question}

Answer clearly in simple business language.
"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

        st.write(resp.choices[0].message.content)
