# ======================================================================
# InsightHub 6.1.3 – Semantic KPI Edition
# Fully fixed version – No emojis – Streamlit Cloud Compatible
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import json
import re

# ======================================================================
# STREAMLIT CONFIG
# ======================================================================
st.set_page_config(
    page_title="InsightHub 6.1.3 – Semantic Auto EDA",
    layout="wide",
)

st.title("InsightHub 6.1.3 – Semantic Auto EDA")
st.caption("Upload dataset → Semantic cleaning → Business KPIs → GPT EDA → Ask questions")


# ======================================================================
# OPENAI CLIENT
# ======================================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ======================================================================
# 1. SEMANTIC MAP
# ======================================================================
SEMANTIC_MAP = {
    "revenue": ["revenue", "sales", "gmv", "amount", "amt", "rev", "turnover", "income"],
    "units_sold": ["units", "units_sold", "qty", "quantity", "sold"],
    "daily_demand": ["demand", "daily_demand", "orders", "order_qty"],
    "inventory_on_hand": ["inventory", "inv_onhand", "stock", "onhand", "available_qty"],
    "stockout_flag": ["stockout", "out_of_stock", "oos"],
    "lead_time_days": ["leadtime", "lead_time"],
    "patients_in": ["admissions", "patients_in", "inflow"],
    "patients_out": ["patients_out", "discharges", "outflow"],
    "surgery_count": ["surgeries", "surgery"],
    "spend": ["spend", "cost", "budget", "ad_spend", "marketing_spend"],
    "profit": ["profit", "net_profit"],
    "expenses": ["expense", "expenses", "costs"]
}


# ======================================================================
# 2. SEMANTIC COLUMN RENAMING
# ======================================================================
def semantic_match(col):
    col_l = col.lower()
    for key, synonyms in SEMANTIC_MAP.items():
        if any(s in col_l for s in synonyms):
            return key
    return None


def harmonize_columns(df):
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        meaning = semantic_match(col)
        rename_map[col] = meaning if meaning else col
    return df.rename(columns=rename_map)


# ======================================================================
# 3. KPI RULE ENGINE
# ======================================================================
KPI_RULES = {
    "retail": {
        "keywords": ["revenue", "units_sold"],
        "kpis": {
            "Total Revenue Generated": lambda df: df["revenue"].sum(),
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
            "ROI (Revenue / Spend)": lambda df: (
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
            "Average Inventory on Hand": lambda df: df["inventory_onhand"].mean()
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
# 4. DETECT KPI GROUP
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
# 5. COMPUTE SEMANTIC KPIs
# ======================================================================
def compute_semantic_kpis(df):
    group = detect_kpi_group(df)

    if group is None:
        num = df.select_dtypes(include="number")
        if len(num.columns) == 0:
            return {}
        c = num.columns[0]
        return {
            f"Total {c.title()}": num[c].sum(),
            f"Average {c.title()}": num[c].mean(),
            f"Maximum {c.title()}": num[c].max()
        }

    kpi_funcs = KPI_RULES[group]["kpis"]
    results = {}

    for label, fn in kpi_funcs.items():
        try:
            value = fn(df)
            if value is not None:
                results[label] = value
        except:
            pass

    return results


# ======================================================================
# 6. AUTO CLEANING
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

        if any(k in col.lower() for k in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# 7. GPT ANALYSIS
# ======================================================================
def ask_gpt_for_analysis(df):
    SAMPLE = df.head(40).to_csv(index=False)

    prompt = f"""
You are a senior data analyst. Based only on the dataset sample below:

{SAMPLE}

Return valid JSON with fields:
1. cleaning_code — must define clean_df(df). Do not read or write any files. Never use pd.read_csv.
2. eda_code — must define make_figures(df) returning a dict of Plotly charts.
3. insights — business-friendly insights.

Return only valid JSON.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = res.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "")
    return json.loads(raw)


# ======================================================================
# 8. EXECUTE GPT CODE SAFELY
# ======================================================================
def run_dynamic_code(df, code, func_name):
    local_vars = {}
    exec(code, {"df": df, "px": px, "pd": pd, "np": np}, local_vars)
    return local_vars[func_name](df)


# ======================================================================
# 9. FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)


# ======================================================================
# 10. PREPROCESS + SEMANTIC ALIGNMENT
# ======================================================================
df_clean = auto_clean_df(df_raw)
df_semantic = harmonize_columns(df_clean)


# ======================================================================
# 11. EXECUTIVE SUMMARY KPI DISPLAY
# ======================================================================
st.subheader("Executive Summary")
kpis = compute_semantic_kpis(df_semantic)

cols = st.columns(len(kpis))
for (label, value), col in zip(kpis.items(), cols):
    with col:
        st.markdown(
            f"""
            <div style="padding:16px; border-radius:10px; background:#10141a; border:1px solid #1f2937;">
                <div style="font-size:14px; color:#9ca3af;">{label}</div>
                <div style="font-size:22px; font-weight:600; margin-top:6px; color:white;">
                    {value:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ======================================================================
# 12. RAW + CLEANED PREVIEW
# ======================================================================
st.subheader("Raw Data")
st.dataframe(df_raw.head(50), use_container_width=True)

st.subheader("Cleaned and Semantic-Aligned Data")
st.dataframe(df_semantic.head(50), use_container_width=True)


# ======================================================================
# 13. GPT AUTO ANALYSIS
# ======================================================================
st.subheader("GPT Auto EDA")

if st.button("Run GPT Analysis"):
    with st.spinner("Processing..."):
        gpt = ask_gpt_for_analysis(df_semantic)

    st.success("Completed")

    st.subheader("Insights")
    st.write(gpt["insights"])

    df2 = run_dynamic_code(df_semantic, gpt["cleaning_code"], "clean_df")
    figs = run_dynamic_code(df2, gpt["eda_code"], "make_figures")

    st.subheader("Charts")
    for fig in figs.values():
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 14. ASK QUESTIONS ABOUT THE DATA
# ======================================================================
st.subheader("Ask Questions About This Dataset")

q = st.text_area("Enter your question")

if st.button("Ask"):
    if q.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            sample = df_semantic.head(50).to_csv(index=False)

            prompt = f"""
You are a business data analyst.
Dataset sample:
{sample}

Question: {q}

Answer clearly in plain business language.
"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

        st.write(resp.choices[0].message.content)
