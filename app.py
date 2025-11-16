# ======================================================================
# DataPilot 7.0 – Ultra-Stable Semantic KPI Edition
# Safe GPT Execution + JSON Hardening + Figure Validation
# Streamlit Cloud Compatible
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
    page_title="DataPilot",
    layout="wide",
)

st.title("DataPilot")


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
            "ROI (Revenue / Spend)": lambda df:
                df["revenue"].sum() / df[[c for c in df.columns if "spend" in c]].sum().sum()
                if "revenue" in df else None
        }
    },

    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Average Daily Demand": lambda df: df["daily_demand"].mean(),
            "Total Stockouts": lambda df: df["stockout_flag"].sum() if "stockout_flag" in df else None,
            "Average Inventory on Hand": lambda df: df["inventory_on_hand"].mean()
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
    scores = {grp: sum(1 for kw in rule["keywords"] if kw in cols) for grp, rule in KPI_RULES.items()}
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
            f"Total {c}": num[c].sum(),
            f"Average {c}": num[c].mean(),
            f"Maximum {c}": num[c].max()
        }

    results = {}
    for label, fn in KPI_RULES[group]["kpis"].items():
        try:
            val = fn(df)
            if val is not None:
                results[label] = val
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
            cleaned = (df[col].astype(str)
                       .str.replace(",", "")
                       .str.replace("$", "")
                       .str.replace("₹", "")
                       .str.replace("Rs", "")
                       .str.strip())
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        if any(k in col.lower() for k in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# 7. GPT JSON SAFE ANALYSIS
# ======================================================================
def ask_gpt_for_analysis(df):
    SAMPLE = df.head(40).to_csv(index=False)

    prompt = f"""
Return ONLY valid JSON with keys:
- "cleaning_code"
- "eda_code"
- "insights"

Rules:
- cleaning_code must define clean_df(df)
- eda_code must define make_figures(df) and return a dict of Plotly figures
- Output ONLY RAW JSON (no backticks)
- No markdown, no explanation
- Only aggregate numeric columns

Dataset sample:
{SAMPLE}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = res.choices[0].message.content.strip()

    # strip markdown if any
    raw = raw.replace("```json", "").replace("```", "").strip()

    # try direct parse
    try:
        return json.loads(raw)
    except:
        pass

    # extract JSON with regex
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except:
            pass

    # remove trailing commas
    cleaned = re.sub(r",\s*}", "}", raw)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    try:
        return json.loads(cleaned)
    except Exception as e:
        st.error("GPT returned invalid JSON. Raw output:")
        st.code(raw)
        raise e


# ======================================================================
# 8. SAFE EXECUTION OF GPT CODE
# ======================================================================
def run_dynamic_code(df, code, func_name):
    df_safe = df.copy()

    # prevent datetime sum errors
    dt_cols = df_safe.select_dtypes(include=["datetime64[ns]", "datetime64[ns, tz]"]).columns
    df_safe[dt_cols] = df_safe[dt_cols].astype(str)

    local_vars = {}
    exec(code, {"df": df_safe, "px": px, "pd": pd, "np": np}, local_vars)

    return local_vars[func_name](df_safe)


# ======================================================================
# 9. FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)


# ======================================================================
# 10. CLEAN + SEMANTIC ALIGN
# ======================================================================
df_clean = auto_clean_df(df_raw)
df_semantic = harmonize_columns(df_clean)


# ======================================================================
# 11. KPI DASHBOARD
# ======================================================================
st.subheader("Executive Summary")
kpis = compute_semantic_kpis(df_semantic)

cols = st.columns(len(kpis) if len(kpis) > 0 else 1)
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
st.dataframe(df_raw.head(30))

st.subheader("Cleaned + Semantic-Aligned")
st.dataframe(df_semantic.head(30))


# ======================================================================
# 13. GPT AUTO EDA
# ======================================================================
st.subheader("GPT Auto EDA")

if st.button("Run GPT Analysis"):
    with st.spinner("Running GPT…"):
        gpt = ask_gpt_for_analysis(df_semantic)

    st.success("Done")

    st.subheader("Insights")
    st.write(gpt["insights"])

    df2 = run_dynamic_code(df_semantic, gpt["cleaning_code"], "clean_df")
    figs = run_dynamic_code(df2, gpt["eda_code"], "make_figures")

    st.subheader("Charts")
    for name, fig in figs.items():
        try:
            if hasattr(fig, "to_plotly_json"):
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Skipping invalid figure: {name}")
        except Exception as e:
            st.warning(f"Error rendering figure {name}: {e}")


# ======================================================================
# 14. QUESTION ANSWERING
# ======================================================================
st.subheader("Ask Questions About This Dataset")
q = st.text_area("Your question")

if st.button("Ask"):
    if not q.strip():
        st.warning("Enter a question")
    else:
        with st.spinner("Thinking..."):
            sample = df_semantic.head(50).to_csv(index=False)
            prompt = f"Dataset:\n{sample}\n\nQuestion: {q}\nAnswer in simple business language."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
        st.write(resp.choices[0].message.content)
gi