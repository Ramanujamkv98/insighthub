# ======================================================================
# DataPilot 7.3 – Stable Semantic KPI Edition
# Streamlit Cloud Ready + OpenAI v1.x
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
st.title("DataPilot")


# ======================================================================
# OPENAI CLIENT (works on Streamlit Cloud & locally)
# ======================================================================
api_key = (
    st.secrets["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets
    else os.getenv("OPENAI_API_KEY")
)

if not api_key:
    st.error(
        "❌ OPENAI_API_KEY not found.\n\n"
        "• On Streamlit Cloud: add it under **Settings → Secrets** as OPENAI_API_KEY\n"
        "• Locally: set environment variable OPENAI_API_KEY"
    )
    st.stop()

client = OpenAI(api_key=api_key)


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
    "expenses": ["expense", "expenses", "costs"],
}


# ======================================================================
# 2. SEMANTIC COLUMN RENAMING
# ======================================================================
def semantic_match(col: str):
    col_l = col.lower()
    for key, synonyms in SEMANTIC_MAP.items():
        if any(s in col_l for s in synonyms):
            return key
    return None


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
            "Total Units Sold": lambda df: df["units_sold"].sum(),
        },
    },
    "marketing": {
        "keywords": ["spend"],
        "kpis": {
            "Total Marketing Spend": lambda df: df[
                [c for c in df.columns if "spend" in c]
            ].sum().sum(),
            "Total Revenue": lambda df: df["revenue"].sum()
            if "revenue" in df
            else None,
            "ROI (Revenue / Spend)": lambda df: (
                df["revenue"].sum()
                / df[[c for c in df.columns if "spend" in c]].sum().sum()
                if "revenue" in df
                else None
            ),
        },
    },
    "inventory": {
        "keywords": ["inventory_on_hand", "daily_demand"],
        "kpis": {
            "Average Daily Demand": lambda df: df["daily_demand"].mean(),
            "Total Stockouts": lambda df: df["stockout_flag"].sum()
            if "stockout_flag" in df
            else None,
            "Average Inventory on Hand": lambda df: df["inventory_on_hand"].mean(),
        },
    },
    "finance": {
        "keywords": ["profit", "expenses"],
        "kpis": {
            "Total Expenses": lambda df: df["expenses"].sum()
            if "expenses" in df
            else None,
            "Total Profit": lambda df: df["profit"].sum()
            if "profit" in df
            else None,
            "Total Revenue": lambda df: df["revenue"].sum()
            if "revenue" in df
            else None,
        },
    },
}


# ======================================================================
# 4. DETECT KPI GROUP
# ======================================================================
def detect_kpi_group(df: pd.DataFrame):
    scores = {
        grp: sum(1 for kw in rule["keywords"] if kw in df.columns)
        for grp, rule in KPI_RULES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ======================================================================
# 5. COMPUTE SEMANTIC KPIs
# ======================================================================
def compute_semantic_kpis(df: pd.DataFrame):
    group = detect_kpi_group(df)

    if group is None:
        num = df.select_dtypes(include="number")
        if len(num.columns) == 0:
            return {}
        c = num.columns[0]
        return {
            f"Total {c}": num[c].sum(),
            f"Average {c}": num[c].mean(),
            f"Maximum {c}": num[c].max(),
        }

    results = {}
    for label, fn in KPI_RULES[group]["kpis"].items():
        try:
            val = fn(df)
            if val is not None:
                results[label] = val
        except Exception:
            # Skip KPIs that error out on a particular dataset
            pass

    return results


# ======================================================================
# 6. AUTO CLEANING
# ======================================================================
def auto_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop unnamed index-like columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("Rs", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="ignore")

        # try to parse date-like columns
        if any(k in col.lower() for k in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# ======================================================================
# 7. GPT JSON-SAFE ANALYSIS
# ======================================================================
def ask_gpt_for_analysis(df: pd.DataFrame):
    sample = df.head(40).to_csv(index=False)

    prompt = f"""
Return ONLY raw JSON with keys:
- "cleaning_code"
- "eda_code"
- "insights"

Rules:
- cleaning_code must define clean_df(df)
- eda_code must define make_figures(df) and return a dict of Plotly figures
- No markdown
- No backticks
Dataset sample:
{sample}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return {"cleaning_code": "", "eda_code": "", "insights": "LLM call failed."}

    raw = res.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # regex extraction
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except Exception:
            raw = cleaned

    # trailing comma clean-up
    cleaned = re.sub(r",\s*}", "}", raw)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "cleaning_code": "",
            "eda_code": "",
            "insights": "GPT returned invalid JSON; showing fallback summary.",
        }


# ======================================================================
# 8. SAFE EXECUTION OF GPT CODE
# ======================================================================
def run_dynamic_code(df: pd.DataFrame, code: str, func_name: str):
    df_safe = df.copy()

    # convert datetime to string for safety
    dt_cols = df_safe.select_dtypes(include=[np.datetime64]).columns
    df_safe[dt_cols] = df_safe[dt_cols].astype(str)

    local_vars: dict = {}
    try:
        exec(code, {"df": df_safe, "px": px, "pd": pd, "np": np}, local_vars)
        return local_vars[func_name](df_safe)
    except Exception as e:
        st.warning(f"Error running generated code ({func_name}): {e}")
        return df_safe if func_name == "clean_df" else {}


# ======================================================================
# 9. FILE UPLOAD
# ======================================================================
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded is None:
    st.info("👈 Upload a CSV or Excel file to start exploring your data.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)


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

if kpis:
    cols = st.columns(len(kpis))
    for (label, value), col in zip(kpis.items(), cols):
        with col:
            st.markdown(
                f"""
                <div style="padding:16px; border-radius:10px;
                background:#10141a; border:1px solid #1f2937;">
                    <div style="font-size:14px; color:#9ca3af;">{label}</div>
                    <div style="font-size:22px; font-weight:600; color:white;">
                        {value:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.write("No numeric KPIs detected yet – try another dataset.")


# ======================================================================
# 12. PREVIEWS
# ======================================================================
st.subheader("Raw Data (sample)")
st.dataframe(df_raw.head(30))

st.subheader("Cleaned + Semantic Aligned (sample)")
st.dataframe(df_semantic.head(30))


# ======================================================================
# 13. GPT AUTO EDA
# ======================================================================
st.subheader("GPT Auto EDA")

if st.button("Run GPT Analysis"):
    with st.spinner("Running LLM-powered analysis…"):
        gpt = ask_gpt_for_analysis(df_semantic)

    st.success("Analysis complete.")

    st.subheader("Insights")
    st.write(gpt.get("insights", "No insights returned."))

    if gpt.get("cleaning_code"):
        df2 = run_dynamic_code(df_semantic, gpt["cleaning_code"], "clean_df")
    else:
        df2 = df_semantic

    if gpt.get("eda_code"):
        figs = run_dynamic_code(df2, gpt["eda_code"], "make_figures")
    else:
        figs = {}

    if isinstance(figs, dict) and figs:
        st.subheader("Charts")
        for name, fig in figs.items():
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Error rendering figure '{name}': {e}")
    else:
        st.info("No charts generated by the model.")


# ======================================================================
# 14. QUESTION ANSWERING
# ======================================================================
st.subheader("Ask Questions About This Dataset")
q = st.text_area("Your question")

if st.button("Ask"):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            sample = df_semantic.head(50).to_csv(index=False)
            prompt = f"Dataset:\n{sample}\n\nQuestion: {q}\nAnswer clearly for a business user."

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"OpenAI error: {e}"

        st.write(answer)
