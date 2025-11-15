# ============================================================
# InsightHub 6.1 – Scientific Analyst Mode + SMB Friendly UI
# Complete Streamlit Application (Plug-and-Play)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import json
import re

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="InsightHub 6.1 – GPT Auto EDA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 InsightHub 6.1 – GPT Auto EDA")
st.caption("Upload your dataset → AI cleans it → AI builds charts → Ask questions in plain English.")


# =============================================================
# GPT CLIENT (requires OPENAI_API_KEY in Streamlit Secrets)
# =============================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =============================================================
# UTILITY: Clean monetary & messy columns
# =============================================================
def auto_clean_df(df):
    df = df.copy()

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # Clean currency-like columns
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

    # Fix date columns
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df


# =============================================================
# DETECT DATASET TYPE
# =============================================================
def infer_dataset_type(df):
    cols = " ".join(df.columns).lower()

    if any(k in cols for k in ["spend", "cpc", "ctr", "campaign", "impressions"]):
        return "marketing"

    if any(k in cols for k in ["unit", "sales", "quantity", "sku", "item"]):
        return "sales"

    if any(k in cols for k in ["stock", "inventory", "reorder", "demand"]):
        return "inventory"

    if any(k in cols for k in ["profit", "margin", "expense", "cost"]):
        return "finance"

    return "general"


# =============================================================
# EXECUTIVE OVERVIEW: DATASET-AWARE CARDS
# =============================================================
def make_overview_cards(df):
    ds = infer_dataset_type(df)
    cards = {}

    # ---------- MARKETING DATA ----------
    if ds == "marketing":
        spend_cols = [c for c in df.columns if "spend" in c.lower()]

        if "revenue" in df.columns:
            cards["Total Revenue"] = f"₹{df['revenue'].sum():,.0f}"

        if spend_cols and "revenue" in df.columns:
            roi_strength = {c: df["revenue"].corr(df[c]) for c in spend_cols if df[c].dtype != "O"}
            best_roi = max(roi_strength, key=roi_strength.get)
            cards["Best ROI Channel"] = best_roi.replace("_", " ").title()

        if spend_cols:
            top_spend = max(spend_cols, key=lambda c: df[c].sum())
            cards["Highest Spend Channel"] = top_spend.replace("_", " ").title()

    # ---------- SALES DATA ----------
    elif ds == "sales":
        if "units" in df.columns:
            cards["Total Units Sold"] = f"{df['units'].sum():,.0f}"

        if "revenue" in df.columns:
            cards["Total Revenue"] = f"₹{df['revenue'].sum():,.0f}"

        if "product" in df.columns and "revenue" in df.columns:
            top_prod = df.loc[df["revenue"].idxmax(), "product"]
            cards["Top Revenue Product"] = top_prod

    # ---------- INVENTORY ----------
    elif ds == "inventory":
        if "stock" in df.columns:
            cards["Total Stock"] = df["stock"].sum()
            cards["Average Stock per Item"] = round(df["stock"].mean(), 2)

    # ---------- FINANCE ----------
    elif ds == "finance":
        if "revenue" in df.columns and "expenses" in df.columns:
            profit = df["revenue"].sum() - df["expenses"].sum()
            margin = (profit / df["revenue"].sum()) * 100
            cards["Net Profit"] = f"₹{profit:,.0f}"
            cards["Profit Margin"] = f"{margin:.1f}%"

    # ---------- GENERAL ----------
    else:
        num = df.select_dtypes(include="number")
        if len(num.columns) >= 2:
            corr = num.corr().abs()
            strongest = (
                corr.where(~np.eye(corr.shape[0], dtype=bool))
                .stack()
                .idxmax()
            )
            cards["Strongest Relationship"] = f"{strongest[0]} ↔ {strongest[1]}"

        var_col = num.std().idxmax()
        cards["Most Changing Column"] = var_col.title()

    return cards


# =============================================================
# GPT CALL FOR AUTO EDA
# =============================================================
def ask_gpt_for_analysis(df):
    SAMPLE = df.head(40).to_csv(index=False)

    prompt = f"""
You are a data analyst assistant. Based on this sample dataset:

{SAMPLE}

1. Write Python code to clean the data safely.
2. Write Python code to generate 3-5 meaningful charts using plotly.
3. Write easy-to-understand insights for small business owners. No jargon.
4. Return output strictly in JSON with keys:
   - cleaning_code
   - eda_code
   - insights
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_text = res.choices[0].message.content

    # Remove accidental markdown fences
    cleaned = raw_text.replace("```json", "").replace("```", "")

    try:
        return json.loads(cleaned)
    except:
        raise ValueError("GPT returned invalid JSON:\n\n" + raw_text)


# =============================================================
# EXECUTE GPT-GENERATED CODE SAFELY
# =============================================================
def run_dynamic_code(df, code, func_name):
    local_vars = {}
    exec(code, {"df": df, "px": px, "pd": pd, "np": np}, local_vars)
    return local_vars[func_name](df)


# ================================
# SIDEBAR: File Upload
# ================================
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("👆 Upload a dataset to begin.")
    st.stop()


# ================================
# LOAD + CLEAN
# ================================
if uploaded_file.name.endswith(".csv"):
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_excel(uploaded_file)

df_clean = auto_clean_df(df_raw)


# =============================================================
# SECTION: OVERVIEW (Dynamic Executive Cards)
# =============================================================
st.subheader("📌 Overview")

cards = make_overview_cards(df_clean)
cols = st.columns(len(cards))

for (label, value), col in zip(cards.items(), cols):
    with col:
        st.markdown(
            f"""
            <div style="padding:18px; border-radius:12px; background:#10141a; border:1px solid #1f2937;">
                <div style="font-size:14px; color:#9ca3af;">{label}</div>
                <div style="font-size:22px; font-weight:600; margin-top:6px; color:white;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================
# RAW + CLEANED PREVIEW
# =============================================================
st.subheader("📄 Raw Data")
st.dataframe(df_raw.head(50), use_container_width=True)

st.subheader("🧹 Cleaned Data")
st.dataframe(df_clean.head(50), use_container_width=True)


# =============================================================
# GPT AUTO EDA
# =============================================================
st.subheader("🤖 GPT Auto Analysis")

if st.button("Run GPT Analysis"):
    with st.spinner("Running GPT…"):
        gpt = ask_gpt_for_analysis(df_clean)

    st.success("AI analysis complete.")

    # Insights
    st.subheader("📘 Insights (Plain English)")
    st.write(gpt["insights"])

    # Execute Cleaning & EDA Code
    try:
        df2 = run_dynamic_code(df_clean, gpt["cleaning_code"], "clean_df")
    except:
        df2 = df_clean

    try:
        figures = run_dynamic_code(df2, gpt["eda_code"], "make_figures")

        st.subheader("📊 Charts")
        for f in figures.values():
            st.plotly_chart(f, use_container_width=True)
    except Exception as e:
        st.error("Chart code failed.")
        st.code(str(e))


# =============================================================
# ASK QUESTIONS ABOUT DATA
# =============================================================
st.subheader("💬 Ask Questions About This Data")
question = st.text_area("Example: 'Which channel is most efficient?' or 'Why did sales drop in March?'")

if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI thinking..."):
            prompt = f"Dataset:\n{df_clean.head(40).to_markdown()}\n\nQuestion: {question}\nExplain answer in simple English."
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
        st.write(res.choices[0].message.content)
