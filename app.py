# ============================================================
#  DataPilot 7.0 — Premium Analytics Workspace UI
#  NON-ChatGPT, SaaS-like Design
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import json

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="DataPilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    body { background-color: #0d1117; color: white; }
    .section-title {
        font-size: 22px; 
        font-weight: 600; 
        margin-top: 25px;
        margin-bottom: 10px;
        color: #e5e7eb;
    }
    .kpi-card {
        padding:18px; 
        border-radius:12px; 
        background:#111827; 
        border:1px solid #1f2937;
        height:100px;
    }
    .kpi-label {
        font-size:14px; 
        color:#9ca3af;
    }
    .kpi-value {
        font-size:28px; 
        font-weight:650;
        margin-top:6px; 
        color:white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("<h1 style='color:white;'>DataPilot — Business Insights Workspace</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9ca3af;'>Upload your data → Clean → Analyze → Get plain-English insights.</p>", unsafe_allow_html=True)


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Overview", "Data Explorer", "AI Visualizations", "Insights", "Ask Your Data"],
    index=0
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
if page == "Upload Data":
    st.markdown("<div class='section-title'>Upload Your Dataset</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded:
        st.session_state["raw"] = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.success("File uploaded successfully! Now go to 'Overview'.")


if "raw" not in st.session_state:
    st.stop()

df_raw = st.session_state["raw"]


# ------------------------------------------------------------
# CLEANING + SEMANTICS
# ------------------------------------------------------------
def clean(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = (df[col]
                       .astype(str)
                       .str.replace(",", "")
                       .str.replace("$", "")
                       .str.replace("₹", "")
                       .str.strip())
            df[col] = pd.to_numeric(cleaned, errors="ignore")
        if any(x in col.lower() for x in ["date", "week", "day"]):
            df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

df_clean = clean(df_raw)


# ------------------------------------------------------------
# PAGE: OVERVIEW (KPI CARDS)
# ------------------------------------------------------------
if page == "Overview":
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)

    num_df = df_clean.select_dtypes(include="number")

    if len(num_df.columns) == 0:
        st.warning("No numeric columns detected.")
    else:
        metrics = {
            "Total of Main Metric": num_df.iloc[:,0].sum(),
            "Average Value": num_df.iloc[:,0].mean(),
            "Maximum Value": num_df.iloc[:,0].max(),
        }

        cols = st.columns(len(metrics))

        for (label, value), col in zip(metrics.items(), cols):
            with col:
                st.markdown(
                    f"""
                    <div class='kpi-card'>
                        <div class='kpi-label'>{label}</div>
                        <div class='kpi-value'>{value:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ------------------------------------------------------------
# PAGE: DATA EXPLORER
# ------------------------------------------------------------
if page == "Data Explorer":
    st.markdown("<div class='section-title'>Raw Data</div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.markdown("<div class='section-title'>Cleaned Data</div>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(100), use_container_width=True)


# ------------------------------------------------------------
# PAGE: AI VISUALIZATIONS
# ------------------------------------------------------------
if page == "AI Visualizations":
    st.markdown("<div class='section-title'>Recommended Visuals</div>", unsafe_allow_html=True)

    sample = df_clean.head(40).to_csv(index=False)

    prompt = f"""
You are a data visualization engine.
Based on this dataset:

{sample}

Return only valid JSON with:
- code for a function make_figures(df)
- create 2–3 clean Plotly charts that are easy for small business owners.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    j = json.loads(res.choices[0].message.content.replace("```json", "").replace("```", ""))

    figs = {}
    exec(j["eda_code"], {"px": px, "pd": pd, "np": np}, figs)
    out = figs["make_figures"](df_clean)

    for f in out.values():
        st.plotly_chart(f, use_container_width=True)


# ------------------------------------------------------------
# PAGE: INSIGHTS
# ------------------------------------------------------------
if page == "Insights":
    st.markdown("<div class='section-title'>Key Takeaways</div>", unsafe_allow_html=True)

    sample = df_clean.head(40).to_csv(index=False)

    prompt = f"""
Explain this data to a small business owner in simple English.
Avoid jargon. Be specific and helpful.

Data:
{sample}

Return 6–8 bullet points.
"""

    ans = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.markdown(ans.choices[0].message.content)


# ------------------------------------------------------------
# PAGE: ASK YOUR DATA
# ------------------------------------------------------------
if page == "Ask Your Data":
    st.markdown("<div class='section-title'>Ask Your Data</div>", unsafe_allow_html=True)

    question = st.text_input("Type a business question (e.g., 'Which channel works best?')")

    if question:
        sample = df_clean.head(40).to_csv(index=False)

        qprompt = f"""
Answer the question in plain English. 
Dataset:
{sample}

Question: {question}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": qprompt}]
        )

        st.markdown(resp.choices[0].message.content)
